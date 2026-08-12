"""같은 leaf image에 대해 원논문식 taxonomy 경로의 Prompt-CAM을 나란히 저장한다.

논문 taxonomy 실험의 핵심은 root에서 leaf로 내려가면서 *같은 이미지*에 대해
서로 다른 node-local prompt가 어떤 trait를 보는지 비교하는 것이다. 이 스크립트는
training summary 또는 checkpoint root에서 node 모델을 복원해 true taxonomy path의
각 trainable node에 대한 top Prompt-CAM heads와 평균 overlay를 생성한다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import datasets


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset.imagefolder import JointImageTransform, load_taxonomy_manifest  # noqa: E402
from data.original_taxonomy import TaxonomyNodeSpec, list_taxonomy_nodes  # noqa: E402
from evaluation.checkpoints import (  # noqa: E402
    _build_model,
    _checkpoint_records,
    _discover_checkpoints,
    _resolve_project_path,
    _validate_checkpoint_compatibility,
    _validate_checkpoint_taxonomy_mappings,
)


def _resolve_species_index(taxonomy: Any, value: str | int) -> int:
    text = str(value).strip()
    if text.isdigit():
        index = int(text)
        if 0 <= index < len(taxonomy.class_names):
            return index
    folded = text.casefold()
    matches = []
    for index, (folder_name, scientific_name) in enumerate(
        zip(taxonomy.class_names, taxonomy.scientific_names)
    ):
        if folded in {str(folder_name).casefold(), str(scientific_name).casefold()}:
            matches.append(index)
    if len(matches) != 1:
        raise ValueError(
            f"species {value!r}를 고유하게 찾지 못했습니다. "
            "folder_name, scientific_name 또는 0-based index를 사용하십시오"
        )
    return matches[0]


def _true_path_nodes(taxonomy: Any, species_index: int) -> list[TaxonomyNodeSpec]:
    all_nodes = list_taxonomy_nodes(taxonomy, trainable_only=False)
    genus_index = int(taxonomy.species_to_genus[species_index])
    family_index = int(taxonomy.genus_to_family[genus_index])
    family_name = taxonomy.family_names[family_index]
    genus_name = taxonomy.genus_names[genus_index]

    result = []
    for rank, name in (("root", "root"), ("family", family_name), ("genus", genus_name)):
        matching = [
            node
            for node in all_nodes
            if node.rank == rank and (rank == "root" or node.name == name)
        ]
        if len(matching) != 1:
            raise RuntimeError(f"taxonomy path node를 고유하게 찾지 못했습니다: {rank}:{name}")
        result.append(matching[0])
    return result


def _denormalize_image(tensor: torch.Tensor, transform: JointImageTransform) -> np.ndarray:
    mean = torch.as_tensor(transform.mean, dtype=tensor.dtype, device=tensor.device)[:, None, None]
    std = torch.as_tensor(transform.std, dtype=tensor.dtype, device=tensor.device)[:, None, None]
    image = (tensor * std + mean).clamp(0.0, 1.0)
    return (
        image.detach().float().cpu().permute(1, 2, 0).numpy() * 255.0
    ).round().astype(np.uint8)


def _overlay(image_rgb: np.ndarray, attention: torch.Tensor, patch_size: int) -> np.ndarray:
    patch_count = int(attention.numel())
    height, width = image_rgb.shape[:2]
    grid_height = height // int(patch_size)
    grid_width = width // int(patch_size)
    if grid_height * grid_width != patch_count:
        raise ValueError(
            f"attention patch 수가 이미지 grid와 다릅니다: "
            f"{patch_count} != {grid_height}x{grid_width}"
        )
    heat = attention.detach().float().cpu().reshape(grid_height, grid_width).numpy()
    heat = np.clip(heat, 0.0, None)
    maximum = float(heat.max())
    if maximum > 1e-8:
        heat = heat / maximum
    heat = cv2.resize(heat, (width, height), interpolation=cv2.INTER_CUBIC)
    heat = np.clip(heat, 0.0, 1.0)
    heat_color = cv2.applyColorMap((heat * 255.0).astype(np.uint8), cv2.COLORMAP_JET)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    blended = cv2.addWeighted(image_bgr, 0.5, heat_color, 0.5, 0.0)
    return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)


def _labeled_panel(image: Image.Image, title: str, width: int = 360) -> Image.Image:
    image = image.convert("RGB")
    scale = width / image.width
    resized = image.resize((width, max(1, round(image.height * scale))), Image.Resampling.BICUBIC)
    header = 42
    panel = Image.new("RGB", (width, resized.height + header), "white")
    panel.paste(resized, (0, header))
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()
    draw.text((8, 8), title[:72], fill="black", font=font)
    return panel


def _save_montage(panels: list[tuple[str, Image.Image]], path: Path) -> None:
    labeled = [_labeled_panel(image, title) for title, image in panels]
    width = max(panel.width for panel in labeled)
    height = sum(panel.height for panel in labeled)
    montage = Image.new("RGB", (width, height), "white")
    offset = 0
    for panel in labeled:
        montage.paste(panel, (0, offset))
        offset += panel.height
    montage.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="동일 이미지의 원논문식 taxonomy path Prompt-CAM 시각화"
    )
    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--species",
        default=None,
        help="folder_name, scientific_name 또는 0-based index. 생략하면 이미지 부모 폴더명 사용",
    )
    parser.add_argument("--checkpoint", action="append", default=[])
    parser.add_argument("--checkpoint-root", default=None)
    parser.add_argument("--training-summary", default=None)
    parser.add_argument(
        "--duplicate-policy", choices=["latest", "error"], default="latest"
    )
    parser.add_argument("--top-traits", type=int, default=2)
    parser.add_argument("--device", default=None)
    parser.add_argument("--require-correct", action="store_true")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    image_path = Path(args.image).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"이미지가 없습니다: {image_path}")

    checkpoint_paths = _discover_checkpoints(
        args.checkpoint,
        args.checkpoint_root,
        args.training_summary,
    )
    records, duplicate_selection = _checkpoint_records(
        checkpoint_paths,
        duplicate_policy=args.duplicate_policy,
    )
    _validate_checkpoint_compatibility(records)
    if "root" not in records:
        raise ValueError("taxonomy path 시각화에는 root checkpoint가 필요합니다")

    config = dict(records["root"]["config"])
    data_root = _resolve_project_path(config.get("resolved_data_path", config["data_path"]))
    train_split = str(config.get("train_split", "train"))
    train_base = datasets.ImageFolder(str(data_root / train_split), transform=None)
    taxonomy = load_taxonomy_manifest(
        _resolve_project_path(config["taxonomy_manifest"]),
        train_base.classes,
        class_column=config.get("taxonomy_class_column"),
    )

    species_value = args.species if args.species is not None else image_path.parent.name
    species_index = _resolve_species_index(taxonomy, species_value)
    path_nodes = _true_path_nodes(taxonomy, species_index)
    required_ids = {node.node_id for node in path_nodes if node.trainable}
    missing = sorted(required_ids - set(records))
    if missing:
        raise ValueError(f"true taxonomy path의 checkpoint가 누락되었습니다: {missing}")

    _validate_checkpoint_taxonomy_mappings(
        records,
        [node for node in path_nodes if node.trainable],
    )

    params = SimpleNamespace(**config)
    transform = JointImageTransform(params, training=False)
    pil_image = Image.open(image_path).convert("RGB")
    image_tensor, _, _ = transform(
        pil_image,
        bbox=None,
        bbox_coordinate_mode="normalized",
    )
    image_rgb = _denormalize_image(image_tensor, transform)
    batch = image_tensor.unsqueeze(0)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    batch = batch.to(device)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_rgb).save(output_dir / "transformed_input.png")

    # timm 의존 모듈은 실제 모델 시각화가 시작될 때만 불러온다.
    from evaluation.cam.species import (  # noqa: WPS433
        _greedy_promptcam_top_heads,
        _species_head_maps,
        _species_logits,
    )

    metadata: dict[str, Any] = {
        "image": str(image_path),
        "species_index": species_index,
        "folder_name": taxonomy.class_names[species_index],
        "scientific_name": taxonomy.scientific_names[species_index],
        "duplicate_checkpoint_selection": duplicate_selection,
        "path": [],
    }
    montage_panels: list[tuple[str, Image.Image]] = [
        (f"input: {taxonomy.scientific_names[species_index]}", Image.fromarray(image_rgb))
    ]

    for node in path_nodes:
        local_target = node.local_target(species_index)
        node_record: dict[str, Any] = {
            "node": node.to_dict(),
            "local_target": local_target,
            "local_target_name": node.child_names[local_target],
            "trainable": node.trainable,
        }
        if not node.trainable:
            node_record.update(
                {
                    "checkpoint": None,
                    "prediction": local_target,
                    "prediction_name": node.child_names[local_target],
                    "correct": True,
                    "note": "singleton direct child: deterministic path, no learned Prompt-CAM",
                }
            )
            metadata["path"].append(node_record)
            continue

        record = records[node.node_id]
        model = _build_model(record, device)
        with torch.inference_mode():
            output, _ = model(batch)
            logits = _species_logits(output)
            prediction = int(logits.argmax(dim=1).item())
        correct = prediction == local_target
        if args.require_correct and not correct:
            raise RuntimeError(
                f"node {node.display_name}에서 이미지를 정확히 분류하지 못했습니다: "
                f"target={node.child_names[local_target]}, "
                f"prediction={node.child_names[prediction]}"
            )

        head_count = int(getattr(model, "num_heads", getattr(model.module, "num_heads", 0) if hasattr(model, "module") else 0))
        top_traits = min(int(args.top_traits), head_count)
        if top_traits <= 0:
            raise ValueError("top-traits는 양수이고 attention head 수 이하여야 합니다")
        selection = _greedy_promptcam_top_heads(
            model,
            batch,
            target_class=local_target,
            top_traits=top_traits,
        )
        with torch.inference_mode():
            pruned_output, pruned_attention = model(
                batch,
                blur_head_lst=selection["blurred_heads"],
                target_cls=local_target,
            )
        maps = _species_head_maps(
            pruned_output,
            pruned_attention,
            local_target,
            SimpleNamespace(vpt_num=node.num_children),
            sample_index=0,
        )
        selected = maps.index_select(
            1,
            torch.as_tensor(selection["selected_heads"], dtype=torch.long, device=maps.device),
        )

        node_dir = output_dir / node.node_id
        node_dir.mkdir(parents=True, exist_ok=True)
        overlays = []
        for map_index, head_index in enumerate(selection["selected_heads"]):
            overlay = _overlay(image_rgb, selected[0, map_index], int(config.get("patch_size", 14)))
            path = node_dir / f"head_{head_index + 1:02d}.png"
            Image.fromarray(overlay).save(path)
            overlays.append(overlay)
        mean_overlay = _overlay(
            image_rgb,
            selected.mean(dim=1)[0],
            int(config.get("patch_size", 14)),
        )
        mean_path = node_dir / "selected_heads_mean.png"
        Image.fromarray(mean_overlay).save(mean_path)
        montage_panels.append(
            (
                f"{node.display_name} -> {node.child_names[local_target]} "
                f"(pred={node.child_names[prediction]})",
                Image.fromarray(mean_overlay),
            )
        )
        node_record.update(
            {
                "checkpoint": str(record["path"]),
                "prediction": prediction,
                "prediction_name": node.child_names[prediction],
                "correct": correct,
                "selected_heads_zero_based": selection["selected_heads"],
                "selected_heads_one_based": [head + 1 for head in selection["selected_heads"]],
                "mean_overlay": str(mean_path),
            }
        )
        metadata["path"].append(node_record)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    _save_montage(montage_panels, output_dir / "taxonomy_path_montage.png")
    (output_dir / "taxonomy_path_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"시각화 저장: {output_dir}")


if __name__ == "__main__":
    main()
