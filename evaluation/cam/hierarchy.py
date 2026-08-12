"""동일 이미지에서 family/genus/species Prompt-CAM top-K를 비교한다."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from timm.utils import get_outdir

from data.loader import get_loader
from model.factory import get_model
from utils.log_utils import logging_env_setup
from utils.setup_logging import get_logger


logger = get_logger("Prompt_CAM")
_LEVELS = ("family", "genus", "species")
_LEVEL_LABELS = {"family": "과", "genus": "속", "species": "종"}


def _core_model(model):
    return model.module if hasattr(model, "module") else model


def _unpack_batch(batch, device):
    if isinstance(batch, dict):
        return (
            batch["image"].to(device),
            batch["species_target"].to(device),
        )
    images, targets = batch
    return images.to(device), targets.to(device)


def _target_indices(model, species_index: int) -> dict[str, int]:
    decoder = _core_model(model).hierarchical_head
    species_index = int(species_index)
    genus_index = int(
        decoder.species_to_genus[species_index].item()
    )
    family_index = int(
        decoder.genus_to_family[genus_index].item()
    )
    return {
        "species": species_index,
        "genus": genus_index,
        "family": family_index,
    }


def _level_probabilities(
    output: dict[str, torch.Tensor],
    level: str,
    *,
    conditional: bool = False,
):
    if conditional and level == "genus":
        key = "genus_conditional_probabilities"
    elif conditional and level == "species":
        key = "species_conditional_probabilities"
    else:
        key = f"{level}_probabilities"

    if key not in output:
        raise KeyError(f"모델 출력에 {key}가 없습니다")
    return output[key]


def _level_prediction(output: dict[str, torch.Tensor], level: str) -> int:
    # 정확도 필터에는 taxonomy 전체에서 비교 가능한 전역 확률을 사용한다.
    return int(
        _level_probabilities(
            output,
            level,
            conditional=False,
        )[0].argmax().item()
    )


def _greedy_top_heads(
    model,
    image: torch.Tensor,
    *,
    level: str,
    target_index: int,
    top_traits: int,
) -> dict[str, Any]:
    """node-local 확률 감소량으로 중요한 Prompt-CAM head를 선택한다."""
    level = str(level).lower()
    if level not in _LEVELS:
        raise ValueError(
            f"지원하지 않는 계층 수준입니다: {level}"
        )

    decoder = _core_model(model).hierarchical_head
    num_heads = int(decoder.num_heads)
    top_traits = int(top_traits)
    if not 1 <= top_traits <= num_heads:
        raise ValueError(
            f"top_traits는 [1,{num_heads}] 범위여야 하지만 "
            f"{top_traits}입니다"
        )

    requested_level = level
    requested_target_index = int(target_index)
    effective_level = requested_level
    effective_target_index = requested_target_index
    contrast_defined = True

    # 자식이 하나뿐인 node에서는 조건부 확률이 항상 1이므로
    # 해당 수준의 head 중요도를 정의할 수 없다. 이 경우 부모 node의
    # CAM/head intervention을 사용하고 메타데이터에 이를 명시한다.
    if requested_level == "species":
        parent_genus = int(
            decoder.species_to_genus[requested_target_index].item()
        )
        sibling_count = int(
            decoder.genus_counts[parent_genus].item()
        )
        if sibling_count == 1:
            effective_level = "genus"
            effective_target_index = parent_genus
            contrast_defined = False

    elif requested_level == "genus":
        parent_family = int(
            decoder.genus_to_family[requested_target_index].item()
        )
        sibling_count = int(
            decoder.family_counts[parent_family].item()
        )
        if sibling_count == 1:
            effective_level = "family"
            effective_target_index = parent_family
            contrast_defined = False

    remaining = list(range(num_heads))
    blurred: list[int] = []
    steps: list[dict[str, Any]] = []

    while len(remaining) > top_traits:
        candidate_probs: dict[int, float] = {}

        for candidate in remaining:
            output, _ = model(
                image,
                blur_head_lst=blurred + [candidate],
                target_cls=effective_target_index,
                target_level=effective_level,
            )

            if effective_level == "family":
                probability = output[
                    "family_probabilities"
                ][0, effective_target_index]
            elif effective_level == "genus":
                probability = output[
                    "genus_conditional_probabilities"
                ][0, effective_target_index]
            else:
                probability = output[
                    "species_conditional_probabilities"
                ][0, effective_target_index]

            candidate_probs[candidate] = float(
                probability.item()
            )

        # blur 후 정답 확률이 가장 높게 남는 head가 가장 덜 중요하다.
        least_important = max(
            remaining,
            key=lambda head: (
                candidate_probs[head],
                -head,
            ),
        )

        steps.append(
            {
                "step": len(steps) + 1,
                "removed_head_zero_based": least_important,
                "removed_head_one_based": least_important + 1,
                "target_probability_after_blur": candidate_probs[
                    least_important
                ],
                "candidate_probabilities_one_based": {
                    str(head + 1): candidate_probs[head]
                    for head in remaining
                },
            }
        )

        blurred.append(least_important)
        remaining.remove(least_important)

    return {
        "selected_heads": remaining,
        "blurred_heads": blurred,
        "pruning_steps": steps,
        "contrast_defined": contrast_defined,
        "requested_level": requested_level,
        "requested_target_index": requested_target_index,
        "effective_level": effective_level,
        "effective_target_index": effective_target_index,
    }


def _clean_level_maps(
    output: dict[str, torch.Tensor],
    model,
    *,
    level: str,
    target_index: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """blur되지 않은 계층별 head map [1,H,P]를 반환한다."""
    if level == "species":
        maps = output["species_attention_heads"][
            0:1,
            :,
            int(target_index),
            :,
        ]
        return maps, {}

    if level == "genus":
        maps = output["genus_attention_heads"][
            0:1,
            :,
            int(target_index),
            :,
        ]
        return maps, {}

    if level == "family":
        maps = output["family_attention_heads"][
            0:1,
            :,
            int(target_index),
            :,
        ]
        return maps, {}

    raise ValueError(f"지원하지 않는 계층 수준입니다: {level}")


def _to_rgb_image(image_tensor: torch.Tensor) -> np.ndarray:
    image = (
        image_tensor[0]
        .detach()
        .float()
        .cpu()
        .permute(1, 2, 0)
        .numpy()
    )
    low = float(image.min())
    high = float(image.max())
    if high > low:
        image = (image - low) / (high - low)
    else:
        image = np.zeros_like(image)
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def _overlay(
    image_rgb: np.ndarray,
    attention: np.ndarray,
) -> np.ndarray:
    attention = np.asarray(attention, dtype=np.float32)
    resized = cv2.resize(
        attention,
        (image_rgb.shape[1], image_rgb.shape[0]),
        interpolation=cv2.INTER_CUBIC,
    )
    low = float(resized.min())
    high = float(resized.max())
    if high > low:
        normalized = (resized - low) / (high - low)
    else:
        normalized = np.zeros_like(resized)
    normalized = cv2.GaussianBlur(normalized, (9, 9), 0)
    heatmap_bgr = cv2.applyColorMap(
        np.clip(normalized * 255.0, 0, 255).astype(np.uint8),
        cv2.COLORMAP_JET,
    )
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    result_bgr = cv2.addWeighted(
        image_bgr,
        0.5,
        heatmap_bgr,
        0.5,
        0.0,
    )
    return cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)


def _font(size: int):
    candidates = (
        "C:/Windows/Fonts/malgunbd.ttf",
        "C:/Windows/Fonts/malgun.ttf",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    )
    for candidate in candidates:
        if os.path.exists(candidate):
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _captioned_tile(
    image_rgb: np.ndarray,
    caption: str,
    *,
    tile_size: int = 256,
    caption_height: int = 30,
) -> Image.Image:
    image = Image.fromarray(image_rgb).resize(
        (tile_size, tile_size),
        Image.Resampling.BICUBIC,
    )
    tile = Image.new(
        "RGB",
        (tile_size, tile_size + caption_height),
        "white",
    )
    tile.paste(image, (0, caption_height))
    draw = ImageDraw.Draw(tile)
    draw.text(
        (8, 5),
        caption,
        fill="black",
        font=_font(17),
    )
    return tile


def _save_comparison(
    image: torch.Tensor,
    level_maps: dict[str, torch.Tensor],
    selections: dict[str, dict[str, Any]],
    targets: dict[str, int],
    path: Path,
) -> None:
    image_rgb = _to_rgb_image(image)
    rows: list[Image.Image] = []

    for level in _LEVELS:
        selected_heads = selections[level]["selected_heads"]
        maps = level_maps[level]
        if maps.shape[1] != len(selected_heads):
            raise RuntimeError(
                f"{level}: 맵 수 {maps.shape[1]}가 선택한 헤드 수 "
                f"{len(selected_heads)}와 다릅니다"
            )

        tiles = [
            _captioned_tile(
                image_rgb,
                f"{_LEVEL_LABELS[level]} 정답={targets[level]}",
            )
        ]
        patch_count = int(maps.shape[-1])
        grid_size = int(round(patch_count ** 0.5))
        if grid_size * grid_size != patch_count:
            raise RuntimeError(
                f"패치 수 {patch_count}로 정사각형 격자를 만들 수 없습니다"
            )

        for map_index, head_index in enumerate(selected_heads):
            attention = (
                maps[0, map_index]
                .reshape(grid_size, grid_size)
                .detach()
                .float()
                .cpu()
                .numpy()
            )
            overlay = _overlay(image_rgb, attention)
            tiles.append(
                _captioned_tile(
                    overlay,
                    f"헤드 {head_index + 1}",
                )
            )

        row_width = sum(tile.width for tile in tiles)
        row_height = max(tile.height for tile in tiles)
        row = Image.new("RGB", (row_width, row_height), "white")
        x_offset = 0
        for tile in tiles:
            row.paste(tile, (x_offset, 0))
            x_offset += tile.width
        rows.append(row)

    canvas = Image.new(
        "RGB",
        (
            max(row.width for row in rows),
            sum(row.height for row in rows),
        ),
        "white",
    )
    y_offset = 0
    for row in rows:
        canvas.paste(row, (0, y_offset))
        y_offset += row.height

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=95)


def _save_metadata(
    path: Path,
    *,
    targets: dict[str, int],
    predictions: dict[str, int],
    selections: dict[str, dict[str, Any]],
    extra: dict[str, dict[str, Any]],
) -> None:
    payload = {
        "method": (
            "과·속·종 수준에 각각 적용한 Prompt-CAM 누적 탐욕적 "
            "균등 어텐션 가지치기"
        ),
        "targets_zero_based": targets,
        "predictions_zero_based": predictions,
        "levels": {},
    }
    for level in _LEVELS:
        payload["levels"][level] = {
            "selected_heads_zero_based": selections[level][
                "selected_heads"
            ],
            "selected_heads_one_based": [
                head + 1
                for head in selections[level]["selected_heads"]
            ],
            "blurred_heads_zero_based": selections[level][
                "blurred_heads"
            ],
            "blurred_heads_one_based": [
                head + 1
                for head in selections[level]["blurred_heads"]
            ],
            "pruning_steps": selections[level]["pruning_steps"],
            **extra[level],
        }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def basic_hierarchy_vis(params) -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not bool(getattr(params, "hierarchical_prompt", False)):
        raise ValueError(
            "계층 CAM 비교에는 hierarchical_prompt=True가 필요합니다"
        )

    target_species = int(params.vis_cls)
    requested_samples = int(params.nmbr_samples)
    top_traits = int(params.top_traits)

    dataset_name = params.data.split("-")[0]
    output_dir = os.path.join(
        params.vis_outdir,
        params.model,
        dataset_name,
        f"class_{target_species}",
        f"family_genus_species_top_{top_traits}",
    )
    params.output_dir = get_outdir(output_dir)
    logging_env_setup(params)

    _, _, test_loader = get_loader(params, logger)
    model, _, _ = get_model(params, visualize=True)
    checkpoint = torch.load(
        params.checkpoint,
        map_location="cpu",
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(params.device).eval()

    targets = _target_indices(model, target_species)
    saved = 0
    skipped = 0

    with torch.no_grad():
        for batch in test_loader:
            images, species_targets = _unpack_batch(
                batch,
                params.device,
            )
            indices = torch.nonzero(
                species_targets.eq(target_species),
                as_tuple=False,
            ).flatten()
            if indices.numel() == 0:
                continue

            clean_batch_output, _ = model(images)

            for index_tensor in indices:
                batch_index = int(index_tensor.item())
                clean_output = {
                    key: (
                        value[batch_index : batch_index + 1]
                        if isinstance(value, torch.Tensor)
                        and value.ndim > 0
                        and value.shape[0] == images.shape[0]
                        else value
                    )
                    for key, value in clean_batch_output.items()
                }
                predictions = {
                    level: _level_prediction(clean_output, level)
                    for level in _LEVELS
                }

                if any(
                    predictions[level] != targets[level]
                    for level in _LEVELS
                ):
                    skipped += 1
                    continue

                image = images[batch_index : batch_index + 1]
                selections: dict[str, dict[str, Any]] = {}
                selected_maps: dict[str, torch.Tensor] = {}
                extra: dict[str, dict[str, Any]] = {}

                for level in _LEVELS:
                    selection = _greedy_top_heads(
                        model,
                        image,
                        level=level,
                        target_index=targets[level],
                        top_traits=top_traits,
                    )
                    clean_single_output, _ = model(image)
                    clean_maps, level_extra = _clean_level_maps(
                        clean_single_output,
                        model,
                        level=level,
                        target_index=targets[level],
                    )
                    index_tensor_heads = torch.as_tensor(
                        selection["selected_heads"],
                        device=clean_maps.device,
                        dtype=torch.long,
                    )
                    selected_maps[level] = clean_maps.index_select(
                        1,
                        index_tensor_heads,
                    )
                    selections[level] = selection
                    extra[level] = level_extra

                saved += 1
                folder = Path(params.output_dir) / f"img_{saved}"
                _save_comparison(
                    image,
                    selected_maps,
                    selections,
                    targets,
                    folder / "hierarchy_comparison.jpg",
                )
                _save_metadata(
                    folder / "hierarchy_top_heads.json",
                    targets=targets,
                    predictions=predictions,
                    selections=selections,
                    extra=extra,
                )

                logger.info(
                    "이미지 %d의 헤드: 과=%s 속=%s 종=%s",
                    saved,
                    [h + 1 for h in selections["family"]["selected_heads"]],
                    [h + 1 for h in selections["genus"]["selected_heads"]],
                    [h + 1 for h in selections["species"]["selected_heads"]],
                )

                if saved >= requested_samples:
                    break

            if saved >= requested_samples:
                break

    logger.info(
        "계층 CAM 비교 %d개를 저장했고 세 수준을 모두 맞히지 못한 "
        "표본 %d개를 건너뛰었습니다",
        saved,
        skipped,
    )
    if saved < requested_samples:
        logger.warning(
            "종 %d에서 세 수준을 모두 맞힌 표본을 %d개만 찾았습니다",
            target_species,
            saved,
        )
