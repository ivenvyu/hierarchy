#!/usr/bin/env python3
"""원논문 Prompt-CAM 방식으로 동일 이미지의 attention heads를 비교한다.

원논문의 시각화 정의를 그대로 따른다.

1. target class prompt가 patch tokens에 배분한 마지막 attention layer의
   head별 attention vector를 사용한다.
2. 각 greedy step에서 아직 남은 head를 하나씩 uniform attention으로 바꾸고,
   target class probability를 가장 높게 유지하는 head를 가장 덜 중요하다고 본다.
3. 누적 greedy ranking의 마지막 top-K heads를 개별 trait map으로 표시한다.
4. 각 attention map은 공식 구현처럼 독립 min-max, bicubic resize,
   9x9 Gaussian blur, JET colormap, alpha=0.5로 렌더링한다.

Flat 및 independent-taxonomy 모델은 ViT 마지막 self-attention의
class-prompt -> image-patch attention을 사용한다. Shared hierarchical 모델은
동일한 원리를 family/genus/species decoder의 query -> patch cross-attention에
적용한다. 후자는 Prompt-CAM 원리를 계층 decoder에 확장한 것이며 원논문에
존재하는 동일 architecture라고 주장하지 않는다.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

import cv2
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
from torchvision import datasets


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset.imagefolder import JointImageTransform  # noqa: E402


def _load_support_module(
    *,
    module_names: Sequence[str],
    relative_paths: Sequence[str],
    label: str,
):
    """프로젝트 패키지 또는 파일 경로에서 평가기 모듈을 불러온다."""

    import importlib
    import importlib.util

    errors: list[str] = []

    for module_name in module_names:
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            # 후보 모듈 자체가 없을 때만 다음 후보로 간다. 후보 내부의 다른
            # dependency가 누락된 경우에는 원인을 숨기지 않고 기록한다.
            errors.append(f"{module_name}: {exc}")
        except Exception as exc:  # pragma: no cover - 실행 환경 진단용
            errors.append(f"{module_name}: {type(exc).__name__}: {exc}")

    for relative_path in relative_paths:
        module_path = (PROJECT_ROOT / relative_path).resolve()
        if not module_path.is_file():
            errors.append(f"{module_path}: file not found")
            continue

        synthetic_name = (
            "_promptcam_support_"
            + re.sub(r"[^a-zA-Z0-9_]+", "_", module_path.stem)
            + "_"
            + str(abs(hash(str(module_path))))
        )
        spec = importlib.util.spec_from_file_location(synthetic_name, module_path)
        if spec is None or spec.loader is None:
            errors.append(f"{module_path}: import spec 생성 실패")
            continue

        module = importlib.util.module_from_spec(spec)
        sys.modules[synthetic_name] = module
        try:
            spec.loader.exec_module(module)
            return module
        except Exception as exc:
            sys.modules.pop(synthetic_name, None)
            errors.append(f"{module_path}: {type(exc).__name__}: {exc}")

    detail = "\n  - ".join(errors)
    raise ImportError(
        f"{label} 평가기 모듈을 찾거나 불러오지 못했습니다. 확인한 후보:\n"
        f"  - {detail}"
    )


_HIERARCHICAL_EVALUATOR = _load_support_module(
    module_names=(
        "evaluation.hierarchy",
    ),
    relative_paths=(
        "evaluation/hierarchy.py",
    ),
    label="hierarchical Prompt-CAM",
)

_load_hierarchical_model = _HIERARCHICAL_EVALUATOR._load_model
_load_hierarchical_taxonomy = _HIERARCHICAL_EVALUATOR._load_taxonomy
_load_hierarchical_yaml = _HIERARCHICAL_EVALUATOR._load_yaml
_prepare_hierarchical_params = _HIERARCHICAL_EVALUATOR._prepare_params


import importlib

_ORIGINAL_EVALUATOR = importlib.import_module(
    "evaluation.independent"
)

LoadedNode = _ORIGINAL_EVALUATOR.LoadedNode
TaxonomyTree = _ORIGINAL_EVALUATOR.TaxonomyTree
_discover_node_specs = _ORIGINAL_EVALUATOR._discover_node_specs
_extract_logits = _ORIGINAL_EVALUATOR._extract_logits
_load_node = _ORIGINAL_EVALUATOR._load_node
_resolve_path = _ORIGINAL_EVALUATOR._resolve_path
_taxonomy_from_manifest = _ORIGINAL_EVALUATOR._taxonomy_from_manifest


EPS = 1e-12
LEVELS = ("family", "genus", "species")


@dataclass(frozen=True)
class CanonicalTargets:
    species: int
    genus: int
    family: int


@dataclass
class HeadResult:
    model_name: str
    level: str
    target_index: int
    target_name: str
    prediction_index: int
    prediction_name: str
    correct: bool
    target_probability: float
    selected_heads: list[int]
    importance_order: list[int]
    blurred_heads: list[int]
    pruning_steps: list[dict[str, Any]]
    maps: np.ndarray  # [K, grid_h, grid_w]
    checkpoint: str
    attention_source: str
    note: str | None = None


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"YAML 최상위 값이 mapping이 아닙니다: {path}")
    return value


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.device(value)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA를 요청했지만 사용할 수 없습니다")
    return result


def _resolve_run_dir(path: str | None, *, required: bool, label: str) -> Path | None:
    if path in (None, "", "none", "null"):
        if required:
            raise ValueError(f"{label} run directory가 필요합니다")
        return None
    result = Path(path).expanduser().resolve()
    if not result.is_dir():
        raise FileNotFoundError(f"{label} run directory가 없습니다: {result}")
    return result


def _namespace_for_flat(args_data: Mapping[str, Any], run_dir: Path) -> SimpleNamespace:
    data = dict(args_data)
    data.update(
        {
            "distributed": False,
            "local_rank": 0,
            "load_pretrained_backbone": False,
            "promptcam_checkpoint": None,
            "resume": None,
            "store_ckp": False,
            "debug": False,
            "vis_attn": True,
            "amp_dtype": "none",
            "output_dir": str(run_dir),
        }
    )
    for field in ("data_path", "taxonomy_manifest", "identifiability_manifest"):
        value = data.get(field)
        if value in (None, "", "null"):
            continue
        candidate = Path(str(value)).expanduser()
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        data[field] = str(candidate.resolve())
    return SimpleNamespace(**data)


def _load_flat_model(run_dir: Path, device: torch.device):
    args_path = run_dir / "args.yaml"
    checkpoint_path = run_dir / "model.pt"
    if not args_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Flat run에는 args.yaml과 model.pt가 모두 필요합니다: {run_dir}"
        )
    args_data = _load_yaml(args_path)
    if bool(args_data.get("hierarchical_prompt", False)):
        raise ValueError("Flat run에 hierarchical_prompt=True가 설정되어 있습니다")
    if bool(args_data.get("original_taxonomy_prompt", False)):
        raise ValueError("Flat run이 original taxonomy node 실행입니다")
    params = _namespace_for_flat(args_data, run_dir)
    from model.factory import get_model

    model, _, _ = get_model(params, visualize=True)
    checkpoint = _torch_load(checkpoint_path)
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Flat checkpoint 구조가 모델과 다릅니다: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    model = model.to(device).eval()
    model.params.vis_attn = True
    return model, params, args_data, checkpoint_path


def _load_shared_model(run_dir: Path, device: torch.device):
    args_path = run_dir / "args.yaml"
    if not args_path.is_file():
        raise FileNotFoundError(f"Shared args.yaml이 없습니다: {args_path}")
    args_data = _load_hierarchical_yaml(args_path)
    dummy_cli = SimpleNamespace(batch_size=1, num_workers=0)
    params = _prepare_hierarchical_params(args_data, PROJECT_ROOT, run_dir, dummy_cli)
    params.amp_dtype = "none"
    params.vis_attn = False  # decoder attention은 output dict에서 직접 반환된다.
    model, _, checkpoint_path, _ = _load_hierarchical_model(
        PROJECT_ROOT,
        run_dir,
        params,
        device,
    )
    taxonomy = _load_hierarchical_taxonomy(run_dir, params)
    return model.eval(), params, args_data, checkpoint_path, taxonomy


def _load_independent_nodes(run_dir: Path, device: torch.device):
    root_config = run_dir / "configs" / "root.yaml"
    if not root_config.is_file():
        raise FileNotFoundError(f"Independent root config가 없습니다: {root_config}")
    root_data = _load_yaml(root_config)
    taxonomy_manifest = _resolve_path(
        root_data["taxonomy_manifest"],
        project_root=PROJECT_ROOT,
        base_dir=root_config.parent,
    )
    tree = _taxonomy_from_manifest(
        taxonomy_manifest,
        str(root_data.get("taxonomy_class_column", "folder_name")),
    )
    specs = _discover_node_specs(run_dir, tree)
    nodes: dict[str, LoadedNode] = {}
    for spec in specs:
        node = _load_node(
            spec,
            project_root=PROJECT_ROOT,
            tree=tree,
            device=device,
            allow_order_fallback=False,
        )
        node.params.amp_dtype = "none"
        node.params.vis_attn = True
        node.model.params.vis_attn = True
        nodes[spec.key] = node
    if "root" not in nodes:
        raise RuntimeError("Independent root node를 불러오지 못했습니다")
    return nodes, tree, root_data


def _preprocess_signature(args_data: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "crop_size",
        "eval_resize_size",
        "normalization",
        "normalization_mean",
        "normalization_std",
    )
    return {key: args_data.get(key) for key in keys}


def _validate_preprocessing(configs: Mapping[str, Mapping[str, Any]]) -> None:
    signatures = {name: _preprocess_signature(config) for name, config in configs.items()}
    reference_name, reference = next(iter(signatures.items()))
    mismatches = {name: sig for name, sig in signatures.items() if sig != reference}
    if mismatches:
        raise ValueError(
            "모델 간 전처리가 다릅니다. 동일 tensor 비교가 불가능합니다: "
            f"reference={reference_name}:{reference}, mismatches={mismatches}"
        )


def _validate_taxonomies(shared_taxonomy: Any, tree: TaxonomyTree) -> None:
    checks = {
        "class_names": (tuple(shared_taxonomy.class_names), tuple(tree.species_names)),
        "scientific_names": (
            tuple(shared_taxonomy.scientific_names),
            tuple(tree.scientific_names),
        ),
        "genus_names": (tuple(shared_taxonomy.genus_names), tuple(tree.genera)),
        "family_names": (tuple(shared_taxonomy.family_names), tuple(tree.families)),
        "species_to_genus": (
            tuple(shared_taxonomy.species_to_genus),
            tuple(tree.species_to_genus),
        ),
        "genus_to_family": (
            tuple(shared_taxonomy.genus_to_family),
            tuple(tree.genus_to_family),
        ),
    }
    bad = {key: pair for key, pair in checks.items() if pair[0] != pair[1]}
    if bad:
        raise ValueError(f"Shared와 independent taxonomy 순서가 다릅니다: {bad}")


def _validate_flat_class_order(flat_params: SimpleNamespace, tree: TaxonomyTree) -> None:
    data_root = Path(str(flat_params.data_path)).expanduser().resolve()
    train_split = str(getattr(flat_params, "train_split", "train"))
    base = datasets.ImageFolder(str(data_root / train_split), transform=None)
    if tuple(base.classes) != tuple(tree.species_names):
        raise ValueError(
            "Flat 출력 class 순서와 taxonomy species 순서가 다릅니다: "
            f"flat={base.classes}, taxonomy={tree.species_names}"
        )


def _resolve_species_index(tree: TaxonomyTree, value: str | None, image_path: Path) -> int:
    candidate = value if value not in (None, "") else image_path.parent.name
    text = str(candidate).strip()
    if text.isdigit():
        index = int(text)
        if 0 <= index < tree.num_species:
            return index
    folded = text.casefold()
    matches = []
    for index, (folder, scientific) in enumerate(
        zip(tree.species_names, tree.scientific_names)
    ):
        aliases = {
            folder.casefold(),
            scientific.casefold(),
            scientific.replace(" ", "_").casefold(),
        }
        if folded in aliases:
            matches.append(index)
    if len(matches) != 1:
        raise ValueError(
            f"species {candidate!r}를 고유하게 찾지 못했습니다. "
            "folder name, scientific name 또는 0-based index를 사용하십시오"
        )
    return matches[0]


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _targets(tree: TaxonomyTree, species_index: int) -> CanonicalTargets:
    return CanonicalTargets(
        species=int(species_index),
        genus=int(tree.species_to_genus[species_index]),
        family=int(tree.species_to_family[species_index]),
    )


def _denormalize(image: torch.Tensor, transform: JointImageTransform) -> np.ndarray:
    mean = torch.as_tensor(transform.mean, dtype=image.dtype, device=image.device)[:, None, None]
    std = torch.as_tensor(transform.std, dtype=image.dtype, device=image.device)[:, None, None]
    restored = (image * std + mean).clamp(0.0, 1.0)
    return (
        restored.detach().float().cpu().permute(1, 2, 0).numpy() * 255.0
    ).round().astype(np.uint8)


def _official_base_image(image: torch.Tensor) -> np.ndarray:
    """공식 visual_utils.py와 같은 per-image min-max base image 복원."""
    value = image[0].detach().float().cpu().permute(1, 2, 0).numpy()
    low = float(value.min())
    high = float(value.max())
    if high > low:
        value = (value - low) / (high - low)
    else:
        value = np.zeros_like(value)
    return np.clip(value * 255.0, 0, 255).astype(np.uint8)


def _official_overlay(attention: np.ndarray, image_rgb: np.ndarray) -> np.ndarray:
    """공식 SuperImposeHeatmap의 연산을 RGB 입출력으로 재현."""
    resized = cv2.resize(
        np.asarray(attention, dtype=np.float32),
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
    result_bgr = (
        image_bgr.astype(np.float32) * 0.5
        + heatmap_bgr.astype(np.float32) * 0.5
    ).astype(np.uint8)
    return cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)


def _flat_logits(output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        output = output[0]
    if isinstance(output, Mapping):
        output = output.get("species_logits", output.get("logits", output))
    if output.ndim == 3 and output.shape[-1] == 1:
        output = output.squeeze(-1)
    if output.ndim != 2:
        raise ValueError(f"logits shape가 [B,C]가 아닙니다: {tuple(output.shape)}")
    return output


def _flat_patch_attention(
    model: torch.nn.Module,
    output: Any,
    attention: torch.Tensor | None,
    target_index: int,
) -> torch.Tensor:
    del output
    if attention is None:
        raise RuntimeError("Prompt-CAM model이 마지막 attention을 반환하지 않았습니다")
    core = model.module if hasattr(model, "module") else model
    prompt_count = int(core.vpt.prompt_count)
    prefix_count = int(core.num_prefix_tokens)
    patch_start = prompt_count + prefix_count
    result = attention[0, :, int(target_index), patch_start:]
    if result.ndim != 2:
        raise ValueError(f"patch attention은 [H,P]여야 합니다: {tuple(result.shape)}")
    return result


def _greedy_rank_heads(
    num_heads: int,
    evaluate_blur: Callable[[list[int]], tuple[float, int]],
) -> dict[str, Any]:
    """원논문 cumulative uniform-blur greedy ranking의 순수 구현."""
    if num_heads <= 0:
        raise ValueError("num_heads는 양수여야 합니다")
    remaining = list(range(int(num_heads)))
    blurred: list[int] = []
    steps: list[dict[str, Any]] = []

    while remaining:
        candidate_results: dict[int, tuple[float, int]] = {}
        for candidate in remaining:
            candidate_results[candidate] = evaluate_blur(blurred + [candidate])

        # 공식 코드의 순회 순서와 같은 tie break: 낮은 head index가 먼저 제거된다.
        least_important = max(
            remaining,
            key=lambda head: (candidate_results[head][0], -head),
        )
        probability, prediction = candidate_results[least_important]
        steps.append(
            {
                "step": len(steps) + 1,
                "removed_head_zero_based": least_important,
                "removed_head_one_based": least_important + 1,
                "target_probability_after_blur": probability,
                "prediction_after_blur": prediction,
                "candidate_results_one_based": {
                    str(head + 1): {
                        "target_probability": candidate_results[head][0],
                        "prediction": candidate_results[head][1],
                    }
                    for head in remaining
                },
            }
        )
        blurred.append(least_important)
        remaining.remove(least_important)

    # 가장 늦게 제거된 head가 가장 중요하다.
    importance_order = list(reversed(blurred))
    return {
        "importance_order": importance_order,
        "removal_order": blurred,
        "pruning_steps": steps,
    }


def _select_top_heads(ranking: Mapping[str, Any], top_traits: int) -> dict[str, Any]:
    importance = list(ranking["importance_order"])
    if not 1 <= int(top_traits) <= len(importance):
        raise ValueError(
            f"top_traits는 [1,{len(importance)}] 범위여야 하지만 {top_traits}입니다"
        )
    selected = importance[: int(top_traits)]
    selected_set = set(selected)
    removal_order = list(ranking["removal_order"])
    blurred = [head for head in removal_order if head not in selected_set]
    return {
        "selected_heads": selected,
        "blurred_heads": blurred,
        "importance_order": importance,
        "removal_order": removal_order,
        "pruning_steps": list(ranking["pruning_steps"]),
    }


def _grid_maps(maps: torch.Tensor) -> np.ndarray:
    maps = maps.detach().float().cpu()
    if maps.ndim != 2:
        raise ValueError(f"head maps는 [H,P]여야 합니다: {tuple(maps.shape)}")
    patch_count = int(maps.shape[1])
    side = int(round(math.sqrt(patch_count)))
    if side * side != patch_count:
        raise ValueError(f"patch count {patch_count}는 정사각형이 아닙니다")
    return maps.reshape(maps.shape[0], side, side).numpy()


def _run_flat_or_node(
    *,
    model_name: str,
    model: torch.nn.Module,
    image: torch.Tensor,
    target_index: int,
    target_name: str,
    label_names: Sequence[str],
    top_traits: int,
    checkpoint: Path,
    level: str,
) -> HeadResult:
    with torch.inference_mode():
        clean_output, clean_attention = model(image)
        clean_logits = _flat_logits(clean_output).float()
        clean_probabilities = torch.softmax(clean_logits, dim=1)
    prediction = int(clean_probabilities.argmax(dim=1).item())
    target_probability = float(clean_probabilities[0, int(target_index)].item())
    num_heads = int(_flat_patch_attention(
        model, clean_output, clean_attention, target_index
    ).shape[0])

    def evaluate(blurred: list[int]) -> tuple[float, int]:
        with torch.inference_mode():
            output, _ = model(
                image,
                blur_head_lst=blurred,
                target_cls=int(target_index),
            )
            probabilities = torch.softmax(_flat_logits(output).float(), dim=1)
        return (
            float(probabilities[0, int(target_index)].item()),
            int(probabilities.argmax(dim=1).item()),
        )

    ranking = _greedy_rank_heads(num_heads, evaluate)
    selection = _select_top_heads(ranking, top_traits)
    with torch.inference_mode():
        pruned_output, pruned_attention = model(
            image,
            blur_head_lst=selection["blurred_heads"],
            target_cls=int(target_index),
        )
    all_maps = _flat_patch_attention(
        model, pruned_output, pruned_attention, target_index
    )
    index = torch.as_tensor(selection["selected_heads"], device=all_maps.device)
    maps = _grid_maps(all_maps.index_select(0, index))

    return HeadResult(
        model_name=model_name,
        level=level,
        target_index=int(target_index),
        target_name=str(target_name),
        prediction_index=prediction,
        prediction_name=str(label_names[prediction]),
        correct=prediction == int(target_index),
        target_probability=target_probability,
        selected_heads=list(selection["selected_heads"]),
        importance_order=list(selection["importance_order"]),
        blurred_heads=list(selection["blurred_heads"]),
        pruning_steps=list(selection["pruning_steps"]),
        maps=maps,
        checkpoint=str(checkpoint),
        attention_source=(
            "last ViT self-attention: target class prompt query -> image patch keys"
        ),
    )


def _shared_sibling_indices(
    tree: TaxonomyTree,
    level: str,
    target_index: int,
) -> list[int]:
    if level == "family":
        return list(range(len(tree.families)))
    if level == "genus":
        family_index = int(tree.genus_to_family[target_index])
        return [
            index
            for index, value in enumerate(tree.genus_to_family)
            if int(value) == family_index
        ]
    if level == "species":
        genus_index = int(tree.species_to_genus[target_index])
        return [
            index
            for index, value in enumerate(tree.species_to_genus)
            if int(value) == genus_index
        ]
    raise ValueError(f"지원하지 않는 level입니다: {level}")


def _shared_level_probabilities(output: Mapping[str, torch.Tensor], level: str) -> torch.Tensor:
    if level == "family":
        return output["family_probabilities"].float()
    if level == "genus":
        return output["genus_conditional_probabilities"].float()
    if level == "species":
        return output["species_conditional_probabilities"].float()
    raise ValueError(f"지원하지 않는 level입니다: {level}")


def _shared_level_maps(
    output: Mapping[str, torch.Tensor],
    level: str,
    target_index: int,
) -> torch.Tensor:
    key = f"{level}_attention_heads"
    maps = output[key][0, :, int(target_index), :]
    if maps.ndim != 2:
        raise ValueError(f"{key} target slice는 [H,P]여야 합니다: {tuple(maps.shape)}")
    return maps


def _shared_name(tree: TaxonomyTree, level: str, index: int) -> str:
    if level == "family":
        return tree.families[index]
    if level == "genus":
        return tree.genera[index]
    return tree.scientific_names[index]


def _run_shared_level(
    *,
    model: torch.nn.Module,
    image: torch.Tensor,
    tree: TaxonomyTree,
    level: str,
    target_index: int,
    top_traits: int,
    checkpoint: Path,
) -> HeadResult | None:
    siblings = _shared_sibling_indices(tree, level, target_index)
    if len(siblings) <= 1:
        return None

    with torch.inference_mode():
        clean_output, _ = model(image)
    probabilities = _shared_level_probabilities(clean_output, level)
    local_values = probabilities[0, siblings]
    prediction = siblings[int(local_values.argmax().item())]
    target_probability = float(probabilities[0, target_index].item())
    num_heads = int(_shared_level_maps(clean_output, level, target_index).shape[0])

    def evaluate(blurred: list[int]) -> tuple[float, int]:
        with torch.inference_mode():
            output, _ = model(
                image,
                blur_head_lst=blurred,
                target_cls=int(target_index),
                target_level=level,
            )
        candidate_probabilities = _shared_level_probabilities(output, level)
        candidate_local = candidate_probabilities[0, siblings]
        candidate_prediction = siblings[int(candidate_local.argmax().item())]
        return (
            float(candidate_probabilities[0, target_index].item()),
            int(candidate_prediction),
        )

    ranking = _greedy_rank_heads(num_heads, evaluate)
    selection = _select_top_heads(ranking, top_traits)
    with torch.inference_mode():
        pruned_output, _ = model(
            image,
            blur_head_lst=selection["blurred_heads"],
            target_cls=int(target_index),
            target_level=level,
        )
    all_maps = _shared_level_maps(pruned_output, level, target_index)
    index = torch.as_tensor(selection["selected_heads"], device=all_maps.device)
    maps = _grid_maps(all_maps.index_select(0, index))

    return HeadResult(
        model_name="Shared hierarchy",
        level=level,
        target_index=int(target_index),
        target_name=_shared_name(tree, level, target_index),
        prediction_index=int(prediction),
        prediction_name=_shared_name(tree, level, prediction),
        correct=int(prediction) == int(target_index),
        target_probability=target_probability,
        selected_heads=list(selection["selected_heads"]),
        importance_order=list(selection["importance_order"]),
        blurred_heads=list(selection["blurred_heads"]),
        pruning_steps=list(selection["pruning_steps"]),
        maps=maps,
        checkpoint=str(checkpoint),
        attention_source=(
            f"shared hierarchical {level} decoder cross-attention: "
            "target taxonomy prompt query -> image patch keys"
        ),
        note=(
            "Prompt-CAM uniform-blur greedy rule applied to the shared decoder; "
            "this is a hierarchical extension, not the original paper architecture"
        ),
    )


def _independent_path_nodes(
    nodes: Mapping[str, LoadedNode],
    tree: TaxonomyTree,
    targets: CanonicalTargets,
) -> dict[str, tuple[LoadedNode, int, str]]:
    result: dict[str, tuple[LoadedNode, int, str]] = {}

    root = nodes["root"]
    family_name = tree.families[targets.family]
    root_target = root.labels.index(family_name)
    result["family"] = (root, root_target, family_name)

    family_children = tree.genera_by_family[family_name]
    genus_name = tree.genera[targets.genus]
    if len(family_children) > 1:
        key = f"family__{_slug(family_name)}"
        node = nodes[key]
        result["genus"] = (node, node.labels.index(genus_name), genus_name)

    genus_children = tree.species_by_genus[genus_name]
    species_folder = tree.species_names[targets.species]
    if len(genus_children) > 1:
        key = f"genus__{_slug(genus_name)}"
        node = nodes[key]
        result["species"] = (
            node,
            node.labels.index(species_folder),
            tree.scientific_names[targets.species],
        )

    return result


def _font(size: int):
    candidates = (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "C:/Windows/Fonts/malgunbd.ttf",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _captioned_tile(
    image: np.ndarray,
    title: str,
    subtitle: str = "",
    *,
    width: int = 300,
) -> Image.Image:
    source = Image.fromarray(image).convert("RGB")
    scale = width / source.width
    resized = source.resize(
        (width, max(1, round(source.height * scale))),
        Image.Resampling.BICUBIC,
    )
    header = 64 if subtitle else 40
    tile = Image.new("RGB", (width, resized.height + header), "white")
    tile.paste(resized, (0, header))
    draw = ImageDraw.Draw(tile)
    draw.text((8, 5), title, fill="black", font=_font(16))
    if subtitle:
        draw.text((8, 31), subtitle, fill="black", font=_font(13))
    return tile


def _row_for_result(
    result: HeadResult,
    *,
    base_image: np.ndarray,
) -> Image.Image:
    status = "correct" if result.correct else f"pred={result.prediction_name}"
    tiles = [
        _captioned_tile(
            base_image,
            f"{result.model_name} — {result.level}",
            f"target={result.target_name}; P={result.target_probability:.4f}; {status}",
        )
    ]
    for rank, (head, attention) in enumerate(
        zip(result.selected_heads, result.maps),
        start=1,
    ):
        overlay = _official_overlay(attention, base_image)
        tiles.append(
            _captioned_tile(
                overlay,
                f"rank {rank} — head {head + 1}",
                "official per-head normalization",
            )
        )
    width = sum(tile.width for tile in tiles)
    height = max(tile.height for tile in tiles)
    row = Image.new("RGB", (width, height), "white")
    x = 0
    for tile in tiles:
        row.paste(tile, (x, 0))
        x += tile.width
    return row


def _save_rows(results: Sequence[HeadResult], base_image: np.ndarray, path: Path) -> None:
    if not results:
        return
    rows = [_row_for_result(result, base_image=base_image) for result in results]
    canvas = Image.new(
        "RGB",
        (max(row.width for row in rows), sum(row.height for row in rows)),
        "white",
    )
    y = 0
    for row in rows:
        canvas.paste(row, (0, y))
        y += row.height
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=95)


def _save_result_files(
    result: HeadResult,
    *,
    official_base: np.ndarray,
    denormalized_base: np.ndarray,
    output_dir: Path,
) -> dict[str, Any]:
    folder = output_dir / _slug(result.model_name) / result.level
    folder.mkdir(parents=True, exist_ok=True)
    head_files = []
    for rank, (head, attention) in enumerate(zip(result.selected_heads, result.maps), start=1):
        raw_path = folder / f"rank_{rank:02d}_head_{head + 1:02d}_attention.npy"
        np.save(raw_path, attention.astype(np.float32))
        official_path = folder / f"rank_{rank:02d}_head_{head + 1:02d}_official.png"
        denorm_path = folder / f"rank_{rank:02d}_head_{head + 1:02d}_denormalized_base.png"
        Image.fromarray(_official_overlay(attention, official_base)).save(official_path)
        Image.fromarray(_official_overlay(attention, denormalized_base)).save(denorm_path)
        head_files.append(
            {
                "rank": rank,
                "head_zero_based": head,
                "head_one_based": head + 1,
                "raw_attention": str(raw_path),
                "official_overlay": str(official_path),
                "denormalized_base_overlay": str(denorm_path),
                "attention_min": float(np.min(attention)),
                "attention_max": float(np.max(attention)),
                "attention_mean": float(np.mean(attention)),
                "attention_std": float(np.std(attention)),
                "attention_entropy": float(
                    -np.sum(
                        np.clip(attention.reshape(-1), EPS, None)
                        * np.log(np.clip(attention.reshape(-1), EPS, None))
                    )
                ),
            }
        )
    return {
        "model": result.model_name,
        "level": result.level,
        "target_index": result.target_index,
        "target_name": result.target_name,
        "prediction_index": result.prediction_index,
        "prediction_name": result.prediction_name,
        "correct": result.correct,
        "target_probability": result.target_probability,
        "selected_heads_zero_based": result.selected_heads,
        "selected_heads_one_based": [value + 1 for value in result.selected_heads],
        "importance_order_zero_based": result.importance_order,
        "importance_order_one_based": [value + 1 for value in result.importance_order],
        "blurred_heads_zero_based": result.blurred_heads,
        "blurred_heads_one_based": [value + 1 for value in result.blurred_heads],
        "pruning_steps": result.pruning_steps,
        "checkpoint": result.checkpoint,
        "attention_source": result.attention_source,
        "note": result.note,
        "head_files": head_files,
    }


def _paper_protocol_check(results: Sequence[HeadResult], allow_misclassified: bool) -> None:
    incorrect = [
        f"{result.model_name}/{result.level}: target={result.target_name}, "
        f"prediction={result.prediction_name}"
        for result in results
        if not result.correct
    ]
    if incorrect and not allow_misclassified:
        raise RuntimeError(
            "원논문 top-head protocol은 correctly classified target image를 요구합니다. "
            "다음 모델/node가 틀렸습니다:\n  - "
            + "\n  - ".join(incorrect)
            + "\n실패 사례의 true-class heads도 보려면 --allow-misclassified를 명시하십시오."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="원논문 Prompt-CAM 방식의 Flat/Independent/Shared head 비교"
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--species", default=None)
    parser.add_argument("--flat-run-dir", default=None)
    parser.add_argument("--independent-run-dir", required=True)
    parser.add_argument("--shared-run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-traits", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--allow-misclassified",
        action="store_true",
        help="원논문 기본 protocol 밖의 실패 사례에 대해 true-class heads를 생성",
    )
    args = parser.parse_args()

    image_path = Path(args.image).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"이미지가 없습니다: {image_path}")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _device(args.device)

    independent_run = _resolve_run_dir(
        args.independent_run_dir, required=True, label="Independent"
    )
    shared_run = _resolve_run_dir(args.shared_run_dir, required=True, label="Shared")
    flat_run = _resolve_run_dir(args.flat_run_dir, required=False, label="Flat")
    assert independent_run is not None and shared_run is not None

    print("[1/5] 모델과 taxonomy 불러오기")
    independent_nodes, tree, independent_config = _load_independent_nodes(
        independent_run, device
    )
    shared_model, shared_params, shared_config, shared_checkpoint, shared_taxonomy = (
        _load_shared_model(shared_run, device)
    )
    _validate_taxonomies(shared_taxonomy, tree)

    flat_model = None
    flat_params = None
    flat_config = None
    flat_checkpoint = None
    if flat_run is not None:
        flat_model, flat_params, flat_config, flat_checkpoint = _load_flat_model(
            flat_run, device
        )
        _validate_flat_class_order(flat_params, tree)

    configs: dict[str, Mapping[str, Any]] = {
        "independent": independent_config,
        "shared": shared_config,
    }
    if flat_config is not None:
        configs["flat"] = flat_config
    _validate_preprocessing(configs)

    species_index = _resolve_species_index(tree, args.species, image_path)
    targets = _targets(tree, species_index)

    print("[2/5] 동일 test transform 적용")
    transform_params = SimpleNamespace(**dict(independent_config))
    transform = JointImageTransform(transform_params, training=False)
    pil_image = Image.open(image_path).convert("RGB")
    image_tensor, _, _ = transform(
        pil_image,
        bbox=None,
        bbox_coordinate_mode="normalized",
    )
    image = image_tensor.unsqueeze(0).to(device)
    official_base = _official_base_image(image)
    denormalized_base = _denormalize(image_tensor, transform)
    Image.fromarray(official_base).save(output_dir / "transformed_input_official_minmax.png")
    Image.fromarray(denormalized_base).save(output_dir / "transformed_input_denormalized.png")

    print("[3/5] 원논문 cumulative uniform-blur greedy head ranking")
    results: list[HeadResult] = []
    independent_results: dict[str, HeadResult] = {}
    shared_results: dict[str, HeadResult] = {}

    independent_path = _independent_path_nodes(independent_nodes, tree, targets)
    for level in LEVELS:
        if level not in independent_path:
            continue
        node, local_target, target_name = independent_path[level]
        result = _run_flat_or_node(
            model_name="Independent taxonomy",
            model=node.model,
            image=image,
            target_index=local_target,
            target_name=target_name,
            label_names=node.labels,
            top_traits=args.top_traits,
            checkpoint=node.checkpoint_path,
            level=level,
        )
        independent_results[level] = result
        results.append(result)

    for level, target_index in (
        ("family", targets.family),
        ("genus", targets.genus),
        ("species", targets.species),
    ):
        result = _run_shared_level(
            model=shared_model,
            image=image,
            tree=tree,
            level=level,
            target_index=target_index,
            top_traits=args.top_traits,
            checkpoint=shared_checkpoint,
        )
        if result is not None:
            shared_results[level] = result
            results.append(result)

    flat_result = None
    if flat_model is not None and flat_checkpoint is not None:
        flat_result = _run_flat_or_node(
            model_name="Flat Prompt-CAM",
            model=flat_model,
            image=image,
            target_index=targets.species,
            target_name=tree.scientific_names[targets.species],
            label_names=tree.scientific_names,
            top_traits=args.top_traits,
            checkpoint=flat_checkpoint,
            level="species",
        )
        results.append(flat_result)

    _paper_protocol_check(results, bool(args.allow_misclassified))

    print("[4/5] head별 raw attention과 공식 overlay 저장")
    records = [
        _save_result_files(
            result,
            official_base=official_base,
            denormalized_base=denormalized_base,
            output_dir=output_dir,
        )
        for result in results
    ]

    # 원논문과 같은 official rendering을 주요 montage로 저장한다.
    species_rows: list[HeadResult] = []
    if flat_result is not None:
        species_rows.append(flat_result)
    if "species" in independent_results:
        species_rows.append(independent_results["species"])
    if "species" in shared_results:
        species_rows.append(shared_results["species"])
    _save_rows(
        species_rows,
        official_base,
        output_dir / "comparison_species_official.png",
    )
    _save_rows(
        species_rows,
        denormalized_base,
        output_dir / "comparison_species_denormalized_base.png",
    )

    _save_rows(
        [independent_results[level] for level in LEVELS if level in independent_results],
        official_base,
        output_dir / "independent_taxonomy_path_official.png",
    )
    _save_rows(
        [shared_results[level] for level in LEVELS if level in shared_results],
        official_base,
        output_dir / "shared_hierarchy_path_official.png",
    )

    for level in LEVELS:
        level_rows = []
        if level == "species" and flat_result is not None:
            level_rows.append(flat_result)
        if level in independent_results:
            level_rows.append(independent_results[level])
        if level in shared_results:
            level_rows.append(shared_results[level])
        _save_rows(
            level_rows,
            official_base,
            output_dir / f"comparison_{level}_official.png",
        )

    print("[5/5] 재현 metadata 저장")
    payload = {
        "method": "Prompt-CAM original cumulative uniform-attention greedy ranking",
        "paper_protocol": {
            "target": "ground-truth class/taxonomy child",
            "correctly_classified_required": not bool(args.allow_misclassified),
            "blur": "replace target query/head patch-attention logits by zeros, yielding uniform softmax over patches",
            "least_important_rule": "head whose cumulative blur leaves the highest target probability",
            "visualization": "individual head maps; no averaging",
            "rendering": "per-head min-max, bicubic resize, GaussianBlur(9x9), JET, alpha=0.5",
        },
        "shared_extension": (
            "The same Prompt-CAM rule is applied to rank-specific decoder query-to-patch "
            "cross-attention. This is an extension to the shared hierarchy, not an original-paper architecture."
        ),
        "image": str(image_path),
        "true_species": {
            "index": targets.species,
            "folder_name": tree.species_names[targets.species],
            "scientific_name": tree.scientific_names[targets.species],
            "genus_index": targets.genus,
            "genus": tree.genera[targets.genus],
            "family_index": targets.family,
            "family": tree.families[targets.family],
        },
        "top_traits": int(args.top_traits),
        "checkpoints": {
            "flat": None if flat_checkpoint is None else str(flat_checkpoint),
            "independent_run": str(independent_run),
            "shared": str(shared_checkpoint),
        },
        "results": records,
        "deterministic_levels": {
            "independent_genus_absent": "genus" not in independent_results,
            "independent_species_absent": "species" not in independent_results,
            "shared_genus_absent": "genus" not in shared_results,
            "shared_species_absent": "species" not in shared_results,
            "reason": "a taxonomy node with one child has no classification contrast and no ranked Prompt-CAM",
        },
        "claim_boundary": {
            "official_overlays": (
                "Reproduce the official Prompt-CAM visualization transform. Colors are independently "
                "normalized per head, so absolute color intensity must not be compared across heads/models."
            ),
            "denormalized_base_overlays": (
                "Use identical attention maps and official heatmap transform, but restore the input color "
                "with the configured normalization mean/std for report readability."
            ),
        },
    }
    (output_dir / "promptcam_official_metadata.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 80)
    print("원논문 방식 Prompt-CAM head 비교 완료")
    print("=" * 80)
    print(f"True species : {tree.scientific_names[targets.species]}")
    print(f"Top traits   : {args.top_traits}")
    print(f"Output       : {output_dir}")
    print("Main figure  : comparison_species_official.png")
    print("Path figures : independent_taxonomy_path_official.png, shared_hierarchy_path_official.png")
    print("Metadata     : promptcam_official_metadata.json")
    print("=" * 80)


if __name__ == "__main__":
    main()
