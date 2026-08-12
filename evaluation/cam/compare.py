#!/usr/bin/env python3
"""Create controlled Prompt-CAM comparisons for one WILD-30 image.

The same deterministic evaluation tensor is passed to the flat Prompt-CAM,
independent taxonomy-node models, and shared hierarchical Prompt-CAM.  Every
CAM uses the same top-K greedy head selection, overlay, and per-map min-max
normalization.  Both deletion and sufficiency confidence drops are recorded.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw, ImageFont
from torchvision import datasets


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset.imagefolder import JointImageTransform, load_taxonomy_manifest  # noqa: E402
from data.original_taxonomy import TaxonomyNodeSpec, list_taxonomy_nodes  # noqa: E402
from evaluation.cam.hierarchy import (  # noqa: E402
    _clean_level_maps,
    _greedy_top_heads,
)
from evaluation.cam.species import (  # noqa: E402
    _greedy_promptcam_top_heads,
    _species_head_maps,
    _species_logits,
)
from evaluation.checkpoints import (  # noqa: E402
    _checkpoint_records,
    _discover_checkpoints,
    _resolve_project_path,
    _squeeze_logits,
    _torch_load,
    _validate_checkpoint_compatibility,
    _validate_checkpoint_taxonomy_mappings,
    aggregate_soft_path_scores,
)
from evaluation.cam.visualize_independent import (  # noqa: E402
    _denormalize_image,
    _resolve_species_index,
    _true_path_nodes,
)


LEVELS = ("family", "genus", "species")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return value


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.device(value)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    return result


def _clear_device_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _checkpoint_state(path: Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    payload = _torch_load(path)
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint must be a mapping: {path}")
    state = payload.get("model_state_dict", payload.get("state_dict", payload))
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint state must be a mapping: {path}")
    if state and all(str(key).startswith("module.") for key in state):
        state = {str(key)[7:]: value for key, value in state.items()}
    return payload, state


def _load_run_model(
    run_dir: Path,
    device: torch.device,
    *,
    expected_hierarchical: bool,
    allow_last: bool = False,
) -> tuple[torch.nn.Module, SimpleNamespace, Path, dict[str, Any]]:
    from model.factory import get_model

    args_path = run_dir / "args.yaml"
    if not args_path.is_file():
        raise FileNotFoundError(f"args.yaml is missing: {args_path}")
    args_data = _load_yaml(args_path)
    actual_hierarchical = bool(args_data.get("hierarchical_prompt", False))
    if actual_hierarchical != expected_hierarchical:
        raise ValueError(
            f"hierarchical_prompt={actual_hierarchical} does not match the requested model: {run_dir}"
        )
    if bool(args_data.get("original_taxonomy_prompt", False)):
        raise ValueError(f"A node-local run cannot be loaded as a flat/shared run: {run_dir}")

    checkpoint_path = run_dir / "model.pt"
    if not checkpoint_path.is_file() and allow_last:
        checkpoint_path = run_dir / "last.pt"
    if not checkpoint_path.is_file():
        suffix = " (last.pt was also allowed)" if allow_last else ""
        raise FileNotFoundError(f"model checkpoint is missing: {run_dir}{suffix}")

    params = SimpleNamespace(**args_data)
    params.load_pretrained_backbone = False
    params.promptcam_checkpoint = None
    params.resume = None
    params.vis_attn = True
    params.debug = False
    params.distributed = False
    model, _, _ = get_model(params, visualize=True)
    payload, state = _checkpoint_state(checkpoint_path)
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint/model mismatch at {checkpoint_path}: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    model = model.float().to(device).eval()
    return model, params, checkpoint_path, payload


def _load_node_model(
    record: dict[str, Any],
    device: torch.device,
    *,
    visualize: bool,
) -> torch.nn.Module:
    """Restore one independent node while preserving CAM attention when requested."""
    from model.factory import get_model

    config = dict(record["config"])
    config.update(
        {
            "load_pretrained_backbone": False,
            "promptcam_checkpoint": str(record["path"]),
            "promptcam_checkpoint_strict": True,
            "resume": None,
            "vis_attn": bool(visualize),
            "debug": False,
            "distributed": False,
        }
    )
    model, _, _ = get_model(SimpleNamespace(**config), visualize=visualize)
    return model.float().to(device).eval()


def _normalization_signature(config: dict[str, Any]) -> dict[str, Any]:
    return {
        key: config.get(key)
        for key in (
            "crop_size",
            "eval_resize_size",
            "normalization",
            "normalization_mean",
            "normalization_std",
            "pretrained_weights",
        )
    }


def _existing_project_path(*values: Any, label: str) -> Path:
    attempted: list[Path] = []
    for value in values:
        if value in (None, "", "null"):
            continue
        path = _resolve_project_path(value)
        attempted.append(path)
        if path.exists():
            return path
    formatted = ", ".join(str(path) for path in attempted) or "<none>"
    raise FileNotFoundError(f"No existing {label} path was found; attempted: {formatted}")


def _validate_common_protocol(
    independent_config: dict[str, Any],
    shared_config: dict[str, Any],
    flat_config: dict[str, Any] | None,
) -> None:
    expected = _normalization_signature(independent_config)
    mismatches: list[str] = []
    for name, config in (("shared", shared_config), ("flat", flat_config)):
        if config is None:
            continue
        actual = _normalization_signature(config)
        for key, value in expected.items():
            if actual[key] != value:
                mismatches.append(f"{name}.{key}: {actual[key]!r} != {value!r}")
    if mismatches:
        raise ValueError("The models do not share one preprocessing/backbone protocol:\n- " + "\n- ".join(mismatches))


def _taxonomy_names(taxonomy: Any, species_index: int) -> dict[str, Any]:
    genus_index = int(taxonomy.species_to_genus[species_index])
    family_index = int(taxonomy.genus_to_family[genus_index])
    return {
        "species_index": int(species_index),
        "species": str(taxonomy.scientific_names[species_index]),
        "folder": str(taxonomy.class_names[species_index]),
        "genus_index": genus_index,
        "genus": str(taxonomy.genus_names[genus_index]),
        "family_index": family_index,
        "family": str(taxonomy.family_names[family_index]),
    }


def _target_probability(output: dict[str, torch.Tensor], level: str, index: int) -> float:
    if level == "family":
        probabilities = output["family_probabilities"]
    elif level == "genus":
        probabilities = output["genus_conditional_probabilities"]
    elif level == "species":
        probabilities = output["species_conditional_probabilities"]
    else:
        raise ValueError(f"Unknown level: {level}")
    return float(probabilities[0, int(index)].float().item())


def _drop_metrics(original: float, retained: float, deleted: float) -> dict[str, float]:
    denominator = max(float(original), 1e-12)
    return {
        "p_original": float(original),
        "p_topk_retained": float(retained),
        "p_topk_deleted": float(deleted),
        "deletion_absolute_drop": float(original - deleted),
        "deletion_relative_drop": float((original - deleted) / denominator),
        "sufficiency_absolute_drop": float(original - retained),
        "sufficiency_relative_drop": float((original - retained) / denominator),
    }


def _flat_cam(
    model: torch.nn.Module,
    batch: torch.Tensor,
    params: SimpleNamespace,
    *,
    target_index: int,
    top_traits: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    with torch.inference_mode():
        output, attention = model(batch)
        logits = _species_logits(output).float()
        probability = float(torch.softmax(logits, dim=-1)[0, target_index].item())
        selection = _greedy_promptcam_top_heads(model, batch, target_index, top_traits)
        retained_output, _ = model(
            batch,
            blur_head_lst=selection["blurred_heads"],
            target_cls=target_index,
        )
        deleted_output, _ = model(
            batch,
            blur_head_lst=selection["selected_heads"],
            target_cls=target_index,
        )
        retained = float(
            torch.softmax(_species_logits(retained_output).float(), dim=-1)[0, target_index].item()
        )
        deleted = float(
            torch.softmax(_species_logits(deleted_output).float(), dim=-1)[0, target_index].item()
        )
        maps = _species_head_maps(
            output,
            attention,
            target_index,
            params,
            sample_index=0,
        )
        mean_map = maps[0, selection["selected_heads"], :].float().mean(dim=0)
    metadata = {
        "target_local_index": int(target_index),
        "selected_heads_zero_based": selection["selected_heads"],
        "selected_heads_one_based": [value + 1 for value in selection["selected_heads"]],
        "blurred_non_topk_heads_zero_based": selection["blurred_heads"],
        "pruning_steps": selection["pruning_steps"],
        "faithfulness": _drop_metrics(probability, retained, deleted),
    }
    return mean_map, metadata


def _shared_cam(
    model: torch.nn.Module,
    batch: torch.Tensor,
    *,
    level: str,
    target_index: int,
    top_traits: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    with torch.inference_mode():
        output, _ = model(batch)
        selection = _greedy_top_heads(
            model,
            batch,
            level=level,
            target_index=target_index,
            top_traits=top_traits,
        )
        effective_level = str(selection["effective_level"])
        effective_target = int(selection["effective_target_index"])
        probability = _target_probability(output, effective_level, effective_target)
        retained_output, _ = model(
            batch,
            blur_head_lst=selection["blurred_heads"],
            target_cls=effective_target,
            target_level=effective_level,
        )
        deleted_output, _ = model(
            batch,
            blur_head_lst=selection["selected_heads"],
            target_cls=effective_target,
            target_level=effective_level,
        )
        retained = _target_probability(retained_output, effective_level, effective_target)
        deleted = _target_probability(deleted_output, effective_level, effective_target)
        maps, _ = _clean_level_maps(output, model, level=level, target_index=target_index)
        mean_map = maps[0, selection["selected_heads"], :].float().mean(dim=0)
    metadata = {
        "target_global_index": int(target_index),
        "effective_level": effective_level,
        "effective_target_index": effective_target,
        "contrast_defined": bool(selection["contrast_defined"]),
        "selected_heads_zero_based": selection["selected_heads"],
        "selected_heads_one_based": [value + 1 for value in selection["selected_heads"]],
        "blurred_non_topk_heads_zero_based": selection["blurred_heads"],
        "pruning_steps": selection["pruning_steps"],
        "faithfulness": _drop_metrics(probability, retained, deleted),
    }
    return mean_map, metadata


def _overlay(image_rgb: np.ndarray, attention: torch.Tensor, patch_size: int) -> Image.Image:
    height, width = image_rgb.shape[:2]
    patch_count = int(attention.numel())
    grid_height = height // int(patch_size)
    grid_width = width // int(patch_size)
    if grid_height * grid_width != patch_count:
        raise ValueError(f"CAM patch count {patch_count} != grid {grid_height}x{grid_width}")
    heat = attention.detach().float().cpu().reshape(grid_height, grid_width).numpy()
    low, high = float(heat.min()), float(heat.max())
    heat = (heat - low) / (high - low) if high > low else np.zeros_like(heat)
    heat = cv2.resize(heat, (width, height), interpolation=cv2.INTER_CUBIC)
    heat = np.clip(heat, 0.0, 1.0)
    color_bgr = cv2.applyColorMap(np.round(heat * 255.0).astype(np.uint8), cv2.COLORMAP_JET)
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    blended = np.round(0.5 * image_rgb.astype(np.float32) + 0.5 * color_rgb.astype(np.float32))
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


def _placeholder(size: tuple[int, int], text: str) -> Image.Image:
    image = Image.new("RGB", size, "#eeeeee")
    draw = ImageDraw.Draw(image)
    draw.multiline_text((12, 12), text, fill="black", font=ImageFont.load_default(), spacing=4)
    return image


@lru_cache(maxsize=8)
def _font(size: int = 14) -> ImageFont.ImageFont:
    candidates = (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibri.ttf"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _panel(image: Image.Image, title: str, width: int = 336) -> Image.Image:
    image = image.convert("RGB")
    target_height = max(1, round(image.height * width / image.width))
    image = image.resize((width, target_height), Image.Resampling.BICUBIC)
    header = 78
    result = Image.new("RGB", (width, target_height + header), "white")
    result.paste(image, (0, header))
    ImageDraw.Draw(result).multiline_text(
        (8, 6), title[:140], fill="black", font=_font(), spacing=2
    )
    return result


def _save_grid(rows: list[list[tuple[str, Image.Image]]], path: Path) -> None:
    rendered = [[_panel(image, title) for title, image in row] for row in rows]
    column_count = max(len(row) for row in rendered)
    cell_width = max(panel.width for row in rendered for panel in row)
    row_heights = [max(panel.height for panel in row) for row in rendered]
    canvas = Image.new("RGB", (column_count * cell_width, sum(row_heights)), "white")
    y = 0
    for row, row_height in zip(rendered, row_heights):
        for x_index, panel in enumerate(row):
            canvas.paste(panel, (x_index * cell_width, y))
        y += row_height
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _save_hierarchy_grid(
    input_image: Image.Image,
    input_title: str,
    rows: list[list[tuple[str, Image.Image]]],
    path: Path,
) -> None:
    """Save a two-column hierarchy grid with one centered, column-spanning input row."""
    rendered = [[_panel(image, title) for title, image in row] for row in rows]
    if any(len(row) != 2 for row in rendered):
        raise ValueError("Hierarchy comparison rows must contain exactly two panels")
    cell_width = max(panel.width for row in rendered for panel in row)
    total_width = cell_width * 2
    source = input_image.convert("RGB")
    source_height = max(1, round(source.height * cell_width / source.width))
    source = source.resize((cell_width, source_height), Image.Resampling.BICUBIC)
    input_header = 58
    input_panel = Image.new("RGB", (total_width, source_height + input_header), "white")
    input_panel.paste(source, ((total_width - cell_width) // 2, input_header))
    ImageDraw.Draw(input_panel).text((8, 8), input_title, fill="black", font=_font())

    row_heights = [max(panel.height for panel in row) for row in rendered]
    canvas = Image.new("RGB", (total_width, input_panel.height + sum(row_heights)), "white")
    canvas.paste(input_panel, (0, 0))
    y = input_panel.height
    for row, row_height in zip(rendered, row_heights):
        for column, panel in enumerate(row):
            canvas.paste(panel, (column * cell_width, y))
        y += row_height
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _independent_predictions(
    records: dict[str, dict[str, Any]],
    trainable_nodes: Iterable[TaxonomyNodeSpec],
    batch: torch.Tensor,
    device: torch.device,
    num_species: int,
) -> tuple[int, dict[str, torch.Tensor], torch.Tensor]:
    node_log_probabilities: dict[str, torch.Tensor] = {}
    for node in trainable_nodes:
        model = _load_node_model(records[node.node_id], device, visualize=False)
        with torch.inference_mode():
            output, _ = model(batch)
            logits = _squeeze_logits(output).float()
        if logits.shape[1] != node.num_children:
            raise ValueError(f"{node.node_id}: output classes {logits.shape[1]} != {node.num_children}")
        node_log_probabilities[node.node_id] = F.log_softmax(logits, dim=-1).cpu()
        del model
        _clear_device_cache()
    scores = aggregate_soft_path_scores(
        node_log_probabilities,
        trainable_nodes,
        num_species=num_species,
    )
    leaf_probabilities = scores.exp()
    probability_sum = float(leaf_probabilities[0].sum().item())
    if abs(probability_sum - 1.0) > 1e-4:
        raise RuntimeError(
            "Independent taxonomy path probabilities do not sum to one: "
            f"{probability_sum:.8f}"
        )
    return int(scores[0].argmax().item()), node_log_probabilities, leaf_probabilities


def _independent_path_cams(
    records: dict[str, dict[str, Any]],
    taxonomy: Any,
    species_index: int,
    batch: torch.Tensor,
    device: torch.device,
    top_traits: int,
) -> tuple[dict[str, torch.Tensor | None], list[dict[str, Any]]]:
    images: dict[str, torch.Tensor | None] = {}
    entries: list[dict[str, Any]] = []
    for node in _true_path_nodes(taxonomy, species_index):
        level = {"root": "family", "family": "genus", "genus": "species"}[node.rank]
        local_target = int(node.species_to_child[species_index])
        if not node.trainable:
            images[level] = None
            entries.append(
                {
                    "model": "independent_taxonomy",
                    "level": level,
                    "node_id": node.node_id,
                    "deterministic_singleton": True,
                    "target_local_index": local_target,
                }
            )
            continue
        model = _load_node_model(records[node.node_id], device, visualize=True)
        params = SimpleNamespace(**records[node.node_id]["config"])
        cam, metadata = _flat_cam(
            model,
            batch,
            params,
            target_index=local_target,
            top_traits=top_traits,
        )
        entries.append(
            {
                "model": "independent_taxonomy",
                "level": level,
                "node_id": node.node_id,
                "deterministic_singleton": False,
                **metadata,
            }
        )
        images[level] = cam
        del model
        _clear_device_cache()
    return images, entries


def _case_label(true_index: int, flat: int | None, independent: int, shared: int, taxonomy: Any) -> str:
    if independent == true_index and shared == true_index and (flat is None or flat == true_index):
        return "A_all_available_correct"
    if shared == true_index and independent != true_index and (flat is None or flat != true_index):
        return "B_shared_only_correct"
    if independent == true_index and shared != true_index and (flat is None or flat != true_index):
        return "C_independent_only_correct"
    if flat == true_index and independent != true_index and shared != true_index:
        return "D_flat_only_correct"
    if independent != true_index and shared != true_index:
        true_genus = int(taxonomy.species_to_genus[true_index])
        if all(int(taxonomy.species_to_genus[p]) == true_genus for p in (independent, shared)):
            return "D_both_wrong_species_siblings"
        true_family = int(taxonomy.genus_to_family[true_genus])
        predicted_families = [
            int(taxonomy.genus_to_family[int(taxonomy.species_to_genus[p])])
            for p in (independent, shared)
        ]
        if all(value == true_family for value in predicted_families):
            return "D_both_wrong_genus_siblings"
    return "other"


def _case_tags(
    true_index: int,
    flat: int | None,
    independent: int,
    shared: int,
    taxonomy: Any,
) -> list[str]:
    predictions = [independent, shared] + ([] if flat is None else [flat])
    true_genus = int(taxonomy.species_to_genus[true_index])
    true_family = int(taxonomy.genus_to_family[true_genus])
    predicted_genera = [int(taxonomy.species_to_genus[index]) for index in predictions]
    predicted_families = [int(taxonomy.genus_to_family[index]) for index in predicted_genera]
    tags: list[str] = []
    if all(index != true_index for index in predictions):
        tags.append("all_available_wrong")
        if all(index == true_genus for index in predicted_genera):
            tags.append("all_available_wrong_same_genus")
    if any(index != true_family for index in predicted_families):
        tags.append("upper_family_error")
    if flat == true_index and independent != true_index and shared != true_index:
        tags.append("flat_only_correct")
    true_family_name = str(taxonomy.family_names[true_family]).casefold()
    true_genus_name = str(taxonomy.genus_names[true_genus]).casefold()
    independent_genus = str(
        taxonomy.genus_names[int(taxonomy.species_to_genus[independent])]
    ).casefold()
    shared_genus = str(taxonomy.genus_names[int(taxonomy.species_to_genus[shared])]).casefold()
    if (
        true_family_name == "ulmaceae"
        and true_genus_name == "ulmus"
        and independent_genus == "zelkova"
        and shared_genus == "zelkova"
    ):
        tags.append("ulmaceae_genus_routing_failure")
    return tags


def _level_target_name(taxonomy: Any, species_index: int, level: str) -> str:
    names = _taxonomy_names(taxonomy, species_index)
    return str(names[level])


def _entry_probability(entry: dict[str, Any]) -> float | None:
    faithfulness = entry.get("faithfulness")
    if not isinstance(faithfulness, dict):
        return None
    value = faithfulness.get("p_original")
    return None if value is None else float(value)


def _entry_leaf_probability(entry: dict[str, Any]) -> float | None:
    value = entry.get("p_leaf")
    return None if value is None else float(value)


def _cam_title(
    model_label: str,
    level: str,
    taxonomy: Any,
    species_index: int,
    entry: dict[str, Any],
    *,
    target_mode: str,
) -> str:
    probability = _entry_probability(entry)
    leaf_probability = _entry_leaf_probability(entry)
    probability_name = "P_family" if level == "family" else "P_cond"
    probability_text = "" if probability is None else f"{probability_name}={probability:.3f}"
    leaf_text = "" if leaf_probability is None else f"P_leaf={leaf_probability:.3f}"
    level_label = level.capitalize()
    target_name = _level_target_name(taxonomy, species_index, level)
    if model_label == "Flat":
        return f"Flat — Species: {target_name}\n{leaf_text}"
    if model_label == "Shared" and not bool(entry.get("contrast_defined", True)):
        effective_level = str(entry.get("effective_level", level))
        effective_name = _level_target_name(taxonomy, species_index, effective_level)
        leaf_prefix = "Predicted leaf" if target_mode == "predicted" else "Leaf"
        return (
            f"Shared — {leaf_prefix}: {taxonomy.scientific_names[species_index]}\n"
            f"Effective {effective_level}: {effective_name}, {probability_text}\n"
            f"{leaf_text}"
        )
    details = ", ".join(value for value in (probability_text, leaf_text) if value)
    return f"{model_label} — {level_label}: {target_name}\n{details}"


def _probability_table(cams: list[dict[str, Any]]) -> dict[str, Any]:
    table: dict[str, Any] = {}
    for entry in cams:
        model = str(entry.get("model"))
        mode = str(entry.get("target_mode"))
        level = str(entry.get("level"))
        model_table = table.setdefault(mode, {}).setdefault(model, {})
        model_table[level] = {
            "p_level_or_conditional": _entry_probability(entry),
            "p_leaf": _entry_leaf_probability(entry),
            "target_semantics": entry.get("target_semantics"),
            "deterministic_singleton": bool(entry.get("deterministic_singleton", False)),
            "effective_level": entry.get("effective_level", level),
            "contrast_defined": entry.get("contrast_defined"),
        }
    return table


def _target_indices(taxonomy: Any, species_index: int) -> dict[str, int]:
    genus = int(taxonomy.species_to_genus[species_index])
    return {
        "species": int(species_index),
        "genus": genus,
        "family": int(taxonomy.genus_to_family[genus]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--species", default=None, help="true folder/scientific name or 0-based index")
    parser.add_argument("--independent-training-summary", required=True)
    parser.add_argument("--hierarchical-run-dir", required=True)
    parser.add_argument("--flat-run-dir", default=None)
    parser.add_argument(
        "--allow-incomplete-flat",
        action="store_true",
        help="allow last.pt when the flat run has no selected model.pt; marked provisional",
    )
    parser.add_argument("--target-mode", choices=("true", "predicted", "both"), default="both")
    parser.add_argument("--top-traits", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    image_path = Path(args.image).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image is missing: {image_path}")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _device(args.device)

    checkpoint_paths = _discover_checkpoints([], None, args.independent_training_summary)
    records, duplicate_selection = _checkpoint_records(checkpoint_paths, duplicate_policy="error")
    _validate_checkpoint_compatibility(records)
    if "root" not in records:
        raise ValueError("The independent experiment has no root checkpoint")
    independent_config = dict(records["root"]["config"])
    data_root = _existing_project_path(
        independent_config.get("resolved_data_path"),
        independent_config.get("data_path"),
        label="data root",
    )
    train_split = str(independent_config.get("train_split", "train"))
    train_base = datasets.ImageFolder(str(data_root / train_split), transform=None)
    taxonomy = load_taxonomy_manifest(
        _existing_project_path(independent_config.get("taxonomy_manifest"), label="taxonomy manifest"),
        train_base.classes,
        class_column=independent_config.get("taxonomy_class_column"),
    )
    all_nodes = list_taxonomy_nodes(taxonomy, trainable_only=False)
    trainable_nodes = [node for node in all_nodes if node.trainable]
    missing = sorted(node.node_id for node in trainable_nodes if node.node_id not in records)
    if missing:
        raise ValueError(f"Independent checkpoints are missing: {missing}")
    _validate_checkpoint_taxonomy_mappings(records, trainable_nodes)

    species_value = args.species if args.species is not None else image_path.parent.name
    true_species = _resolve_species_index(taxonomy, species_value)
    transform = JointImageTransform(SimpleNamespace(**independent_config), training=False)
    transformed, _, _ = transform(
        Image.open(image_path).convert("RGB"),
        bbox=None,
        bbox_coordinate_mode="normalized",
    )
    transformed = transformed.float()
    image_rgb = _denormalize_image(transformed, transform)
    input_image = Image.fromarray(image_rgb)
    input_image.save(output_dir / "input.png")
    batch = transformed.unsqueeze(0).to(device)

    hierarchical_dir = Path(args.hierarchical_run_dir).expanduser()
    if not hierarchical_dir.is_absolute():
        hierarchical_dir = PROJECT_ROOT / hierarchical_dir
    hierarchical_dir = hierarchical_dir.resolve()
    shared_model, shared_params, shared_checkpoint, _ = _load_run_model(
        hierarchical_dir, device, expected_hierarchical=True
    )

    flat_model = None
    flat_params = None
    flat_checkpoint = None
    flat_config = None
    if args.flat_run_dir:
        flat_dir = Path(args.flat_run_dir).expanduser()
        if not flat_dir.is_absolute():
            flat_dir = PROJECT_ROOT / flat_dir
        flat_dir = flat_dir.resolve()
        flat_model, flat_params, flat_checkpoint, _ = _load_run_model(
            flat_dir,
            device,
            expected_hierarchical=False,
            allow_last=args.allow_incomplete_flat,
        )
        flat_config = vars(flat_params)

    _validate_common_protocol(independent_config, vars(shared_params), flat_config)
    if list(getattr(shared_params, "class_names", [])) != list(taxonomy.class_names):
        raise ValueError("Shared and independent species orders differ")
    if flat_params is not None and list(getattr(flat_params, "class_names", [])) != list(taxonomy.class_names):
        raise ValueError("Flat and independent species orders differ")

    with torch.inference_mode():
        shared_output, _ = shared_model(batch)
        shared_leaf_probabilities = shared_output["species_probabilities"][0].detach().float().cpu()
        shared_prediction = int(shared_leaf_probabilities.argmax().item())
        if flat_model is not None:
            flat_output, _ = flat_model(batch)
            flat_leaf_probabilities = torch.softmax(
                _species_logits(flat_output).float(), dim=-1
            )[0].detach().cpu()
            flat_prediction = int(flat_leaf_probabilities.argmax().item())
        else:
            flat_leaf_probabilities = None
            flat_prediction = None
    independent_prediction, _, independent_leaf_probabilities = _independent_predictions(
        records, trainable_nodes, batch, device, len(taxonomy.class_names)
    )

    patch_size = int(independent_config.get("patch_size", 14))
    if transformed.shape[-2] % patch_size or transformed.shape[-1] % patch_size:
        raise ValueError(f"Transformed size {tuple(transformed.shape[-2:])} is not divisible by patch_size={patch_size}")

    modes = ("true", "predicted") if args.target_mode == "both" else (args.target_mode,)
    metadata: dict[str, Any] = {
        "schema_version": 3,
        "image": str(image_path),
        "true": _taxonomy_names(taxonomy, true_species),
        "predictions": {
            "flat_promptcam": None if flat_prediction is None else _taxonomy_names(taxonomy, flat_prediction),
            "independent_taxonomy_joint_map": _taxonomy_names(taxonomy, independent_prediction),
            "shared_hierarchical_joint_map": _taxonomy_names(taxonomy, shared_prediction),
        },
        "case": _case_label(true_species, flat_prediction, independent_prediction, shared_prediction, taxonomy),
        "case_tags": _case_tags(
            true_species,
            flat_prediction,
            independent_prediction,
            shared_prediction,
            taxonomy,
        ),
        "protocol": {
            "same_transformed_tensor": True,
            "dtype": str(batch.dtype),
            "shape": list(batch.shape),
            "resize": int(independent_config["eval_resize_size"]),
            "center_crop": int(independent_config["crop_size"]),
            "patch_size": patch_size,
            "patch_grid": [int(transformed.shape[-2] // patch_size), int(transformed.shape[-1] // patch_size)],
            "normalization_mean": list(transform.mean),
            "normalization_std": list(transform.std),
            "augmentation": False,
            "snapmix": False,
            "autocast": False,
            "top_traits": int(args.top_traits),
            "cam_normalization": "independent per-CAM min-max",
            "colormap": "OpenCV JET",
            "overlay_alpha": 0.5,
            "head_selection": "cumulative greedy uniform-blur pruning",
            "deletion_definition": "blur selected top-K heads",
            "sufficiency_definition": "blur all non-top-K heads",
            "probability_labels": {
                "P_leaf": "final species probability P(species | x)",
                "P_family": "family probability P(family | x)",
                "P_cond_at_genus": "conditional probability P(genus | family, x)",
                "P_cond_at_species": "conditional probability P(species | genus, x)",
            },
        },
        "checkpoints": {
            "flat": None if flat_checkpoint is None else str(flat_checkpoint),
            "flat_provisional": bool(flat_checkpoint is not None and flat_checkpoint.name == "last.pt"),
            "independent_training_summary": str(Path(args.independent_training_summary).expanduser().resolve()),
            "shared": str(shared_checkpoint),
            "duplicate_selection": duplicate_selection,
        },
        "cams": [],
    }

    for mode in modes:
        shared_target_species = true_species if mode == "true" else shared_prediction
        independent_target_species = true_species if mode == "true" else independent_prediction
        flat_target_species = true_species if mode == "true" else flat_prediction
        shared_target_leaf_probability = float(
            shared_leaf_probabilities[shared_target_species].item()
        )
        independent_target_leaf_probability = float(
            independent_leaf_probabilities[0, independent_target_species].item()
        )
        flat_target_leaf_probability = (
            None
            if flat_target_species is None or flat_leaf_probabilities is None
            else float(flat_leaf_probabilities[flat_target_species].item())
        )
        mode_dir = output_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)

        independent_maps, independent_entries = _independent_path_cams(
            records,
            taxonomy,
            independent_target_species,
            batch,
            device,
            args.top_traits,
        )
        independent_images: dict[str, Image.Image | None] = {}
        for level in LEVELS:
            cam = independent_maps[level]
            if cam is None:
                independent_images[level] = None
                continue
            image = _overlay(image_rgb, cam, patch_size)
            path = mode_dir / "independent" / f"{level}_cam.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            image.save(path)
            independent_images[level] = image
        for entry in independent_entries:
            entry["target_mode"] = mode
            entry["target_species"] = _taxonomy_names(taxonomy, independent_target_species)
            entry["p_leaf"] = independent_target_leaf_probability
            entry["target_semantics"] = {
                "family": "P(family | x)",
                "genus": "P(genus | family, x)",
                "species": "P(species | genus, x)",
            }[entry["level"]]
        metadata["cams"].extend(independent_entries)
        independent_by_level = {entry["level"]: entry for entry in independent_entries}

        shared_images: dict[str, Image.Image] = {}
        shared_by_level: dict[str, dict[str, Any]] = {}
        shared_indices = _target_indices(taxonomy, shared_target_species)
        for level in LEVELS:
            cam, entry = _shared_cam(
                shared_model,
                batch,
                level=level,
                target_index=shared_indices[level],
                top_traits=args.top_traits,
            )
            image = _overlay(image_rgb, cam, patch_size)
            path = mode_dir / "shared" / f"{level}_cam.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            image.save(path)
            shared_images[level] = image
            full_entry = {
                "model": "shared_hierarchical",
                "level": level,
                "target_mode": mode,
                "target_species": _taxonomy_names(taxonomy, shared_target_species),
                "p_leaf": shared_target_leaf_probability,
                "target_semantics": {
                    "family": "P(family | x)",
                    "genus": "P(genus | family, x)",
                    "species": "P(species | genus, x)",
                }[level],
                **entry,
            }
            metadata["cams"].append(full_entry)
            shared_by_level[level] = full_entry

        flat_image = None
        flat_entry = None
        if flat_model is not None and flat_target_species is not None:
            cam, entry = _flat_cam(
                flat_model,
                batch,
                flat_params,
                target_index=flat_target_species,
                top_traits=args.top_traits,
            )
            flat_image = _overlay(image_rgb, cam, patch_size)
            path = mode_dir / "flat" / "species_cam.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            flat_image.save(path)
            flat_entry = {
                "model": "flat_promptcam",
                "level": "species",
                "target_mode": mode,
                "target_species": _taxonomy_names(taxonomy, flat_target_species),
                "p_leaf": flat_target_leaf_probability,
                "target_semantics": "P(species | x) over all 30 species",
                **entry,
            }
            metadata["cams"].append(flat_entry)

        species_row: list[tuple[str, Image.Image]] = [
            (f"Input — {mode.capitalize()}-class comparison", input_image)
        ]
        if flat_image is not None:
            species_row.append(
                (
                    _cam_title(
                        "Flat",
                        "species",
                        taxonomy,
                        flat_target_species,
                        flat_entry,
                        target_mode=mode,
                    ),
                    flat_image,
                )
            )
        independent_species_entry = independent_by_level["species"]
        independent_species = independent_images["species"] or _placeholder(
            input_image.size,
            "No species-level classifier\n(singleton genus)",
        )
        species_row.extend(
            [
                (
                    (
                        "Independent — No species-level classifier\n(singleton genus)"
                        f"\nP_leaf={independent_target_leaf_probability:.3f}"
                        if independent_species_entry.get("deterministic_singleton")
                        else _cam_title(
                            "Independent",
                            "species",
                            taxonomy,
                            independent_target_species,
                            independent_species_entry,
                            target_mode=mode,
                        )
                    ),
                    independent_species,
                ),
                (
                    _cam_title(
                        "Shared",
                        "species",
                        taxonomy,
                        shared_target_species,
                        shared_by_level["species"],
                        target_mode=mode,
                    ),
                    shared_images["species"],
                ),
            ]
        )
        _save_grid([species_row], mode_dir / "comparison_species.png")
        _save_grid([species_row], mode_dir / "comparison_montage.png")

        hierarchy_rows: list[list[tuple[str, Image.Image]]] = []
        for level in LEVELS:
            independent_entry = independent_by_level[level]
            independent_image = independent_images[level] or _placeholder(
                input_image.size,
                f"No {level}-level classifier\n(singleton parent node)",
            )
            independent_title = (
                f"Independent — No {level}-level classifier\n(singleton parent node)\n"
                f"P_leaf={independent_target_leaf_probability:.3f}"
                if independent_entry.get("deterministic_singleton")
                else _cam_title(
                    "Independent",
                    level,
                    taxonomy,
                    independent_target_species,
                    independent_entry,
                    target_mode=mode,
                )
            )
            hierarchy_rows.append(
                [
                    (independent_title, independent_image),
                    (
                        _cam_title(
                            "Shared",
                            level,
                            taxonomy,
                            shared_target_species,
                            shared_by_level[level],
                            target_mode=mode,
                        ),
                        shared_images[level],
                    ),
                ]
            )
        _save_hierarchy_grid(
            input_image,
            f"Input — same transformed tensor — {mode.capitalize()}-class targets",
            hierarchy_rows,
            mode_dir / "comparison_hierarchy.png",
        )

        mode_entries = [entry for entry in metadata["cams"] if entry.get("target_mode") == mode]
        for folder_name, model_name in (
            ("flat", "flat_promptcam"),
            ("independent", "independent_taxonomy"),
            ("shared", "shared_hierarchical"),
        ):
            folder = mode_dir / folder_name
            matching = [entry for entry in mode_entries if entry.get("model") == model_name]
            if not matching and folder_name == "flat":
                continue
            folder.mkdir(parents=True, exist_ok=True)
            with (folder / "metadata.json").open("w", encoding="utf-8") as handle:
                json.dump(matching, handle, ensure_ascii=False, indent=2, default=_json_value)

    metadata["target_probability_table"] = _probability_table(metadata["cams"])
    if "ulmaceae_genus_routing_failure" in metadata["case_tags"]:
        metadata["report_interpretation"] = {
            "recommended_section": "Failure analysis: error propagation at the Ulmaceae genus node",
            "claim_boundary": (
                "This case demonstrates genus-routing error propagation; it must not be presented "
                "as evidence that the shared hierarchy has a superior CAM."
            ),
            "bottleneck_level": "genus",
            "summary": (
                "The taxonomy models can strongly support the true species conditional on Ulmus, "
                "while the final leaf prediction still fails because the Ulmaceae branch is routed "
                "from Ulmus to Zelkova."
            ),
        }

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, default=_json_value)

    del flat_model
    del shared_model
    _clear_device_cache()
    print(json.dumps({"output_dir": str(output_dir), "case": metadata["case"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
