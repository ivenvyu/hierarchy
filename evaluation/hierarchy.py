#!/usr/bin/env python3
"""공유 계층 Prompt-CAM 체크포인트를 Wild-30에서 평가한다.

모델이 한 번의 forward에서 반환하는

    P(family | x),
    P(genus | family, x),
    P(species | genus, x)

를 사용해 joint species MAP, greedy routing, taxonomy 수준 정확도, node-conditional
정확도, 종별 F1 및 confusion matrix를 계산한다. 클래스와 taxonomy 순서는
``args.yaml``/``taxonomy.json``/ImageFolder에서 읽으며 하드코딩하지 않는다.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import torch
import yaml
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class TaxonomyInfo:
    class_names: tuple[str, ...]
    scientific_names: tuple[str, ...]
    genus_names: tuple[str, ...]
    family_names: tuple[str, ...]
    species_to_genus: tuple[int, ...]
    genus_to_family: tuple[int, ...]

    @property
    def num_species(self) -> int:
        return len(self.class_names)

    @property
    def num_genera(self) -> int:
        return len(self.genus_names)

    @property
    def num_families(self) -> int:
        return len(self.family_names)

    @property
    def species_to_family(self) -> tuple[int, ...]:
        return tuple(self.genus_to_family[g] for g in self.species_to_genus)


# ---------------------------------------------------------------------------
# 일반 유틸리티
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"YAML 최상위 값이 mapping이 아닙니다: {path}")
    return value


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def _slug(value: str) -> str:
    result = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    return result or "unnamed"


def _resolve_path(value: Any, *, project_root: Path, field: str, required: bool = True) -> Path | None:
    if value in (None, "", "null"):
        if required:
            raise ValueError(f"{field} 값이 비어 있습니다")
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = project_root / path
    path = path.resolve()
    if required and not path.exists():
        raise FileNotFoundError(f"{field} 경로가 없습니다: {path}")
    return path


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _device_from_arg(value: str) -> torch.device:
    normalized = str(value).strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 장치를 요청했지만 torch.cuda.is_available()가 False입니다")
    return device


def _autocast_context(device: torch.device, amp_dtype: str):
    if device.type != "cuda":
        return nullcontext()
    normalized = str(amp_dtype).lower()
    if normalized == "float16":
        return torch.amp.autocast("cuda", dtype=torch.float16)
    if normalized == "bfloat16":
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def _discover_run_dir(project_root: Path, search_root: Path | None) -> Path:
    root = search_root or (project_root / "output")
    if not root.is_absolute():
        root = (project_root / root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"실험 검색 디렉터리가 없습니다: {root}")

    candidates: list[tuple[float, Path]] = []
    for args_path in root.rglob("args.yaml"):
        run_dir = args_path.parent
        checkpoint = run_dir / "model.pt"
        final_result = run_dir / "final_result.json"
        if not checkpoint.is_file() or not final_result.is_file():
            continue
        try:
            args_data = _load_yaml(args_path)
        except Exception:
            continue
        if not bool(args_data.get("hierarchical_prompt", False)):
            continue
        if bool(args_data.get("original_taxonomy_prompt", False)):
            continue
        candidates.append((checkpoint.stat().st_mtime, run_dir))

    if not candidates:
        raise FileNotFoundError(
            f"{root} 아래에서 args.yaml, model.pt, final_result.json을 모두 가진 완료된 계층 Prompt-CAM 실행을 찾지 못했습니다"
        )
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1].resolve()


def _load_taxonomy(run_dir: Path, params: SimpleNamespace) -> TaxonomyInfo:
    taxonomy_path = run_dir / "taxonomy.json"
    if taxonomy_path.is_file():
        data = _load_json(taxonomy_path)
    else:
        data = getattr(params, "taxonomy", None)
    if not isinstance(data, dict):
        raise ValueError(f"taxonomy 정보를 찾지 못했습니다: {taxonomy_path}")

    required = {
        "class_names",
        "scientific_names",
        "genus_names",
        "family_names",
        "species_to_genus",
        "genus_to_family",
    }
    missing = required - set(data)
    if missing:
        raise ValueError(f"taxonomy 정보에 필요한 필드가 없습니다: {sorted(missing)}")

    taxonomy = TaxonomyInfo(
        class_names=tuple(str(x) for x in data["class_names"]),
        scientific_names=tuple(str(x) for x in data["scientific_names"]),
        genus_names=tuple(str(x) for x in data["genus_names"]),
        family_names=tuple(str(x) for x in data["family_names"]),
        species_to_genus=tuple(int(x) for x in data["species_to_genus"]),
        genus_to_family=tuple(int(x) for x in data["genus_to_family"]),
    )
    if taxonomy.num_species != len(taxonomy.scientific_names):
        raise ValueError("class_names와 scientific_names 길이가 다릅니다")
    if len(taxonomy.species_to_genus) != taxonomy.num_species:
        raise ValueError("species_to_genus 길이가 species 수와 다릅니다")
    if len(taxonomy.genus_to_family) != taxonomy.num_genera:
        raise ValueError("genus_to_family 길이가 genus 수와 다릅니다")
    return taxonomy


# ---------------------------------------------------------------------------
# 지표 계산: 단위 테스트 가능한 순수 함수
# ---------------------------------------------------------------------------


def classification_metrics(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    predictions: torch.Tensor | None = None,
) -> tuple[dict[str, float], torch.Tensor, list[dict[str, float | int]]]:
    probabilities = probabilities.float().cpu()
    targets = targets.long().cpu()
    if probabilities.ndim != 2:
        raise ValueError("probabilities는 [N,C]여야 합니다")
    if targets.shape != (probabilities.shape[0],):
        raise ValueError("targets 형태가 sample 수와 일치하지 않습니다")

    predictions = (
        probabilities.argmax(dim=1)
        if predictions is None
        else predictions.long().cpu()
    )
    class_count = probabilities.shape[1]
    indices = targets * class_count + predictions
    confusion = torch.bincount(
        indices,
        minlength=class_count * class_count,
    ).reshape(class_count, class_count)

    true_count = confusion.sum(dim=1).float()
    pred_count = confusion.sum(dim=0).float()
    true_positive = confusion.diag().float()
    precision = true_positive / pred_count.clamp_min(1.0)
    recall = true_positive / true_count.clamp_min(1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-12)
    present = true_count.gt(0)

    top_k = min(5, class_count)
    top5_correct = probabilities.topk(top_k, dim=1).indices.eq(targets[:, None]).any(dim=1)
    metrics = {
        "top1": float(predictions.eq(targets).float().mean().item() * 100.0),
        "top5": float(top5_correct.float().mean().item() * 100.0),
        "balanced_accuracy": float(recall[present].mean().item() * 100.0),
        "macro_f1": float(f1[present].mean().item() * 100.0),
    }

    per_class: list[dict[str, float | int]] = []
    for index in range(class_count):
        per_class.append(
            {
                "support": int(true_count[index].item()),
                "predicted_count": int(pred_count[index].item()),
                "correct": int(true_positive[index].item()),
                "precision": float(precision[index].item() * 100.0),
                "recall": float(recall[index].item() * 100.0),
                "f1": float(f1[index].item() * 100.0),
            }
        )
    return metrics, confusion, per_class


def taxonomy_metrics(
    taxonomy: TaxonomyInfo,
    targets: torch.Tensor,
    predictions: torch.Tensor,
) -> dict[str, float]:
    targets = targets.long().cpu()
    predictions = predictions.long().cpu()
    species_to_genus = torch.tensor(taxonomy.species_to_genus, dtype=torch.long)
    species_to_family = torch.tensor(taxonomy.species_to_family, dtype=torch.long)

    true_genus = species_to_genus.index_select(0, targets)
    pred_genus = species_to_genus.index_select(0, predictions)
    true_family = species_to_family.index_select(0, targets)
    pred_family = species_to_family.index_select(0, predictions)

    same_species = predictions.eq(targets)
    same_genus = pred_genus.eq(true_genus)
    same_family = pred_family.eq(true_family)
    distance = torch.where(
        same_species,
        torch.zeros_like(targets),
        torch.where(
            same_genus,
            torch.ones_like(targets),
            torch.where(same_family, torch.full_like(targets, 2), torch.full_like(targets, 3)),
        ),
    )
    return {
        "genus_accuracy": float(same_genus.float().mean().item() * 100.0),
        "family_accuracy": float(same_family.float().mean().item() * 100.0),
        "mean_taxonomic_distance": float(distance.float().mean().item()),
    }


def greedy_predictions(
    taxonomy: TaxonomyInfo,
    family_probabilities: torch.Tensor,
    genus_conditional_probabilities: torch.Tensor,
    species_conditional_probabilities: torch.Tensor,
) -> torch.Tensor:
    family_probabilities = family_probabilities.cpu()
    genus_conditional_probabilities = genus_conditional_probabilities.cpu()
    species_conditional_probabilities = species_conditional_probabilities.cpu()
    sample_count = family_probabilities.shape[0]

    genus_to_family = torch.tensor(taxonomy.genus_to_family, dtype=torch.long)
    species_to_genus = torch.tensor(taxonomy.species_to_genus, dtype=torch.long)
    predicted_family = family_probabilities.argmax(dim=1)
    predicted_genus = torch.empty(sample_count, dtype=torch.long)

    for family_index in range(taxonomy.num_families):
        sample_mask = predicted_family.eq(family_index)
        if not bool(sample_mask.any()):
            continue
        members = torch.nonzero(genus_to_family.eq(family_index), as_tuple=False).flatten()
        local_prediction = genus_conditional_probabilities[sample_mask][:, members].argmax(dim=1)
        predicted_genus[sample_mask] = members.index_select(0, local_prediction)

    predicted_species = torch.empty(sample_count, dtype=torch.long)
    for genus_index in range(taxonomy.num_genera):
        sample_mask = predicted_genus.eq(genus_index)
        if not bool(sample_mask.any()):
            continue
        members = torch.nonzero(species_to_genus.eq(genus_index), as_tuple=False).flatten()
        local_prediction = species_conditional_probabilities[sample_mask][:, members].argmax(dim=1)
        predicted_species[sample_mask] = members.index_select(0, local_prediction)
    return predicted_species


def node_conditional_metrics(
    taxonomy: TaxonomyInfo,
    targets: torch.Tensor,
    family_probabilities: torch.Tensor,
    genus_conditional_probabilities: torch.Tensor,
    species_conditional_probabilities: torch.Tensor,
) -> dict[str, dict[str, Any]]:
    targets = targets.long().cpu()
    family_probabilities = family_probabilities.cpu()
    genus_conditional_probabilities = genus_conditional_probabilities.cpu()
    species_conditional_probabilities = species_conditional_probabilities.cpu()

    species_to_genus = torch.tensor(taxonomy.species_to_genus, dtype=torch.long)
    genus_to_family = torch.tensor(taxonomy.genus_to_family, dtype=torch.long)
    true_genus = species_to_genus.index_select(0, targets)
    true_family = genus_to_family.index_select(0, true_genus)

    result: dict[str, dict[str, Any]] = {}
    root_prediction = family_probabilities.argmax(dim=1)
    result["root"] = {
        "rank": "family",
        "name": "root",
        "sample_count": int(targets.numel()),
        "child_count": taxonomy.num_families,
        "accuracy": float(root_prediction.eq(true_family).float().mean().item() * 100.0),
    }

    for family_index, family_name in enumerate(taxonomy.family_names):
        genus_members = torch.nonzero(genus_to_family.eq(family_index), as_tuple=False).flatten()
        if genus_members.numel() <= 1:
            continue
        sample_mask = true_family.eq(family_index)
        local = genus_conditional_probabilities[sample_mask][:, genus_members]
        predicted = genus_members.index_select(0, local.argmax(dim=1))
        result[f"family__{_slug(family_name)}"] = {
            "rank": "genus",
            "name": family_name,
            "sample_count": int(sample_mask.sum().item()),
            "child_count": int(genus_members.numel()),
            "accuracy": float(predicted.eq(true_genus[sample_mask]).float().mean().item() * 100.0),
        }

    for genus_index, genus_name in enumerate(taxonomy.genus_names):
        species_members = torch.nonzero(species_to_genus.eq(genus_index), as_tuple=False).flatten()
        if species_members.numel() <= 1:
            continue
        sample_mask = true_genus.eq(genus_index)
        local = species_conditional_probabilities[sample_mask][:, species_members]
        predicted = species_members.index_select(0, local.argmax(dim=1))
        result[f"genus__{_slug(genus_name)}"] = {
            "rank": "species",
            "name": genus_name,
            "sample_count": int(sample_mask.sum().item()),
            "child_count": int(species_members.numel()),
            "accuracy": float(predicted.eq(targets[sample_mask]).float().mean().item() * 100.0),
        }
    return result


def conditional_probability_sum_error(
    taxonomy: TaxonomyInfo,
    genus_conditional_probabilities: torch.Tensor,
    species_conditional_probabilities: torch.Tensor,
) -> float:
    genus_to_family = torch.tensor(taxonomy.genus_to_family, dtype=torch.long)
    species_to_genus = torch.tensor(taxonomy.species_to_genus, dtype=torch.long)
    errors: list[torch.Tensor] = []
    for family_index in range(taxonomy.num_families):
        members = torch.nonzero(genus_to_family.eq(family_index), as_tuple=False).flatten()
        errors.append(
            (genus_conditional_probabilities[:, members].sum(dim=1) - 1.0).abs().max()
        )
    for genus_index in range(taxonomy.num_genera):
        members = torch.nonzero(species_to_genus.eq(genus_index), as_tuple=False).flatten()
        errors.append(
            (species_conditional_probabilities[:, members].sum(dim=1) - 1.0).abs().max()
        )
    return float(torch.stack(errors).max().item()) if errors else 0.0


# ---------------------------------------------------------------------------
# 실제 모델 평가
# ---------------------------------------------------------------------------


def _prepare_params(args_data: dict[str, Any], project_root: Path, run_dir: Path, cli_args) -> SimpleNamespace:
    params = SimpleNamespace(**args_data)
    params.distributed = False
    params.local_rank = 0
    params.load_pretrained_backbone = False
    params.store_ckp = False
    params.debug = False
    params.vis_attn = False
    params.output_dir = str(run_dir)
    params.test_batch_size = int(cli_args.batch_size or getattr(params, "test_batch_size", 64))
    params.num_workers = int(
        cli_args.num_workers if cli_args.num_workers is not None else getattr(params, "num_workers", 4)
    )

    data_path = _resolve_path(getattr(params, "data_path", None), project_root=project_root, field="data_path")
    params.data_path = str(data_path)
    taxonomy_manifest = _resolve_path(
        getattr(params, "taxonomy_manifest", None),
        project_root=project_root,
        field="taxonomy_manifest",
    )
    params.taxonomy_manifest = str(taxonomy_manifest)

    optional_manifest = _resolve_path(
        getattr(params, "identifiability_manifest", None),
        project_root=project_root,
        field="identifiability_manifest",
        required=False,
    )
    params.identifiability_manifest = None if optional_manifest is None else str(optional_manifest)
    return params


def _load_model(project_root: Path, run_dir: Path, params: SimpleNamespace, device: torch.device):
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from data.dataset.imagefolder import get_imagefolder
    from model.factory import get_model

    test_dataset = get_imagefolder(params, mode="test")
    if test_dataset is None:
        raise RuntimeError("test dataset을 생성하지 못했습니다")

    model, _, _ = get_model(params)
    checkpoint_path = run_dir / "model.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"최선 체크포인트가 없습니다: {checkpoint_path}")
    checkpoint = _torch_load(checkpoint_path)
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "체크포인트와 현재 모델 구조가 일치하지 않습니다: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    model = model.to(device)
    model.eval()
    return model, test_dataset, checkpoint_path, checkpoint


def _collect_outputs(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    amp_dtype: str,
) -> dict[str, Any]:
    tensor_keys = [
        "species_probabilities",
        "genus_probabilities",
        "family_probabilities",
        "genus_conditional_probabilities",
        "species_conditional_probabilities",
        "rank_logits",
    ]
    collected: dict[str, list[torch.Tensor]] = {key: [] for key in tensor_keys}
    targets: list[torch.Tensor] = []
    genus_targets: list[torch.Tensor] = []
    family_targets: list[torch.Tensor] = []
    rank_targets: list[torch.Tensor] = []
    paths: list[str] = []

    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, start=1):
            images = batch["image"].to(device, non_blocking=True)
            with _autocast_context(device, amp_dtype):
                output, _ = model(images, patch_prior=None)
            if not isinstance(output, dict):
                raise TypeError("계층 모델 출력이 dict가 아닙니다")
            for key in tensor_keys:
                if key not in output:
                    raise KeyError(f"계층 모델 출력에 {key!r}가 없습니다")
                collected[key].append(output[key].detach().float().cpu())
            targets.append(batch["species_target"].long().cpu())
            genus_targets.append(batch["genus_target"].long().cpu())
            family_targets.append(batch["family_target"].long().cpu())
            rank_targets.append(batch["rank_target"].long().cpu())
            paths.extend(str(path) for path in batch["relative_path"])

            if batch_index == 1 or batch_index % 10 == 0 or batch_index == len(loader):
                print(f"[추론] batch {batch_index}/{len(loader)}")

    result: dict[str, Any] = {
        key: torch.cat(value, dim=0)
        for key, value in collected.items()
    }
    result["targets"] = torch.cat(targets)
    result["genus_targets"] = torch.cat(genus_targets)
    result["family_targets"] = torch.cat(family_targets)
    result["rank_targets"] = torch.cat(rank_targets)
    result["relative_paths"] = paths
    return result


def _checkpoint_metadata(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "epoch": checkpoint.get("epoch"),
        "selection_metric": checkpoint.get("selection_metric"),
        "best_value": checkpoint.get("best_value"),
        "best_metrics": checkpoint.get("best_metrics", {}),
        "current_metrics": checkpoint.get("current_metrics", {}),
    }


def evaluate(cli_args) -> dict[str, Any]:
    project_root = Path(cli_args.project_root).expanduser().resolve()
    if not project_root.is_dir():
        raise FileNotFoundError(f"프로젝트 루트가 없습니다: {project_root}")

    if cli_args.run_dir:
        run_dir = Path(cli_args.run_dir).expanduser()
        if not run_dir.is_absolute():
            run_dir = project_root / run_dir
        run_dir = run_dir.resolve()
    else:
        search_root = Path(cli_args.search_root).expanduser() if cli_args.search_root else None
        run_dir = _discover_run_dir(project_root, search_root)
        print(f"[자동 선택] 가장 최근 계층 모델 실행: {run_dir}")

    args_path = run_dir / "args.yaml"
    if not args_path.is_file():
        raise FileNotFoundError(f"args.yaml이 없습니다: {args_path}")
    args_data = _load_yaml(args_path)
    if not bool(args_data.get("hierarchical_prompt", False)):
        raise ValueError(f"선택한 실행은 hierarchical_prompt=True가 아닙니다: {run_dir}")

    params = _prepare_params(args_data, project_root, run_dir, cli_args)
    device = _device_from_arg(cli_args.device)
    model, test_dataset, checkpoint_path, checkpoint = _load_model(
        project_root, run_dir, params, device
    )
    taxonomy = _load_taxonomy(run_dir, params)

    if tuple(test_dataset.classes) != taxonomy.class_names:
        raise ValueError(
            "test ImageFolder 클래스 순서와 저장된 taxonomy 순서가 다릅니다: "
            f"test={list(test_dataset.classes)}, taxonomy={list(taxonomy.class_names)}"
        )

    loader = DataLoader(
        test_dataset,
        batch_size=int(params.test_batch_size),
        shuffle=False,
        num_workers=int(params.num_workers),
        pin_memory=device.type == "cuda",
        persistent_workers=int(params.num_workers) > 0,
    )
    print(
        f"[평가 대상] run={run_dir}\n"
        f"[체크포인트] {checkpoint_path}\n"
        f"[데이터] split={getattr(params, 'test_split', 'test')}, "
        f"samples={len(test_dataset)}, batch={params.test_batch_size}, device={device}"
    )

    outputs = _collect_outputs(
        model,
        loader,
        device=device,
        amp_dtype="none",
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    species_probabilities = outputs["species_probabilities"]
    genus_probabilities = outputs["genus_probabilities"]
    family_probabilities = outputs["family_probabilities"]
    rank_logits = outputs["rank_logits"]
    genus_conditional = outputs["genus_conditional_probabilities"]
    species_conditional = outputs["species_conditional_probabilities"]
    targets = outputs["targets"]

    joint_sum_error = float((species_probabilities.sum(dim=1) - 1.0).abs().max().item())
    conditional_sum_error = conditional_probability_sum_error(
        taxonomy, genus_conditional, species_conditional
    )
    species_probabilities = species_probabilities / species_probabilities.sum(dim=1, keepdim=True).clamp_min(1e-12)

    joint_prediction = species_probabilities.argmax(dim=1)
    greedy_prediction = greedy_predictions(
        taxonomy,
        family_probabilities,
        genus_conditional,
        species_conditional,
    )

    joint_classification, confusion, per_class = classification_metrics(
        species_probabilities, targets, joint_prediction
    )
    joint_taxonomy = taxonomy_metrics(taxonomy, targets, joint_prediction)
    greedy_classification, _, _ = classification_metrics(
        species_probabilities, targets, greedy_prediction
    )
    greedy_taxonomy = taxonomy_metrics(taxonomy, targets, greedy_prediction)
    greedy_classification["routing_disagreement_with_joint_map"] = float(
        greedy_prediction.ne(joint_prediction).float().mean().item() * 100.0
    )
    conditional_metrics = node_conditional_metrics(
        taxonomy,
        targets,
        family_probabilities,
        genus_conditional,
        species_conditional,
    )

    direct_genus_prediction = genus_probabilities.argmax(dim=1)
    direct_family_prediction = family_probabilities.argmax(dim=1)
    direct_rank_metrics = {
        "genus_accuracy": float(
            direct_genus_prediction.eq(outputs["genus_targets"]).float().mean().item() * 100.0
        ),
        "family_accuracy": float(
            direct_family_prediction.eq(outputs["family_targets"]).float().mean().item() * 100.0
        ),
        "rank_accuracy": (
            float(
                rank_logits.argmax(dim=1).eq(outputs["rank_targets"]).float().mean().item() * 100.0
            )
            if bool(getattr(params, "identifiability_enabled", False))
            else 0.0
        ),
    }

    output_dir = (
        Path(cli_args.output_dir).expanduser().resolve()
        if cli_args.output_dir
        else run_dir / "evaluation" / f"hierarchical_{getattr(params, 'test_split', 'test')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    species_to_genus = torch.tensor(taxonomy.species_to_genus, dtype=torch.long)
    species_to_family = torch.tensor(taxonomy.species_to_family, dtype=torch.long)
    true_probability = species_probabilities.gather(1, targets[:, None]).squeeze(1)
    joint_confidence, _ = species_probabilities.max(dim=1)
    top5_indices = species_probabilities.topk(min(5, taxonomy.num_species), dim=1).indices

    prediction_rows: list[dict[str, Any]] = []
    for index in range(targets.numel()):
        true_index = int(targets[index])
        joint_index = int(joint_prediction[index])
        greedy_index = int(greedy_prediction[index])
        true_genus_index = int(species_to_genus[true_index])
        pred_genus_index = int(species_to_genus[joint_index])
        true_family_index = int(species_to_family[true_index])
        pred_family_index = int(species_to_family[joint_index])
        distance = (
            0 if joint_index == true_index
            else 1 if pred_genus_index == true_genus_index
            else 2 if pred_family_index == true_family_index
            else 3
        )
        prediction_rows.append(
            {
                "relative_path": outputs["relative_paths"][index],
                "true_folder": taxonomy.class_names[true_index],
                "true_scientific_name": taxonomy.scientific_names[true_index],
                "true_genus": taxonomy.genus_names[true_genus_index],
                "true_family": taxonomy.family_names[true_family_index],
                "joint_pred_folder": taxonomy.class_names[joint_index],
                "joint_pred_scientific_name": taxonomy.scientific_names[joint_index],
                "joint_pred_genus": taxonomy.genus_names[pred_genus_index],
                "joint_pred_family": taxonomy.family_names[pred_family_index],
                "joint_correct": int(joint_index == true_index),
                "genus_correct": int(pred_genus_index == true_genus_index),
                "family_correct": int(pred_family_index == true_family_index),
                "taxonomic_distance": distance,
                "joint_confidence": float(joint_confidence[index]),
                "true_leaf_probability": float(true_probability[index]),
                "greedy_pred_folder": taxonomy.class_names[greedy_index],
                "greedy_correct": int(greedy_index == true_index),
                "top5_folders": "|".join(
                    taxonomy.class_names[int(class_index)]
                    for class_index in top5_indices[index]
                ),
            }
        )
    _write_csv(
        output_dir / "predictions.csv",
        tuple(prediction_rows[0].keys()),
        prediction_rows,
    )

    per_class_rows: list[dict[str, Any]] = []
    for index, metrics in enumerate(per_class):
        per_class_rows.append(
            {
                "class_index": index,
                "folder_name": taxonomy.class_names[index],
                "scientific_name": taxonomy.scientific_names[index],
                "genus": taxonomy.genus_names[taxonomy.species_to_genus[index]],
                "family": taxonomy.family_names[taxonomy.species_to_family[index]],
                **metrics,
            }
        )
    _write_csv(
        output_dir / "per_class_metrics.csv",
        tuple(per_class_rows[0].keys()),
        per_class_rows,
    )

    confusion_rows: list[dict[str, Any]] = []
    for true_index, true_name in enumerate(taxonomy.class_names):
        row: dict[str, Any] = {"true_class": true_name}
        for pred_index, pred_name in enumerate(taxonomy.class_names):
            row[pred_name] = int(confusion[true_index, pred_index])
        confusion_rows.append(row)
    _write_csv(
        output_dir / "confusion_matrix.csv",
        ("true_class", *taxonomy.class_names),
        confusion_rows,
    )

    result = {
        "model_type": "single_hierarchical_promptcam",
        "run_dir": str(run_dir),
        "split": str(getattr(params, "test_split", "test")),
        "sample_count": int(targets.numel()),
        "class_count": taxonomy.num_species,
        "checkpoint": str(checkpoint_path),
        "checkpoint_metadata": _checkpoint_metadata(checkpoint),
        "probability_definition": (
            "P(species|x)=P(family|x)*P(genus|family,x)*P(species|genus,x), "
            "computed in one shared hierarchical model forward"
        ),
        "probability_sum_max_absolute_error": joint_sum_error,
        "conditional_probability_sum_max_absolute_error": conditional_sum_error,
        "joint_map": {**joint_classification, **joint_taxonomy},
        "greedy_routing": {**greedy_classification, **greedy_taxonomy},
        "node_conditional_metrics": conditional_metrics,
        "direct_rank_predictions": direct_rank_metrics,
        "taxonomy": {
            "families": list(taxonomy.family_names),
            "genera": list(taxonomy.genus_names),
            "species": list(taxonomy.class_names),
        },
        "artifacts": {
            "predictions_csv": str(output_dir / "predictions.csv"),
            "per_class_metrics_csv": str(output_dir / "per_class_metrics.csv"),
            "confusion_matrix_csv": str(output_dir / "confusion_matrix.csv"),
        },
    }
    result_path = output_dir / "evaluation_summary.json"
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)

    print("\n" + "=" * 72)
    print("단일 계층 Prompt-CAM 평가 결과")
    print("=" * 72)
    print(f"Joint MAP Top-1          : {joint_classification['top1']:.4f}%")
    print(f"Joint MAP Macro-F1       : {joint_classification['macro_f1']:.4f}%")
    print(f"Joint MAP Top-5          : {joint_classification['top5']:.4f}%")
    print(f"Joint genus accuracy     : {joint_taxonomy['genus_accuracy']:.4f}%")
    print(f"Joint family accuracy    : {joint_taxonomy['family_accuracy']:.4f}%")
    print(f"Mean taxonomic distance  : {joint_taxonomy['mean_taxonomic_distance']:.6f}")
    print(f"Greedy routing Top-1     : {greedy_classification['top1']:.4f}%")
    print(f"Direct genus accuracy    : {direct_rank_metrics['genus_accuracy']:.4f}%")
    print(f"Direct family accuracy   : {direct_rank_metrics['family_accuracy']:.4f}%")
    if bool(getattr(params, "identifiability_enabled", False)):
        print(f"Direct rank accuracy     : {direct_rank_metrics['rank_accuracy']:.4f}%")
    print(f"Joint 확률합 최대 오차    : {joint_sum_error:.6g}")
    print(f"조건부 확률합 최대 오차   : {conditional_sum_error:.6g}")
    print(f"결과 저장                : {result_path}")
    print("=" * 72)

    print("\n[노드별 조건부 정확도]")
    for key, metrics in conditional_metrics.items():
        print(
            f"{key:<22} rank={metrics['rank']:<8} "
            f"N={metrics['sample_count']:<4} accuracy={metrics['accuracy']:.4f}%"
        )

    print("\n[F1이 낮은 species 10개]")
    for row in sorted(per_class_rows, key=lambda item: float(item["f1"]))[:10]:
        print(
            f"{row['scientific_name']:<30} genus={row['genus']:<12} "
            f"precision={float(row['precision']):6.2f}%  "
            f"recall={float(row['recall']):6.2f}%  f1={float(row['f1']):6.2f}%"
        )

    pairs: list[tuple[int, str, str]] = []
    for true_index, true_name in enumerate(taxonomy.class_names):
        for pred_index, pred_name in enumerate(taxonomy.class_names):
            count = int(confusion[true_index, pred_index])
            if true_index != pred_index and count > 0:
                pairs.append((count, true_name, pred_name))
    pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
    print("\n[가장 빈번한 오분류 쌍 20개]")
    for count, true_name, pred_name in pairs[:20]:
        print(f"{count:>3}장: {true_name} -> {pred_name}")

    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="단일 계층 Prompt-CAM의 joint/conditional taxonomy 성능을 전체 test split에서 계산"
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="PromptCAM-SnapMix 프로젝트 루트",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="평가할 실행 디렉터리(args.yaml과 model.pt가 있는 경로). 생략하면 최신 계층 실행 자동 선택",
    )
    parser.add_argument(
        "--search-root",
        default="output",
        help="--run-dir 생략 시 계층 실행을 검색할 디렉터리",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, cpu")
    parser.add_argument("--output-dir", default=None)
    return parser


def main() -> None:
    evaluate(build_parser().parse_args())


if __name__ == "__main__":
    main()
