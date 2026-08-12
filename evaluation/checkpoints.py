"""독립 taxonomy node 체크포인트를 직접 지정해 결합 평가한다.

두 종류의 end-to-end 예측을 함께 계산한다.

1. soft path: 각 종 경로의 조건부 log-probability를 더해 전역 argmax
2. hard traversal: root에서 자식 argmax를 선택하며 leaf까지 greedy 이동

singleton node는 별도 모델 없이 확률 1의 결정적 경로로 처리한다.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset.imagefolder import (  # noqa: E402
    FlatImageFolder,
    JointImageTransform,
    load_taxonomy_manifest,
)
from data.original_taxonomy import (  # noqa: E402
    TaxonomyNodeSpec,
    list_taxonomy_nodes,
    node_lookup,
)


def _torch_load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _summary_checkpoint_paths(summary_path: str | Path) -> list[Path]:
    path = Path(summary_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"training summary가 없습니다: {path}")
    summary = json.loads(path.read_text(encoding="utf-8"))
    paths: list[Path] = []
    missing: list[str] = []
    for record in summary.get("nodes", []):
        node_id = str(record.get("node", {}).get("node_id", "unknown"))
        checkpoint = record.get("checkpoint")
        returncode = record.get("returncode")
        if returncode not in (None, 0):
            missing.append(f"{node_id}(returncode={returncode})")
            continue
        if checkpoint in (None, "", "null"):
            missing.append(f"{node_id}(checkpoint 미기록)")
            continue
        checkpoint_path = Path(str(checkpoint)).expanduser()
        if not checkpoint_path.is_absolute():
            checkpoint_path = path.parent / checkpoint_path
        paths.append(checkpoint_path.resolve())
    if missing:
        raise ValueError(
            "training summary에 평가 가능한 checkpoint가 없는 node가 있습니다: "
            + ", ".join(missing)
        )
    if not paths:
        raise ValueError(f"training summary에 checkpoint가 없습니다: {path}")
    return paths


def _discover_checkpoints(
    explicit: Iterable[str],
    checkpoint_root: str | None,
    training_summary: str | None = None,
) -> list[Path]:
    paths = [Path(value).expanduser().resolve() for value in explicit]
    if training_summary:
        paths.extend(_summary_checkpoint_paths(training_summary))
    if checkpoint_root:
        root = Path(checkpoint_root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"checkpoint root가 없습니다: {root}")
        paths.extend(sorted(root.rglob("model.pt")))
    unique = []
    seen = set()
    for path in paths:
        if path in seen:
            continue
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint가 없습니다: {path}")
        seen.add(path)
        unique.append(path)
    if not unique:
        raise ValueError(
            "--checkpoint, --training-summary 또는 --checkpoint-root로 "
            "checkpoint를 지정해야 합니다"
        )
    return unique


def _checkpoint_records(
    paths: Iterable[Path],
    *,
    duplicate_policy: str = "latest",
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    if duplicate_policy not in {"latest", "error"}:
        raise ValueError(f"지원하지 않는 duplicate policy입니다: {duplicate_policy}")

    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in paths:
        payload = _torch_load(path)
        config = dict(payload.get("config", {}))
        node = dict(payload.get("taxonomy_node", config.get("taxonomy_node", {})))
        if not node:
            raise ValueError(f"원논문식 taxonomy node 정보가 없는 checkpoint입니다: {path}")
        node_id = str(node.get("node_id", "")).strip()
        if not node_id:
            raise ValueError(f"taxonomy_node.node_id가 없습니다: {path}")
        candidates[node_id].append(
            {
                "path": path,
                "payload": payload,
                "config": config,
                "node": node,
            }
        )

    records: dict[str, dict[str, Any]] = {}
    duplicate_selection: dict[str, dict[str, Any]] = {}
    for node_id, node_candidates in candidates.items():
        ordered = sorted(
            node_candidates,
            key=lambda item: (
                item["path"].stat().st_mtime_ns,
                str(item["path"]),
            ),
            reverse=True,
        )
        if len(ordered) > 1 and duplicate_policy == "error":
            formatted = "\n".join(f"- {item['path']}" for item in ordered)
            raise ValueError(
                f"같은 node checkpoint가 둘 이상 발견되었습니다: {node_id}\n{formatted}"
            )
        records[node_id] = ordered[0]
        if len(ordered) > 1:
            duplicate_selection[node_id] = {
                "selected": str(ordered[0]["path"]),
                "discarded": [str(item["path"]) for item in ordered[1:]],
                "criterion": "filesystem modification time",
            }
    return records, duplicate_selection


def _normalized_config_value(config: dict[str, Any], key: str) -> Any:
    value = config.get(key)
    if key in {"data_path", "resolved_data_path", "taxonomy_manifest"}:
        if value in (None, "", "null"):
            return None
        return str(_resolve_project_path(value))
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return value


def _validate_checkpoint_compatibility(records: dict[str, dict[str, Any]]) -> None:
    """서로 다른 데이터/백본/전처리 실험 checkpoint가 섞이는 것을 차단한다."""
    if "root" not in records:
        return
    reference = records["root"]["config"]
    # resolved_data_path가 존재하면 data_path보다 우선한다.
    data_key = "resolved_data_path" if reference.get("resolved_data_path") else "data_path"
    keys = [
        data_key,
        "taxonomy_manifest",
        "taxonomy_class_column",
        "pretrained_weights",
        "model",
        "crop_size",
        "eval_resize_size",
        "normalization",
        "normalization_mean",
        "normalization_std",
        "train_split",
        "val_split",
        "test_split",
    ]
    # 새 orchestrator가 만든 checkpoint는 같은 run_id여야 한다. 최신 node별
    # checkpoint를 임의 조합해 서로 다른 seed/run이 섞이는 것을 방지한다.
    if reference.get("taxonomy_experiment_run_id") not in (None, "", "null"):
        keys.append("taxonomy_experiment_run_id")
    mismatches: list[str] = []
    for node_id, record in records.items():
        config = record["config"]
        for key in keys:
            expected = _normalized_config_value(reference, key)
            actual = _normalized_config_value(config, key)
            if actual != expected:
                mismatches.append(
                    f"{node_id}.{key}: {actual!r} != root의 {expected!r}"
                )
    if mismatches:
        raise ValueError(
            "서로 호환되지 않는 taxonomy node checkpoint가 섞였습니다:\n- "
            + "\n- ".join(mismatches)
        )

def _validate_checkpoint_taxonomy_mappings(
    records: dict[str, dict[str, Any]],
    nodes: Iterable[TaxonomyNodeSpec],
) -> None:
    """checkpoint의 local class 순서가 현재 taxonomy manifest와 같은지 검증한다."""
    keys = (
        "rank",
        "name",
        "child_rank",
        "child_names",
        "child_identifiers",
        "descendant_species_indices",
        "species_to_child",
        "child_to_species_indices",
    )
    mismatches: list[str] = []
    for node in nodes:
        record = records.get(node.node_id)
        if record is None:
            mismatches.append(f"{node.node_id}: checkpoint 누락")
            continue
        stored = record["node"]
        expected = node.to_dict()
        for key in keys:
            if stored.get(key) != expected.get(key):
                mismatches.append(f"{node.node_id}.{key}")
    if mismatches:
        raise ValueError(
            "checkpoint taxonomy mapping이 현재 manifest와 다릅니다: "
            + ", ".join(mismatches)
        )


def _squeeze_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 3 and logits.shape[-1] == 1:
        logits = logits.squeeze(-1)
    if logits.ndim != 2:
        raise ValueError(f"node logits는 [B,K]여야 하지만 {tuple(logits.shape)}입니다")
    return logits


def _confusion_metrics(confusion: torch.Tensor) -> dict[str, float]:
    matrix = confusion.to(torch.float64)
    total = matrix.sum()
    if total <= 0:
        return {
            "top1": 0.0,
            "balanced_accuracy": 0.0,
            "macro_f1": 0.0,
        }
    true_positive = matrix.diag()
    support = matrix.sum(dim=1)
    predicted = matrix.sum(dim=0)
    active = support > 0
    recall = true_positive / support.clamp_min(1.0)
    precision = true_positive / predicted.clamp_min(1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-12)
    return {
        "top1": float((true_positive.sum() / total * 100.0).item()),
        "balanced_accuracy": float((recall[active].mean() * 100.0).item()),
        "macro_f1": float((f1[active].mean() * 100.0).item()),
    }


def _confusion_update(
    confusion: torch.Tensor,
    targets: torch.Tensor,
    predictions: torch.Tensor,
) -> None:
    if targets.numel() == 0:
        return
    class_count = int(confusion.shape[0])
    indices = targets.to(torch.long) * class_count + predictions.to(torch.long)
    confusion += torch.bincount(
        indices.cpu(),
        minlength=class_count * class_count,
    ).reshape(class_count, class_count)


def aggregate_soft_path_scores(
    node_log_probabilities: dict[str, torch.Tensor],
    trainable_nodes: Iterable[TaxonomyNodeSpec],
    *,
    num_species: int,
) -> torch.Tensor:
    """node 조건부 log-probability를 species 경로별로 합산한다."""
    nodes = list(trainable_nodes)
    if not nodes:
        raise ValueError("trainable taxonomy node가 없습니다")
    first = node_log_probabilities[nodes[0].node_id]
    scores = first.new_zeros((first.shape[0], int(num_species)))

    for node in nodes:
        log_prob = node_log_probabilities[node.node_id]
        if log_prob.shape[1] != node.num_children:
            raise ValueError(
                f"{node.node_id} 출력 클래스 수가 다릅니다: "
                f"{log_prob.shape[1]} != {node.num_children}"
            )
        mapping = torch.as_tensor(
            node.species_to_child,
            dtype=torch.long,
            device=log_prob.device,
        )
        eligible = mapping.ge(0)
        species_indices = torch.nonzero(eligible, as_tuple=False).flatten()
        local_indices = mapping.index_select(0, species_indices)
        scores[:, species_indices] += log_prob.index_select(1, local_indices)
    return scores


def greedy_taxonomy_predictions(
    node_log_probabilities: dict[str, torch.Tensor],
    taxonomy: Any,
    all_nodes: Iterable[TaxonomyNodeSpec],
) -> torch.Tensor:
    """root부터 local argmax를 따라가며 최종 species index를 반환한다."""
    nodes = node_lookup(all_nodes)
    root = nodes["root"]
    if root.node_id not in node_log_probabilities:
        raise ValueError("hard traversal에는 root checkpoint가 필요합니다")

    species_to_genus = [int(value) for value in taxonomy.species_to_genus]
    genus_to_family = [int(value) for value in taxonomy.genus_to_family]
    family_names = list(taxonomy.family_names)
    genus_names = list(taxonomy.genus_names)

    family_to_genera: dict[int, list[int]] = defaultdict(list)
    for genus_index, family_index in enumerate(genus_to_family):
        family_to_genera[family_index].append(genus_index)
    genus_to_species: dict[int, list[int]] = defaultdict(list)
    for species_index, genus_index in enumerate(species_to_genus):
        genus_to_species[genus_index].append(species_index)

    family_predictions = node_log_probabilities["root"].argmax(dim=1)
    predictions = []
    for batch_index, family_index_tensor in enumerate(family_predictions):
        family_index = int(family_index_tensor.item())
        genera = family_to_genera[family_index]
        if len(genera) == 1:
            genus_index = genera[0]
        else:
            family_node_id = next(
                node.node_id
                for node in nodes.values()
                if node.rank == "family" and node.name == family_names[family_index]
            )
            if family_node_id not in node_log_probabilities:
                raise ValueError(f"family node checkpoint가 없습니다: {family_node_id}")
            local_genus = int(
                node_log_probabilities[family_node_id][batch_index].argmax().item()
            )
            family_node = nodes[family_node_id]
            representative_species = family_node.child_to_species_indices[local_genus][0]
            genus_index = species_to_genus[representative_species]

        species = genus_to_species[genus_index]
        if len(species) == 1:
            species_index = species[0]
        else:
            genus_node_id = next(
                node.node_id
                for node in nodes.values()
                if node.rank == "genus" and node.name == genus_names[genus_index]
            )
            if genus_node_id not in node_log_probabilities:
                raise ValueError(f"genus node checkpoint가 없습니다: {genus_node_id}")
            local_species = int(
                node_log_probabilities[genus_node_id][batch_index].argmax().item()
            )
            genus_node = nodes[genus_node_id]
            species_index = genus_node.child_to_species_indices[local_species][0]
        predictions.append(species_index)

    return torch.tensor(
        predictions,
        dtype=torch.long,
        device=family_predictions.device,
    )


def _taxonomic_distance(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    taxonomy: Any,
) -> torch.Tensor:
    species_to_genus = torch.as_tensor(
        taxonomy.species_to_genus,
        dtype=torch.long,
        device=targets.device,
    )
    genus_to_family = torch.as_tensor(
        taxonomy.genus_to_family,
        dtype=torch.long,
        device=targets.device,
    )
    true_genus = species_to_genus[targets]
    pred_genus = species_to_genus[predictions]
    true_family = genus_to_family[true_genus]
    pred_family = genus_to_family[pred_genus]
    distance = torch.full_like(targets, 3, dtype=torch.long)
    distance[true_family.eq(pred_family)] = 2
    distance[true_genus.eq(pred_genus)] = 1
    distance[targets.eq(predictions)] = 0
    return distance


def _build_model(record: dict[str, Any], device: torch.device):
    # 평가 수학 함수만 import하는 테스트에서는 timm이 필요하지 않도록 지연 import한다.
    from model.factory import get_model

    config = dict(record["config"])
    config.update(
        {
            "load_pretrained_backbone": False,
            "promptcam_checkpoint": str(record["path"]),
            "promptcam_checkpoint_strict": True,
            "resume": None,
            "vis_attn": False,
            "debug": False,
            "distributed": False,
        }
    )
    params = SimpleNamespace(**config)
    model, _, _ = get_model(params)
    model = model.to(device).eval()
    return model


def _dataset_and_taxonomy(config: dict[str, Any], split: str):
    data_root_value = config.get("resolved_data_path", config.get("data_path"))
    data_root = _resolve_project_path(data_root_value)
    train_split = str(config.get("train_split", "train"))
    split_name = str(config.get(f"{split}_split", split))
    train_base = datasets.ImageFolder(str(data_root / train_split), transform=None)
    eval_base = datasets.ImageFolder(str(data_root / split_name), transform=None)
    if eval_base.class_to_idx != train_base.class_to_idx:
        raise ValueError("평가 split의 class_to_idx가 train split과 다릅니다")

    taxonomy = load_taxonomy_manifest(
        _resolve_project_path(config["taxonomy_manifest"]),
        train_base.classes,
        class_column=config.get("taxonomy_class_column"),
    )
    params = SimpleNamespace(**config)
    transform = JointImageTransform(params, training=False)
    dataset = FlatImageFolder(eval_base, transform)
    return dataset, taxonomy


def _model_autocast(device: torch.device, amp_dtype: str):
    name = str(amp_dtype).lower()
    if device.type != "cuda" or name == "none":
        return torch.autocast(device_type=device.type, enabled=False)
    dtype = torch.float16 if name == "float16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _infer_node_log_probabilities(
    record: dict[str, Any],
    node: TaxonomyNodeSpec,
    dataset: Any,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    amp_dtype: str,
) -> tuple[torch.Tensor, int, int]:
    """한 node model만 device에 올려 전체 split의 local log-probability를 계산한다.

    모든 node backbone을 동시에 GPU에 상주시킬 필요가 없도록 node별로 순차
    추론하고 결과만 CPU에 보관한다. 평가 표본 순서는 shuffle=False로 고정된다.
    """
    model = _build_model(record, device)
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    chunks: list[torch.Tensor] = []
    with torch.inference_mode():
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            with _model_autocast(device, amp_dtype):
                output, _ = model(images)
                logits = _squeeze_logits(output)
            if logits.shape[1] != node.num_children:
                raise ValueError(
                    f"{node.node_id} 출력 클래스 수가 taxonomy와 다릅니다: "
                    f"{logits.shape[1]} != {node.num_children}"
                )
            chunks.append(F.log_softmax(logits.float(), dim=1).cpu())

    if not chunks:
        raise RuntimeError(f"평가 dataset이 비어 있습니다: node={node.node_id}")
    log_probabilities = torch.cat(chunks, dim=0)
    if log_probabilities.shape[0] != len(dataset):
        raise RuntimeError(
            f"{node.node_id} 추론 표본 수가 dataset과 다릅니다: "
            f"{log_probabilities.shape[0]} != {len(dataset)}"
        )

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return log_probabilities, trainable_parameters, total_parameters


def main() -> None:
    parser = argparse.ArgumentParser(
        description="원논문식 taxonomy node checkpoint의 end-to-end 평가"
    )
    parser.add_argument("--checkpoint", action="append", default=[])
    parser.add_argument("--checkpoint-root", default=None)
    parser.add_argument(
        "--training-summary",
        default=None,
        help="training.independent가 기록한 training_summary.json",
    )
    parser.add_argument(
        "--duplicate-policy",
        choices=["latest", "error"],
        default="latest",
        help="checkpoint-root에 같은 node의 여러 실행이 있을 때 선택 방식",
    )
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp-dtype", choices=["none", "float16", "bfloat16"], default="none")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

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
        raise ValueError("end-to-end 평가에는 root node model.pt가 필요합니다")

    reference_config = dict(records["root"]["config"])
    dataset, taxonomy = _dataset_and_taxonomy(reference_config, args.split)
    all_nodes = list_taxonomy_nodes(taxonomy, trainable_only=False)
    trainable_nodes = list_taxonomy_nodes(taxonomy, trainable_only=True)
    expected = {node.node_id for node in trainable_nodes}
    supplied = set(records)
    missing = sorted(expected - supplied)
    extra = sorted(supplied - expected)
    if missing or extra:
        raise ValueError(
            "checkpoint node 구성이 taxonomy와 일치하지 않습니다: "
            f"누락={missing}, 추가={extra}"
        )

    _validate_checkpoint_taxonomy_mappings(records, trainable_nodes)

    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    batch_size = int(
        args.batch_size or reference_config.get("test_batch_size", 64)
    )
    num_workers = int(
        args.num_workers
        if args.num_workers is not None
        else reference_config.get("num_workers", 4)
    )

    # 독립 node 모델은 동일한 frozen backbone을 포함하므로, 모두 GPU에 동시에
    # 올리면 불필요하게 메모리를 소모한다. 각 node를 한 번씩 순차 추론하고
    # local log-probability만 CPU에 캐시한다.
    node_log_probabilities: dict[str, torch.Tensor] = {}
    trainable_parameter_counts: dict[str, int] = {}
    total_parameter_counts: dict[str, int] = {}
    for node in trainable_nodes:
        log_probabilities, trainable_count, total_count = (
            _infer_node_log_probabilities(
                records[node.node_id],
                node,
                dataset,
                device=device,
                batch_size=batch_size,
                num_workers=num_workers,
                amp_dtype=args.amp_dtype,
            )
        )
        node_log_probabilities[node.node_id] = log_probabilities
        trainable_parameter_counts[node.node_id] = trainable_count
        total_parameter_counts[node.node_id] = total_count
        print(
            f"[taxonomy eval] {node.node_id}: "
            f"samples={len(dataset)}, classes={node.num_children}"
        )

    species_count = len(taxonomy.class_names)
    sample_count = len(dataset)
    if sample_count <= 0:
        raise RuntimeError("평가 dataset이 비어 있습니다")
    targets = torch.as_tensor(dataset.targets, dtype=torch.long)
    if targets.numel() != sample_count:
        raise RuntimeError(
            f"dataset targets 수가 표본 수와 다릅니다: "
            f"{targets.numel()} != {sample_count}"
        )

    soft_scores = aggregate_soft_path_scores(
        node_log_probabilities,
        trainable_nodes,
        num_species=species_count,
    )
    soft_predictions = soft_scores.argmax(dim=1)
    hard_predictions = greedy_taxonomy_predictions(
        node_log_probabilities,
        taxonomy,
        all_nodes,
    )

    soft_confusion = torch.zeros(species_count, species_count, dtype=torch.long)
    hard_confusion = torch.zeros_like(soft_confusion)
    _confusion_update(soft_confusion, targets, soft_predictions)
    _confusion_update(hard_confusion, targets, hard_predictions)

    topk = min(5, species_count)
    soft_top5 = int(
        soft_scores.topk(topk, dim=1)
        .indices.eq(targets[:, None])
        .any(dim=1)
        .sum()
        .item()
    )
    species_to_genus = torch.as_tensor(
        taxonomy.species_to_genus,
        dtype=torch.long,
    )
    genus_to_family = torch.as_tensor(
        taxonomy.genus_to_family,
        dtype=torch.long,
    )
    true_genus = species_to_genus[targets]
    true_family = genus_to_family[true_genus]
    soft_genus = species_to_genus[soft_predictions]
    hard_genus = species_to_genus[hard_predictions]
    soft_family = genus_to_family[soft_genus]
    hard_family = genus_to_family[hard_genus]
    soft_genus_correct = int(soft_genus.eq(true_genus).sum().item())
    hard_genus_correct = int(hard_genus.eq(true_genus).sum().item())
    soft_family_correct = int(soft_family.eq(true_family).sum().item())
    hard_family_correct = int(hard_family.eq(true_family).sum().item())
    soft_distance_sum = int(
        _taxonomic_distance(targets, soft_predictions, taxonomy).sum().item()
    )
    hard_distance_sum = int(
        _taxonomic_distance(targets, hard_predictions, taxonomy).sum().item()
    )

    local_confusions: dict[str, torch.Tensor] = {}
    for node in trainable_nodes:
        confusion = torch.zeros(
            node.num_children,
            node.num_children,
            dtype=torch.long,
        )
        mapping = torch.as_tensor(node.species_to_child, dtype=torch.long)
        local_targets = mapping[targets]
        eligible = local_targets.ge(0)
        local_predictions = node_log_probabilities[node.node_id].argmax(dim=1)
        _confusion_update(
            confusion,
            local_targets[eligible],
            local_predictions[eligible],
        )
        local_confusions[node.node_id] = confusion

    prediction_rows: list[dict[str, Any]] = []
    for row_index, (sample_path, _) in enumerate(dataset.samples):
        target_index = int(targets[row_index].item())
        soft_index = int(soft_predictions[row_index].item())
        hard_index = int(hard_predictions[row_index].item())
        prediction_rows.append(
            {
                "path": str(sample_path),
                "target_index": target_index,
                "target_species": taxonomy.scientific_names[target_index],
                "soft_prediction_index": soft_index,
                "soft_prediction_species": taxonomy.scientific_names[soft_index],
                "hard_prediction_index": hard_index,
                "hard_prediction_species": taxonomy.scientific_names[hard_index],
            }
        )

    def end_to_end_metrics(
        confusion: torch.Tensor,
        *,
        top5_correct: int | None,
        genus_correct: int,
        family_correct: int,
        distance_sum: int,
    ) -> dict[str, float]:
        metrics = _confusion_metrics(confusion)
        metrics.update(
            {
                "top5": (
                    100.0 * top5_correct / max(1, sample_count)
                    if top5_correct is not None
                    else None
                ),
                "genus_accuracy": 100.0 * genus_correct / max(1, sample_count),
                "family_accuracy": 100.0 * family_correct / max(1, sample_count),
                "mean_taxonomic_distance": distance_sum / max(1, sample_count),
                "samples": sample_count,
            }
        )
        return {
            key: round(value, 6) if isinstance(value, float) else value
            for key, value in metrics.items()
        }

    result = {
        "mode": "prompt_cam_original_taxonomy",
        "split": args.split,
        "soft_path": end_to_end_metrics(
            soft_confusion,
            top5_correct=soft_top5,
            genus_correct=soft_genus_correct,
            family_correct=soft_family_correct,
            distance_sum=soft_distance_sum,
        ),
        "hard_traversal": end_to_end_metrics(
            hard_confusion,
            top5_correct=None,
            genus_correct=hard_genus_correct,
            family_correct=hard_family_correct,
            distance_sum=hard_distance_sum,
        ),
        "node_local": {
            node.node_id: {
                **_confusion_metrics(local_confusions[node.node_id]),
                "samples": int(local_confusions[node.node_id].sum().item()),
                "node": node.to_dict(),
                "checkpoint": str(records[node.node_id]["path"]),
            }
            for node in trainable_nodes
        },
        "taxonomy": taxonomy.to_dict(),
        "checkpoints": {
            node_id: str(record["path"])
            for node_id, record in records.items()
        },
        "checkpoint_selection": {
            "training_summary": (
                str(Path(args.training_summary).expanduser().resolve())
                if args.training_summary
                else None
            ),
            "checkpoint_root": (
                str(Path(args.checkpoint_root).expanduser().resolve())
                if args.checkpoint_root
                else None
            ),
            "duplicate_policy": args.duplicate_policy,
            "duplicates": duplicate_selection,
        },
        "resources": {
            "num_independent_node_models": len(trainable_nodes),
            "peak_resident_node_models": 1,
            "evaluation_strategy": "sequential_node_inference_with_cpu_log_probability_cache",
            "trainable_parameters_by_node": trainable_parameter_counts,
            "total_parameters_by_node": total_parameter_counts,
            "total_trainable_parameters": sum(
                trainable_parameter_counts.values()
            ),
            "total_trainable_parameters_across_nodes": sum(
                trainable_parameter_counts.values()
            ),
            "stored_parameters_across_node_checkpoints": sum(
                total_parameter_counts.values()
            ),
            "soft_path_forward_passes_per_image": len(trainable_nodes),
            "conceptual_hard_traversal_max_forward_passes_per_image": 3,
        },
    }

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else PROJECT_ROOT / "output" / "independent" / f"{args.split}_evaluation.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if not prediction_rows:
        raise RuntimeError("평가 dataset이 비어 있어 표본별 예측을 저장할 수 없습니다")
    predictions_path = output_path.with_name(output_path.stem + "_predictions.csv")
    with predictions_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0].keys()))
        writer.writeheader()
        writer.writerows(prediction_rows)

    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"평가 결과: {output_path}")
    print(f"표본별 예측: {predictions_path}")


if __name__ == "__main__":
    main()
