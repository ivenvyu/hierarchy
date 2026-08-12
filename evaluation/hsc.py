#!/usr/bin/env python3
"""Validation-calibrated hierarchical fallback comparison.

Compares
  1) one shared hierarchical Prompt-CAM, and
  2) independent taxonomy-node Prompt-CAMs
under the same adaptive stopping rule.

For each reliability constraint q, species/genus thresholds are selected ONLY
on the validation split to maximize mean emitted taxonomy depth subject to
validation emitted-rank accuracy >= q. The fixed thresholds are then evaluated
on the test split.

Emitted depths:
    family=1, genus=2, species=3

The default confidence_mode="joint" uses path-consistent confidence:
    genus score   = P(f_hat|x) P(g_hat|f_hat,x)
    species score = genus score P(s_hat|g_hat,x)
This is recommended because uncertainty in ancestors propagates downward and
singleton branches do not receive artificial confidence 1 at the global rank.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation import checkpoints as independent_eval  # noqa: E402
from evaluation import hierarchy as hierarchy_eval  # noqa: E402


NO_SELECT_THRESHOLD = math.nextafter(1.0, math.inf)


@dataclass
class PathData:
    family_confidence: torch.Tensor
    genus_confidence_conditional: torch.Tensor
    species_confidence_conditional: torch.Tensor
    genus_confidence_joint: torch.Tensor
    species_confidence_joint: torch.Tensor
    family_correct: torch.Tensor
    genus_correct: torch.Tensor
    species_correct: torch.Tensor
    sample_count: int

    def scores(self, mode: str) -> tuple[torch.Tensor, torch.Tensor]:
        if mode == "joint":
            return self.species_confidence_joint, self.genus_confidence_joint
        if mode == "conditional":
            return (
                self.species_confidence_conditional,
                self.genus_confidence_conditional,
            )
        raise ValueError(f"지원하지 않는 confidence mode입니다: {mode}")

    def to_cache(self) -> dict[str, Any]:
        return {
            "family_confidence": self.family_confidence,
            "genus_confidence_conditional": self.genus_confidence_conditional,
            "species_confidence_conditional": self.species_confidence_conditional,
            "genus_confidence_joint": self.genus_confidence_joint,
            "species_confidence_joint": self.species_confidence_joint,
            "family_correct": self.family_correct,
            "genus_correct": self.genus_correct,
            "species_correct": self.species_correct,
            "sample_count": self.sample_count,
        }

    @classmethod
    def from_cache(cls, payload: dict[str, Any]) -> "PathData":
        return cls(**payload)


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _save_cache(path: Path, *, fingerprint: dict[str, Any], data: PathData) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "fingerprint": fingerprint,
            "path_data": data.to_cache(),
        },
        path,
    )


def _load_cache(path: Path, fingerprint: dict[str, Any]) -> PathData | None:
    if not path.is_file():
        return None
    payload = _torch_load(path)
    if payload.get("fingerprint") != fingerprint:
        return None
    return PathData.from_cache(payload["path_data"])


def _shared_fingerprint(run_dir: Path, split: str) -> dict[str, Any]:
    checkpoint = run_dir / "model.pt"
    args_path = run_dir / "args.yaml"
    return {
        "kind": "shared",
        "run_dir": str(run_dir.resolve()),
        "split": split,
        "checkpoint_mtime_ns": checkpoint.stat().st_mtime_ns,
        "args_mtime_ns": args_path.stat().st_mtime_ns,
    }


def _independent_fingerprint(summary_path: Path, split: str) -> dict[str, Any]:
    checkpoint_paths = independent_eval._summary_checkpoint_paths(summary_path)
    return {
        "kind": "independent",
        "training_summary": str(summary_path.resolve()),
        "split": split,
        "summary_mtime_ns": summary_path.stat().st_mtime_ns,
        "checkpoints": [
            [str(path.resolve()), path.stat().st_mtime_ns]
            for path in sorted(checkpoint_paths)
        ],
    }


def _build_family_to_genera(genus_to_family: list[int]) -> dict[int, list[int]]:
    result: dict[int, list[int]] = {}
    for genus_index, family_index in enumerate(genus_to_family):
        result.setdefault(int(family_index), []).append(int(genus_index))
    return result


def _build_genus_to_species(species_to_genus: list[int]) -> dict[int, list[int]]:
    result: dict[int, list[int]] = {}
    for species_index, genus_index in enumerate(species_to_genus):
        result.setdefault(int(genus_index), []).append(int(species_index))
    return result


def _path_data_from_predictions(
    *,
    targets: torch.Tensor,
    species_to_genus: list[int],
    genus_to_family: list[int],
    family_prediction: torch.Tensor,
    genus_prediction: torch.Tensor,
    species_prediction: torch.Tensor,
    family_confidence: torch.Tensor,
    genus_conditional_confidence: torch.Tensor,
    species_conditional_confidence: torch.Tensor,
) -> PathData:
    targets = targets.long().cpu()
    family_prediction = family_prediction.long().cpu()
    genus_prediction = genus_prediction.long().cpu()
    species_prediction = species_prediction.long().cpu()
    family_confidence = family_confidence.float().cpu()
    genus_conditional_confidence = genus_conditional_confidence.float().cpu()
    species_conditional_confidence = species_conditional_confidence.float().cpu()

    s2g = torch.tensor(species_to_genus, dtype=torch.long)
    g2f = torch.tensor(genus_to_family, dtype=torch.long)
    true_genus = s2g[targets]
    true_family = g2f[true_genus]

    genus_joint = family_confidence * genus_conditional_confidence
    species_joint = genus_joint * species_conditional_confidence

    return PathData(
        family_confidence=family_confidence,
        genus_confidence_conditional=genus_conditional_confidence,
        species_confidence_conditional=species_conditional_confidence,
        genus_confidence_joint=genus_joint,
        species_confidence_joint=species_joint,
        family_correct=family_prediction.eq(true_family),
        genus_correct=genus_prediction.eq(true_genus),
        species_correct=species_prediction.eq(targets),
        sample_count=int(targets.numel()),
    )


def collect_shared(
    run_dir: Path,
    split: str,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> PathData:
    args_path = run_dir / "args.yaml"
    if not args_path.is_file():
        raise FileNotFoundError(f"shared args.yaml이 없습니다: {args_path}")
    args_data = hierarchy_eval._load_yaml(args_path)
    if not bool(args_data.get("hierarchical_prompt", False)):
        raise ValueError(f"shared run이 hierarchical_prompt=True가 아닙니다: {run_dir}")

    dummy_cli = SimpleNamespace(batch_size=batch_size, num_workers=num_workers)
    params = hierarchy_eval._prepare_params(args_data, PROJECT_ROOT, run_dir, dummy_cli)
    if split == "val":
        params.test_split = str(getattr(params, "val_split", "val"))
    elif split == "test":
        params.test_split = str(getattr(params, "test_split", "test"))
    else:
        raise ValueError(split)

    model, dataset, _, _ = hierarchy_eval._load_model(
        PROJECT_ROOT,
        run_dir,
        params,
        device,
    )
    taxonomy = hierarchy_eval._load_taxonomy(run_dir, params)
    if tuple(dataset.classes) != taxonomy.class_names:
        raise ValueError("shared dataset class 순서와 taxonomy가 다릅니다")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    outputs = hierarchy_eval._collect_outputs(
        model,
        loader,
        device=device,
        amp_dtype="none",
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    family_probs = outputs["family_probabilities"].float().cpu()
    genus_cond = outputs["genus_conditional_probabilities"].float().cpu()
    species_cond = outputs["species_conditional_probabilities"].float().cpu()
    targets = outputs["targets"].long().cpu()

    species_to_genus = [int(x) for x in taxonomy.species_to_genus]
    genus_to_family = [int(x) for x in taxonomy.genus_to_family]
    family_to_genera = _build_family_to_genera(genus_to_family)
    genus_to_species = _build_genus_to_species(species_to_genus)

    n = targets.numel()
    family_prediction = family_probs.argmax(dim=1)
    family_confidence = family_probs.gather(
        1, family_prediction[:, None]
    ).squeeze(1)

    genus_prediction = torch.empty(n, dtype=torch.long)
    genus_confidence = torch.empty(n, dtype=torch.float32)
    for family_index, genera in family_to_genera.items():
        mask = family_prediction.eq(family_index)
        if not mask.any():
            continue
        members = torch.tensor(genera, dtype=torch.long)
        local = genus_cond[mask][:, members]
        best_prob, best_local = local.max(dim=1)
        genus_prediction[mask] = members[best_local]
        genus_confidence[mask] = best_prob

    species_prediction = torch.empty(n, dtype=torch.long)
    species_confidence = torch.empty(n, dtype=torch.float32)
    for genus_index, species in genus_to_species.items():
        mask = genus_prediction.eq(genus_index)
        if not mask.any():
            continue
        members = torch.tensor(species, dtype=torch.long)
        local = species_cond[mask][:, members]
        best_prob, best_local = local.max(dim=1)
        species_prediction[mask] = members[best_local]
        species_confidence[mask] = best_prob

    return _path_data_from_predictions(
        targets=targets,
        species_to_genus=species_to_genus,
        genus_to_family=genus_to_family,
        family_prediction=family_prediction,
        genus_prediction=genus_prediction,
        species_prediction=species_prediction,
        family_confidence=family_confidence,
        genus_conditional_confidence=genus_confidence,
        species_conditional_confidence=species_confidence,
    )


def collect_independent(
    summary_path: Path,
    split: str,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> PathData:
    checkpoint_paths = independent_eval._discover_checkpoints(
        [], None, str(summary_path)
    )
    records, _ = independent_eval._checkpoint_records(
        checkpoint_paths,
        duplicate_policy="latest",
    )
    independent_eval._validate_checkpoint_compatibility(records)
    if "root" not in records:
        raise ValueError("independent taxonomy에 root checkpoint가 없습니다")

    reference_config = dict(records["root"]["config"])
    dataset, taxonomy = independent_eval._dataset_and_taxonomy(
        reference_config, split
    )
    all_nodes = independent_eval.list_taxonomy_nodes(
        taxonomy, trainable_only=False
    )
    trainable_nodes = independent_eval.list_taxonomy_nodes(
        taxonomy, trainable_only=True
    )
    independent_eval._validate_checkpoint_taxonomy_mappings(
        records, trainable_nodes
    )

    expected = {node.node_id for node in trainable_nodes}
    supplied = set(records)
    if expected != supplied:
        raise ValueError(
            f"independent checkpoint node 불일치: "
            f"missing={sorted(expected - supplied)}, extra={sorted(supplied - expected)}"
        )

    node_log_probabilities: dict[str, torch.Tensor] = {}
    for node in trainable_nodes:
        log_probabilities, _, _ = independent_eval._infer_node_log_probabilities(
            records[node.node_id],
            node,
            dataset,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            amp_dtype="none",
        )
        node_log_probabilities[node.node_id] = log_probabilities.float().cpu()
        print(
            f"[stopping cache] independent {split} {node.node_id}: "
            f"N={len(dataset)}, K={node.num_children}"
        )

    species_to_genus = [int(x) for x in taxonomy.species_to_genus]
    genus_to_family = [int(x) for x in taxonomy.genus_to_family]
    family_names = list(taxonomy.family_names)
    genus_names = list(taxonomy.genus_names)
    family_to_genera = _build_family_to_genera(genus_to_family)
    genus_to_species = _build_genus_to_species(species_to_genus)

    node_by_id = {node.node_id: node for node in all_nodes}
    family_node_by_name = {
        str(node.name): node
        for node in trainable_nodes
        if node.rank == "family"
    }
    genus_node_by_name = {
        str(node.name): node
        for node in trainable_nodes
        if node.rank == "genus"
    }
    root_node = node_by_id["root"]

    root_probs = node_log_probabilities["root"].exp()
    root_local_prediction = root_probs.argmax(dim=1)
    family_confidence = root_probs.gather(
        1, root_local_prediction[:, None]
    ).squeeze(1)

    n = len(dataset)
    family_prediction = torch.empty(n, dtype=torch.long)
    genus_prediction = torch.empty(n, dtype=torch.long)
    species_prediction = torch.empty(n, dtype=torch.long)
    genus_confidence = torch.empty(n, dtype=torch.float32)
    species_confidence = torch.empty(n, dtype=torch.float32)

    for i in range(n):
        local_family = int(root_local_prediction[i].item())
        representative_species = int(
            root_node.child_to_species_indices[local_family][0]
        )
        predicted_genus_of_rep = species_to_genus[representative_species]
        predicted_family = genus_to_family[predicted_genus_of_rep]
        family_prediction[i] = predicted_family

        genera = family_to_genera[predicted_family]
        if len(genera) == 1:
            predicted_genus = genera[0]
            genus_confidence[i] = 1.0
        else:
            family_name = family_names[predicted_family]
            family_node = family_node_by_name.get(family_name)
            if family_node is None:
                raise ValueError(
                    f"family node가 없습니다: family={family_name!r}"
                )
            probabilities = node_log_probabilities[family_node.node_id][i].exp()
            local_genus = int(probabilities.argmax().item())
            representative_species = int(
                family_node.child_to_species_indices[local_genus][0]
            )
            predicted_genus = species_to_genus[representative_species]
            genus_confidence[i] = float(probabilities[local_genus].item())
        genus_prediction[i] = predicted_genus

        species_members = genus_to_species[predicted_genus]
        if len(species_members) == 1:
            predicted_species = species_members[0]
            species_confidence[i] = 1.0
        else:
            genus_name = genus_names[predicted_genus]
            genus_node = genus_node_by_name.get(genus_name)
            if genus_node is None:
                raise ValueError(
                    f"genus node가 없습니다: genus={genus_name!r}"
                )
            probabilities = node_log_probabilities[genus_node.node_id][i].exp()
            local_species = int(probabilities.argmax().item())
            predicted_species = int(
                genus_node.child_to_species_indices[local_species][0]
            )
            species_confidence[i] = float(probabilities[local_species].item())
        species_prediction[i] = predicted_species

    targets = torch.as_tensor(dataset.targets, dtype=torch.long)
    return _path_data_from_predictions(
        targets=targets,
        species_to_genus=species_to_genus,
        genus_to_family=genus_to_family,
        family_prediction=family_prediction,
        genus_prediction=genus_prediction,
        species_prediction=species_prediction,
        family_confidence=family_confidence,
        genus_conditional_confidence=genus_confidence,
        species_conditional_confidence=species_confidence,
    )


def evaluate_thresholds(
    data: PathData,
    *,
    species_threshold: float,
    genus_threshold: float,
    confidence_mode: str,
) -> dict[str, float | int]:
    species_score, genus_score = data.scores(confidence_mode)
    species_mask = species_score.ge(float(species_threshold))
    genus_mask = (~species_mask) & genus_score.ge(float(genus_threshold))
    family_mask = ~(species_mask | genus_mask)

    correct = (
        int(data.species_correct[species_mask].sum().item())
        + int(data.genus_correct[genus_mask].sum().item())
        + int(data.family_correct[family_mask].sum().item())
    )
    ns = int(species_mask.sum().item())
    ng = int(genus_mask.sum().item())
    nf = int(family_mask.sum().item())
    n = data.sample_count
    depth_sum = 3 * ns + 2 * ng + nf

    return {
        "correct": correct,
        "samples": n,
        "emitted_accuracy": 100.0 * correct / n,
        "species_output_pct": 100.0 * ns / n,
        "genus_output_pct": 100.0 * ng / n,
        "family_output_pct": 100.0 * nf / n,
        "mean_depth": depth_sum / n,
        "normalized_specificity": depth_sum / (3.0 * n),
        "species_count": ns,
        "genus_count": ng,
        "family_count": nf,
    }


def _best_genus_for_fixed_species(
    data: PathData,
    species_mask: torch.Tensor,
    *,
    required_correct: int,
    confidence_mode: str,
) -> tuple[float, int, int] | None:
    _, genus_score = data.scores(confidence_mode)
    remaining = torch.nonzero(~species_mask, as_tuple=False).flatten()
    species_correct = int(data.species_correct[species_mask].sum().item())

    # No genus output: all remaining samples emit family.
    base_correct = species_correct + int(
        data.family_correct[remaining].sum().item()
    )
    best: tuple[float, int, int] | None = None
    if base_correct >= required_correct:
        best = (NO_SELECT_THRESHOLD, 0, base_correct)

    if remaining.numel() == 0:
        return best

    scores = genus_score[remaining]
    order = torch.argsort(scores, descending=True, stable=True)
    ordered_index = remaining[order]
    ordered_scores = scores[order]
    delta = (
        data.genus_correct[ordered_index].to(torch.int64)
        - data.family_correct[ordered_index].to(torch.int64)
    )
    cumulative_delta = torch.cumsum(delta, dim=0)

    # Threshold >= score includes all ties, so evaluate only at tie-group ends.
    group_end = torch.ones(ordered_scores.numel(), dtype=torch.bool)
    if ordered_scores.numel() > 1:
        group_end[:-1] = ordered_scores[:-1].ne(ordered_scores[1:])
    candidate_positions = torch.nonzero(group_end, as_tuple=False).flatten()

    for position_tensor in candidate_positions:
        position = int(position_tensor.item())
        genus_count = position + 1
        correct = base_correct + int(cumulative_delta[position].item())
        if correct < required_correct:
            continue
        threshold = float(ordered_scores[position].item())
        candidate = (threshold, genus_count, correct)
        # For fixed species output count, more genus outputs means greater depth.
        # Accuracy is used only as a tie-breaker.
        if best is None or (candidate[1], candidate[2]) > (best[1], best[2]):
            best = candidate
    return best


def optimize_thresholds(
    data: PathData,
    *,
    reliability: float,
    confidence_mode: str,
) -> dict[str, Any]:
    n = data.sample_count
    required_correct = int(math.ceil(float(reliability) * n - 1e-12))
    species_score, _ = data.scores(confidence_mode)

    unique_species = torch.unique(species_score).sort(descending=True).values
    candidate_species_thresholds = [NO_SELECT_THRESHOLD] + [
        float(value.item()) for value in unique_species
    ]

    best: dict[str, Any] | None = None
    for species_threshold in candidate_species_thresholds:
        species_mask = species_score.ge(species_threshold)
        ns = int(species_mask.sum().item())
        genus_choice = _best_genus_for_fixed_species(
            data,
            species_mask,
            required_correct=required_correct,
            confidence_mode=confidence_mode,
        )
        if genus_choice is None:
            continue
        genus_threshold, ng, correct = genus_choice
        depth_units = 2 * ns + ng  # total depth = N + this quantity
        candidate = {
            "species_threshold": float(species_threshold),
            "genus_threshold": float(genus_threshold),
            "species_count": ns,
            "genus_count": ng,
            "correct": correct,
            "depth_units": depth_units,
        }
        if best is None:
            best = candidate
            continue
        # Maximize mean depth. On exact ties prefer more species outputs, then
        # greater validation accuracy, then lower thresholds.
        candidate_key = (
            candidate["depth_units"],
            candidate["species_count"],
            candidate["correct"],
            -candidate["species_threshold"],
            -candidate["genus_threshold"],
        )
        best_key = (
            best["depth_units"],
            best["species_count"],
            best["correct"],
            -best["species_threshold"],
            -best["genus_threshold"],
        )
        if candidate_key > best_key:
            best = candidate

    max_reliability = float(data.family_correct.float().mean().item())
    if best is None:
        return {
            "status": "infeasible",
            "reliability_constraint": float(reliability),
            "max_validation_emitted_accuracy": 100.0 * max_reliability,
            "reason": (
                "family가 최저 출력 rank이므로 hard path에서 달성 가능한 "
                "최대 emitted-rank accuracy는 family prediction accuracy를 넘지 못합니다."
            ),
        }

    metrics = evaluate_thresholds(
        data,
        species_threshold=best["species_threshold"],
        genus_threshold=best["genus_threshold"],
        confidence_mode=confidence_mode,
    )
    if metrics["correct"] < required_correct:
        raise AssertionError("optimizer 내부 오류: reliability constraint 위반")

    return {
        "status": "ok",
        "reliability_constraint": float(reliability),
        "species_threshold": best["species_threshold"],
        "genus_threshold": best["genus_threshold"],
        "validation": metrics,
        "max_validation_emitted_accuracy": 100.0 * max_reliability,
    }


def _fmt_pct(value: float) -> str:
    return f"{value:.2f}%"


def _fmt_threshold(value: float) -> str:
    if value > 1.0:
        return ">1 (none)"
    return f"{value:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Shared vs independent taxonomy hierarchical fallback evaluation"
    )
    parser.add_argument("--shared-run-dir", required=True)
    parser.add_argument("--independent-training-summary", required=True)
    parser.add_argument(
        "--reliability",
        nargs="+",
        type=float,
        default=[90.0, 95.0, 97.5],
        help="percentage constraints, e.g. 90 95 97.5",
    )
    parser.add_argument(
        "--confidence-mode",
        choices=["joint", "conditional"],
        default="joint",
        help="joint is recommended for ancestor-aware confidence",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--refresh-cache", action="store_true")
    args = parser.parse_args()

    shared_run_dir = Path(args.shared_run_dir).expanduser().resolve()
    summary_path = Path(args.independent_training_summary).expanduser().resolve()
    if not shared_run_dir.is_dir():
        raise FileNotFoundError(shared_run_dir)
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)

    reliabilities = []
    for value in args.reliability:
        normalized = value / 100.0 if value > 1.0 else value
        if not 0.0 < normalized <= 1.0:
            raise ValueError(f"reliability는 (0,100] 범위여야 합니다: {value}")
        reliabilities.append(normalized)

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else PROJECT_ROOT / "output" / "hierarchical_fallback_comparison"
    )
    cache_dir = output_dir / "cache"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA를 요청했지만 사용할 수 없습니다")

    datasets: dict[str, dict[str, PathData]] = {
        "Independent": {},
        "Shared": {},
    }

    for split in ("val", "test"):
        independent_fp = _independent_fingerprint(summary_path, split)
        independent_cache = cache_dir / f"independent_{split}.pt"
        independent_data = None if args.refresh_cache else _load_cache(
            independent_cache, independent_fp
        )
        if independent_data is None:
            print(f"\n[추론] Independent {split}")
            independent_data = collect_independent(
                summary_path,
                split,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=device,
            )
            _save_cache(
                independent_cache,
                fingerprint=independent_fp,
                data=independent_data,
            )
        else:
            print(f"[cache] Independent {split}: {independent_cache}")
        datasets["Independent"][split] = independent_data

        shared_fp = _shared_fingerprint(shared_run_dir, split)
        shared_cache = cache_dir / f"shared_{split}.pt"
        shared_data = None if args.refresh_cache else _load_cache(
            shared_cache, shared_fp
        )
        if shared_data is None:
            print(f"\n[추론] Shared {split}")
            shared_data = collect_shared(
                shared_run_dir,
                split,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=device,
            )
            _save_cache(
                shared_cache,
                fingerprint=shared_fp,
                data=shared_data,
            )
        else:
            print(f"[cache] Shared {split}: {shared_cache}")
        datasets["Shared"][split] = shared_data

    rows: list[dict[str, Any]] = []
    for reliability in reliabilities:
        for model_name in ("Independent", "Shared"):
            val_data = datasets[model_name]["val"]
            test_data = datasets[model_name]["test"]
            optimization = optimize_thresholds(
                val_data,
                reliability=reliability,
                confidence_mode=args.confidence_mode,
            )

            row: dict[str, Any] = {
                "reliability_constraint_pct": 100.0 * reliability,
                "model": model_name,
                "confidence_mode": args.confidence_mode,
                "status": optimization["status"],
                "max_validation_emitted_accuracy": optimization[
                    "max_validation_emitted_accuracy"
                ],
            }
            if optimization["status"] == "ok":
                test_metrics = evaluate_thresholds(
                    test_data,
                    species_threshold=optimization["species_threshold"],
                    genus_threshold=optimization["genus_threshold"],
                    confidence_mode=args.confidence_mode,
                )
                row.update(
                    {
                        "species_threshold": optimization["species_threshold"],
                        "genus_threshold": optimization["genus_threshold"],
                        "val_emitted_accuracy": optimization["validation"][
                            "emitted_accuracy"
                        ],
                        "test_emitted_accuracy": test_metrics["emitted_accuracy"],
                        "species_output_pct": test_metrics["species_output_pct"],
                        "genus_output_pct": test_metrics["genus_output_pct"],
                        "family_output_pct": test_metrics["family_output_pct"],
                        "mean_depth": test_metrics["mean_depth"],
                        "normalized_specificity": test_metrics[
                            "normalized_specificity"
                        ],
                    }
                )
            else:
                row["reason"] = optimization["reason"]
            rows.append(row)

    json_path = output_dir / "fallback_results.json"
    json_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    csv_path = output_dir / "fallback_results.csv"
    import csv

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Exact report table requested by the user.
    table_lines = [
        "| Reliability constraint | Model | Species 출력 | Genus 출력 | Family 출력 | Mean depth |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        constraint = f"{row['reliability_constraint_pct']:g}%"
        if row["status"] != "ok":
            table_lines.append(
                f"| {constraint} | {row['model']} | INFEASIBLE | INFEASIBLE | "
                f"INFEASIBLE | INFEASIBLE |"
            )
        else:
            table_lines.append(
                f"| {constraint} | {row['model']} | "
                f"{row['species_output_pct']:.2f}% | "
                f"{row['genus_output_pct']:.2f}% | "
                f"{row['family_output_pct']:.2f}% | "
                f"{row['mean_depth']:.4f} |"
            )
    table_text = "\n".join(table_lines) + "\n"
    table_path = output_dir / "fallback_table.md"
    table_path.write_text(table_text, encoding="utf-8")

    print("\n" + "=" * 100)
    print("보고서용 표 (threshold는 validation에서만 선택, 출력 비율/MeanDepth는 test)")
    print("=" * 100)
    print(table_text)

    print("[threshold / reliability audit]")
    for row in rows:
        q = row["reliability_constraint_pct"]
        if row["status"] != "ok":
            print(
                f"{q:5.1f}%  {row['model']:<11} INFEASIBLE  "
                f"max-val={row['max_validation_emitted_accuracy']:.4f}%"
            )
            continue
        print(
            f"{q:5.1f}%  {row['model']:<11} "
            f"tauS={_fmt_threshold(row['species_threshold']):>12} "
            f"tauG={_fmt_threshold(row['genus_threshold']):>12} "
            f"val-acc={row['val_emitted_accuracy']:.4f}% "
            f"test-acc={row['test_emitted_accuracy']:.4f}% "
            f"depth={row['mean_depth']:.4f}"
        )

    print(f"\nJSON: {json_path}")
    print(f"CSV : {csv_path}")
    print(f"MD  : {table_path}")


if __name__ == "__main__":
    main()
