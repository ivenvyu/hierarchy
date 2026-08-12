#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared 계층 Prompt-CAM의 decision-boundary deletion을 계산한다.

목적
----
기존 confidence-drop만이 아니라, species CAM 상위 patch를 제거했을 때

1) 실제 species argmax prediction이 바뀌는가?
2) target-vs-best-competitor log-probability margin이 얼마나 무너지는가?
3) validation에서 고정한 hierarchical fallback threshold 아래에서
   species -> genus/family로 내려가는가?
4) 같은 크기의 random patch deletion보다 위 효과가 큰가?
5) CAM 순서대로 점점 더 지울 때 최초 decision flip/fallback은 언제 일어나는가?

를 correctly classified test images 전체에서 측정한다.

중요
----
- backbone은 이미지당 한 번만 통과한다.
- intervention은 decoder 입력의 patch token에 직접 수행한다.
- default replacement='mean'.
- fallback thresholds는 validation에서 선택된 기존 fallback_results.csv에서 읽는다.
- test에서 threshold를 다시 최적화하지 않는다.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def resolve(root: Path, value: str | Path) -> Path:
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = root / p
    return p.resolve()


def load_fixed_fallback_thresholds(
    csv_path: Path,
    *,
    reliability_pct: float,
    model_name: str = "Shared",
    confidence_mode: str = "joint",
) -> tuple[float, float, pd.Series]:
    df = pd.read_csv(csv_path)

    required = {
        "model",
        "confidence_mode",
        "reliability_constraint_pct",
        "species_threshold",
        "genus_threshold",
        "status",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"fallback 결과 CSV에 필요한 열이 없습니다: {sorted(missing)}"
        )

    mask = (
        df["model"].astype(str).str.lower().eq(str(model_name).lower())
        & df["confidence_mode"].astype(str).str.lower().eq(str(confidence_mode).lower())
        & np.isclose(
            pd.to_numeric(df["reliability_constraint_pct"], errors="coerce"),
            float(reliability_pct),
            rtol=0.0,
            atol=1e-9,
        )
        & df["status"].astype(str).str.lower().eq("ok")
    )

    rows = df.loc[mask]
    if len(rows) != 1:
        raise RuntimeError(
            "고정 fallback threshold 행을 정확히 하나 찾지 못했습니다: "
            f"model={model_name}, mode={confidence_mode}, "
            f"reliability={reliability_pct}, matches={len(rows)}"
        )

    row = rows.iloc[0]
    return float(row["species_threshold"]), float(row["genus_threshold"]), row


def _members_by_parent(mapping: torch.Tensor, parent_count: int) -> list[torch.Tensor]:
    result: list[torch.Tensor] = []
    for parent in range(int(parent_count)):
        members = torch.nonzero(mapping.eq(parent), as_tuple=False).flatten()
        if members.numel() == 0:
            raise RuntimeError(f"parent={parent}에 child가 없습니다")
        result.append(members)
    return result


def decision_metrics(
    output: Mapping[str, torch.Tensor],
    decoder,
    *,
    target_species: int,
    species_threshold: float,
    genus_threshold: float,
) -> dict[str, torch.Tensor]:
    """batch output에서 species decision / margin / fallback metrics 계산."""
    target_species = int(target_species)

    species_probs = output["species_probabilities"].float()
    if "species_log_probabilities" in output:
        species_logp = output["species_log_probabilities"].float()
    else:
        species_logp = species_probs.clamp_min(1e-12).log()

    batch = species_probs.shape[0]
    device = species_probs.device

    joint_species_prediction = species_probs.argmax(dim=1)
    target_probability = species_probs[:, target_species]
    target_logp = species_logp[:, target_species]

    competitor = species_logp.clone()
    competitor[:, target_species] = -torch.inf
    best_competitor_logp, best_competitor_species = competitor.max(dim=1)
    log_probability_margin = target_logp - best_competitor_logp

    family_probs = output["family_probabilities"].float()
    genus_cond = output["genus_conditional_probabilities"].float()
    species_cond = output["species_conditional_probabilities"].float()

    family_prediction = family_probs.argmax(dim=1)
    family_confidence = family_probs.gather(
        1, family_prediction[:, None]
    ).squeeze(1)

    genus_to_family = decoder.genus_to_family.to(device=device)
    species_to_genus = decoder.species_to_genus.to(device=device)

    num_families = int(family_probs.shape[1])
    num_genera = int(genus_cond.shape[1])

    family_to_genera = _members_by_parent(genus_to_family, num_families)
    genus_to_species = _members_by_parent(species_to_genus, num_genera)

    genus_prediction = torch.empty(batch, dtype=torch.long, device=device)
    genus_conditional_confidence = torch.empty(
        batch, dtype=torch.float32, device=device
    )

    for family_index, members in enumerate(family_to_genera):
        mask = family_prediction.eq(family_index)
        if not mask.any():
            continue
        local = genus_cond[mask].index_select(1, members)
        best_prob, best_local = local.max(dim=1)
        genus_prediction[mask] = members[best_local]
        genus_conditional_confidence[mask] = best_prob

    species_prediction_greedy = torch.empty(
        batch, dtype=torch.long, device=device
    )
    species_conditional_confidence = torch.empty(
        batch, dtype=torch.float32, device=device
    )

    for genus_index, members in enumerate(genus_to_species):
        mask = genus_prediction.eq(genus_index)
        if not mask.any():
            continue
        local = species_cond[mask].index_select(1, members)
        best_prob, best_local = local.max(dim=1)
        species_prediction_greedy[mask] = members[best_local]
        species_conditional_confidence[mask] = best_prob

    genus_joint_confidence = (
        family_confidence * genus_conditional_confidence
    )
    species_joint_confidence = (
        genus_joint_confidence * species_conditional_confidence
    )

    emitted_depth = torch.ones(batch, dtype=torch.long, device=device)
    emitted_depth[
        genus_joint_confidence.ge(float(genus_threshold))
    ] = 2
    emitted_depth[
        species_joint_confidence.ge(float(species_threshold))
    ] = 3

    true_genus = int(species_to_genus[target_species].item())
    true_family = int(genus_to_family[true_genus].item())

    emitted_correct = torch.where(
        emitted_depth.eq(3),
        species_prediction_greedy.eq(target_species),
        torch.where(
            emitted_depth.eq(2),
            genus_prediction.eq(true_genus),
            family_prediction.eq(true_family),
        ),
    )

    return {
        "joint_species_prediction": joint_species_prediction,
        "target_probability": target_probability,
        "target_log_probability": target_logp,
        "best_competitor_species": best_competitor_species,
        "best_competitor_log_probability": best_competitor_logp,
        "log_probability_margin": log_probability_margin,
        "family_prediction": family_prediction,
        "genus_prediction": genus_prediction,
        "species_prediction_greedy": species_prediction_greedy,
        "family_confidence": family_confidence,
        "genus_joint_confidence": genus_joint_confidence,
        "species_joint_confidence": species_joint_confidence,
        "emitted_depth": emitted_depth,
        "emitted_correct": emitted_correct,
    }


def evaluate_deleted_sets(
    decoder,
    family,
    genus,
    species,
    patches,
    global_feature,
    *,
    index_sets: list[np.ndarray],
    replacement: str,
    batch_size: int,
    target_species: int,
    species_threshold: float,
    genus_threshold: float,
    replacement_vector,
    decoder_forward,
) -> dict[str, np.ndarray]:
    repl = replacement_vector(patches, replacement)[0, 0]
    buffers: dict[str, list[np.ndarray]] = {}

    with torch.inference_mode():
        for start in range(0, len(index_sets), int(batch_size)):
            chunk = index_sets[start : start + int(batch_size)]
            b = len(chunk)

            p = patches.expand(b, -1, -1).clone()
            for j, idx_np in enumerate(chunk):
                idx = torch.as_tensor(
                    idx_np,
                    dtype=torch.long,
                    device=p.device,
                )
                p[j, idx] = repl

            out = decoder_forward(
                decoder,
                family.expand(b, -1, -1),
                genus.expand(b, -1, -1),
                species.expand(b, -1, -1),
                p,
                global_feature.expand(b, -1),
            )

            m = decision_metrics(
                out,
                decoder,
                target_species=target_species,
                species_threshold=species_threshold,
                genus_threshold=genus_threshold,
            )

            for key, value in m.items():
                arr = value.detach().cpu().numpy()
                buffers.setdefault(key, []).append(arr)

    return {
        key: np.concatenate(parts, axis=0)
        for key, parts in buffers.items()
    }


def depth_name(depth: int) -> str:
    return {1: "family", 2: "genus", 3: "species"}[int(depth)]


def first_true_fraction(
    fractions: list[float],
    indicators: list[bool],
) -> float:
    for fraction, flag in zip(fractions, indicators):
        if bool(flag):
            return float(fraction)
    return math.nan


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Shared species-CAM decision-boundary deletion analysis"
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--shared-run-dir", required=True)
    parser.add_argument(
        "--fallback-results-csv",
        default="output/hierarchical_fallback_comparison_dense/fallback_results.csv",
    )
    parser.add_argument(
        "--reliability",
        type=float,
        default=94.0,
        help="validation에서 선택된 fallback reliability constraint(%)",
    )
    parser.add_argument("--test-root", default="data/dataset/imagefolder/test")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        default=[0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90],
    )
    parser.add_argument("--replacement", choices=["zero", "mean"], default="mean")
    parser.add_argument("--random-repeats", type=int, default=50)
    parser.add_argument("--deletion-batch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="smoke test 전용. 전체 분석에서는 지정하지 않는다.",
    )
    args = parser.parse_args()

    fractions = sorted(set(float(x) for x in args.fractions))
    if not fractions:
        raise ValueError("--fractions가 비어 있습니다")
    if any((x <= 0.0 or x > 1.0) for x in fractions):
        raise ValueError("--fractions는 (0,1] 범위여야 합니다")

    project_root = Path(args.project_root).expanduser().resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from evaluation.cam.faithfulness import (
        contrast_defined,
        decoder_forward,
        load_image,
        load_shared,
        rank_cam,
        replacement_vector,
        split_decoder_inputs,
    )

    run_dir = resolve(project_root, args.shared_run_dir)
    fallback_csv = resolve(project_root, args.fallback_results_csv)
    test_root = resolve(project_root, args.test_root)
    output_dir = resolve(project_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    species_threshold, genus_threshold, threshold_row = (
        load_fixed_fallback_thresholds(
            fallback_csv,
            reliability_pct=float(args.reliability),
            model_name="Shared",
            confidence_mode="joint",
        )
    )

    print(
        "[fallback thresholds] "
        f"reliability={args.reliability:.1f}% "
        f"tauS={species_threshold:.9f} "
        f"tauG={genus_threshold:.9f}"
    )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA를 요청했지만 사용할 수 없습니다.")

    model, transform, taxonomy, checkpoint_path = load_shared(
        project_root,
        run_dir,
        device,
    )
    print(f"[checkpoint] {checkpoint_path}")

    class_dirs = sorted(p for p in test_root.iterdir() if p.is_dir())
    all_items: list[tuple[int, str, Path]] = []
    for species_index, class_dir in enumerate(class_dirs):
        paths = sorted(
            p for p in class_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
        for path in paths:
            all_items.append(
                (species_index, class_dir.name, path.resolve())
            )

    if args.max_images is not None:
        all_items = all_items[: int(args.max_images)]

    rng = np.random.default_rng(int(args.seed))

    curve_rows: list[dict[str, Any]] = []
    minimum_rows: list[dict[str, Any]] = []

    correct_count = 0
    incorrect_skip = 0
    singleton_skip = 0

    for image_no, (species_index, species_name, image_path) in enumerate(
        all_items, start=1
    ):
        image = load_image(image_path, transform, device)

        (
            decoder,
            family,
            genus,
            species,
            patches,
            global_feature,
        ) = split_decoder_inputs(model, image)

        with torch.inference_mode():
            baseline_output = decoder_forward(
                decoder,
                family,
                genus,
                species,
                patches,
                global_feature,
            )

        baseline_metrics = decision_metrics(
            baseline_output,
            decoder,
            target_species=species_index,
            species_threshold=species_threshold,
            genus_threshold=genus_threshold,
        )

        baseline_joint_pred = int(
            baseline_metrics["joint_species_prediction"][0].item()
        )
        if baseline_joint_pred != int(species_index):
            incorrect_skip += 1
            continue

        if not contrast_defined(
            baseline_output,
            "species",
            int(species_index),
        ):
            singleton_skip += 1
            continue

        correct_count += 1

        baseline_prob = float(
            baseline_metrics["target_probability"][0].item()
        )
        baseline_margin = float(
            baseline_metrics["log_probability_margin"][0].item()
        )
        baseline_depth = int(
            baseline_metrics["emitted_depth"][0].item()
        )
        baseline_emitted_correct = bool(
            baseline_metrics["emitted_correct"][0].item()
        )
        baseline_species_emitted_correct = bool(
            baseline_depth == 3 and baseline_emitted_correct
        )

        cam = (
            rank_cam(
                baseline_output,
                "species",
                int(species_index),
            )
            .numpy()
            .astype(np.float64)
        )
        patch_count = int(cam.size)
        cam_order = np.argsort(cam)

        # 한 random repeat 안에서는 fraction이 커질수록 deletion set이 nested가 되도록
        # random permutation을 한 번 생성하고 prefix를 사용한다.
        random_permutations = np.stack(
            [
                rng.permutation(patch_count).astype(np.int64)
                for _ in range(int(args.random_repeats))
            ],
            axis=0,
        )

        top_flip_flags: list[bool] = []
        top_fallback_flags: list[bool] = []

        random_first_flip = np.full(
            int(args.random_repeats),
            np.nan,
            dtype=np.float64,
        )
        random_first_fallback = np.full(
            int(args.random_repeats),
            np.nan,
            dtype=np.float64,
        )

        for fraction in fractions:
            k = max(1, int(round(float(fraction) * patch_count)))
            k = min(k, patch_count)

            top_idx = cam_order[-k:].copy()
            random_sets = [
                random_permutations[r, :k].copy()
                for r in range(int(args.random_repeats))
            ]

            deleted = evaluate_deleted_sets(
                decoder,
                family,
                genus,
                species,
                patches,
                global_feature,
                index_sets=[top_idx] + random_sets,
                replacement=args.replacement,
                batch_size=int(args.deletion_batch),
                target_species=int(species_index),
                species_threshold=species_threshold,
                genus_threshold=genus_threshold,
                replacement_vector=replacement_vector,
                decoder_forward=decoder_forward,
            )

            # first row = top-CAM deletion, rest = random repeats
            top_pred = int(deleted["joint_species_prediction"][0])
            random_pred = deleted["joint_species_prediction"][1:].astype(np.int64)

            top_flip = bool(top_pred != int(species_index))
            random_flip = random_pred != int(species_index)
            random_flip_rate = float(random_flip.mean())

            top_margin = float(deleted["log_probability_margin"][0])
            random_margin = deleted["log_probability_margin"][1:].astype(np.float64)

            top_margin_drop = baseline_margin - top_margin
            random_margin_drop = baseline_margin - random_margin
            random_margin_drop_mean = float(random_margin_drop.mean())

            top_depth = int(deleted["emitted_depth"][0])
            random_depth = deleted["emitted_depth"][1:].astype(np.int64)

            top_depth_loss = float(baseline_depth - top_depth)
            random_depth_loss = baseline_depth - random_depth
            random_depth_loss_mean = float(random_depth_loss.mean())

            top_emitted_correct = bool(deleted["emitted_correct"][0])
            random_emitted_correct = (
                deleted["emitted_correct"][1:].astype(bool)
            )

            if baseline_species_emitted_correct:
                top_species_to_fallback = float(top_depth < 3)
                random_species_to_fallback = (random_depth < 3)
                random_species_to_fallback_rate = float(
                    random_species_to_fallback.mean()
                )
                top_fallback_keeps_correct = (
                    float(top_emitted_correct)
                    if top_depth < 3
                    else math.nan
                )
                random_fallback_keeps_correct_rate = (
                    float(
                        random_emitted_correct[
                            random_species_to_fallback
                        ].mean()
                    )
                    if random_species_to_fallback.any()
                    else math.nan
                )
            else:
                top_species_to_fallback = math.nan
                random_species_to_fallback_rate = math.nan
                top_fallback_keeps_correct = math.nan
                random_fallback_keeps_correct_rate = math.nan

            top_flip_flags.append(top_flip)
            top_fallback_flags.append(
                bool(
                    baseline_species_emitted_correct
                    and top_depth < 3
                )
            )

            newly_flipped = (
                np.isnan(random_first_flip)
                & random_flip
            )
            random_first_flip[newly_flipped] = float(fraction)

            if baseline_species_emitted_correct:
                random_fb = random_depth < 3
                newly_fallback = (
                    np.isnan(random_first_fallback)
                    & random_fb
                )
                random_first_fallback[newly_fallback] = float(
                    fraction
                )

            curve_rows.append(
                {
                    "image_path": str(image_path),
                    "species": species_name,
                    "species_index": int(species_index),
                    "fraction": float(fraction),
                    "deleted_patch_count": int(k),
                    "patch_count": patch_count,
                    "replacement": args.replacement,
                    "random_repeats": int(args.random_repeats),
                    "fallback_reliability_pct": float(args.reliability),
                    "species_threshold": species_threshold,
                    "genus_threshold": genus_threshold,
                    "baseline_probability": baseline_prob,
                    "baseline_margin": baseline_margin,
                    "baseline_emitted_depth": baseline_depth,
                    "baseline_emitted_rank": depth_name(baseline_depth),
                    "baseline_emitted_correct": baseline_emitted_correct,
                    "baseline_species_emitted_correct": (
                        baseline_species_emitted_correct
                    ),
                    "top_species_prediction": top_pred,
                    "top_species_flip": float(top_flip),
                    "random_species_flip_rate": random_flip_rate,
                    "top_minus_random_flip": (
                        float(top_flip) - random_flip_rate
                    ),
                    "top_margin": top_margin,
                    "random_margin_mean": float(random_margin.mean()),
                    "top_margin_drop": top_margin_drop,
                    "random_margin_drop_mean": random_margin_drop_mean,
                    "top_minus_random_margin_drop": (
                        top_margin_drop - random_margin_drop_mean
                    ),
                    "top_emitted_depth": top_depth,
                    "top_emitted_rank": depth_name(top_depth),
                    "random_emitted_depth_mean": float(random_depth.mean()),
                    "top_depth_loss": top_depth_loss,
                    "random_depth_loss_mean": random_depth_loss_mean,
                    "top_minus_random_depth_loss": (
                        top_depth_loss - random_depth_loss_mean
                    ),
                    "top_emitted_correct": float(top_emitted_correct),
                    "random_emitted_correct_rate": float(
                        random_emitted_correct.mean()
                    ),
                    "top_species_to_fallback": top_species_to_fallback,
                    "random_species_to_fallback_rate": (
                        random_species_to_fallback_rate
                    ),
                    "top_minus_random_species_fallback": (
                        top_species_to_fallback
                        - random_species_to_fallback_rate
                        if not math.isnan(top_species_to_fallback)
                        else math.nan
                    ),
                    "top_fallback_keeps_correct": (
                        top_fallback_keeps_correct
                    ),
                    "random_fallback_keeps_correct_rate": (
                        random_fallback_keeps_correct_rate
                    ),
                }
            )

        top_first_flip_fraction = first_true_fraction(
            fractions,
            top_flip_flags,
        )

        if baseline_species_emitted_correct:
            top_first_fallback_fraction = first_true_fraction(
                fractions,
                top_fallback_flags,
            )
        else:
            top_first_fallback_fraction = math.nan

        minimum_rows.append(
            {
                "image_path": str(image_path),
                "species": species_name,
                "species_index": int(species_index),
                "baseline_probability": baseline_prob,
                "baseline_margin": baseline_margin,
                "baseline_emitted_depth": baseline_depth,
                "baseline_emitted_rank": depth_name(baseline_depth),
                "baseline_emitted_correct": baseline_emitted_correct,
                "baseline_species_emitted_correct": (
                    baseline_species_emitted_correct
                ),
                "top_first_flip_fraction": top_first_flip_fraction,
                "top_no_flip_through_max": bool(
                    math.isnan(top_first_flip_fraction)
                ),
                "random_first_flip_fraction_mean_observed": (
                    float(np.nanmean(random_first_flip))
                    if np.isfinite(random_first_flip).any()
                    else math.nan
                ),
                "random_no_flip_through_max_rate": float(
                    np.isnan(random_first_flip).mean()
                ),
                "top_first_fallback_fraction": (
                    top_first_fallback_fraction
                ),
                "top_no_fallback_through_max": (
                    bool(math.isnan(top_first_fallback_fraction))
                    if baseline_species_emitted_correct
                    else math.nan
                ),
                "random_first_fallback_fraction_mean_observed": (
                    float(np.nanmean(random_first_fallback))
                    if (
                        baseline_species_emitted_correct
                        and np.isfinite(random_first_fallback).any()
                    )
                    else math.nan
                ),
                "random_no_fallback_through_max_rate": (
                    float(np.isnan(random_first_fallback).mean())
                    if baseline_species_emitted_correct
                    else math.nan
                ),
                "max_fraction_tested": float(max(fractions)),
            }
        )

        if (
            image_no % 25 == 0
            or image_no == len(all_items)
        ):
            print(
                f"[{image_no}/{len(all_items)}] "
                f"usable={correct_count}, "
                f"incorrect-skip={incorrect_skip}, "
                f"singleton-skip={singleton_skip}"
            )

    if not curve_rows:
        raise RuntimeError("분석 가능한 결과가 없습니다.")

    curve = pd.DataFrame(curve_rows)
    minima = pd.DataFrame(minimum_rows)

    curve_path = output_dir / "decision_deletion_per_image_fraction.csv"
    minima_path = output_dir / "decision_minimal_fraction_per_image.csv"

    curve.to_csv(curve_path, index=False)
    minima.to_csv(minima_path, index=False)

    summary = (
        curve.groupby("fraction")
        .agg(
            n_images=("image_path", "size"),
            n_species=("species", "nunique"),
            top_flip_rate=("top_species_flip", "mean"),
            random_flip_rate=("random_species_flip_rate", "mean"),
            mean_top_minus_random_flip=(
                "top_minus_random_flip",
                "mean",
            ),
            mean_top_margin_drop=("top_margin_drop", "mean"),
            mean_random_margin_drop=(
                "random_margin_drop_mean",
                "mean",
            ),
            mean_top_minus_random_margin_drop=(
                "top_minus_random_margin_drop",
                "mean",
            ),
            mean_baseline_depth=("baseline_emitted_depth", "mean"),
            mean_top_depth=("top_emitted_depth", "mean"),
            mean_random_depth=("random_emitted_depth_mean", "mean"),
            mean_top_minus_random_depth_loss=(
                "top_minus_random_depth_loss",
                "mean",
            ),
            baseline_species_emitted_n=(
                "baseline_species_emitted_correct",
                "sum",
            ),
            top_species_to_fallback_rate=(
                "top_species_to_fallback",
                "mean",
            ),
            random_species_to_fallback_rate=(
                "random_species_to_fallback_rate",
                "mean",
            ),
            mean_top_minus_random_species_fallback=(
                "top_minus_random_species_fallback",
                "mean",
            ),
            top_emitted_accuracy=("top_emitted_correct", "mean"),
            random_emitted_accuracy=(
                "random_emitted_correct_rate",
                "mean",
            ),
        )
        .reset_index()
    )

    summary_path = output_dir / "decision_deletion_summary.csv"
    summary.to_csv(summary_path, index=False)

    threshold_audit = pd.DataFrame(
        [
            {
                "model": "Shared",
                "confidence_mode": "joint",
                "reliability_constraint_pct": float(args.reliability),
                "species_threshold": species_threshold,
                "genus_threshold": genus_threshold,
                "source_csv": str(fallback_csv),
                "source_val_emitted_accuracy": threshold_row.get(
                    "val_emitted_accuracy",
                    math.nan,
                ),
                "source_test_emitted_accuracy": threshold_row.get(
                    "test_emitted_accuracy",
                    math.nan,
                ),
            }
        ]
    )
    threshold_audit.to_csv(
        output_dir / "fallback_threshold_audit.csv",
        index=False,
    )

    print()
    print("===== DECISION-DELETION SUMMARY =====")
    print(summary.to_string(index=False))
    print()
    print(f"usable correct non-singleton images : {correct_count}")
    print(f"incorrect skipped                   : {incorrect_skip}")
    print(f"singleton-contrast skipped          : {singleton_skip}")
    print(f"[저장] {curve_path}")
    print(f"[저장] {minima_path}")
    print(f"[저장] {summary_path}")


if __name__ == "__main__":
    main()
