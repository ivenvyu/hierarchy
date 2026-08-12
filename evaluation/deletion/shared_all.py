#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""모든 모델이 맞힌 표본에서 shared CAM deletion을 계산한다."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def resolve(root: Path, value: str | Path) -> Path:
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = root / p
    return p.resolve()


def batched_delete_sets(
    decoder,
    family,
    genus,
    species,
    patches,
    global_feature,
    *,
    rank: str,
    target: int,
    index_sets: list[np.ndarray],
    replacement: str,
    batch_size: int,
    rank_probability,
    rank_log_probability,
    replacement_vector,
    decoder_forward,
) -> tuple[np.ndarray, np.ndarray]:
    """여러 patch-set intervention을 batch로 계산한다."""
    repl = replacement_vector(patches, replacement)[0, 0]
    probs: list[float] = []
    logs: list[float] = []

    with torch.inference_mode():
        for start in range(0, len(index_sets), int(batch_size)):
            chunk = index_sets[start : start + int(batch_size)]
            b = len(chunk)

            p = patches.expand(b, -1, -1).clone()
            for j, idx_np in enumerate(chunk):
                idx = torch.as_tensor(idx_np, dtype=torch.long, device=p.device)
                p[j, idx] = repl

            out = decoder_forward(
                decoder,
                family.expand(b, -1, -1),
                genus.expand(b, -1, -1),
                species.expand(b, -1, -1),
                p,
                global_feature.expand(b, -1),
            )

            probs.extend(
                rank_probability(out, rank, target).detach().float().cpu().tolist()
            )
            logs.extend(
                rank_log_probability(out, rank, target).detach().float().cpu().tolist()
            )

    return np.asarray(probs, dtype=np.float64), np.asarray(logs, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Shared hierarchical Prompt-CAM의 correctly classified test images 전체에서 "
            "top-CAM / random / bottom patch-set deletion만 빠르게 계산한다."
        )
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--shared-run-dir", required=True)
    parser.add_argument("--test-root", default="data/dataset/imagefolder/test")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--ranks",
        nargs="+",
        choices=["family", "genus", "species"],
        default=["species"],
    )
    parser.add_argument("--replacement", choices=["zero", "mean"], default="mean")
    parser.add_argument("--top-fraction", type=float, default=0.10)
    parser.add_argument("--random-repeats", type=int, default=50)
    parser.add_argument("--deletion-batch", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="smoke test용. 전체 분석에서는 지정하지 않는다.",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # 이미 서버에서 검증된 shared faithfulness 구현의 loader/decoder helper를 그대로 재사용.
    from evaluation.cam.faithfulness import (
        contrast_defined,
        decoder_forward,
        load_image,
        load_shared,
        rank_cam,
        rank_log_probability,
        rank_probability,
        rank_target,
        replacement_vector,
        split_decoder_inputs,
    )

    run_dir = resolve(project_root, args.shared_run_dir)
    test_root = resolve(project_root, args.test_root)
    output_dir = resolve(project_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA를 요청했지만 사용할 수 없습니다.")

    model, transform, taxonomy, checkpoint_path = load_shared(
        project_root, run_dir, device
    )
    print(f"[checkpoint] {checkpoint_path}")

    class_dirs = sorted(p for p in test_root.iterdir() if p.is_dir())
    if not class_dirs:
        raise RuntimeError(f"test class directory가 없습니다: {test_root}")

    all_items: list[tuple[int, str, Path]] = []
    for species_index, class_dir in enumerate(class_dirs):
        paths = sorted(
            p for p in class_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
        for p in paths:
            all_items.append((species_index, class_dir.name, p.resolve()))

    if args.max_images is not None:
        all_items = all_items[: int(args.max_images)]

    print(f"[test images] {len(all_items)}")
    print(f"[ranks] {args.ranks}")
    print(f"[replacement] {args.replacement}")
    print(f"[random repeats] {args.random_repeats}")

    rng = np.random.default_rng(int(args.seed))
    rows: list[dict[str, Any]] = []
    correct_count = 0
    skipped_incorrect = 0

    for image_no, (species_index, species_name, image_path) in enumerate(all_items, start=1):
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
            baseline = decoder_forward(
                decoder,
                family,
                genus,
                species,
                patches,
                global_feature,
            )

        pred_species = int(
            baseline["species_probabilities"][0].argmax().item()
        )
        if pred_species != int(species_index):
            skipped_incorrect += 1
            if image_no % 50 == 0:
                print(
                    f"[{image_no}/{len(all_items)}] correct={correct_count}, "
                    f"incorrect-skip={skipped_incorrect}"
                )
            continue

        correct_count += 1

        for rank in args.ranks:
            target = rank_target(decoder, species_index, rank)
            if not contrast_defined(baseline, rank, target):
                continue

            base_prob = float(rank_probability(baseline, rank, target)[0].item())
            base_log = float(rank_log_probability(baseline, rank, target)[0].item())
            cam = rank_cam(baseline, rank, target).numpy().astype(np.float64)

            patch_count = int(cam.size)
            k = max(1, int(round(float(args.top_fraction) * patch_count)))
            order = np.argsort(cam)

            top_idx = order[-k:].copy()
            bottom_idx = order[:k].copy()

            random_sets = [
                rng.choice(patch_count, size=k, replace=False).astype(np.int64)
                for _ in range(int(args.random_repeats))
            ]

            # 0=top, 1=bottom, 2...=random
            index_sets = [top_idx, bottom_idx] + random_sets

            probs, logs = batched_delete_sets(
                decoder,
                family,
                genus,
                species,
                patches,
                global_feature,
                rank=rank,
                target=target,
                index_sets=index_sets,
                replacement=args.replacement,
                batch_size=args.deletion_batch,
                rank_probability=rank_probability,
                rank_log_probability=rank_log_probability,
                replacement_vector=replacement_vector,
                decoder_forward=decoder_forward,
            )

            top_prob = float(probs[0])
            bottom_prob = float(probs[1])
            random_probs = probs[2:]

            top_log = float(logs[0])
            bottom_log = float(logs[1])
            random_logs = logs[2:]

            top_drop = base_prob - top_prob
            bottom_drop = base_prob - bottom_prob
            random_drops = base_prob - random_probs

            top_log_drop = base_log - top_log
            bottom_log_drop = base_log - bottom_log
            random_log_drops = base_log - random_logs

            rows.append(
                {
                    "image_path": str(image_path),
                    "species": species_name,
                    "species_index": int(species_index),
                    "rank": rank,
                    "target_index": int(target),
                    "predicted_species_index": pred_species,
                    "correct_species": True,
                    "baseline_confidence": base_prob,
                    "baseline_log_probability": base_log,
                    "replacement": args.replacement,
                    "top_fraction": float(args.top_fraction),
                    "patch_count": patch_count,
                    "top_patch_count": k,
                    "random_repeats": int(args.random_repeats),
                    "top_cam_set_drop": top_drop,
                    "bottom_cam_set_drop": bottom_drop,
                    "random_set_drop_mean": float(random_drops.mean()),
                    "random_set_drop_std": (
                        float(random_drops.std(ddof=1))
                        if len(random_drops) > 1
                        else math.nan
                    ),
                    "top_minus_random_drop": float(
                        top_drop - random_drops.mean()
                    ),
                    "top_minus_bottom_drop": float(
                        top_drop - bottom_drop
                    ),
                    "top_cam_set_log_drop": top_log_drop,
                    "bottom_cam_set_log_drop": bottom_log_drop,
                    "random_set_log_drop_mean": float(random_log_drops.mean()),
                    "random_set_log_drop_std": (
                        float(random_log_drops.std(ddof=1))
                        if len(random_log_drops) > 1
                        else math.nan
                    ),
                    "top_minus_random_log_drop": float(
                        top_log_drop - random_log_drops.mean()
                    ),
                    "top_minus_bottom_log_drop": float(
                        top_log_drop - bottom_log_drop
                    ),
                }
            )

        if image_no % 25 == 0 or image_no == len(all_items):
            print(
                f"[{image_no}/{len(all_items)}] correct={correct_count}, "
                f"incorrect-skip={skipped_incorrect}, rows={len(rows)}"
            )

    if not rows:
        raise RuntimeError("유효한 분석 결과가 없습니다.")

    per_image = pd.DataFrame(rows)
    per_image_path = output_dir / "set_deletion_per_image.csv"
    per_image.to_csv(per_image_path, index=False)

    summary = (
        per_image.groupby("rank", dropna=False)
        .agg(
            n_images=("image_path", "size"),
            n_species=("species", "nunique"),
            mean_baseline_confidence=("baseline_confidence", "mean"),
            mean_top_cam_set_drop=("top_cam_set_drop", "mean"),
            mean_random_set_drop=("random_set_drop_mean", "mean"),
            mean_bottom_cam_set_drop=("bottom_cam_set_drop", "mean"),
            mean_top_minus_random_drop=("top_minus_random_drop", "mean"),
            median_top_minus_random_drop=("top_minus_random_drop", "median"),
            mean_top_minus_bottom_drop=("top_minus_bottom_drop", "mean"),
            median_top_minus_bottom_drop=("top_minus_bottom_drop", "median"),
        )
        .reset_index()
    )
    summary_path = output_dir / "set_deletion_summary.csv"
    summary.to_csv(summary_path, index=False)

    species_counts = (
        per_image.groupby(["rank", "species", "species_index"])
        .size()
        .rename("n_images")
        .reset_index()
    )
    species_counts_path = output_dir / "species_counts.csv"
    species_counts.to_csv(species_counts_path, index=False)

    print()
    print("===== SET-DELETION SUMMARY =====")
    print(summary.to_string(index=False))
    print()
    print(f"correct species images: {correct_count}")
    print(f"incorrect skipped      : {skipped_incorrect}")
    print(f"[저장] {per_image_path}")
    print(f"[저장] {summary_path}")
    print(f"[저장] {species_counts_path}")


if __name__ == "__main__":
    main()
