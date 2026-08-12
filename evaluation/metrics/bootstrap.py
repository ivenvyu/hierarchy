#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""종 단위 cluster bootstrap 신뢰구간을 계산한다."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ENDPOINTS = [
    "top_minus_random_drop",
    "top_minus_bottom_drop",
]


def percentile_ci(draws: np.ndarray, alpha: float) -> tuple[float, float]:
    lo = float(np.quantile(draws, alpha / 2.0))
    hi = float(np.quantile(draws, 1.0 - alpha / 2.0))
    return lo, hi


def cluster_bootstrap_pooled(
    species_names: np.ndarray,
    cluster_sum: np.ndarray,
    cluster_n: np.ndarray,
    *,
    reps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    사용자가 요청한 방식 그대로:
    species를 cluster 단위로 복원추출하고, 선택된 species에 속한 이미지를
    통째로 포함한 뒤 전체 이미지 평균을 계산한다.

    cluster가 여러 번 선택되면 그 cluster의 모든 이미지도 그 횟수만큼 복제된다.
    """
    s = len(species_names)
    draws = np.empty(int(reps), dtype=np.float64)

    for b in range(int(reps)):
        sampled = rng.integers(0, s, size=s)
        total_sum = cluster_sum[sampled].sum()
        total_n = cluster_n[sampled].sum()
        draws[b] = total_sum / total_n

    return draws


def cluster_bootstrap_species_equal(
    species_means: np.ndarray,
    *,
    reps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    sensitivity analysis:
    species별 평균을 먼저 계산하고 species를 동일 가중치로 bootstrap한다.
    correct-only에서는 species별 이미지 수가 달라질 수 있으므로 함께 보고한다.
    """
    s = len(species_means)
    draws = np.empty(int(reps), dtype=np.float64)

    for b in range(int(reps)):
        sampled = rng.integers(0, s, size=s)
        draws[b] = species_means[sampled].mean()

    return draws


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Species-cluster bootstrap 95% CI for Prompt-CAM set-deletion effects"
    )
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument(
        "--endpoints",
        nargs="+",
        default=DEFAULT_ENDPOINTS,
    )
    args = parser.parse_args()

    input_csv = Path(args.input_csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)

    required = {"species", "rank"} | set(args.endpoints)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"입력 CSV에 필요한 열이 없습니다: {sorted(missing)}")

    df = df.copy()
    df["species"] = df["species"].astype(str)
    df["rank"] = df["rank"].astype(str)

    summary_rows = []
    species_rows = []
    draw_rows = []

    master_rng = np.random.default_rng(int(args.seed))

    for rank in sorted(df["rank"].dropna().unique()):
        rank_df = df[df["rank"] == rank].copy()

        for endpoint in args.endpoints:
            g = rank_df[["species", endpoint]].dropna().copy()
            if g.empty:
                continue

            species_stats = (
                g.groupby("species")[endpoint]
                .agg(["count", "sum", "mean", "median"])
                .reset_index()
                .rename(
                    columns={
                        "count": "n_images",
                        "sum": "effect_sum",
                        "mean": "effect_mean",
                        "median": "effect_median",
                    }
                )
                .sort_values("species")
                .reset_index(drop=True)
            )

            n_species = len(species_stats)
            n_images = len(g)
            if n_species < 2:
                raise RuntimeError(
                    f"{rank}/{endpoint}: cluster 수가 {n_species}개라 bootstrap CI를 계산할 수 없습니다."
                )

            species_names = species_stats["species"].to_numpy()
            cluster_sum = species_stats["effect_sum"].to_numpy(dtype=np.float64)
            cluster_n = species_stats["n_images"].to_numpy(dtype=np.float64)
            species_means = species_stats["effect_mean"].to_numpy(dtype=np.float64)

            # endpoint/rank마다 독립된 재현가능 RNG stream.
            seed_pooled = int(master_rng.integers(0, 2**32 - 1))
            seed_equal = int(master_rng.integers(0, 2**32 - 1))

            pooled_draws = cluster_bootstrap_pooled(
                species_names,
                cluster_sum,
                cluster_n,
                reps=args.bootstrap_reps,
                rng=np.random.default_rng(seed_pooled),
            )
            equal_draws = cluster_bootstrap_species_equal(
                species_means,
                reps=args.bootstrap_reps,
                rng=np.random.default_rng(seed_equal),
            )

            pooled_obs = float(g[endpoint].mean())
            equal_obs = float(species_means.mean())

            pooled_lo, pooled_hi = percentile_ci(
                pooled_draws, float(args.alpha)
            )
            equal_lo, equal_hi = percentile_ci(
                equal_draws, float(args.alpha)
            )

            positive_species = int((species_means > 0).sum())
            zero_species = int((species_means == 0).sum())
            negative_species = int((species_means < 0).sum())

            for estimand, obs, draws, lo, hi in (
                (
                    "cluster_pooled_image_mean",
                    pooled_obs,
                    pooled_draws,
                    pooled_lo,
                    pooled_hi,
                ),
                (
                    "species_equal_mean",
                    equal_obs,
                    equal_draws,
                    equal_lo,
                    equal_hi,
                ),
            ):
                summary_rows.append(
                    {
                        "rank": rank,
                        "endpoint": endpoint,
                        "estimand": estimand,
                        "n_images": int(n_images),
                        "n_species": int(n_species),
                        "estimate": float(obs),
                        "bootstrap_se": float(draws.std(ddof=1)),
                        "ci95_lower": float(lo),
                        "ci95_upper": float(hi),
                        "ci_excludes_zero_positive": bool(lo > 0),
                        "positive_species": positive_species,
                        "zero_species": zero_species,
                        "negative_species": negative_species,
                        "positive_species_fraction": float(
                            positive_species / n_species
                        ),
                        "bootstrap_reps": int(args.bootstrap_reps),
                        "seed": int(args.seed),
                    }
                )

                for b, value in enumerate(draws):
                    draw_rows.append(
                        {
                            "rank": rank,
                            "endpoint": endpoint,
                            "estimand": estimand,
                            "bootstrap_index": int(b),
                            "estimate": float(value),
                        }
                    )

            for _, row in species_stats.iterrows():
                species_rows.append(
                    {
                        "rank": rank,
                        "endpoint": endpoint,
                        "species": row["species"],
                        "n_images": int(row["n_images"]),
                        "effect_mean": float(row["effect_mean"]),
                        "effect_median": float(row["effect_median"]),
                    }
                )

    summary = pd.DataFrame(summary_rows)
    species_effects = pd.DataFrame(species_rows)
    bootstrap_draws = pd.DataFrame(draw_rows)

    if summary.empty:
        raise RuntimeError("분석 가능한 rank/endpoint가 없습니다.")

    summary_path = output_dir / "species_cluster_bootstrap_summary.csv"
    species_path = output_dir / "species_effects.csv"
    draws_path = output_dir / "species_cluster_bootstrap_draws.csv"

    summary.to_csv(summary_path, index=False)
    species_effects.to_csv(species_path, index=False)
    bootstrap_draws.to_csv(draws_path, index=False)

    print("===== SPECIES-CLUSTER BOOTSTRAP =====")
    print(summary.to_string(index=False))
    print()
    print(f"[저장] {summary_path}")
    print(f"[저장] {species_path}")
    print(f"[저장] {draws_path}")


if __name__ == "__main__":
    main()
