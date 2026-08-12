#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Decision-boundary CAM deletion 곡선의 종 단위 bootstrap을 계산한다."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ENDPOINTS = [
    "top_minus_random_flip",
    "top_minus_random_margin_drop",
    "top_minus_random_depth_loss",
    "top_minus_random_species_fallback",
]


def percentile_ci(draws: np.ndarray, alpha: float) -> tuple[float, float]:
    return (
        float(np.quantile(draws, alpha / 2.0)),
        float(np.quantile(draws, 1.0 - alpha / 2.0)),
    )


def pooled_cluster_draws(
    stats: pd.DataFrame,
    *,
    reps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    cluster_sum = stats["effect_sum"].to_numpy(dtype=np.float64)
    cluster_n = stats["n_images"].to_numpy(dtype=np.float64)
    s = len(stats)

    draws = np.empty(int(reps), dtype=np.float64)
    for b in range(int(reps)):
        sampled = rng.integers(0, s, size=s)
        draws[b] = (
            cluster_sum[sampled].sum()
            / cluster_n[sampled].sum()
        )
    return draws


def species_equal_draws(
    species_means: np.ndarray,
    *,
    reps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    s = len(species_means)
    draws = np.empty(int(reps), dtype=np.float64)
    for b in range(int(reps)):
        sampled = rng.integers(0, s, size=s)
        draws[b] = species_means[sampled].mean()
    return draws


def main() -> None:
    parser = argparse.ArgumentParser()
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

    required = {"species", "fraction"} | set(args.endpoints)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"입력 CSV에 필요한 열이 없습니다: {sorted(missing)}"
        )

    master_rng = np.random.default_rng(int(args.seed))

    summary_rows = []
    species_rows = []

    for fraction in sorted(df["fraction"].dropna().unique()):
        fdf = df[np.isclose(df["fraction"], fraction)].copy()

        for endpoint in args.endpoints:
            g = fdf[["species", endpoint]].dropna().copy()
            if g.empty:
                continue

            stats = (
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

            n_species = len(stats)
            if n_species < 2:
                continue

            species_means = stats["effect_mean"].to_numpy(
                dtype=np.float64
            )

            pooled_draws = pooled_cluster_draws(
                stats,
                reps=int(args.bootstrap_reps),
                rng=np.random.default_rng(
                    int(master_rng.integers(0, 2**32 - 1))
                ),
            )
            equal_draws = species_equal_draws(
                species_means,
                reps=int(args.bootstrap_reps),
                rng=np.random.default_rng(
                    int(master_rng.integers(0, 2**32 - 1))
                ),
            )

            pooled_estimate = float(g[endpoint].mean())
            equal_estimate = float(species_means.mean())

            for estimand, estimate, draws in (
                (
                    "cluster_pooled_image_mean",
                    pooled_estimate,
                    pooled_draws,
                ),
                (
                    "species_equal_mean",
                    equal_estimate,
                    equal_draws,
                ),
            ):
                lo, hi = percentile_ci(
                    draws,
                    float(args.alpha),
                )
                summary_rows.append(
                    {
                        "fraction": float(fraction),
                        "endpoint": endpoint,
                        "estimand": estimand,
                        "n_images": int(len(g)),
                        "n_species": int(n_species),
                        "estimate": estimate,
                        "bootstrap_se": float(
                            draws.std(ddof=1)
                        ),
                        "ci95_lower": lo,
                        "ci95_upper": hi,
                        "ci_excludes_zero_positive": bool(
                            lo > 0.0
                        ),
                        "positive_species": int(
                            (species_means > 0).sum()
                        ),
                        "zero_species": int(
                            (species_means == 0).sum()
                        ),
                        "negative_species": int(
                            (species_means < 0).sum()
                        ),
                        "positive_species_fraction": float(
                            (species_means > 0).mean()
                        ),
                        "bootstrap_reps": int(
                            args.bootstrap_reps
                        ),
                        "seed": int(args.seed),
                    }
                )

            for _, row in stats.iterrows():
                species_rows.append(
                    {
                        "fraction": float(fraction),
                        "endpoint": endpoint,
                        "species": row["species"],
                        "n_images": int(row["n_images"]),
                        "effect_mean": float(
                            row["effect_mean"]
                        ),
                        "effect_median": float(
                            row["effect_median"]
                        ),
                    }
                )

    summary = pd.DataFrame(summary_rows)
    species_effects = pd.DataFrame(species_rows)

    if summary.empty:
        raise RuntimeError("bootstrap 결과가 없습니다.")

    summary_path = (
        output_dir
        / "decision_species_cluster_bootstrap_summary.csv"
    )
    species_path = (
        output_dir
        / "decision_species_effects.csv"
    )

    summary.to_csv(summary_path, index=False)
    species_effects.to_csv(species_path, index=False)

    print("===== DECISION SPECIES-CLUSTER BOOTSTRAP =====")
    print(summary.to_string(index=False))
    print()
    print(f"[저장] {summary_path}")
    print(f"[저장] {species_path}")


if __name__ == "__main__":
    main()
