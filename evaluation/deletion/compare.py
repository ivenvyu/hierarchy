#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared, flat, independent deletion 결과를 비교한다.

- model-own correct non-singleton sample summary
- 세 모델 모두 정분류한 common-image subset의 paired comparison
- species-cluster paired bootstrap 95% CI
- Shared vs Independent의 common baseline-species-emitted subset fallback comparison
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def cluster_bootstrap_difference(
    df: pd.DataFrame,
    effect_col: str,
    *,
    reps: int,
    seed: int,
) -> dict:
    stats = (
        df.groupby("species")[effect_col]
        .agg(["count", "sum", "mean"])
        .reset_index()
    )
    if len(stats) < 2:
        raise RuntimeError(
            f"{effect_col}: species cluster가 {len(stats)}개뿐입니다"
        )

    rng = np.random.default_rng(int(seed))
    cluster_sum = stats["sum"].to_numpy(dtype=float)
    cluster_n = stats["count"].to_numpy(dtype=float)
    cluster_mean = stats["mean"].to_numpy(dtype=float)
    s = len(stats)

    pooled = np.empty(reps, dtype=float)
    equal = np.empty(reps, dtype=float)

    for b in range(reps):
        idx = rng.integers(0, s, size=s)
        pooled[b] = cluster_sum[idx].sum() / cluster_n[idx].sum()
        equal[b] = cluster_mean[idx].mean()

    return {
        "n_images": len(df),
        "n_species": s,
        "pooled_estimate": float(df[effect_col].mean()),
        "pooled_ci_lower": float(np.quantile(pooled, 0.025)),
        "pooled_ci_upper": float(np.quantile(pooled, 0.975)),
        "species_equal_estimate": float(cluster_mean.mean()),
        "species_equal_ci_lower": float(np.quantile(equal, 0.025)),
        "species_equal_ci_upper": float(np.quantile(equal, 0.975)),
        "positive_species": int((cluster_mean > 0).sum()),
        "zero_species": int((cluster_mean == 0).sum()),
        "negative_species": int((cluster_mean < 0).sum()),
    }


def own_summary(name: str, df: pd.DataFrame) -> dict:
    row = {
        "model": name,
        "n_images": len(df),
        "n_species": df["species"].nunique(),
        "top_flip_rate": df["top_species_flip"].mean(),
        "random_flip_rate": df["random_species_flip_rate"].mean(),
        "top_minus_random_flip": df["top_minus_random_flip"].mean(),
        "top_margin_drop": df["top_margin_drop"].mean(),
        "random_margin_drop": df["random_margin_drop_mean"].mean(),
        "top_minus_random_margin_drop": (
            df["top_minus_random_margin_drop"].mean()
        ),
    }

    if "baseline_species_emitted_correct" in df.columns:
        s = df[
            df["baseline_species_emitted_correct"].fillna(False).astype(bool)
        ]
        row.update(
            {
                "baseline_species_emitted_n": len(s),
                "top_species_to_fallback_rate": (
                    s["top_species_to_fallback"].mean()
                    if len(s) else np.nan
                ),
                "random_species_to_fallback_rate": (
                    s["random_species_to_fallback_rate"].mean()
                    if len(s) else np.nan
                ),
                "top_minus_random_species_fallback": (
                    s["top_minus_random_species_fallback"].mean()
                    if len(s) else np.nan
                ),
                "top_fallback_correct_rate": (
                    s.loc[
                        s["top_species_to_fallback"].eq(1),
                        "top_emitted_correct",
                    ].mean()
                    if s["top_species_to_fallback"].eq(1).any()
                    else np.nan
                ),
            }
        )
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-csv", required=True)
    parser.add_argument("--flat-csv", required=True)
    parser.add_argument("--independent-csv", required=True)
    parser.add_argument("--fraction", type=float, default=0.10)
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    paths = {
        "Shared": Path(args.shared_csv).expanduser().resolve(),
        "Flat": Path(args.flat_csv).expanduser().resolve(),
        "Independent": Path(args.independent_csv).expanduser().resolve(),
    }

    frames = {}
    for name, path in paths.items():
        df = pd.read_csv(path)
        df = df[np.isclose(df["fraction"], float(args.fraction))].copy()
        if df.empty:
            raise RuntimeError(
                f"{name}: fraction={args.fraction} 행이 없습니다"
            )
        frames[name] = df

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    own = pd.DataFrame(
        [own_summary(name, frames[name]) for name in ("Flat", "Independent", "Shared")]
    )
    own.to_csv(out_dir / "model_own_sample_summary.csv", index=False)

    # 세 모델 모두 baseline forced-species가 correct이고 non-singleton인 공통 image.
    columns = [
        "image_path",
        "species",
        "species_index",
        "top_species_flip",
        "random_species_flip_rate",
        "top_minus_random_flip",
        "top_margin_drop",
        "random_margin_drop_mean",
        "top_minus_random_margin_drop",
    ]

    common = None
    for name in ("Flat", "Independent", "Shared"):
        x = frames[name][columns].copy()
        rename = {
            c: f"{name.lower()}__{c}"
            for c in columns
            if c not in {"image_path", "species", "species_index"}
        }
        x = x.rename(columns=rename)
        if common is None:
            common = x
        else:
            common = common.merge(
                x,
                on=["image_path", "species", "species_index"],
                how="inner",
                validate="one_to_one",
            )

    common.to_csv(
        out_dir / "common_correct_images.csv",
        index=False,
    )

    paired_rows = []
    endpoint_map = {
        "flip_excess": "top_minus_random_flip",
        "margin_drop_excess": "top_minus_random_margin_drop",
    }

    for endpoint_name, suffix in endpoint_map.items():
        for left, right in (
            ("Shared", "Flat"),
            ("Shared", "Independent"),
            ("Independent", "Flat"),
        ):
            work = common[
                ["species", f"{left.lower()}__{suffix}", f"{right.lower()}__{suffix}"]
            ].copy()
            work["paired_difference"] = (
                work[f"{left.lower()}__{suffix}"]
                - work[f"{right.lower()}__{suffix}"]
            )
            result = cluster_bootstrap_difference(
                work,
                "paired_difference",
                reps=int(args.bootstrap_reps),
                seed=int(args.seed)
                + len(paired_rows) * 1009,
            )
            paired_rows.append(
                {
                    "fraction": float(args.fraction),
                    "endpoint": endpoint_name,
                    "contrast": f"{left} - {right}",
                    **result,
                }
            )

    paired = pd.DataFrame(paired_rows)
    paired.to_csv(
        out_dir / "paired_species_cluster_bootstrap.csv",
        index=False,
    )

    # Shared vs Independent: 둘 다 baseline에서 species rank를 정확히 emit한 공통 subset.
    shared = frames["Shared"].copy()
    independent = frames["Independent"].copy()

    required_h = [
        "baseline_species_emitted_correct",
        "top_species_to_fallback",
        "random_species_to_fallback_rate",
        "top_minus_random_species_fallback",
        "top_emitted_correct",
    ]
    for name, df in (("Shared", shared), ("Independent", independent)):
        missing = [c for c in required_h if c not in df.columns]
        if missing:
            raise ValueError(f"{name} fallback column 누락: {missing}")

    sh = shared[
        shared["baseline_species_emitted_correct"].fillna(False).astype(bool)
    ][
        ["image_path", "species", "species_index"] + required_h[1:]
    ].copy()
    ind = independent[
        independent["baseline_species_emitted_correct"].fillna(False).astype(bool)
    ][
        ["image_path", "species", "species_index"] + required_h[1:]
    ].copy()

    sh = sh.rename(
        columns={
            c: f"shared__{c}"
            for c in required_h[1:]
        }
    )
    ind = ind.rename(
        columns={
            c: f"independent__{c}"
            for c in required_h[1:]
        }
    )

    hcommon = sh.merge(
        ind,
        on=["image_path", "species", "species_index"],
        how="inner",
        validate="one_to_one",
    )
    hcommon.to_csv(
        out_dir / "common_species_emitted_shared_independent.csv",
        index=False,
    )

    if len(hcommon):
        hcommon["paired_fallback_excess_difference"] = (
            hcommon["shared__top_minus_random_species_fallback"]
            - hcommon["independent__top_minus_random_species_fallback"]
        )
        hb = cluster_bootstrap_difference(
            hcommon,
            "paired_fallback_excess_difference",
            reps=int(args.bootstrap_reps),
            seed=int(args.seed) + 90001,
        )
        fallback_compare = pd.DataFrame(
            [
                {
                    "fraction": float(args.fraction),
                    "contrast": "Shared - Independent",
                    "endpoint": "fallback_excess",
                    "shared_top_fallback_rate": (
                        hcommon["shared__top_species_to_fallback"].mean()
                    ),
                    "independent_top_fallback_rate": (
                        hcommon["independent__top_species_to_fallback"].mean()
                    ),
                    "shared_random_fallback_rate": (
                        hcommon[
                            "shared__random_species_to_fallback_rate"
                        ].mean()
                    ),
                    "independent_random_fallback_rate": (
                        hcommon[
                            "independent__random_species_to_fallback_rate"
                        ].mean()
                    ),
                    "shared_top_fallback_correct_rate": (
                        hcommon.loc[
                            hcommon["shared__top_species_to_fallback"].eq(1),
                            "shared__top_emitted_correct",
                        ].mean()
                    ),
                    "independent_top_fallback_correct_rate": (
                        hcommon.loc[
                            hcommon["independent__top_species_to_fallback"].eq(1),
                            "independent__top_emitted_correct",
                        ].mean()
                    ),
                    **hb,
                }
            ]
        )
    else:
        fallback_compare = pd.DataFrame()

    fallback_compare.to_csv(
        out_dir / "paired_fallback_comparison.csv",
        index=False,
    )

    print("===== MODEL-OWN SAMPLE SUMMARY =====")
    print(own.to_string(index=False))
    print()
    print("===== COMMON CORRECT SAMPLE =====")
    print(
        f"N={len(common)}, species={common['species'].nunique()}"
    )
    print(paired.to_string(index=False))
    print()
    print("===== COMMON SPECIES-EMITTED: SHARED vs INDEPENDENT =====")
    print(
        f"N={len(hcommon)}, "
        f"species={hcommon['species'].nunique() if len(hcommon) else 0}"
    )
    if len(fallback_compare):
        print(fallback_compare.to_string(index=False))
    else:
        print("공통 baseline species-emitted 표본이 없습니다.")

    print()
    print(f"[저장] {out_dir}")


if __name__ == "__main__":
    main()
