#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared와 independent 모델의 reliability-specificity 및 CAM intervention을 비교한다.

입력:
  - Shared decision_deletion_per_image_fraction.csv
  - Independent decision_deletion_per_image_fraction.csv

두 파일은 이미:
  * 전체 test set에서
  * forced species prediction이 원래 정답이고
  * species sibling contrast가 정의되는 이미지
만 포함한다.

이 스크립트는 두 모델 모두에서 사용 가능한 동일 이미지 교집합을 만들고,
validation에서 각 모델별 94% reliability operating point로 고정된 hierarchical
stopping 결과를 apples-to-apples로 비교한다.

주요 비교:
1) baseline specificity
   - emitted species rate
   - mean emitted depth
   - emitted accuracy
2) top-CAM 10% 삭제 후
   - emitted species rate
   - mean emitted depth
   - emitted accuracy
3) random 10% 삭제 후
   - mean emitted depth
   - emitted accuracy
4) intervention selectivity
   - top minus random depth loss
5) 둘 다 baseline에서 species를 emit했던 동일 이미지 subset
   - top-CAM fallback rate
   - random fallback rate
   - fallback excess
   - top-CAM fallback accuracy
6) 모든 차이는 species-cluster paired bootstrap 95% CI

차이의 부호는 항상 Shared - Independent.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


RANK_TO_DEPTH = {"family": 1, "genus": 2, "species": 3}


def bootstrap_cluster_paired(
    df: pd.DataFrame,
    diff_col: str,
    *,
    reps: int,
    seed: int,
) -> dict:
    g = df[["species", diff_col]].dropna().copy()
    if g.empty:
        raise RuntimeError(f"{diff_col}: 유효한 관측치가 없습니다.")

    stats = (
        g.groupby("species")[diff_col]
        .agg(["count", "sum", "mean"])
        .reset_index()
        .sort_values("species")
        .reset_index(drop=True)
    )

    n_species = len(stats)
    if n_species < 2:
        raise RuntimeError(
            f"{diff_col}: species cluster 수가 {n_species}개뿐입니다."
        )

    cluster_sum = stats["sum"].to_numpy(dtype=np.float64)
    cluster_n = stats["count"].to_numpy(dtype=np.float64)
    cluster_mean = stats["mean"].to_numpy(dtype=np.float64)

    rng = np.random.default_rng(int(seed))
    pooled_draws = np.empty(int(reps), dtype=np.float64)
    equal_draws = np.empty(int(reps), dtype=np.float64)

    for b in range(int(reps)):
        idx = rng.integers(0, n_species, size=n_species)
        pooled_draws[b] = (
            cluster_sum[idx].sum() / cluster_n[idx].sum()
        )
        equal_draws[b] = cluster_mean[idx].mean()

    return {
        "n_images": int(len(g)),
        "n_species": int(n_species),
        "pooled_estimate": float(g[diff_col].mean()),
        "pooled_bootstrap_se": float(pooled_draws.std(ddof=1)),
        "pooled_ci95_lower": float(np.quantile(pooled_draws, 0.025)),
        "pooled_ci95_upper": float(np.quantile(pooled_draws, 0.975)),
        "species_equal_estimate": float(cluster_mean.mean()),
        "species_equal_bootstrap_se": float(equal_draws.std(ddof=1)),
        "species_equal_ci95_lower": float(np.quantile(equal_draws, 0.025)),
        "species_equal_ci95_upper": float(np.quantile(equal_draws, 0.975)),
        "positive_species": int((cluster_mean > 0).sum()),
        "zero_species": int((cluster_mean == 0).sum()),
        "negative_species": int((cluster_mean < 0).sum()),
    }


def load_fraction(path: Path, fraction: float, prefix: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "fraction" not in df.columns:
        raise ValueError(f"{path}: fraction 열이 없습니다.")

    df = df[np.isclose(df["fraction"], float(fraction))].copy()
    if df.empty:
        raise RuntimeError(
            f"{path}: fraction={fraction}에 해당하는 행이 없습니다."
        )

    key_cols = ["image_path", "species", "species_index"]

    needed = [
        "baseline_emitted_depth",
        "baseline_emitted_rank",
        "baseline_emitted_correct",
        "baseline_species_emitted_correct",
        "top_emitted_depth",
        "top_emitted_rank",
        "top_emitted_correct",
        "random_emitted_depth_mean",
        "random_emitted_correct_rate",
        "top_depth_loss",
        "random_depth_loss_mean",
        "top_minus_random_depth_loss",
        "top_species_to_fallback",
        "random_species_to_fallback_rate",
        "top_minus_random_species_fallback",
        "top_species_flip",
        "random_species_flip_rate",
        "top_minus_random_flip",
        "top_margin_drop",
        "random_margin_drop_mean",
        "top_minus_random_margin_drop",
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path}: 필요한 열이 없습니다: {missing}"
        )

    keep = key_cols + needed
    df = df[keep].copy()

    rename = {
        c: f"{prefix}__{c}"
        for c in needed
    }
    return df.rename(columns=rename)


def model_summary(common: pd.DataFrame, prefix: str, name: str) -> dict:
    bdepth = common[f"{prefix}__baseline_emitted_depth"].astype(float)
    tdepth = common[f"{prefix}__top_emitted_depth"].astype(float)

    return {
        "model": name,
        "n_images": int(len(common)),
        "n_species": int(common["species"].nunique()),
        "baseline_species_emit_rate": float((bdepth == 3).mean()),
        "baseline_genus_emit_rate": float((bdepth == 2).mean()),
        "baseline_family_emit_rate": float((bdepth == 1).mean()),
        "baseline_mean_depth": float(bdepth.mean()),
        "baseline_emitted_accuracy": float(
            common[f"{prefix}__baseline_emitted_correct"].astype(float).mean()
        ),
        "top_species_emit_rate": float((tdepth == 3).mean()),
        "top_genus_emit_rate": float((tdepth == 2).mean()),
        "top_family_emit_rate": float((tdepth == 1).mean()),
        "top_mean_depth": float(tdepth.mean()),
        "top_emitted_accuracy": float(
            common[f"{prefix}__top_emitted_correct"].astype(float).mean()
        ),
        "random_mean_depth": float(
            common[f"{prefix}__random_emitted_depth_mean"].astype(float).mean()
        ),
        "random_emitted_accuracy": float(
            common[f"{prefix}__random_emitted_correct_rate"].astype(float).mean()
        ),
        "top_minus_random_depth_loss": float(
            common[f"{prefix}__top_minus_random_depth_loss"].astype(float).mean()
        ),
        "top_minus_random_flip": float(
            common[f"{prefix}__top_minus_random_flip"].astype(float).mean()
        ),
        "top_minus_random_margin_drop": float(
            common[f"{prefix}__top_minus_random_margin_drop"].astype(float).mean()
        ),
    }


def add_common_differences(common: pd.DataFrame) -> pd.DataFrame:
    x = common.copy()

    for prefix in ("shared", "independent"):
        x[f"{prefix}__baseline_species_emit"] = (
            x[f"{prefix}__baseline_emitted_depth"].astype(int).eq(3).astype(float)
        )
        x[f"{prefix}__top_species_emit"] = (
            x[f"{prefix}__top_emitted_depth"].astype(int).eq(3).astype(float)
        )

    endpoint_pairs = {
        "baseline_species_emit_diff": (
            "shared__baseline_species_emit",
            "independent__baseline_species_emit",
        ),
        "baseline_depth_diff": (
            "shared__baseline_emitted_depth",
            "independent__baseline_emitted_depth",
        ),
        "baseline_emitted_accuracy_diff": (
            "shared__baseline_emitted_correct",
            "independent__baseline_emitted_correct",
        ),
        "top_species_emit_diff": (
            "shared__top_species_emit",
            "independent__top_species_emit",
        ),
        "top_depth_diff": (
            "shared__top_emitted_depth",
            "independent__top_emitted_depth",
        ),
        "top_emitted_accuracy_diff": (
            "shared__top_emitted_correct",
            "independent__top_emitted_correct",
        ),
        "random_depth_diff": (
            "shared__random_emitted_depth_mean",
            "independent__random_emitted_depth_mean",
        ),
        "random_emitted_accuracy_diff": (
            "shared__random_emitted_correct_rate",
            "independent__random_emitted_correct_rate",
        ),
        "top_minus_random_depth_loss_diff": (
            "shared__top_minus_random_depth_loss",
            "independent__top_minus_random_depth_loss",
        ),
        "top_minus_random_flip_diff": (
            "shared__top_minus_random_flip",
            "independent__top_minus_random_flip",
        ),
        "top_minus_random_margin_drop_diff": (
            "shared__top_minus_random_margin_drop",
            "independent__top_minus_random_margin_drop",
        ),
    }

    for out, (left, right) in endpoint_pairs.items():
        x[out] = x[left].astype(float) - x[right].astype(float)

    return x


def rank_crosstab(
    df: pd.DataFrame,
    left_col: str,
    right_col: str,
    *,
    left_name: str,
    right_name: str,
) -> pd.DataFrame:
    tab = pd.crosstab(
        df[left_col].astype(str),
        df[right_col].astype(str),
        rownames=[left_name],
        colnames=[right_name],
        dropna=False,
    )
    # rank 순서 고정
    order = ["family", "genus", "species"]
    tab = tab.reindex(index=order, columns=order, fill_value=0)
    return tab


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-csv", required=True)
    parser.add_argument("--independent-csv", required=True)
    parser.add_argument("--fraction", type=float, default=0.10)
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    shared = load_fraction(
        Path(args.shared_csv).expanduser().resolve(),
        args.fraction,
        "shared",
    )
    independent = load_fraction(
        Path(args.independent_csv).expanduser().resolve(),
        args.fraction,
        "independent",
    )

    keys = ["image_path", "species", "species_index"]
    common = shared.merge(
        independent,
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    if common.empty:
        raise RuntimeError("Shared/Independent 공통 이미지가 없습니다.")

    common = add_common_differences(common)

    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    common.to_csv(out / "common_correct_images.csv", index=False)

    summaries = pd.DataFrame(
        [
            model_summary(common, "shared", "Shared"),
            model_summary(common, "independent", "Independent"),
        ]
    )
    summaries.to_csv(
        out / "common_model_summary.csv",
        index=False,
    )

    # 공통 전체 표본에서 paired cluster bootstrap
    endpoints = [
        "baseline_species_emit_diff",
        "baseline_depth_diff",
        "baseline_emitted_accuracy_diff",
        "top_species_emit_diff",
        "top_depth_diff",
        "top_emitted_accuracy_diff",
        "random_depth_diff",
        "random_emitted_accuracy_diff",
        "top_minus_random_depth_loss_diff",
        "top_minus_random_flip_diff",
        "top_minus_random_margin_drop_diff",
    ]

    bootstrap_rows = []
    for j, endpoint in enumerate(endpoints):
        result = bootstrap_cluster_paired(
            common,
            endpoint,
            reps=int(args.bootstrap_reps),
            seed=int(args.seed) + 1009 * j,
        )
        bootstrap_rows.append(
            {
                "fraction": float(args.fraction),
                "contrast": "Shared - Independent",
                "endpoint": endpoint,
                **result,
            }
        )

    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap.to_csv(
        out / "common_paired_species_cluster_bootstrap.csv",
        index=False,
    )

    # baseline rank cross-tab
    baseline_tab = rank_crosstab(
        common,
        "shared__baseline_emitted_rank",
        "independent__baseline_emitted_rank",
        left_name="Shared baseline rank",
        right_name="Independent baseline rank",
    )
    baseline_tab.to_csv(
        out / "baseline_rank_crosstab.csv"
    )

    # CAM deletion 후 rank cross-tab
    top_tab = rank_crosstab(
        common,
        "shared__top_emitted_rank",
        "independent__top_emitted_rank",
        left_name="Shared top-CAM rank",
        right_name="Independent top-CAM rank",
    )
    top_tab.to_csv(
        out / "top_cam_rank_crosstab.csv"
    )

    # 둘 다 baseline species-emitted + correct였던 동일 이미지 subset
    both_species = common[
        common["shared__baseline_species_emitted_correct"].fillna(False).astype(bool)
        & common["independent__baseline_species_emitted_correct"].fillna(False).astype(bool)
    ].copy()

    both_species.to_csv(
        out / "common_both_species_emitted.csv",
        index=False,
    )

    if not both_species.empty:
        for prefix in ("shared", "independent"):
            both_species[f"{prefix}__top_fallback"] = (
                both_species[f"{prefix}__top_species_to_fallback"].astype(float)
            )
            both_species[f"{prefix}__random_fallback"] = (
                both_species[f"{prefix}__random_species_to_fallback_rate"].astype(float)
            )
            both_species[f"{prefix}__fallback_excess"] = (
                both_species[f"{prefix}__top_minus_random_species_fallback"].astype(float)
            )

        both_species["top_fallback_rate_diff"] = (
            both_species["shared__top_fallback"]
            - both_species["independent__top_fallback"]
        )
        both_species["random_fallback_rate_diff"] = (
            both_species["shared__random_fallback"]
            - both_species["independent__random_fallback"]
        )
        both_species["fallback_excess_diff"] = (
            both_species["shared__fallback_excess"]
            - both_species["independent__fallback_excess"]
        )
        both_species["top_fallback_correct_diff"] = (
            both_species["shared__top_emitted_correct"].astype(float)
            - both_species["independent__top_emitted_correct"].astype(float)
        )

        both_summary = pd.DataFrame(
            [
                {
                    "model": "Shared",
                    "n_images": len(both_species),
                    "n_species": both_species["species"].nunique(),
                    "top_fallback_rate": both_species[
                        "shared__top_fallback"
                    ].mean(),
                    "random_fallback_rate": both_species[
                        "shared__random_fallback"
                    ].mean(),
                    "fallback_excess": both_species[
                        "shared__fallback_excess"
                    ].mean(),
                    "top_emitted_accuracy": both_species[
                        "shared__top_emitted_correct"
                    ].astype(float).mean(),
                    "top_fallback_correct_rate": both_species.loc[
                        both_species["shared__top_fallback"].eq(1),
                        "shared__top_emitted_correct",
                    ].astype(float).mean()
                    if both_species["shared__top_fallback"].eq(1).any()
                    else np.nan,
                },
                {
                    "model": "Independent",
                    "n_images": len(both_species),
                    "n_species": both_species["species"].nunique(),
                    "top_fallback_rate": both_species[
                        "independent__top_fallback"
                    ].mean(),
                    "random_fallback_rate": both_species[
                        "independent__random_fallback"
                    ].mean(),
                    "fallback_excess": both_species[
                        "independent__fallback_excess"
                    ].mean(),
                    "top_emitted_accuracy": both_species[
                        "independent__top_emitted_correct"
                    ].astype(float).mean(),
                    "top_fallback_correct_rate": both_species.loc[
                        both_species["independent__top_fallback"].eq(1),
                        "independent__top_emitted_correct",
                    ].astype(float).mean()
                    if both_species["independent__top_fallback"].eq(1).any()
                    else np.nan,
                },
            ]
        )
        both_summary.to_csv(
            out / "common_both_species_emitted_summary.csv",
            index=False,
        )

        fb_rows = []
        for j, endpoint in enumerate(
            [
                "top_fallback_rate_diff",
                "random_fallback_rate_diff",
                "fallback_excess_diff",
                "top_fallback_correct_diff",
            ]
        ):
            result = bootstrap_cluster_paired(
                both_species,
                endpoint,
                reps=int(args.bootstrap_reps),
                seed=int(args.seed) + 50000 + 1009 * j,
            )
            fb_rows.append(
                {
                    "fraction": float(args.fraction),
                    "contrast": "Shared - Independent",
                    "endpoint": endpoint,
                    **result,
                }
            )

        fb_boot = pd.DataFrame(fb_rows)
        fb_boot.to_csv(
            out / "common_both_species_emitted_bootstrap.csv",
            index=False,
        )
    else:
        both_summary = pd.DataFrame()
        fb_boot = pd.DataFrame()

    print("===== COMMON CORRECT IMAGES =====")
    print(
        f"N={len(common)}, species={common['species'].nunique()}, "
        f"fraction={args.fraction}"
    )
    print()
    print("===== MODEL SUMMARY =====")
    print(summaries.to_string(index=False))
    print()

    print("===== BASELINE RANK CROSS-TAB =====")
    print(baseline_tab.to_string())
    print()

    print("===== TOP-CAM DELETION RANK CROSS-TAB =====")
    print(top_tab.to_string())
    print()

    print("===== PAIRED SPECIES-CLUSTER BOOTSTRAP =====")
    show = bootstrap[
        [
            "endpoint",
            "n_images",
            "n_species",
            "pooled_estimate",
            "pooled_ci95_lower",
            "pooled_ci95_upper",
            "species_equal_estimate",
            "species_equal_ci95_lower",
            "species_equal_ci95_upper",
        ]
    ]
    print(show.to_string(index=False))
    print()

    print("===== BOTH BASELINE SPECIES-EMITTED =====")
    print(
        f"N={len(both_species)}, "
        f"species={both_species['species'].nunique() if len(both_species) else 0}"
    )
    if not both_summary.empty:
        print(both_summary.to_string(index=False))
        print()
        print("===== FALLBACK PAIRED BOOTSTRAP =====")
        print(
            fb_boot[
                [
                    "endpoint",
                    "n_images",
                    "n_species",
                    "pooled_estimate",
                    "pooled_ci95_lower",
                    "pooled_ci95_upper",
                    "species_equal_estimate",
                    "species_equal_ci95_lower",
                    "species_equal_ci95_upper",
                ]
            ].to_string(index=False)
        )

    print()
    print(f"[저장] {out}")


if __name__ == "__main__":
    main()
