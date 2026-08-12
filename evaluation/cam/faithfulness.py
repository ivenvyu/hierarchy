#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared hierarchical Prompt-CAM의 patch-level causal faithfulness를 측정한다.

Backbone을 한 번만 통과해 마지막 family/genus/species prompt와 576개 patch token을
고정한다. 그 뒤 decoder 입력의 patch token을 하나씩 제거하여 target confidence 변화량을
계산한다. 따라서 pixel masking처럼 backbone 전체 표현을 다시 바꾸지 않고, 실제
prompt-to-patch 분류 경로에 직접 개입한다.

각 patch p에 대해

    D_p = C(X) - C(X with patch p replaced)

를 계산하고, Prompt-CAM A_p와 D_p의 Spearman 상관 및 top-CAM deletion 효과를 저장한다.

기본 replacement='zero'. sensitivity check에는 --replacement mean 을 추가로 사용한다.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from PIL import Image

try:
    from scipy.stats import spearmanr
except Exception:
    spearmanr = None


def resolve(project_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def core_model(model):
    return model.module if hasattr(model, "module") else model


def load_inputs(input_csv: Path, *, max_images: int | None) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    required = {"image_path", "species_index", "species"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"input CSV에 필요한 열이 없습니다: {sorted(missing)}")

    df = (
        df[["image_path", "species_index", "species"]]
        .drop_duplicates(subset=["image_path", "species_index"])
        .sort_values(["species_index", "image_path"])
        .reset_index(drop=True)
    )
    if max_images is not None:
        df = df.iloc[: int(max_images)].copy()
    if df.empty:
        raise ValueError("분석할 이미지가 없습니다")
    return df


def load_shared(project_root: Path, run_dir: Path, device: torch.device):
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from evaluation import hierarchy as hierarchy_eval
    from data.dataset.imagefolder import JointImageTransform

    args_path = run_dir / "args.yaml"
    if not args_path.is_file():
        raise FileNotFoundError(args_path)
    args_data = hierarchy_eval._load_yaml(args_path)

    cli = SimpleNamespace(batch_size=1, num_workers=0, device=str(device))
    params = hierarchy_eval._prepare_params(args_data, project_root, run_dir, cli)
    model, _, checkpoint_path, _ = hierarchy_eval._load_model(
        project_root,
        run_dir,
        params,
        device,
    )
    taxonomy = hierarchy_eval._load_taxonomy(run_dir, params)
    transform = JointImageTransform(params, training=False)

    return model.eval(), transform, taxonomy, checkpoint_path


def load_image(path: Path, transform, device: torch.device) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    tensor, _, _ = transform(
        image,
        bbox=None,
        bbox_coordinate_mode="normalized",
    )
    return tensor.unsqueeze(0).to(device)


def split_decoder_inputs(model, image: torch.Tensor):
    """forward_features 결과를 hierarchical decoder의 실제 입력으로 분해한다."""
    core = core_model(model)
    with torch.inference_mode():
        features, _ = core.forward_features(image)

    family_count = int(core.params.num_families)
    genus_count = int(core.params.num_genera)
    species_count = int(core.params.class_num)
    family_end = family_count
    genus_end = family_end + genus_count
    prompt_count = genus_end + species_count

    family_tokens = features[:, :family_end]
    genus_tokens = features[:, family_end:genus_end]
    species_tokens = features[:, genus_end:prompt_count]
    backbone_tokens = features[:, prompt_count:]
    patch_tokens = backbone_tokens[:, core.num_prefix_tokens:]
    if core.num_prefix_tokens > 0:
        global_feature = backbone_tokens[:, 0]
    else:
        global_feature = patch_tokens.mean(dim=1)

    if patch_tokens.ndim != 3 or patch_tokens.shape[0] != 1:
        raise RuntimeError(f"예상하지 못한 patch token shape: {tuple(patch_tokens.shape)}")

    return (
        core.hierarchical_head,
        family_tokens,
        genus_tokens,
        species_tokens,
        patch_tokens,
        global_feature,
    )


def decoder_forward(decoder, family, genus, species, patches, global_feature):
    return decoder(
        family,
        genus,
        species,
        patches,
        global_feature,
        patch_prior=None,
    )


def rank_target(decoder, species_index: int, rank: str) -> int:
    rank = rank.lower()
    species_index = int(species_index)
    if rank == "species":
        return species_index
    genus_index = int(decoder.species_to_genus[species_index].item())
    if rank == "genus":
        return genus_index
    if rank == "family":
        return int(decoder.genus_to_family[genus_index].item())
    raise ValueError(rank)


def rank_probability(output: Mapping[str, torch.Tensor], rank: str, target: int) -> torch.Tensor:
    return output[f"{rank}_probabilities"][:, int(target)]


def rank_log_probability(output: Mapping[str, torch.Tensor], rank: str, target: int) -> torch.Tensor:
    key = f"{rank}_log_probabilities"
    if key in output:
        return output[key][:, int(target)]
    return rank_probability(output, rank, target).clamp_min(1e-12).log()


def rank_cam(output: Mapping[str, torch.Tensor], rank: str, target: int) -> torch.Tensor:
    return output[f"{rank}_cam"][0, int(target)].detach().float().cpu()


def contrast_defined(output: Mapping[str, torch.Tensor], rank: str, target: int) -> bool:
    if rank == "family":
        return True
    key = f"{rank}_contrast_defined"
    value = output.get(key)
    if value is None:
        return True
    return bool(value[int(target)].item())


def replacement_vector(patches: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "zero":
        return torch.zeros_like(patches[:, :1, :])
    if mode == "mean":
        return patches.mean(dim=1, keepdim=True)
    raise ValueError(mode)


def single_patch_causal_map(
    decoder,
    family,
    genus,
    species,
    patches,
    global_feature,
    *,
    rank: str,
    target: int,
    baseline_prob: float,
    baseline_log_prob: float,
    replacement: str,
    ablation_batch: int,
):
    patch_count = int(patches.shape[1])
    repl = replacement_vector(patches, replacement)
    probs = torch.empty(patch_count, dtype=torch.float32)
    log_probs = torch.empty(patch_count, dtype=torch.float32)

    with torch.inference_mode():
        for start in range(0, patch_count, int(ablation_batch)):
            indices = torch.arange(
                start,
                min(start + int(ablation_batch), patch_count),
                device=patches.device,
            )
            batch = int(indices.numel())
            p = patches.expand(batch, -1, -1).clone()
            row = torch.arange(batch, device=patches.device)
            p[row, indices] = repl.expand(batch, -1, -1)[:, 0]

            out = decoder_forward(
                decoder,
                family.expand(batch, -1, -1),
                genus.expand(batch, -1, -1),
                species.expand(batch, -1, -1),
                p,
                global_feature.expand(batch, -1),
            )
            probs[start : start + batch] = rank_probability(out, rank, target).detach().float().cpu()
            log_probs[start : start + batch] = rank_log_probability(out, rank, target).detach().float().cpu()

    drop = float(baseline_prob) - probs
    log_drop = float(baseline_log_prob) - log_probs
    return drop, log_drop


def delete_patch_set(
    decoder,
    family,
    genus,
    species,
    patches,
    global_feature,
    *,
    rank: str,
    target: int,
    indices: torch.Tensor,
    replacement: str,
):
    p = patches.clone()
    repl = replacement_vector(patches, replacement)[0, 0]
    p[0, indices.to(p.device)] = repl
    with torch.inference_mode():
        out = decoder_forward(decoder, family, genus, species, p, global_feature)
    return (
        float(rank_probability(out, rank, target)[0].item()),
        float(rank_log_probability(out, rank, target)[0].item()),
    )


def corr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return math.nan, math.nan
    if spearmanr is not None:
        result = spearmanr(x, y)
        return float(result.statistic), float(result.pvalue)
    return float(pd.Series(x).rank().corr(pd.Series(y).rank())), math.nan


def save_maps(output_dir: Path, stem: str, cam: np.ndarray, causal: np.ndarray, log_causal: np.ndarray):
    """24x24 map을 간단한 PNG로 저장한다. matplotlib가 없으면 npy만 저장한다."""
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f"{stem}_cam.npy", cam)
    np.save(output_dir / f"{stem}_causal_drop.npy", causal)
    np.save(output_dir / f"{stem}_causal_log_drop.npy", log_causal)
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    patch_count = int(cam.size)
    grid = int(round(patch_count ** 0.5))
    if grid * grid != patch_count:
        return

    for suffix, values, title in (
        ("cam", cam, "Prompt-CAM"),
        ("causal_drop", causal, "Causal confidence drop"),
        ("causal_log_drop", log_causal, "Causal log-probability drop"),
    ):
        fig = plt.figure(figsize=(5, 5))
        plt.imshow(values.reshape(grid, grid), interpolation="nearest")
        plt.title(title)
        plt.colorbar()
        plt.tight_layout()
        fig.savefig(output_dir / f"{stem}_{suffix}.png", dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Shared Prompt-CAM patch-token causal faithfulness")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--shared-run-dir", required=True)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ranks", nargs="+", choices=["family", "genus", "species"], default=["species"])
    parser.add_argument("--replacement", choices=["zero", "mean"], default="zero")
    parser.add_argument("--ablation-batch", type=int, default=48)
    parser.add_argument("--top-fraction", type=float, default=0.10)
    parser.add_argument("--random-repeats", type=int, default=50)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    run_dir = resolve(project_root, args.shared_run_dir)
    input_csv = resolve(project_root, args.input_csv)
    output_dir = resolve(project_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA를 요청했지만 사용할 수 없습니다")

    df = load_inputs(input_csv, max_images=args.max_images)
    model, transform, taxonomy, checkpoint_path = load_shared(project_root, run_dir, device)
    print(f"[checkpoint] {checkpoint_path}")
    print(f"[images] {len(df)}")
    print(f"[ranks] {args.ranks}")

    rng = np.random.default_rng(int(args.seed))
    summary_rows: list[dict[str, Any]] = []

    for image_number, row in df.iterrows():
        image_path = Path(str(row["image_path"])).expanduser().resolve()
        species_index = int(row["species_index"])
        species_name = str(row["species"])
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
            baseline = decoder_forward(decoder, family, genus, species, patches, global_feature)
            # full model output과 cached decoder output이 같은지 확인
            full_output, _ = model(image, patch_prior=None)

        max_diff = float(
            (baseline["species_probabilities"] - full_output["species_probabilities"])
            .abs()
            .max()
            .item()
        )
        if max_diff > 1e-5:
            raise RuntimeError(
                f"cached decoder가 full model과 일치하지 않습니다: max_diff={max_diff}"
            )

        for rank in args.ranks:
            target = rank_target(decoder, species_index, rank)
            defined = contrast_defined(baseline, rank, target)
            if not defined:
                print(f"[SKIP] {species_name} / {rank}: sibling contrast 없음")
                continue

            base_prob = float(rank_probability(baseline, rank, target)[0].item())
            base_log = float(rank_log_probability(baseline, rank, target)[0].item())
            cam = rank_cam(baseline, rank, target).numpy().astype(np.float64)

            causal, causal_log = single_patch_causal_map(
                decoder,
                family,
                genus,
                species,
                patches,
                global_feature,
                rank=rank,
                target=target,
                baseline_prob=base_prob,
                baseline_log_prob=base_log,
                replacement=args.replacement,
                ablation_batch=args.ablation_batch,
            )
            causal_np = causal.numpy().astype(np.float64)
            causal_log_np = causal_log.numpy().astype(np.float64)

            rho_signed, p_signed = corr(cam, causal_np)
            rho_positive, p_positive = corr(cam, np.clip(causal_np, 0.0, None))
            rho_abs, p_abs = corr(cam, np.abs(causal_np))
            rho_log, p_log = corr(cam, causal_log_np)

            patch_count = int(cam.size)
            k = max(1, int(round(float(args.top_fraction) * patch_count)))
            order = np.argsort(cam)
            top_indices = torch.tensor(order[-k:].copy(), dtype=torch.long)
            bottom_indices = torch.tensor(order[:k].copy(), dtype=torch.long)

            top_prob, top_log = delete_patch_set(
                decoder, family, genus, species, patches, global_feature,
                rank=rank, target=target, indices=top_indices, replacement=args.replacement,
            )
            bottom_prob, bottom_log = delete_patch_set(
                decoder, family, genus, species, patches, global_feature,
                rank=rank, target=target, indices=bottom_indices, replacement=args.replacement,
            )

            random_drops = []
            random_log_drops = []
            for _ in range(int(args.random_repeats)):
                idx = torch.tensor(rng.choice(patch_count, size=k, replace=False), dtype=torch.long)
                prob, log_prob = delete_patch_set(
                    decoder, family, genus, species, patches, global_feature,
                    rank=rank, target=target, indices=idx, replacement=args.replacement,
                )
                random_drops.append(base_prob - prob)
                random_log_drops.append(base_log - log_prob)

            top_drop = base_prob - top_prob
            bottom_drop = base_prob - bottom_prob
            top_log_drop = base_log - top_log
            bottom_log_drop = base_log - bottom_log
            random_mean = float(np.mean(random_drops))
            random_std = float(np.std(random_drops, ddof=1)) if len(random_drops) > 1 else math.nan
            random_log_mean = float(np.mean(random_log_drops))

            stem = f"{int(species_index):02d}_{rank}_{image_number:02d}"
            map_dir = output_dir / "maps" / species_name
            save_maps(map_dir, stem, cam, causal_np, causal_log_np)

            grid = int(round(patch_count ** 0.5))
            patch_rows = []
            for patch_index in range(patch_count):
                patch_rows.append(
                    {
                        "image_path": str(image_path),
                        "species": species_name,
                        "species_index": species_index,
                        "rank": rank,
                        "target_index": target,
                        "patch_index": patch_index,
                        "patch_row": patch_index // grid if grid * grid == patch_count else math.nan,
                        "patch_col": patch_index % grid if grid * grid == patch_count else math.nan,
                        "cam": cam[patch_index],
                        "causal_drop": causal_np[patch_index],
                        "causal_positive_drop": max(causal_np[patch_index], 0.0),
                        "causal_abs_drop": abs(causal_np[patch_index]),
                        "causal_log_drop": causal_log_np[patch_index],
                    }
                )
            pd.DataFrame(patch_rows).to_csv(
                output_dir / f"patches_{stem}.csv", index=False
            )

            summary_rows.append(
                {
                    "image_path": str(image_path),
                    "species": species_name,
                    "species_index": species_index,
                    "rank": rank,
                    "target_index": target,
                    "contrast_defined": defined,
                    "patch_count": patch_count,
                    "baseline_confidence": base_prob,
                    "baseline_log_probability": base_log,
                    "replacement": args.replacement,
                    "rho_cam_signed_drop": rho_signed,
                    "p_cam_signed_drop": p_signed,
                    "rho_cam_positive_drop": rho_positive,
                    "p_cam_positive_drop": p_positive,
                    "rho_cam_abs_drop": rho_abs,
                    "p_cam_abs_drop": p_abs,
                    "rho_cam_log_drop": rho_log,
                    "p_cam_log_drop": p_log,
                    "top_fraction": float(args.top_fraction),
                    "top_patch_count": k,
                    "top_cam_set_drop": top_drop,
                    "bottom_cam_set_drop": bottom_drop,
                    "random_set_drop_mean": random_mean,
                    "random_set_drop_std": random_std,
                    "top_minus_random_drop": top_drop - random_mean,
                    "top_minus_bottom_drop": top_drop - bottom_drop,
                    "top_cam_set_log_drop": top_log_drop,
                    "bottom_cam_set_log_drop": bottom_log_drop,
                    "random_set_log_drop_mean": random_log_mean,
                    "top_minus_random_log_drop": top_log_drop - random_log_mean,
                    "top_minus_bottom_log_drop": top_log_drop - bottom_log_drop,
                    "cached_full_max_probability_diff": max_diff,
                }
            )
            print(
                f"[{image_number+1}/{len(df)}] {species_name} / {rank} | "
                f"C={base_prob:.6f} rho={rho_signed:+.3f} "
                f"top10_drop={top_drop:+.6f} random={random_mean:+.6f} "
                f"top-random={top_drop-random_mean:+.6f}"
            )

    summary = pd.DataFrame(summary_rows)
    summary_path = output_dir / "faithfulness_per_image.csv"
    summary.to_csv(summary_path, index=False)

    if summary.empty:
        raise RuntimeError("유효한 rank/image 조합이 없습니다")

    group = (
        summary.groupby("rank", dropna=False)
        .agg(
            n=("rho_cam_signed_drop", "size"),
            mean_rho_signed=("rho_cam_signed_drop", "mean"),
            median_rho_signed=("rho_cam_signed_drop", "median"),
            mean_rho_positive=("rho_cam_positive_drop", "mean"),
            mean_rho_abs=("rho_cam_abs_drop", "mean"),
            mean_rho_log=("rho_cam_log_drop", "mean"),
            mean_top_cam_set_drop=("top_cam_set_drop", "mean"),
            mean_random_set_drop=("random_set_drop_mean", "mean"),
            mean_top_minus_random_drop=("top_minus_random_drop", "mean"),
            mean_top_minus_bottom_drop=("top_minus_bottom_drop", "mean"),
            mean_top_minus_random_log_drop=("top_minus_random_log_drop", "mean"),
        )
        .reset_index()
    )
    group_path = output_dir / "faithfulness_summary.csv"
    group.to_csv(group_path, index=False)

    report = output_dir / "faithfulness_report.md"
    report.write_text(
        "# Shared Prompt-CAM patch-token causal faithfulness\n\n"
        + f"- checkpoint: `{checkpoint_path}`\n"
        + f"- replacement: `{args.replacement}`\n"
        + f"- top_fraction: `{args.top_fraction}`\n"
        + "- causal intervention: cached backbone의 decoder 입력 patch token을 직접 제거\n\n"
        + "## Summary\n\n"
        + group.to_markdown(index=False)
        + "\n\n"
        + "## Interpretation\n\n"
        + "- `rho_cam_signed_drop > 0`: 높은 CAM patch일수록 제거 시 target confidence가 더 감소.\n"
        + "- `top_minus_random_drop > 0`: CAM top patch 집합 제거가 같은 크기의 random patch 제거보다 더 큰 confidence 감소를 유발.\n"
        + "- `top_minus_bottom_drop > 0`: CAM top patch 집합이 bottom patch보다 더 decision-critical.\n",
        encoding="utf-8",
    )

    print("\n===== SUMMARY =====")
    print(group.to_string(index=False))
    print(f"\n[저장] {summary_path}")
    print(f"[저장] {group_path}")
    print(f"[저장] {report}")


if __name__ == "__main__":
    main()
