#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared 모델의 정분류 표본을 종별로 균형 선택한다."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def resolve(root: Path, value: str | Path) -> Path:
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = root / p
    return p.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Shared hierarchical Prompt-CAM에서 species별 correct test image를 균형 추출한다."
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--shared-run-dir", required=True)
    parser.add_argument("--test-root", default="data/dataset/imagefolder/test")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--per-species", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # 현재 서버에 이미 사용 중인 faithfulness 스크립트의 검증된 로더를 재사용한다.
    from evaluation.cam.faithfulness import load_shared

    run_dir = resolve(project_root, args.shared_run_dir)
    test_root = resolve(project_root, args.test_root)
    output_csv = resolve(project_root, args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA를 요청했지만 torch.cuda.is_available() == False 입니다.")

    model, transform, taxonomy, checkpoint_path = load_shared(
        project_root,
        run_dir,
        device,
    )
    print(f"[checkpoint] {checkpoint_path}")

    class_dirs = sorted(p for p in test_root.iterdir() if p.is_dir())
    if not class_dirs:
        raise RuntimeError(f"test class directory가 없습니다: {test_root}")

    core = model.module if hasattr(model, "module") else model
    expected_classes = int(core.params.class_num)
    if len(class_dirs) != expected_classes:
        raise RuntimeError(
            f"test class 수({len(class_dirs)}) != model class 수({expected_classes})"
        )

    rng = np.random.default_rng(int(args.seed))
    rows: list[dict] = []

    for species_index, class_dir in enumerate(class_dirs):
        candidates = sorted(
            p for p in class_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
        if not candidates:
            raise RuntimeError(f"이미지가 없습니다: {class_dir}")

        order = rng.permutation(len(candidates))
        candidates = [candidates[int(i)] for i in order]

        selected: list[dict] = []
        cursor = 0

        while cursor < len(candidates) and len(selected) < int(args.per_species):
            chunk_paths = candidates[cursor : cursor + int(args.batch_size)]
            cursor += len(chunk_paths)

            tensors = []
            valid_paths = []
            for image_path in chunk_paths:
                try:
                    image = Image.open(image_path).convert("RGB")
                    tensor, _, _ = transform(
                        image,
                        bbox=None,
                        bbox_coordinate_mode="normalized",
                    )
                except Exception as exc:
                    print(f"[WARN] image load/transform 실패: {image_path} | {exc}")
                    continue
                tensors.append(tensor)
                valid_paths.append(image_path)

            if not tensors:
                continue

            batch = torch.stack(tensors, dim=0).to(device)

            with torch.inference_mode():
                output, _ = model(batch, patch_prior=None)
                probs = output["species_probabilities"].detach().float().cpu()

            preds = probs.argmax(dim=1)

            for image_path, pred, prob_row in zip(valid_paths, preds.tolist(), probs):
                if int(pred) != int(species_index):
                    continue

                selected.append(
                    {
                        "image_path": str(image_path.resolve()),
                        "species_index": int(species_index),
                        "species": class_dir.name,
                        "shared_predicted_index": int(pred),
                        "shared_species_confidence": float(prob_row[species_index].item()),
                    }
                )

                if len(selected) >= int(args.per_species):
                    break

        if len(selected) < int(args.per_species):
            raise RuntimeError(
                f"{class_dir.name}: correct image를 {args.per_species}개 찾지 못했습니다. "
                f"찾은 수={len(selected)}, 후보={len(candidates)}"
            )

        rows.extend(selected)
        confs = [r["shared_species_confidence"] for r in selected]
        print(
            f"[{species_index + 1:02d}/{len(class_dirs)}] "
            f"{class_dir.name}: selected={len(selected)} "
            f"mean_conf={np.mean(confs):.6f}"
        )

    df = pd.DataFrame(rows)
    expected_rows = len(class_dirs) * int(args.per_species)
    if len(df) != expected_rows:
        raise RuntimeError(f"예상 {expected_rows}행이지만 실제 {len(df)}행입니다.")

    counts = df.groupby(["species_index", "species"]).size()
    if not (counts == int(args.per_species)).all():
        raise RuntimeError(f"species 균형 추출 실패:\n{counts}")

    df.to_csv(output_csv, index=False)

    print()
    print("===== BALANCED CORRECT SAMPLE =====")
    print(f"classes        : {len(class_dirs)}")
    print(f"per species    : {args.per_species}")
    print(f"total images   : {len(df)}")
    print(f"mean confidence: {df['shared_species_confidence'].mean():.6f}")
    print(f"[저장] {output_csv}")


if __name__ == "__main__":
    main()
