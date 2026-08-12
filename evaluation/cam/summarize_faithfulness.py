#!/usr/bin/env python3
"""CAM deletion 및 sufficiency 결과와 bootstrap 신뢰구간을 요약한다."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * float(probability)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap_mean_interval(
    values: Iterable[float],
    *,
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, float | int]:
    data = [float(value) for value in values]
    if not data:
        raise ValueError("bootstrap interval requires at least one value")
    if samples < 1:
        raise ValueError("samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    rng = random.Random(seed)
    if len(data) == 1:
        return {
            "count": 1,
            "mean": data[0],
            # 어떤 복원 표본을 뽑아도 같은 관측값이므로 percentile interval은
            # 그 값 하나로 퇴화한다. 추론적 근거가 늘어나는 것은 아니므로
            # 아래 상태와 안내 문구는 단일 관측임을 계속 명시한다.
            "ci_low": data[0],
            "ci_high": data[0],
            "confidence": confidence,
            "bootstrap_samples": 0,
            "inferential_status": "descriptive_only_single_observation",
            "reporting_guidance": (
                "Use only as an illustrative per-image value; a bootstrap interval is not reported."
            ),
        }
    means = [
        statistics.fmean(data[rng.randrange(len(data))] for _ in data)
        for _ in range(samples)
    ]
    tail = (1.0 - confidence) / 2.0
    return {
        "count": len(data),
        "mean": statistics.fmean(data),
        "ci_low": _percentile(means, tail),
        "ci_high": _percentile(means, 1.0 - tail),
        "confidence": confidence,
        "bootstrap_samples": samples,
        "inferential_status": (
            "descriptive_small_sample" if len(data) < 30 else "aggregate_sample"
        ),
        "reporting_guidance": (
            "Treat cautiously; collect at least 30 prespecified images."
            if len(data) < 30
            else "Suitable for aggregate reporting with the stated sampling protocol."
        ),
    }


def _metadata_paths(inputs: Iterable[str]) -> list[Path]:
    result: list[Path] = []
    for value in inputs:
        path = Path(value).expanduser().resolve()
        if path.is_dir():
            result.extend(sorted(path.rglob("metadata.json")))
        elif path.is_file():
            result.append(path)
        else:
            raise FileNotFoundError(path)
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in result:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    if not unique:
        raise ValueError("No metadata.json files were found")
    return unique


def _rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        for entry in payload.get("cams", []):
            faithfulness = entry.get("faithfulness")
            if not isinstance(faithfulness, dict):
                continue
            rows.append(
                {
                    "metadata_path": str(path),
                    "image": payload.get("image"),
                    "case": payload.get("case"),
                    "model": entry.get("model"),
                    "level": entry.get("level"),
                    "target_mode": entry.get("target_mode"),
                    **{key: float(value) for key, value in faithfulness.items()},
                }
            )
    if not rows:
        raise ValueError("No CAM faithfulness records were found")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="comparison directories or metadata.json files")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    paths = _metadata_paths(args.inputs)
    rows = _rows(paths)
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["model"], row["level"], row["target_mode"])].append(row)

    summary: list[dict[str, Any]] = []
    for group_index, (key, group_rows) in enumerate(sorted(groups.items())):
        model, level, target_mode = key
        entry: dict[str, Any] = {
            "model": model,
            "level": level,
            "target_mode": target_mode,
        }
        for metric in (
            "deletion_absolute_drop",
            "deletion_relative_drop",
            "sufficiency_absolute_drop",
            "sufficiency_relative_drop",
        ):
            entry[metric] = bootstrap_mean_interval(
                [row[metric] for row in group_rows],
                samples=args.bootstrap_samples,
                confidence=args.confidence,
                seed=args.seed + group_index,
            )
        summary.append(entry)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "faithfulness_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "schema_version": 1,
                "metadata_files": [str(path) for path in paths],
                "groups": summary,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    fieldnames = list(rows[0])
    with (output_dir / "faithfulness_per_image.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"output_dir": str(output_dir), "records": len(rows), "groups": len(summary)}))


if __name__ == "__main__":
    main()
