"""독립 taxonomy와 공유 계층 Prompt-CAM 지표를 비교한다."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


METRICS = (
    "top1",
    "top5",
    "balanced_accuracy",
    "macro_f1",
    "genus_accuracy",
    "family_accuracy",
    "mean_taxonomic_distance",
    "trainable_parameters",
)


def _load(path: str) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved, json.loads(resolved.read_text(encoding="utf-8"))


def _metric_row(method: str, source: Path, metrics: dict[str, Any]) -> dict[str, Any]:
    row = {"method": method, "source": str(source)}
    row.update({name: metrics.get(name) for name in METRICS})
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="taxonomy 구현 비교 결과 CSV 생성")
    parser.add_argument("--original", required=True, help="독립 모델 평가 결과 JSON")
    parser.add_argument("--hierarchical", required=True, help="현재 공동 계층 모델 final_result.json")
    parser.add_argument("--output", default="taxonomy_comparison.csv")
    args = parser.parse_args()

    original_path, original = _load(args.original)
    hierarchical_path, hierarchical = _load(args.hierarchical)
    hierarchical_metrics = hierarchical.get(
        "final_test_metrics",
        hierarchical.get("final_metrics", hierarchical),
    )
    if not isinstance(hierarchical_metrics, dict):
        raise ValueError("공동 계층 결과에서 test metric dictionary를 찾지 못했습니다")
    hierarchical_metrics = dict(hierarchical_metrics)
    hierarchical_metrics["trainable_parameters"] = hierarchical.get(
        "trainable_parameters",
        hierarchical.get("trainable_parameters_without_head"),
    )

    resources = original.get("resources", {})
    original_total_parameters = resources.get(
        "total_trainable_parameters",
        resources.get("total_trainable_parameters_across_nodes"),
    )
    original_soft = dict(original["soft_path"])
    original_soft["trainable_parameters"] = original_total_parameters
    original_hard = dict(original["hard_traversal"])
    original_hard["trainable_parameters"] = original_total_parameters

    rows = [
        _metric_row(
            "current_shared_hierarchical",
            hierarchical_path,
            hierarchical_metrics,
        ),
        _metric_row(
            "original_taxonomy_soft_path",
            original_path,
            original_soft,
        ),
        _metric_row(
            "original_taxonomy_hard_traversal",
            original_path,
            original_hard,
        ),
    ]

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["method", "source", *METRICS],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"비교표 저장: {output}")


if __name__ == "__main__":
    main()
