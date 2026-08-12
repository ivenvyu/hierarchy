#!/usr/bin/env python3
"""완료된 shared, flat, independent 실행 디렉터리를 탐색한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def _yaml(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = yaml.safe_load(handle) or {}
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _latest(candidates: list[Path]) -> Path | None:
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path / "model.pt").stat().st_mtime)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    root = Path(args.project_root).expanduser().resolve()
    output = root / "output"
    if not output.is_dir():
        raise FileNotFoundError(f"output directory가 없습니다: {output}")

    shared: list[Path] = []
    flat: list[Path] = []
    for args_path in output.rglob("args.yaml"):
        run_dir = args_path.parent
        if not (run_dir / "model.pt").is_file():
            continue
        config = _yaml(args_path)
        if bool(config.get("original_taxonomy_prompt", False)):
            continue
        if bool(config.get("hierarchical_prompt", False)):
            shared.append(run_dir)
        elif (
            bool(config.get("semantic_snapmix", False))
            and int(config.get("vpt_num", -1)) == int(config.get("class_num", -2))
            and int(config.get("class_num", -1)) > 1
        ):
            flat.append(run_dir)

    independent: list[Path] = []
    for summary in output.rglob("training_summary.json"):
        run_dir = summary.parent
        if (run_dir / "configs" / "root.yaml").is_file():
            # 마지막 node까지 model.pt가 있는 run만 완료 후보로 둔다.
            node_models = list((run_dir / "nodes").rglob("model.pt")) if (run_dir / "nodes").is_dir() else []
            if node_models:
                independent.append(run_dir)

    independent_latest = None
    if independent:
        independent_latest = max(
            independent,
            key=lambda path: max(model.stat().st_mtime for model in (path / "nodes").rglob("model.pt")),
        )

    result = {
        "project_root": str(root),
        "shared_run_dir": None if _latest(shared) is None else str(_latest(shared)),
        "flat_run_dir": None if _latest(flat) is None else str(_latest(flat)),
        "independent_run_dir": None if independent_latest is None else str(independent_latest),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
