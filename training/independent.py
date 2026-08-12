"""독립 taxonomy node Prompt-CAM 모델을 순차적으로 학습한다.

각 trainable 내부 node마다 별도 YAML, output root, Python process, optimizer,
checkpoint를 사용한다. 하나의 실행 묶음(run)은 고유한 run directory 아래에
격리되며, ``training_summary.json``에 실제 checkpoint 경로까지 기록된다.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset.imagefolder import load_taxonomy_manifest  # noqa: E402
from data.original_taxonomy import (  # noqa: E402
    TaxonomyNodeSpec,
    list_taxonomy_nodes,
    node_lookup,
)


def _resolve_from_project(value: str | Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _training_class_names(config: dict) -> list[str]:
    data_root = _resolve_from_project(config["data_path"])
    train_split = str(config.get("train_split", "train"))
    candidates = [data_root / train_split, data_root / "imagefolder" / train_split]
    train_root = next((path for path in candidates if path.is_dir()), None)
    if train_root is None:
        checked = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"ImageFolder 학습 split이 없습니다. 확인 경로: {checked}")
    classes = sorted(path.name for path in train_root.iterdir() if path.is_dir())
    if not classes:
        raise ValueError(f"ImageFolder 학습 split에 클래스 폴더가 없습니다: {train_root}")
    return classes


def _parse_node_filter(values: Iterable[str]) -> set[tuple[str, str | None]]:
    result: set[tuple[str, str | None]] = set()
    for raw in values:
        text = str(raw).strip()
        if not text:
            continue
        if text.casefold() == "root":
            result.add(("root", None))
            continue
        if ":" not in text:
            raise ValueError(
                f"--node {text!r} 형식이 잘못되었습니다. "
                "root 또는 family:이름, genus:이름을 사용하십시오"
            )
        rank, name = text.split(":", 1)
        rank = rank.strip().lower()
        name = name.strip()
        if rank not in {"family", "genus"} or not name:
            raise ValueError(f"--node 값이 잘못되었습니다: {text!r}")
        result.add((rank, name.casefold()))
    return result


def _selected(node: TaxonomyNodeSpec, filters: set[tuple[str, str | None]]) -> bool:
    if not filters:
        return True
    if node.rank == "root":
        return ("root", None) in filters
    return (node.rank, node.name.casefold()) in filters


def _generated_config(
    base: dict,
    node: TaxonomyNodeSpec,
    *,
    node_output_root: Path,
    run_id: str,
) -> dict:
    config = dict(base)
    config.update(
        {
            "train_type": "prompt_cam",
            "hierarchical_prompt": False,
            "original_taxonomy_prompt": True,
            "taxonomy_node_rank": node.rank,
            "taxonomy_node_name": None if node.rank == "root" else node.name,
            # 실제 값은 데이터 로더가 taxonomy에서 계산한다.
            "class_num": 0,
            "vpt_num": 0,
            # 실행 묶음마다 node output을 격리하여 재실행 checkpoint가 섞이지 않게 한다.
            "output_root": str(node_output_root),
            "taxonomy_experiment_run_id": run_id,
        }
    )
    return config


def _find_node_checkpoint(node_output_root: Path, node_id: str) -> tuple[Path, Path]:
    """새 node 전용 output root에서 정확히 하나의 검증된 model.pt를 찾는다."""
    matches: list[tuple[Path, Path]] = []
    for checkpoint in sorted(node_output_root.rglob("model.pt")):
        sidecar = checkpoint.parent / "taxonomy_node.json"
        if not sidecar.is_file():
            continue
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        if str(payload.get("node_id", "")).strip() == node_id:
            matches.append((checkpoint.resolve(), checkpoint.parent.resolve()))

    if not matches:
        raise FileNotFoundError(
            f"학습은 종료됐지만 node {node_id}의 model.pt를 찾지 못했습니다: "
            f"{node_output_root}"
        )
    if len(matches) > 1:
        formatted = "\n".join(f"- {path}" for path, _ in matches)
        raise RuntimeError(
            f"격리된 node output에 model.pt가 둘 이상 생성되었습니다: {node_id}\n{formatted}"
        )
    return matches[0]


def _write_summary(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="원논문식 taxonomy의 모든 trainable node Prompt-CAM 학습"
    )
    parser.add_argument("--base-config", required=True, type=str)
    parser.add_argument(
        "--node",
        action="append",
        default=[],
        help="선택 학습: root, family:Ulmaceae, genus:Quercus. 여러 번 지정 가능",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--generated-config-dir", default=None, type=str)
    parser.add_argument(
        "--run-id",
        default=None,
        help="실행 묶음 식별자. 생략하면 충돌 방지를 위해 microsecond 포함 시각을 사용",
    )
    args = parser.parse_args()

    base_path = Path(args.base_config).expanduser().resolve()
    if not base_path.is_file():
        raise FileNotFoundError(f"기준 YAML이 없습니다: {base_path}")
    with base_path.open("r", encoding="utf-8") as handle:
        base_config = yaml.safe_load(handle) or {}

    for key in ("data_path", "taxonomy_manifest"):
        if base_config.get(key) in (None, "", "null"):
            raise ValueError(f"기준 YAML에 {key}가 필요합니다")

    if not bool(base_config.get("store_ckp", False)):
        raise ValueError(
            "모든 node를 결합 평가하려면 기준 YAML에서 store_ckp: true가 필요합니다"
        )

    class_names = _training_class_names(base_config)
    taxonomy = load_taxonomy_manifest(
        _resolve_from_project(base_config["taxonomy_manifest"]),
        class_names,
        class_column=base_config.get("taxonomy_class_column"),
    )
    all_nodes = list_taxonomy_nodes(taxonomy, trainable_only=False)
    # 파일 경로용 slug가 충돌하면 YAML/checkpoint가 덮어써지므로 학습 전에 차단한다.
    node_lookup(all_nodes)
    filters = _parse_node_filter(args.node)
    nodes = [
        node
        for node in all_nodes
        if node.trainable and _selected(node, filters)
    ]
    if not nodes:
        raise ValueError("조건에 맞는 trainable taxonomy node가 없습니다")

    run_id = str(
        args.run_id
        or datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    ).strip()
    if not run_id or any(char in run_id for char in ("/", "\\")):
        raise ValueError("--run-id는 비어 있거나 경로 구분자를 포함할 수 없습니다")

    base_output_root = _resolve_from_project(base_config.get("output_root", "./output"))
    run_root = base_output_root / "runs" / run_id
    if run_root.exists() and any(run_root.iterdir()):
        raise FileExistsError(
            f"동일한 taxonomy run directory가 이미 존재합니다: {run_root}. "
            "새 --run-id를 사용하십시오."
        )
    run_root.mkdir(parents=True, exist_ok=True)

    if args.generated_config_dir:
        config_dir = Path(args.generated_config_dir).expanduser().resolve()
    else:
        config_dir = run_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "schema_version": 2,
        "base_config": str(base_path),
        "created_at": datetime.now().isoformat(timespec="microseconds"),
        "run_id": run_id,
        "run_root": str(run_root.resolve()),
        "dry_run": bool(args.dry_run),
        "nodes": [],
    }
    summary_path = run_root / "training_summary.json"

    for node in nodes:
        node_output_root = run_root / "nodes" / node.node_id
        generated = _generated_config(
            base_config,
            node,
            node_output_root=node_output_root,
            run_id=run_id,
        )
        config_path = config_dir / f"{node.node_id}.yaml"
        with config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(generated, handle, sort_keys=False, allow_unicode=True)

        command = [
            sys.executable,
            str(PROJECT_ROOT / "train.py"),
            "--config",
            str(config_path),
        ]
        print(
            f"[{node.node_id}] {node.display_name}: "
            f"{node.num_children}개 {node.child_rank} 분류"
        )
        print("  " + " ".join(command))

        record = {
            "node": node.to_dict(),
            "config": str(config_path.resolve()),
            "command": command,
            "node_output_root": str(node_output_root.resolve()),
            "returncode": None,
            "output_dir": None,
            "checkpoint": None,
            "final_result": None,
        }
        summary["nodes"].append(record)
        _write_summary(summary_path, summary)

        if args.dry_run:
            continue

        completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
        record["returncode"] = int(completed.returncode)
        if completed.returncode == 0:
            checkpoint, output_dir = _find_node_checkpoint(
                node_output_root,
                node.node_id,
            )
            record["checkpoint"] = str(checkpoint)
            record["output_dir"] = str(output_dir)
            final_result = output_dir / "final_result.json"
            record["final_result"] = (
                str(final_result.resolve()) if final_result.is_file() else None
            )
        _write_summary(summary_path, summary)

        if completed.returncode != 0 and not args.continue_on_error:
            raise SystemExit(completed.returncode)

    summary["completed_at"] = datetime.now().isoformat(timespec="microseconds")
    _write_summary(summary_path, summary)
    print(f"요약 저장: {summary_path}")


if __name__ == "__main__":
    main()
