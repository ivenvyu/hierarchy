#!/usr/bin/env python3
"""독립 taxonomy-node Prompt-CAM의 조건부 확률을 결합해 평가한다.

각 이미지 x와 species s의 최종 확률을 다음처럼 계산한다.

    P(s | x)
      = P(family(s) | x)
        P(genus(s) | family(s), x)
        P(s | genus(s), x)

자식이 하나뿐인 family/genus 단계의 조건부 확률은 1로 처리한다. 실행 중 생성된
``args.yaml``, ``class_to_idx.json``, ``model.pt``를 이용하므로 클래스 순서를
하드코딩하지 않는다.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets


@dataclass(frozen=True)
class TaxonRow:
    folder_name: str
    scientific_name: str
    genus: str
    family: str


@dataclass(frozen=True)
class TaxonomyTree:
    rows: tuple[TaxonRow, ...]
    species_names: tuple[str, ...]
    scientific_names: tuple[str, ...]
    genera: tuple[str, ...]
    families: tuple[str, ...]
    species_to_genus: tuple[int, ...]
    species_to_family: tuple[int, ...]
    genus_to_family: tuple[int, ...]
    species_by_genus: Mapping[str, tuple[str, ...]]
    genera_by_family: Mapping[str, tuple[str, ...]]

    @property
    def num_species(self) -> int:
        return len(self.species_names)


@dataclass(frozen=True)
class NodeSpec:
    key: str
    rank: str
    name: str | None
    config_path: Path
    expected_labels: tuple[str, ...]


@dataclass
class LoadedNode:
    spec: NodeSpec
    checkpoint_path: Path
    args_path: Path
    output_dir: Path
    model: torch.nn.Module
    params: SimpleNamespace
    labels: tuple[str, ...]
    validation_metric: float | None
    selection_metric: str | None


@dataclass(frozen=True)
class NodeOutput:
    labels: tuple[str, ...]
    probabilities: torch.Tensor
    checkpoint_path: str
    validation_metric: float | None
    selection_metric: str | None


class FullEvaluationDataset(Dataset):
    """전체 ImageFolder split을 하나의 공통 평가 순서로 반환한다."""

    def __init__(
        self,
        base_dataset: datasets.ImageFolder,
        transform,
        *,
        dataset_root: Path,
        global_index_by_class: Mapping[str, int],
    ) -> None:
        self.base_dataset = base_dataset
        self.transform = transform
        self.dataset_root = dataset_root.resolve()
        self.global_index_by_class = dict(global_index_by_class)

        missing = [name for name in base_dataset.classes if name not in self.global_index_by_class]
        if missing:
            raise ValueError(f"taxonomy에 없는 ImageFolder 클래스가 있습니다: {missing}")

    def __len__(self) -> int:
        return len(self.base_dataset.samples)

    def __getitem__(self, index: int):
        path_string, local_target = self.base_dataset.samples[index]
        image = self.base_dataset.loader(path_string)
        image_tensor, _, _ = self.transform(
            image,
            bbox=None,
            bbox_coordinate_mode="normalized",
        )
        class_name = self.base_dataset.classes[int(local_target)]
        global_target = self.global_index_by_class[class_name]
        relative_path = Path(path_string).resolve().relative_to(self.dataset_root).as_posix()
        return image_tensor, int(global_target), int(index), relative_path


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _display_path(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _resolve_path(value: Any, *, project_root: Path, base_dir: Path | None = None) -> Path:
    if value in (None, "", "null"):
        raise ValueError("비어 있는 경로는 해석할 수 없습니다")
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    candidates = []
    if base_dir is not None:
        candidates.append((base_dir / path).resolve())
    candidates.append((project_root / path).resolve())
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML 최상위 값이 mapping이 아닙니다: {path}")
    return data


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _taxonomy_from_manifest(path: Path, class_column: str) -> TaxonomyTree:
    rows: list[TaxonRow] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {class_column, "scientific_name", "genus", "family"}
        if reader.fieldnames is None:
            raise ValueError(f"taxonomy CSV에 헤더가 없습니다: {path}")
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"taxonomy CSV에 필요한 열이 없습니다: {sorted(missing)}")
        for line_number, row in enumerate(reader, start=2):
            values = {
                key: str(row.get(key, "")).strip()
                for key in required
            }
            empty = [key for key, value in values.items() if not value]
            if empty:
                raise ValueError(f"{path}:{line_number}에 빈 값이 있습니다: {empty}")
            rows.append(
                TaxonRow(
                    folder_name=values[class_column],
                    scientific_name=values["scientific_name"],
                    genus=values["genus"],
                    family=values["family"],
                )
            )

    if not rows:
        raise ValueError(f"taxonomy CSV에 데이터 행이 없습니다: {path}")

    species_names = tuple(row.folder_name for row in rows)
    if len(set(species_names)) != len(species_names):
        raise ValueError("taxonomy folder_name 값이 중복되었습니다")

    genera = tuple(dict.fromkeys(row.genus for row in rows))
    families = tuple(dict.fromkeys(row.family for row in rows))
    genus_index = {name: index for index, name in enumerate(genera)}
    family_index = {name: index for index, name in enumerate(families)}

    genus_family_name: dict[str, str] = {}
    for row in rows:
        previous = genus_family_name.setdefault(row.genus, row.family)
        if previous != row.family:
            raise ValueError(
                f"속 {row.genus!r}가 둘 이상의 family에 연결됩니다: {previous}, {row.family}"
            )

    species_by_genus: dict[str, tuple[str, ...]] = {
        genus: tuple(row.folder_name for row in rows if row.genus == genus)
        for genus in genera
    }
    genera_by_family: dict[str, tuple[str, ...]] = {
        family: tuple(genus for genus in genera if genus_family_name[genus] == family)
        for family in families
    }

    return TaxonomyTree(
        rows=tuple(rows),
        species_names=species_names,
        scientific_names=tuple(row.scientific_name for row in rows),
        genera=genera,
        families=families,
        species_to_genus=tuple(genus_index[row.genus] for row in rows),
        species_to_family=tuple(family_index[row.family] for row in rows),
        genus_to_family=tuple(family_index[genus_family_name[genus]] for genus in genera),
        species_by_genus=species_by_genus,
        genera_by_family=genera_by_family,
    )


def _match_named_taxon(raw_name: str, canonical_names: Sequence[str], *, rank: str) -> str:
    matches = [name for name in canonical_names if _slug(name) == _slug(raw_name)]
    if len(matches) != 1:
        raise ValueError(
            f"{rank} 노드 이름 {raw_name!r}을 taxonomy에서 유일하게 찾지 못했습니다. "
            f"후보={list(canonical_names)}"
        )
    return matches[0]


def _discover_node_specs(run_dir: Path, tree: TaxonomyTree) -> list[NodeSpec]:
    config_dir = run_dir / "configs"
    if not config_dir.is_dir():
        raise FileNotFoundError(f"노드 설정 디렉터리가 없습니다: {config_dir}")

    specs: list[NodeSpec] = []
    root_config = config_dir / "root.yaml"
    if not root_config.is_file():
        raise FileNotFoundError(f"root 설정이 없습니다: {root_config}")
    specs.append(
        NodeSpec(
            key="root",
            rank="root",
            name=None,
            config_path=root_config,
            expected_labels=tree.families,
        )
    )

    for path in sorted(config_dir.glob("family__*.yaml")):
        raw_name = path.stem.split("__", 1)[1]
        family = _match_named_taxon(raw_name, tree.families, rank="family")
        expected = tree.genera_by_family[family]
        if len(expected) <= 1:
            raise ValueError(
                f"자식 genus가 {len(expected)}개인 family 노드가 생성되었습니다: {path}"
            )
        specs.append(
            NodeSpec(
                key=path.stem.lower(),
                rank="family",
                name=family,
                config_path=path,
                expected_labels=expected,
            )
        )

    for path in sorted(config_dir.glob("genus__*.yaml")):
        raw_name = path.stem.split("__", 1)[1]
        genus = _match_named_taxon(raw_name, tree.genera, rank="genus")
        expected = tree.species_by_genus[genus]
        if len(expected) <= 1:
            raise ValueError(
                f"자식 species가 {len(expected)}개인 genus 노드가 생성되었습니다: {path}"
            )
        specs.append(
            NodeSpec(
                key=path.stem.lower(),
                rank="genus",
                name=genus,
                config_path=path,
                expected_labels=expected,
            )
        )

    expected_family_nodes = {
        f"family__{_slug(family)}"
        for family in tree.families
        if len(tree.genera_by_family[family]) > 1
    }
    expected_genus_nodes = {
        f"genus__{_slug(genus)}"
        for genus in tree.genera
        if len(tree.species_by_genus[genus]) > 1
    }
    observed_family_nodes = {spec.key for spec in specs if spec.rank == "family"}
    observed_genus_nodes = {spec.key for spec in specs if spec.rank == "genus"}

    if expected_family_nodes != observed_family_nodes:
        raise ValueError(
            "필요한 family 노드 설정과 실제 설정이 다릅니다: "
            f"필요={sorted(expected_family_nodes)}, 실제={sorted(observed_family_nodes)}"
        )
    if expected_genus_nodes != observed_genus_nodes:
        raise ValueError(
            "필요한 genus 노드 설정과 실제 설정이 다릅니다: "
            f"필요={sorted(expected_genus_nodes)}, 실제={sorted(observed_genus_nodes)}"
        )
    return specs


def _metric_from_result(result_path: Path) -> tuple[float | None, str | None]:
    if not result_path.is_file():
        return None, None
    data = _load_json(result_path)
    selection = str(data.get("selection_metric", "macro_f1"))
    metrics = data.get("best_validation_metrics", {})
    try:
        return float(metrics[selection]), selection
    except (KeyError, TypeError, ValueError):
        return None, selection


def _checkpoint_candidates(spec: NodeSpec, project_root: Path) -> list[tuple[Path, Path, float | None, str | None]]:
    config = _load_yaml(spec.config_path)
    output_root_value = config.get("output_root")
    search_roots: list[Path] = []
    if output_root_value not in (None, "", "null"):
        search_roots.append(
            _resolve_path(output_root_value, project_root=project_root, base_dir=spec.config_path.parent)
        )
    search_roots.append(spec.config_path.parent.parent / "nodes" / spec.key)

    candidates: list[tuple[Path, Path, float | None, str | None]] = []
    seen: set[Path] = set()
    for root in search_roots:
        if not root.is_dir():
            continue
        for checkpoint in root.rglob("model.pt"):
            checkpoint = checkpoint.resolve()
            if checkpoint in seen:
                continue
            seen.add(checkpoint)
            output_dir = checkpoint.parent
            args_path = output_dir / "args.yaml"
            result_path = output_dir / "final_result.json"
            if not args_path.is_file() or not result_path.is_file():
                continue
            metric, selection = _metric_from_result(result_path)
            candidates.append((checkpoint, args_path, metric, selection))
    return candidates


def _select_checkpoint(spec: NodeSpec, project_root: Path) -> tuple[Path, Path, float | None, str | None]:
    candidates = _checkpoint_candidates(spec, project_root)
    if not candidates:
        raise FileNotFoundError(
            f"{spec.key}의 완료된 model.pt/args.yaml/final_result.json 묶음을 찾지 못했습니다"
        )

    def score(item):
        checkpoint, _, metric, _ = item
        metric_score = metric if metric is not None and math.isfinite(metric) else float("-inf")
        return metric_score, checkpoint.stat().st_mtime

    return max(candidates, key=score)


def _walk_metadata_candidates(value: Any, *, prefix: str = "") -> Iterable[tuple[str, tuple[str, ...]]]:
    """class/label/name 관련 metadata에서 출력 순서 후보를 재귀적으로 추출한다."""
    if isinstance(value, Mapping):
        if value and all(isinstance(key, str) for key in value) and all(
            isinstance(index, int) and not isinstance(index, bool) for index in value.values()
        ):
            indices = sorted(value.values())
            if indices == list(range(len(indices))):
                ordered = tuple(key for key, _ in sorted(value.items(), key=lambda item: item[1]))
                yield prefix or "mapping", ordered
        for key, child in value.items():
            key_text = str(key).lower()
            if any(token in key_text for token in ("class", "label", "name", "child", "target", "family", "genus", "species")):
                yield from _walk_metadata_candidates(child, prefix=f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(value, (list, tuple)) and value and all(
        isinstance(item, (str, int)) and not isinstance(item, bool) for item in value
    ):
        yield prefix or "list", tuple(str(item) for item in value)


def _species_aliases(tree: TaxonomyTree, canonical: str) -> set[str]:
    index = tree.species_names.index(canonical)
    scientific = tree.scientific_names[index]
    return {
        _slug(canonical),
        _slug(scientific),
        _slug(scientific.replace(" ", "_")),
    }


def _canonicalize_output_labels(
    raw_order: Sequence[str],
    expected_labels: Sequence[str],
    *,
    rank: str,
    tree: TaxonomyTree,
) -> tuple[str, ...] | None:
    if len(raw_order) != len(expected_labels):
        return None

    alias_by_expected: dict[str, set[str]] = {}
    for expected in expected_labels:
        if rank == "genus":
            alias_by_expected[expected] = _species_aliases(tree, expected)
        else:
            alias_by_expected[expected] = {_slug(expected)}

    resolved: list[str] = []
    used: set[str] = set()
    for raw in raw_order:
        normalized = _slug(str(raw))
        matches = [
            expected
            for expected, aliases in alias_by_expected.items()
            if normalized in aliases and expected not in used
        ]
        if len(matches) != 1:
            return None
        resolved.append(matches[0])
        used.add(matches[0])
    return tuple(resolved)


def _resolve_output_order(
    spec: NodeSpec,
    args_data: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    class_to_idx_path: Path,
    tree: TaxonomyTree,
    *,
    allow_order_fallback: bool,
) -> tuple[str, ...]:
    sources: list[tuple[str, Any]] = [("args.yaml", args_data)]
    if class_to_idx_path.is_file():
        sources.append(("class_to_idx.json", _load_json(class_to_idx_path)))
    for key in ("class_to_idx", "config", "taxonomy"):
        if key in checkpoint:
            sources.append((f"checkpoint.{key}", checkpoint[key]))

    examined: list[str] = []
    expected_count = len(spec.expected_labels)
    for source_name, source in sources:
        for metadata_key, raw_order in _walk_metadata_candidates(source):
            if len(raw_order) != expected_count:
                continue
            examined.append(f"{source_name}:{metadata_key}={list(raw_order)}")
            canonical = _canonicalize_output_labels(
                raw_order,
                spec.expected_labels,
                rank=spec.rank,
                tree=tree,
            )
            if canonical is not None:
                return canonical

    if allow_order_fallback:
        print(
            f"[경고] {spec.key}: 출력 class metadata를 찾지 못해 taxonomy 순서를 사용합니다: "
            f"{list(spec.expected_labels)}",
            file=sys.stderr,
        )
        return spec.expected_labels

    diagnostic = "\n  ".join(examined[:20]) if examined else "유효한 후보 없음"
    raise ValueError(
        f"{spec.key}의 로짓 열 순서를 안전하게 결정하지 못했습니다. "
        "args.yaml 또는 class_to_idx.json에 로컬 class mapping이 필요합니다.\n  "
        f"검사 후보: {diagnostic}\n"
        "정말 생성 코드의 taxonomy 순서와 동일함을 확인한 경우에만 "
        "--allow-order-fallback을 사용하십시오."
    )


def _namespace_from_args(args_data: Mapping[str, Any]) -> SimpleNamespace:
    data = dict(args_data)
    data.update(
        {
            "distributed": False,
            "local_rank": 0,
            "vis_attn": False,
            "load_pretrained_backbone": False,
            "promptcam_checkpoint": None,
            "resume": None,
            "debug": False,
        }
    )
    return SimpleNamespace(**data)


def _strip_module_prefix(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if state and all(str(key).startswith("module.") for key in state):
        return {str(key)[7:]: value for key, value in state.items()}
    return dict(state)


def _load_node(
    spec: NodeSpec,
    *,
    project_root: Path,
    tree: TaxonomyTree,
    device: torch.device,
    allow_order_fallback: bool,
) -> LoadedNode:
    checkpoint_path, args_path, metric, selection = _select_checkpoint(spec, project_root)
    args_data = _load_yaml(args_path)
    params = _namespace_from_args(args_data)

    from model.factory import get_model

    try:
        model, _, _ = get_model(params, visualize=True)
    except TypeError:
        model, _, _ = get_model(params)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"checkpoint가 mapping이 아닙니다: {checkpoint_path}")
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    if not isinstance(state, Mapping):
        raise TypeError(f"model state를 찾지 못했습니다: {checkpoint_path}")
    state = _strip_module_prefix(state)
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"{spec.key} checkpoint 구조가 모델과 일치하지 않습니다. "
            f"누락={incompatible.missing_keys}, 예상 외={incompatible.unexpected_keys}"
        )

    labels = _resolve_output_order(
        spec,
        args_data,
        checkpoint,
        args_path.parent / "class_to_idx.json",
        tree,
        allow_order_fallback=allow_order_fallback,
    )
    if int(getattr(params, "class_num")) != len(labels):
        raise ValueError(
            f"{spec.key}: class_num={getattr(params, 'class_num')}과 출력 label 수={len(labels)}가 다릅니다"
        )

    model = model.to(device).eval()
    return LoadedNode(
        spec=spec,
        checkpoint_path=checkpoint_path,
        args_path=args_path,
        output_dir=args_path.parent,
        model=model,
        params=params,
        labels=labels,
        validation_metric=metric,
        selection_metric=selection,
    )


def _extract_logits(output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        output = output[0]
    if isinstance(output, Mapping):
        for key in ("logits", "species_logits", "output", "predictions"):
            if key in output:
                output = output[key]
                break
        else:
            raise KeyError(f"모델 출력 dict에서 logits를 찾지 못했습니다: {sorted(output)}")
    if not torch.is_tensor(output):
        raise TypeError(f"모델 출력이 Tensor가 아닙니다: {type(output)!r}")
    if output.ndim == 3 and output.shape[-1] == 1:
        output = output.squeeze(-1)
    if output.ndim != 2:
        raise ValueError(f"로짓 shape은 [B,C]여야 합니다: {tuple(output.shape)}")
    return output


def _amp_context(device: torch.device, amp_name: str):
    amp_name = str(amp_name).lower()
    if device.type != "cuda" or amp_name == "none":
        return nullcontext()
    if amp_name == "float16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    if amp_name == "bfloat16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    raise ValueError(f"지원하지 않는 amp_dtype입니다: {amp_name}")


def _run_node_inference(
    node: LoadedNode,
    loader: DataLoader,
    *,
    sample_count: int,
    device: torch.device,
) -> torch.Tensor:
    class_count = len(node.labels)
    raw_probabilities = torch.empty((sample_count, class_count), dtype=torch.float32)
    seen = torch.zeros(sample_count, dtype=torch.bool)
    amp_name = str(getattr(node.params, "amp_dtype", "none"))

    with torch.inference_mode():
        for images, _, sample_indices, _ in loader:
            images = images.to(device, non_blocking=True)
            with _amp_context(device, amp_name):
                logits = _extract_logits(node.model(images))
            if logits.shape[1] != class_count:
                raise ValueError(
                    f"{node.spec.key}: 로짓 class 수={logits.shape[1]}, metadata class 수={class_count}"
                )
            probs = torch.softmax(logits.float(), dim=1).cpu()
            indices = sample_indices.to(dtype=torch.long)
            raw_probabilities.index_copy_(0, indices, probs)
            seen.index_fill_(0, indices, True)

    if not bool(seen.all()):
        missing = torch.nonzero(~seen, as_tuple=False).flatten().tolist()
        raise RuntimeError(f"{node.spec.key}: 추론되지 않은 sample index가 있습니다: {missing[:20]}")

    canonical_index = {label: index for index, label in enumerate(node.spec.expected_labels)}
    reordered = torch.empty_like(raw_probabilities)
    for model_column, label in enumerate(node.labels):
        reordered[:, canonical_index[label]] = raw_probabilities[:, model_column]
    return reordered


def combine_leaf_probabilities(
    tree: TaxonomyTree,
    node_outputs: Mapping[str, NodeOutput],
) -> tuple[torch.Tensor, float]:
    """node conditional 확률을 모든 species leaf의 joint probability로 결합한다."""
    if "root" not in node_outputs:
        raise KeyError("root node 확률이 없습니다")
    root = node_outputs["root"]
    if tuple(root.labels) != tree.families:
        raise ValueError(f"root label 순서가 taxonomy family 순서와 다릅니다: {root.labels}")

    sample_count = root.probabilities.shape[0]
    leaf = torch.empty((sample_count, tree.num_species), dtype=torch.float32)
    family_index = {name: index for index, name in enumerate(tree.families)}
    genus_index = {name: index for index, name in enumerate(tree.genera)}

    for species_index, row in enumerate(tree.rows):
        leaf[:, species_index] = root.probabilities[:, family_index[row.family]]

    for family in tree.families:
        child_genera = tree.genera_by_family[family]
        if len(child_genera) == 1:
            continue
        key = f"family__{_slug(family)}"
        if key not in node_outputs:
            raise KeyError(f"필요한 family node 확률이 없습니다: {key}")
        output = node_outputs[key]
        if tuple(output.labels) != tuple(child_genera):
            raise ValueError(f"{key} label 순서가 taxonomy와 다릅니다")
        local_index = {name: index for index, name in enumerate(child_genera)}
        for species_index, row in enumerate(tree.rows):
            if row.family == family:
                leaf[:, species_index] *= output.probabilities[:, local_index[row.genus]]

    for genus in tree.genera:
        child_species = tree.species_by_genus[genus]
        if len(child_species) == 1:
            continue
        key = f"genus__{_slug(genus)}"
        if key not in node_outputs:
            raise KeyError(f"필요한 genus node 확률이 없습니다: {key}")
        output = node_outputs[key]
        if tuple(output.labels) != tuple(child_species):
            raise ValueError(f"{key} label 순서가 taxonomy와 다릅니다")
        local_index = {name: index for index, name in enumerate(child_species)}
        for species_name in child_species:
            species_index = tree.species_names.index(species_name)
            leaf[:, species_index] *= output.probabilities[:, local_index[species_name]]

    row_sums = leaf.sum(dim=1)
    max_error = float((row_sums - 1.0).abs().max().item())
    if not torch.isfinite(leaf).all() or (leaf < 0).any():
        raise FloatingPointError("결합 leaf 확률에 음수 또는 비유한 값이 있습니다")
    if max_error > 1e-3:
        raise ValueError(
            f"결합 leaf 확률의 행 합이 1에서 크게 벗어납니다: 최대 오차={max_error:.6g}"
        )
    leaf = leaf / row_sums[:, None].clamp_min(1e-12)
    return leaf, max_error


def greedy_predictions(tree: TaxonomyTree, node_outputs: Mapping[str, NodeOutput]) -> torch.Tensor:
    root = node_outputs["root"].probabilities
    family_choice = root.argmax(dim=1)
    predictions = torch.empty(root.shape[0], dtype=torch.long)
    species_index = {name: index for index, name in enumerate(tree.species_names)}

    for sample_index in range(root.shape[0]):
        family = tree.families[int(family_choice[sample_index])]
        child_genera = tree.genera_by_family[family]
        if len(child_genera) == 1:
            genus = child_genera[0]
        else:
            family_output = node_outputs[f"family__{_slug(family)}"]
            genus = child_genera[int(family_output.probabilities[sample_index].argmax())]

        child_species = tree.species_by_genus[genus]
        if len(child_species) == 1:
            species = child_species[0]
        else:
            genus_output = node_outputs[f"genus__{_slug(genus)}"]
            species = child_species[int(genus_output.probabilities[sample_index].argmax())]
        predictions[sample_index] = species_index[species]
    return predictions


def _confusion_matrix(targets: torch.Tensor, predictions: torch.Tensor, class_count: int) -> torch.Tensor:
    flat = targets.to(torch.long) * class_count + predictions.to(torch.long)
    return torch.bincount(flat, minlength=class_count * class_count).reshape(class_count, class_count)


def _classification_metrics(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    predictions: torch.Tensor,
) -> tuple[dict[str, float], torch.Tensor, list[dict[str, float]]]:
    class_count = probabilities.shape[1]
    confusion = _confusion_matrix(targets, predictions, class_count).to(torch.float64)
    support = confusion.sum(dim=1)
    predicted_count = confusion.sum(dim=0)
    true_positive = confusion.diag()
    active = support > 0
    recall = true_positive / support.clamp_min(1.0)
    precision = true_positive / predicted_count.clamp_min(1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-12)

    top5 = probabilities.topk(min(5, class_count), dim=1).indices.eq(targets[:, None]).any(dim=1)
    true_probability = probabilities.gather(1, targets[:, None]).squeeze(1).clamp_min(1e-12)
    metrics = {
        "top1": float(predictions.eq(targets).float().mean().item() * 100.0),
        "top5": float(top5.float().mean().item() * 100.0),
        "balanced_accuracy": float(recall[active].mean().item() * 100.0),
        "macro_f1": float(f1[active].mean().item() * 100.0),
        "negative_log_likelihood": float((-true_probability.log()).mean().item()),
        "mean_confidence": float(probabilities.max(dim=1).values.mean().item()),
    }
    per_class = [
        {
            "support": int(support[index].item()),
            "precision": float(precision[index].item() * 100.0),
            "recall": float(recall[index].item() * 100.0),
            "f1": float(f1[index].item() * 100.0),
        }
        for index in range(class_count)
    ]
    return metrics, confusion.to(torch.long), per_class


def _taxonomy_metrics(tree: TaxonomyTree, targets: torch.Tensor, predictions: torch.Tensor) -> dict[str, float]:
    species_to_genus = torch.tensor(tree.species_to_genus, dtype=torch.long)
    species_to_family = torch.tensor(tree.species_to_family, dtype=torch.long)
    true_genus = species_to_genus[targets]
    pred_genus = species_to_genus[predictions]
    true_family = species_to_family[targets]
    pred_family = species_to_family[predictions]

    same_species = predictions.eq(targets)
    same_genus = pred_genus.eq(true_genus)
    same_family = pred_family.eq(true_family)
    distance = torch.where(
        same_species,
        torch.zeros_like(targets),
        torch.where(
            same_genus,
            torch.ones_like(targets),
            torch.where(same_family, torch.full_like(targets, 2), torch.full_like(targets, 3)),
        ),
    )
    return {
        "species_accuracy": float(same_species.float().mean().item() * 100.0),
        "genus_accuracy": float(same_genus.float().mean().item() * 100.0),
        "family_accuracy": float(same_family.float().mean().item() * 100.0),
        "mean_taxonomic_distance": float(distance.float().mean().item()),
        "same_genus_but_wrong_species": float((same_genus & ~same_species).float().mean().item() * 100.0),
        "same_family_but_wrong_genus": float((same_family & ~same_genus).float().mean().item() * 100.0),
        "wrong_family": float((~same_family).float().mean().item() * 100.0),
    }


def _node_conditional_metrics(
    tree: TaxonomyTree,
    node_outputs: Mapping[str, NodeOutput],
    targets: torch.Tensor,
) -> dict[str, Any]:
    species_to_genus = torch.tensor(tree.species_to_genus, dtype=torch.long)
    species_to_family = torch.tensor(tree.species_to_family, dtype=torch.long)
    true_genus_global = species_to_genus[targets]
    true_family_global = species_to_family[targets]

    result: dict[str, Any] = {}
    root = node_outputs["root"]
    root_prediction = root.probabilities.argmax(dim=1)
    result["root"] = {
        "rank": "family",
        "sample_count": int(targets.numel()),
        "accuracy": float(root_prediction.eq(true_family_global).float().mean().item() * 100.0),
    }

    genus_global_index = {name: index for index, name in enumerate(tree.genera)}
    family_global_index = {name: index for index, name in enumerate(tree.families)}
    species_global_index = {name: index for index, name in enumerate(tree.species_names)}

    for key, output in node_outputs.items():
        if key == "root":
            continue
        if key.startswith("family__"):
            family = _match_named_taxon(key.split("__", 1)[1], tree.families, rank="family")
            mask = true_family_global.eq(family_global_index[family])
            local_truth_lookup = {genus_global_index[name]: index for index, name in enumerate(output.labels)}
            true_local = torch.tensor(
                [local_truth_lookup[int(value)] for value in true_genus_global[mask]],
                dtype=torch.long,
            )
            pred_local = output.probabilities[mask].argmax(dim=1)
            rank = "genus"
        elif key.startswith("genus__"):
            genus = _match_named_taxon(key.split("__", 1)[1], tree.genera, rank="genus")
            mask = true_genus_global.eq(genus_global_index[genus])
            local_truth_lookup = {species_global_index[name]: index for index, name in enumerate(output.labels)}
            true_local = torch.tensor(
                [local_truth_lookup[int(value)] for value in targets[mask]],
                dtype=torch.long,
            )
            pred_local = output.probabilities[mask].argmax(dim=1)
            rank = "species"
        else:
            continue
        result[key] = {
            "rank": rank,
            "sample_count": int(mask.sum().item()),
            "accuracy": float(pred_local.eq(true_local).float().mean().item() * 100.0),
        }
    return result


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def _validate_preprocessing(nodes: Sequence[LoadedNode]) -> None:
    fields = (
        "crop_size",
        "eval_resize_size",
        "normalization",
        "normalization_mean",
        "normalization_std",
    )
    root = next(node for node in nodes if node.spec.key == "root")
    mismatches = []
    for node in nodes:
        for field in fields:
            if getattr(node.params, field, None) != getattr(root.params, field, None):
                mismatches.append(
                    f"{node.spec.key}.{field}={getattr(node.params, field, None)!r} "
                    f"!= root.{field}={getattr(root.params, field, None)!r}"
                )
    if mismatches:
        raise ValueError("노드별 평가 전처리가 다릅니다:\n" + "\n".join(mismatches))


def _resolve_dataset_root(params: SimpleNamespace, project_root: Path) -> Path:
    configured = _resolve_path(getattr(params, "data_path"), project_root=project_root)
    split = str(getattr(params, "train_split", "train"))
    for candidate in (configured, configured / "imagefolder"):
        if (candidate / split).is_dir():
            return candidate
    raise FileNotFoundError(f"데이터셋 루트를 찾지 못했습니다: {configured}")


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    run_id = args.run_id
    run_dir = project_root / "output" / "independent" / "runs" / run_id
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run 디렉터리가 없습니다: {run_dir}")

    root_config = _load_yaml(run_dir / "configs" / "root.yaml")
    taxonomy_path = _resolve_path(
        root_config.get("taxonomy_manifest"),
        project_root=project_root,
        base_dir=(run_dir / "configs"),
    )
    class_column = str(root_config.get("taxonomy_class_column") or "folder_name")
    tree = _taxonomy_from_manifest(taxonomy_path, class_column)
    specs = _discover_node_specs(run_dir, tree)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device를 요청했지만 torch.cuda.is_available()이 False입니다")

    print(f"[실험] {run_id}")
    print(f"[taxonomy] species={tree.num_species}, genera={len(tree.genera)}, families={len(tree.families)}")
    print(f"[노드] 총 {len(specs)}개: {', '.join(spec.key for spec in specs)}")
    print(f"[장치] {device}")

    loaded_nodes: list[LoadedNode] = []
    for index, spec in enumerate(specs, start=1):
        print(f"[{index}/{len(specs)}] {spec.key} 체크포인트 로드")
        node = _load_node(
            spec,
            project_root=project_root,
            tree=tree,
            device=device,
            allow_order_fallback=bool(args.allow_order_fallback),
        )
        loaded_nodes.append(node)
        print(
            f"  checkpoint={_display_path(node.checkpoint_path, project_root)}\n"
            f"  labels={list(node.labels)}"
        )
    _validate_preprocessing(loaded_nodes)

    root_node = next(node for node in loaded_nodes if node.spec.key == "root")
    dataset_root = _resolve_dataset_root(root_node.params, project_root)
    split_name = args.split or str(getattr(root_node.params, "test_split", "test"))
    split_root = dataset_root / split_name
    if not split_root.is_dir():
        raise FileNotFoundError(f"평가 split이 없습니다: {split_root}")

    base_dataset = datasets.ImageFolder(str(split_root), transform=None)
    taxonomy_classes = set(tree.species_names)
    imagefolder_classes = set(base_dataset.classes)
    if taxonomy_classes != imagefolder_classes:
        raise ValueError(
            "taxonomy와 평가 ImageFolder 클래스가 다릅니다: "
            f"누락={sorted(taxonomy_classes - imagefolder_classes)}, "
            f"추가={sorted(imagefolder_classes - taxonomy_classes)}"
        )

    from data.dataset.imagefolder import JointImageTransform

    transform = JointImageTransform(root_node.params, training=False)
    dataset = FullEvaluationDataset(
        base_dataset,
        transform,
        dataset_root=dataset_root,
        global_index_by_class={name: index for index, name in enumerate(tree.species_names)},
    )
    batch_size = int(args.batch_size or getattr(root_node.params, "test_batch_size", 64))
    num_workers = int(args.num_workers if args.num_workers is not None else getattr(root_node.params, "num_workers", 4))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    print(f"[데이터] {_display_path(split_root, project_root)}, samples={len(dataset)}, batch={batch_size}")

    targets = torch.empty(len(dataset), dtype=torch.long)
    relative_paths: list[str | None] = [None] * len(dataset)
    for _, batch_targets, sample_indices, batch_paths in loader:
        targets.index_copy_(0, sample_indices.long(), batch_targets.long())
        for sample_index, relative_path in zip(sample_indices.tolist(), batch_paths):
            relative_paths[int(sample_index)] = str(relative_path)
    if any(path is None for path in relative_paths):
        raise RuntimeError("평가 sample path 수집이 완전하지 않습니다")

    node_outputs: dict[str, NodeOutput] = {}
    checkpoint_records: dict[str, Any] = {}
    for index, node in enumerate(loaded_nodes, start=1):
        print(f"[{index}/{len(loaded_nodes)}] {node.spec.key} 전체 {split_name} 추론")
        probabilities = _run_node_inference(
            node,
            loader,
            sample_count=len(dataset),
            device=device,
        )
        node_outputs[node.spec.key] = NodeOutput(
            labels=node.spec.expected_labels,
            probabilities=probabilities,
            checkpoint_path=str(node.checkpoint_path),
            validation_metric=node.validation_metric,
            selection_metric=node.selection_metric,
        )
        checkpoint_records[node.spec.key] = {
            "checkpoint": str(node.checkpoint_path),
            "args": str(node.args_path),
            "labels": list(node.spec.expected_labels),
            "model_output_order": list(node.labels),
            "selection_metric": node.selection_metric,
            "best_validation_value": node.validation_metric,
        }
        del node.model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    leaf_probabilities, probability_sum_max_error = combine_leaf_probabilities(tree, node_outputs)
    joint_prediction = leaf_probabilities.argmax(dim=1)
    greedy_prediction = greedy_predictions(tree, node_outputs)

    joint_metrics, confusion, per_class = _classification_metrics(
        leaf_probabilities,
        targets,
        joint_prediction,
    )
    joint_taxonomy_metrics = _taxonomy_metrics(tree, targets, joint_prediction)
    greedy_metrics = {
        "top1": float(greedy_prediction.eq(targets).float().mean().item() * 100.0),
        "routing_disagreement_with_joint_map": float(
            greedy_prediction.ne(joint_prediction).float().mean().item() * 100.0
        ),
    }
    greedy_taxonomy_metrics = _taxonomy_metrics(tree, targets, greedy_prediction)
    conditional_metrics = _node_conditional_metrics(tree, node_outputs, targets)

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else (
        run_dir / "evaluation" / f"combined_{split_name}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    species_to_genus = torch.tensor(tree.species_to_genus, dtype=torch.long)
    species_to_family = torch.tensor(tree.species_to_family, dtype=torch.long)
    true_probability = leaf_probabilities.gather(1, targets[:, None]).squeeze(1)
    confidence, _ = leaf_probabilities.max(dim=1)

    prediction_rows = []
    for index in range(len(dataset)):
        true_index = int(targets[index])
        joint_index = int(joint_prediction[index])
        greedy_index = int(greedy_prediction[index])
        prediction_rows.append(
            {
                "relative_path": relative_paths[index],
                "true_folder": tree.species_names[true_index],
                "true_scientific_name": tree.scientific_names[true_index],
                "true_genus": tree.genera[int(species_to_genus[true_index])],
                "true_family": tree.families[int(species_to_family[true_index])],
                "joint_pred_folder": tree.species_names[joint_index],
                "joint_pred_scientific_name": tree.scientific_names[joint_index],
                "joint_pred_genus": tree.genera[int(species_to_genus[joint_index])],
                "joint_pred_family": tree.families[int(species_to_family[joint_index])],
                "joint_correct": int(joint_index == true_index),
                "joint_confidence": float(confidence[index]),
                "true_leaf_probability": float(true_probability[index]),
                "greedy_pred_folder": tree.species_names[greedy_index],
                "greedy_correct": int(greedy_index == true_index),
            }
        )

    _write_csv(
        output_dir / "predictions.csv",
        tuple(prediction_rows[0].keys()),
        prediction_rows,
    )

    per_class_rows = []
    for index, metrics in enumerate(per_class):
        row = {
            "class_index": index,
            "folder_name": tree.species_names[index],
            "scientific_name": tree.scientific_names[index],
            "genus": tree.genera[tree.species_to_genus[index]],
            "family": tree.families[tree.species_to_family[index]],
            **metrics,
        }
        per_class_rows.append(row)
    _write_csv(
        output_dir / "per_class_metrics.csv",
        tuple(per_class_rows[0].keys()),
        per_class_rows,
    )

    confusion_rows = []
    for true_index, true_name in enumerate(tree.species_names):
        row: dict[str, Any] = {"true_class": true_name}
        for pred_index, pred_name in enumerate(tree.species_names):
            row[pred_name] = int(confusion[true_index, pred_index])
        confusion_rows.append(row)
    _write_csv(
        output_dir / "confusion_matrix.csv",
        ("true_class", *tree.species_names),
        confusion_rows,
    )

    result = {
        "run_id": run_id,
        "split": split_name,
        "sample_count": len(dataset),
        "class_count": tree.num_species,
        "probability_combination": (
            "P(species|x)=P(family|x)*P(genus|family,x)*P(species|genus,x); "
            "singleton conditional factors are 1"
        ),
        "probability_sum_max_absolute_error_before_normalization": probability_sum_max_error,
        "joint_map": {
            **joint_metrics,
            **joint_taxonomy_metrics,
        },
        "greedy_routing": {
            **greedy_metrics,
            **greedy_taxonomy_metrics,
        },
        "node_conditional_metrics": conditional_metrics,
        "checkpoints": checkpoint_records,
        "taxonomy": {
            "families": list(tree.families),
            "genera": list(tree.genera),
            "species": list(tree.species_names),
        },
        "artifacts": {
            "predictions_csv": str(output_dir / "predictions.csv"),
            "per_class_metrics_csv": str(output_dir / "per_class_metrics.csv"),
            "confusion_matrix_csv": str(output_dir / "confusion_matrix.csv"),
        },
    }
    result_path = output_dir / "evaluation_summary.json"
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)

    print("\n" + "=" * 68)
    print("Independent taxonomy 결합 평가 결과")
    print("=" * 68)
    print(f"Joint MAP Top-1          : {joint_metrics['top1']:.4f}%")
    print(f"Joint MAP Macro-F1       : {joint_metrics['macro_f1']:.4f}%")
    print(f"Joint MAP Top-5          : {joint_metrics['top5']:.4f}%")
    print(f"Joint genus accuracy     : {joint_taxonomy_metrics['genus_accuracy']:.4f}%")
    print(f"Joint family accuracy    : {joint_taxonomy_metrics['family_accuracy']:.4f}%")
    print(f"Mean taxonomic distance  : {joint_taxonomy_metrics['mean_taxonomic_distance']:.6f}")
    print(f"Greedy routing Top-1     : {greedy_metrics['top1']:.4f}%")
    print(f"확률합 최대 오차         : {probability_sum_max_error:.6g}")
    print(f"결과 저장                : {result_path}")
    print("=" * 68)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="독립 taxonomy-node Prompt-CAM의 조건부 확률을 결합해 최종 species 성능을 계산"
    )
    parser.add_argument("--run-id", required=True, help="평가할 independent run ID")
    parser.add_argument("--split", default=None, help="기본값은 root args.yaml의 test_split")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, cpu")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--allow-order-fallback",
        action="store_true",
        help="class mapping metadata가 없을 때 taxonomy 순서를 사용. 순서가 동일함을 검증한 경우에만 사용",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
