"""Prompt-CAM 원논문식 taxonomy node 정의와 경로 계산 도구.

원논문식 구현은 하나의 모델에 과·속·종 prompt를 동시에 넣지 않는다.
대신 taxonomy의 각 내부 node에서 그 node의 직접 자식만 분류하는 독립적인
flat Prompt-CAM을 학습한다. 이 모듈은 특정 데이터셋 이름이나 클래스 수를
하드코딩하지 않고 taxonomy manifest에서 node-local 분류 문제를 구성한다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence


_VALID_NODE_RANKS = {"root", "family", "genus"}


def _slug(value: str) -> str:
    """파일명과 실험 경로에 안전한 소문자 식별자를 만든다."""
    normalized = re.sub(r"[^0-9A-Za-z._-]+", "-", str(value).strip())
    normalized = re.sub(r"-+", "-", normalized).strip("-._")
    return normalized.lower() or "node"


def _validate_taxonomy(taxonomy: Any) -> None:
    """필요한 taxonomy 배열의 길이와 index 범위를 검증한다."""
    required = (
        "class_names",
        "scientific_names",
        "genus_names",
        "family_names",
        "species_to_genus",
        "genus_to_family",
    )
    missing = [name for name in required if not hasattr(taxonomy, name)]
    if missing:
        raise TypeError(f"taxonomy 객체에 필요한 속성이 없습니다: {missing}")

    class_names = list(taxonomy.class_names)
    scientific_names = list(taxonomy.scientific_names)
    genus_names = list(taxonomy.genus_names)
    family_names = list(taxonomy.family_names)
    species_to_genus = [int(value) for value in taxonomy.species_to_genus]
    genus_to_family = [int(value) for value in taxonomy.genus_to_family]

    if not class_names:
        raise ValueError("taxonomy에는 적어도 하나의 species가 필요합니다")
    if len(scientific_names) != len(class_names):
        raise ValueError("scientific_names 길이가 class_names와 일치하지 않습니다")
    if len(species_to_genus) != len(class_names):
        raise ValueError("species_to_genus 길이가 class_names와 일치하지 않습니다")
    if len(genus_to_family) != len(genus_names):
        raise ValueError("genus_to_family 길이가 genus_names와 일치하지 않습니다")
    if not genus_names or not family_names:
        raise ValueError("taxonomy에는 적어도 하나의 genus와 family가 필요합니다")

    for species_index, genus_index in enumerate(species_to_genus):
        if not 0 <= genus_index < len(genus_names):
            raise ValueError(
                f"species_to_genus[{species_index}]={genus_index}가 genus 범위를 벗어났습니다"
            )
    for genus_index, family_index in enumerate(genus_to_family):
        if not 0 <= family_index < len(family_names):
            raise ValueError(
                f"genus_to_family[{genus_index}]={family_index}가 family 범위를 벗어났습니다"
            )


def _resolve_name(query: str | None, names: Sequence[str], *, rank: str) -> int:
    """대소문자를 무시한 정확 일치로 family/genus 이름을 index로 변환한다."""
    if query in (None, "", "null"):
        raise ValueError(f"taxonomy_node_rank={rank!r}에는 taxonomy_node_name이 필요합니다")

    normalized = str(query).strip().casefold()
    matches = [
        index
        for index, name in enumerate(names)
        if str(name).strip().casefold() == normalized
    ]
    if not matches:
        raise ValueError(
            f"taxonomy에서 {rank} {query!r}를 찾지 못했습니다. 가능한 값: {list(names)}"
        )
    if len(matches) > 1:
        raise ValueError(f"{rank} 이름 {query!r}가 대소문자 무시 기준으로 중복됩니다")
    return matches[0]


@dataclass(frozen=True)
class TaxonomyNodeSpec:
    """하나의 원논문식 node-local Prompt-CAM 분류 문제."""

    rank: str
    name: str
    child_rank: str
    child_names: tuple[str, ...]
    child_identifiers: tuple[str, ...]
    descendant_species_indices: tuple[int, ...]
    species_to_child: tuple[int, ...]
    child_to_species_indices: tuple[tuple[int, ...], ...]

    @property
    def num_children(self) -> int:
        return len(self.child_names)

    @property
    def trainable(self) -> bool:
        return self.num_children >= 2

    @property
    def node_id(self) -> str:
        if self.rank == "root":
            return "root"
        return f"{self.rank}__{_slug(self.name)}"

    @property
    def display_name(self) -> str:
        return "taxonomy root" if self.rank == "root" else f"{self.rank}:{self.name}"

    @property
    def class_to_idx(self) -> dict[str, int]:
        return {
            identifier: index
            for index, identifier in enumerate(self.child_identifiers)
        }

    def local_target(self, species_index: int) -> int:
        """전역 species index를 이 node의 local child index로 바꾼다."""
        index = int(species_index)
        if not 0 <= index < len(self.species_to_child):
            raise IndexError(
                f"species index {index}가 범위를 벗어났습니다: [0,{len(self.species_to_child)})"
            )
        target = int(self.species_to_child[index])
        if target < 0:
            raise ValueError(
                f"species index {index}는 node {self.display_name}의 descendant가 아닙니다"
            )
        return target

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "name": self.name,
            "child_rank": self.child_rank,
            "child_names": list(self.child_names),
            "child_identifiers": list(self.child_identifiers),
            "descendant_species_indices": list(self.descendant_species_indices),
            "species_to_child": list(self.species_to_child),
            "child_to_species_indices": [
                list(indices) for indices in self.child_to_species_indices
            ],
            "num_children": self.num_children,
            "trainable": self.trainable,
            "node_id": self.node_id,
            "display_name": self.display_name,
            "class_to_idx": self.class_to_idx,
        }


def _root_node(taxonomy: Any) -> TaxonomyNodeSpec:
    class_names = list(taxonomy.class_names)
    family_names = list(taxonomy.family_names)
    species_to_genus = [int(value) for value in taxonomy.species_to_genus]
    genus_to_family = [int(value) for value in taxonomy.genus_to_family]

    species_to_child = tuple(
        genus_to_family[genus_index]
        for genus_index in species_to_genus
    )
    child_to_species = tuple(
        tuple(
            species_index
            for species_index, family_index in enumerate(species_to_child)
            if family_index == child_index
        )
        for child_index in range(len(family_names))
    )
    return TaxonomyNodeSpec(
        rank="root",
        name="root",
        child_rank="family",
        child_names=tuple(family_names),
        child_identifiers=tuple(family_names),
        descendant_species_indices=tuple(range(len(class_names))),
        species_to_child=species_to_child,
        child_to_species_indices=child_to_species,
    )


def _family_node(taxonomy: Any, family_index: int) -> TaxonomyNodeSpec:
    class_names = list(taxonomy.class_names)
    genus_names = list(taxonomy.genus_names)
    family_names = list(taxonomy.family_names)
    species_to_genus = [int(value) for value in taxonomy.species_to_genus]
    genus_to_family = [int(value) for value in taxonomy.genus_to_family]

    child_genera = [
        genus_index
        for genus_index, parent_family in enumerate(genus_to_family)
        if parent_family == family_index
    ]
    genus_to_local = {
        genus_index: local_index
        for local_index, genus_index in enumerate(child_genera)
    }
    species_to_child = tuple(
        genus_to_local.get(genus_index, -1)
        for genus_index in species_to_genus
    )
    child_to_species = tuple(
        tuple(
            species_index
            for species_index, local_child in enumerate(species_to_child)
            if local_child == local_index
        )
        for local_index in range(len(child_genera))
    )
    descendants = tuple(
        species_index
        for species_index, local_child in enumerate(species_to_child)
        if local_child >= 0
    )
    child_names = tuple(genus_names[index] for index in child_genera)

    return TaxonomyNodeSpec(
        rank="family",
        name=family_names[family_index],
        child_rank="genus",
        child_names=child_names,
        child_identifiers=child_names,
        descendant_species_indices=descendants,
        species_to_child=species_to_child,
        child_to_species_indices=child_to_species,
    )


def _genus_node(taxonomy: Any, genus_index: int) -> TaxonomyNodeSpec:
    class_names = list(taxonomy.class_names)
    scientific_names = list(taxonomy.scientific_names)
    genus_names = list(taxonomy.genus_names)
    species_to_genus = [int(value) for value in taxonomy.species_to_genus]

    child_species = [
        species_index
        for species_index, parent_genus in enumerate(species_to_genus)
        if parent_genus == genus_index
    ]
    species_to_local = {
        species_index: local_index
        for local_index, species_index in enumerate(child_species)
    }
    species_to_child = tuple(
        species_to_local.get(species_index, -1)
        for species_index in range(len(class_names))
    )
    child_to_species = tuple((species_index,) for species_index in child_species)

    return TaxonomyNodeSpec(
        rank="genus",
        name=genus_names[genus_index],
        child_rank="species",
        child_names=tuple(scientific_names[index] for index in child_species),
        # checkpoint와 ImageFolder 폴더를 연결할 때는 안정적인 folder_name을 쓴다.
        child_identifiers=tuple(class_names[index] for index in child_species),
        descendant_species_indices=tuple(child_species),
        species_to_child=species_to_child,
        child_to_species_indices=child_to_species,
    )


def resolve_taxonomy_node(
    taxonomy: Any,
    rank: str,
    name: str | None = None,
    *,
    require_trainable: bool = True,
) -> TaxonomyNodeSpec:
    """설정의 rank/name으로 원논문식 node-local 문제를 만든다."""
    _validate_taxonomy(taxonomy)
    normalized_rank = str(rank).strip().lower()
    if normalized_rank not in _VALID_NODE_RANKS:
        raise ValueError(
            f"taxonomy_node_rank는 {sorted(_VALID_NODE_RANKS)} 중 하나여야 하지만 "
            f"{rank!r}입니다"
        )

    if normalized_rank == "root":
        node = _root_node(taxonomy)
    elif normalized_rank == "family":
        family_index = _resolve_name(
            name,
            list(taxonomy.family_names),
            rank="family",
        )
        node = _family_node(taxonomy, family_index)
    else:
        genus_index = _resolve_name(
            name,
            list(taxonomy.genus_names),
            rank="genus",
        )
        node = _genus_node(taxonomy, genus_index)

    if require_trainable and not node.trainable:
        raise ValueError(
            f"node {node.display_name}의 직접 자식이 {node.num_children}개뿐이므로 "
            "분류용 Prompt-CAM을 학습할 수 없습니다. singleton node는 추론에서 "
            "확률 1의 결정적 경로로 처리합니다."
        )
    return node


def list_taxonomy_nodes(
    taxonomy: Any,
    *,
    trainable_only: bool = False,
) -> list[TaxonomyNodeSpec]:
    """root, 모든 family, 모든 genus node를 taxonomy 순서대로 반환한다."""
    _validate_taxonomy(taxonomy)
    nodes = [_root_node(taxonomy)]
    nodes.extend(
        _family_node(taxonomy, family_index)
        for family_index in range(len(taxonomy.family_names))
    )
    nodes.extend(
        _genus_node(taxonomy, genus_index)
        for genus_index in range(len(taxonomy.genus_names))
    )
    if trainable_only:
        nodes = [node for node in nodes if node.trainable]
    return nodes


def node_lookup(nodes: Iterable[TaxonomyNodeSpec]) -> dict[str, TaxonomyNodeSpec]:
    """node_id 중복을 검증하며 lookup dictionary를 만든다."""
    result: dict[str, TaxonomyNodeSpec] = {}
    for node in nodes:
        if node.node_id in result:
            raise ValueError(f"taxonomy node_id가 중복되었습니다: {node.node_id}")
        result[node.node_id] = node
    return result
