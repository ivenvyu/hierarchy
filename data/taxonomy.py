from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class Taxonomy:
    class_names: list[str]
    scientific_names: list[str]
    genus_names: list[str]
    family_names: list[str]
    species_to_genus: list[int]
    genus_to_family: list[int]
    genus_counts: list[int]

    @property
    def num_species(self) -> int:
        return len(self.class_names)

    @property
    def num_genera(self) -> int:
        return len(self.genus_names)

    @property
    def num_families(self) -> int:
        return len(self.family_names)

    @property
    def prompt_count(self) -> int:
        return self.num_families + self.num_genera + self.num_species

    def to_dict(self) -> dict:
        return {
            "class_names": list(self.class_names),
            "scientific_names": list(self.scientific_names),
            "genus_names": list(self.genus_names),
            "family_names": list(self.family_names),
            "species_to_genus": list(self.species_to_genus),
            "genus_to_family": list(self.genus_to_family),
            "genus_counts": list(self.genus_counts),
            "num_species": self.num_species,
            "num_genera": self.num_genera,
            "num_families": self.num_families,
            "prompt_count": self.prompt_count,
        }


def _clean(value: object, *, field: str, row_number: int) -> str:
    result = str(value).strip()

    if not result:
        raise ValueError(
            f"분류 체계 값이 비어 있습니다: 필드={field}, 행={row_number}"
        )

    return result


def load_taxonomy_manifest(
    path: str | Path,
    class_names: Sequence[str],
    *,
    class_column: str | None = None,
) -> Taxonomy:
    manifest_path = Path(path).expanduser().resolve()

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"분류 체계 매니페스트가 존재하지 않습니다: {manifest_path}"
        )

    class_column = class_column or "folder_name"

    with manifest_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames is None:
            raise ValueError(
                f"분류 체계 매니페스트에 헤더가 없습니다: {manifest_path}"
            )

        required = {
            class_column,
            "scientific_name",
            "genus",
            "family",
        }
        missing = required - set(reader.fieldnames)

        if missing:
            raise ValueError(
                f"분류 체계 매니페스트에 필요한 열이 없습니다: {sorted(missing)}"
            )

        rows_by_class: dict[str, dict[str, str]] = {}

        for row_number, row in enumerate(reader, start=2):
            class_name = _clean(
                row[class_column],
                field=class_column,
                row_number=row_number,
            )

            if class_name in rows_by_class:
                raise ValueError(
                    f"분류 체계 클래스가 중복되었습니다: {class_name}"
                )

            rows_by_class[class_name] = {
                "scientific_name": _clean(
                    row["scientific_name"],
                    field="scientific_name",
                    row_number=row_number,
                ),
                "genus": _clean(
                    row["genus"],
                    field="genus",
                    row_number=row_number,
                ),
                "family": _clean(
                    row["family"],
                    field="family",
                    row_number=row_number,
                ),
            }

    ordered_class_names = [str(name) for name in class_names]

    missing_classes = [
        name
        for name in ordered_class_names
        if name not in rows_by_class
    ]
    extra_classes = [
        name
        for name in rows_by_class
        if name not in set(ordered_class_names)
    ]

    if missing_classes or extra_classes:
        raise ValueError(
            "분류 체계와 ImageFolder 클래스가 일치하지 않습니다: "
            f"누락={missing_classes}, 추가={extra_classes}"
        )

    rows = [
        rows_by_class[class_name]
        for class_name in ordered_class_names
    ]

    scientific_names = [
        row["scientific_name"]
        for row in rows
    ]

    if len(set(scientific_names)) != len(scientific_names):
        raise ValueError(
            "scientific_name 값은 고유해야 합니다"
        )

    genus_names = list(
        dict.fromkeys(
            row["genus"]
            for row in rows
        )
    )
    genus_index = {
        name: index
        for index, name in enumerate(genus_names)
    }

    genus_family: dict[str, str] = {}

    for row in rows:
        genus = row["genus"]
        family = row["family"]

        previous = genus_family.setdefault(genus, family)

        if previous != family:
            raise ValueError(
                f"속 {genus}가 여러 과에 대응됩니다: "
                f"{previous}, {family}"
            )

    family_names = list(
        dict.fromkeys(
            genus_family[genus]
            for genus in genus_names
        )
    )
    family_index = {
        name: index
        for index, name in enumerate(family_names)
    }

    species_to_genus = [
        genus_index[row["genus"]]
        for row in rows
    ]

    genus_to_family = [
        family_index[genus_family[genus]]
        for genus in genus_names
    ]

    genus_counts = [
        species_to_genus.count(genus_id)
        for genus_id in range(len(genus_names))
    ]

    taxonomy = Taxonomy(
        class_names=ordered_class_names,
        scientific_names=scientific_names,
        genus_names=genus_names,
        family_names=family_names,
        species_to_genus=species_to_genus,
        genus_to_family=genus_to_family,
        genus_counts=genus_counts,
    )

    if taxonomy.num_species != 30:
        raise ValueError(
            f"종 30개가 필요하지만 {taxonomy.num_species}개를 찾았습니다"
        )

    return taxonomy
