"""계층 분류 체계 라벨을 지원하는 범용 ImageFolder 로더.

예상 데이터셋 구조::

    data_path/
        train/<class_folder>/*
        val/<class_folder>/*
        test/<class_folder>/*

계층적 Prompt-CAM을 사용하려면 ``taxonomy_manifest``에 최소한 다음 열이 필요하다::

    folder_name,scientific_name,genus,family

클래스 순서는 항상 torchvision ImageFolder가 정렬한 학습 폴더 순서로 고정한다.
검증·테스트 분할도 완전히 같은 대응 관계를 사용해야 한다.
"""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


RANK_TO_INDEX = {
    "species": 0,
    "genus": 1,
    "family": 2,
    "unidentifiable": 3,
}

INCEPTION_MEAN = (0.5, 0.5, 0.5)
INCEPTION_STD = (0.5, 0.5, 0.5)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class Taxonomy:
    """ImageFolder 클래스 순서에 맞춘 분류 체계 인덱스."""

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

    def to_dict(self) -> dict[str, Any]:
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


@dataclass(frozen=True)
class SampleMetadata:
    """선택적인 이미지별 식별 가능 등급과 경계 상자."""

    rank_target: int
    bbox: tuple[float, float, float, float] | None


def _required_text(
    row: Mapping[str, Any],
    key: str,
    *,
    row_number: int,
    file_path: Path,
) -> str:
    value = str(row.get(key, "")).strip()
    if not value:
        raise ValueError(
            f"{file_path}의 {row_number}행에서 {key!r} 값이 비어 있습니다"
        )
    return value


def load_taxonomy_manifest(
    path: str | Path,
    class_names: Sequence[str],
    *,
    class_column: str | None = None,
) -> Taxonomy:
    """분류 체계 행을 읽어 학습 ImageFolder 순서에 맞춘다."""

    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"분류 체계 매니페스트가 존재하지 않습니다: {manifest_path}"
        )

    class_column = class_column or "folder_name"
    required_columns = {
        class_column,
        "scientific_name",
        "genus",
        "family",
    }

    rows_by_class: dict[str, dict[str, str]] = {}
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

        missing_columns = required_columns - set(reader.fieldnames)
        if missing_columns:
            raise ValueError(
                "분류 체계 매니페스트에 필요한 열이 없습니다: "
                f"{sorted(missing_columns)}"
            )

        for row_number, row in enumerate(reader, start=2):
            class_name = _required_text(
                row,
                class_column,
                row_number=row_number,
                file_path=manifest_path,
            )
            if class_name in rows_by_class:
                raise ValueError(
                    f"{manifest_path}에 분류 체계 클래스 {class_name!r}가 "
                    "중복되었습니다"
                )

            rows_by_class[class_name] = {
                "scientific_name": _required_text(
                    row,
                    "scientific_name",
                    row_number=row_number,
                    file_path=manifest_path,
                ),
                "genus": _required_text(
                    row,
                    "genus",
                    row_number=row_number,
                    file_path=manifest_path,
                ),
                "family": _required_text(
                    row,
                    "family",
                    row_number=row_number,
                    file_path=manifest_path,
                ),
            }

    ordered_class_names = [str(name) for name in class_names]
    class_name_set = set(ordered_class_names)

    missing_classes = [
        name for name in ordered_class_names if name not in rows_by_class
    ]
    extra_classes = [
        name for name in rows_by_class if name not in class_name_set
    ]
    if missing_classes or extra_classes:
        raise ValueError(
            "분류 체계와 ImageFolder 클래스가 일치하지 않습니다: "
            f"누락={missing_classes}, 추가={extra_classes}"
        )

    ordered_rows = [rows_by_class[name] for name in ordered_class_names]
    scientific_names = [row["scientific_name"] for row in ordered_rows]
    if len(set(scientific_names)) != len(scientific_names):
        raise ValueError(
            "분류 체계 매니페스트의 scientific_name 값은 고유해야 합니다"
        )

    # ImageFolder 클래스에서 처음 등장한 순서를 유지한다.
    genus_names = list(dict.fromkeys(row["genus"] for row in ordered_rows))
    genus_index = {
        genus_name: index
        for index, genus_name in enumerate(genus_names)
    }

    genus_to_family_name: dict[str, str] = {}
    for row in ordered_rows:
        genus_name = row["genus"]
        family_name = row["family"]
        previous = genus_to_family_name.setdefault(genus_name, family_name)
        if previous != family_name:
            raise ValueError(
                f"속 {genus_name!r}가 여러 과에 대응됩니다: "
                f"{previous!r}, {family_name!r}"
            )

    family_names = list(
        dict.fromkeys(
            genus_to_family_name[genus_name]
            for genus_name in genus_names
        )
    )
    family_index = {
        family_name: index
        for index, family_name in enumerate(family_names)
    }

    species_to_genus = [
        genus_index[row["genus"]]
        for row in ordered_rows
    ]
    genus_to_family = [
        family_index[genus_to_family_name[genus_name]]
        for genus_name in genus_names
    ]
    genus_counts = [
        species_to_genus.count(genus_index_value)
        for genus_index_value in range(len(genus_names))
    ]

    return Taxonomy(
        class_names=ordered_class_names,
        scientific_names=scientific_names,
        genus_names=genus_names,
        family_names=family_names,
        species_to_genus=species_to_genus,
        genus_to_family=genus_to_family,
        genus_counts=genus_counts,
    )


def _normalization(params) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """명시적 또는 이름으로 지정한 RGB 정규화 통계를 확정한다."""

    explicit_mean = getattr(params, "normalization_mean", None)
    explicit_std = getattr(params, "normalization_std", None)

    if (explicit_mean is None) != (explicit_std is None):
        raise ValueError(
            "normalization_mean과 normalization_std는 함께 설정해야 합니다"
        )

    if explicit_mean is not None:
        if len(explicit_mean) != 3 or len(explicit_std) != 3:
            raise ValueError(
                "normalization_mean/std는 각각 정확히 값 3개를 포함해야 합니다"
            )
        mean = tuple(float(value) for value in explicit_mean)
        std = tuple(float(value) for value in explicit_std)
        if any(value <= 0.0 for value in std):
            raise ValueError("normalization_std 값은 양수여야 합니다")
        return mean, std

    name = str(getattr(params, "normalization", "inception")).lower()
    if name == "inception":
        return INCEPTION_MEAN, INCEPTION_STD
    if name == "imagenet":
        return IMAGENET_MEAN, IMAGENET_STD

    raise ValueError(
        f"지원하지 않는 정규화 방식 {name!r}입니다. 'inception' 또는 'imagenet'을 사용하십시오"
    )


def _resolve_dataset_root(params) -> Path:
    """설정된 분할 폴더를 직접 포함하는 루트 경로를 찾는다."""

    configured = Path(str(params.data_path)).expanduser().resolve()
    train_split = str(getattr(params, "train_split", "train"))

    candidates = [configured, configured / "imagefolder"]
    for candidate in candidates:
        if (candidate / train_split).is_dir():
            return candidate

    checked = [str(candidate / train_split) for candidate in candidates]
    raise FileNotFoundError(
        "ImageFolder 학습 분할을 찾지 못했습니다. 확인한 경로: "
        + ", ".join(checked)
    )


def _load_imagefolder(split_path: Path) -> datasets.ImageFolder:
    """변환 없는 ImageFolder를 만들고 비어 있는 분할을 거부한다."""

    if not split_path.is_dir():
        raise FileNotFoundError(f"데이터셋 분할이 존재하지 않습니다: {split_path}")

    dataset = datasets.ImageFolder(str(split_path), transform=None)
    if len(dataset.classes) == 0:
        raise ValueError(f"{split_path}에서 클래스 디렉터리를 찾지 못했습니다")
    if len(dataset.samples) == 0:
        raise ValueError(f"{split_path}에서 이미지를 찾지 못했습니다")
    return dataset


def _assert_same_class_mapping(
    reference: datasets.ImageFolder,
    candidate: datasets.ImageFolder,
    *,
    split_name: str,
) -> None:
    """모든 분할에서 클래스 이름과 인덱스가 같은지 확인한다."""

    if candidate.class_to_idx != reference.class_to_idx:
        raise ValueError(
            f"ImageFolder 클래스 대응이 {split_name!r} 분할에서 다릅니다. "
            f"train={reference.class_to_idx}, "
            f"{split_name}={candidate.class_to_idx}"
        )


def _relative_path_key(value: str | Path) -> str:
    """매니페스트와 파일 시스템 경로를 이식 가능한 POSIX 키로 정규화한다."""

    text = str(value).strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return Path(text).as_posix().lstrip("/")


def _optional_float(row: Mapping[str, Any], key: str) -> float | None:
    value = str(row.get(key, "")).strip()
    if not value:
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{key}에 유한하지 않은 값이 있습니다: {value!r}")
    return result


def load_identifiability_manifest(
    path: str | Path | None,
) -> dict[str, SampleMetadata]:
    """상대 경로로 색인된 선택적 등급과 경계 상자 주석을 불러온다."""

    if path in (None, "", "null"):
        return {}

    manifest_path = Path(str(path)).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"식별 가능성 매니페스트가 존재하지 않습니다: {manifest_path}"
        )

    result: dict[str, SampleMetadata] = {}
    bbox_columns = ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")

    with manifest_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(
                f"식별 가능성 매니페스트에 헤더가 없습니다: {manifest_path}"
            )

        required = {"relative_path", "identifiable_rank"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(
                "식별 가능성 매니페스트에 필요한 열이 없습니다: "
                f"{sorted(missing)}"
            )

        for row_number, row in enumerate(reader, start=2):
            key = _relative_path_key(
                _required_text(
                    row,
                    "relative_path",
                    row_number=row_number,
                    file_path=manifest_path,
                )
            )
            if key in result:
                raise ValueError(
                    f"{manifest_path}에서 relative_path {key!r}가 중복되었습니다"
                )

            rank_name = _required_text(
                row,
                "identifiable_rank",
                row_number=row_number,
                file_path=manifest_path,
            ).lower()
            if rank_name not in RANK_TO_INDEX:
                raise ValueError(
                    f"{row_number}행의 identifiable_rank {rank_name!r}는 지원하지 않습니다. "
                    "다음 값 중 하나가 필요합니다: "
                    f"{sorted(RANK_TO_INDEX)}"
                )

            bbox_values = tuple(
                _optional_float(row, column)
                for column in bbox_columns
            )
            nonempty_count = sum(value is not None for value in bbox_values)
            if nonempty_count not in (0, 4):
                raise ValueError(
                    f"{manifest_path}의 {row_number}행에서 bbox 열 4개를 모두 "
                    "함께 입력해야 합니다"
                )

            bbox = None
            if nonempty_count == 4:
                bbox = tuple(float(value) for value in bbox_values)  # type: ignore[arg-type]
                x1, y1, x2, y2 = bbox
                if not (x2 > x1 and y2 > y1):
                    raise ValueError(
                        f"{row_number}행의 bbox가 잘못되었습니다: {bbox}"
                    )

            result[key] = SampleMetadata(
                rank_target=RANK_TO_INDEX[rank_name],
                bbox=bbox,
            )

    return result


class JointImageTransform:
    """이미지 증강과 동일한 기하 변환을 경계 상자에도 적용한다."""

    def __init__(self, params, *, training: bool) -> None:
        self.training = bool(training)
        self.crop_size = int(params.crop_size)
        if self.crop_size <= 0:
            raise ValueError("crop_size는 양수여야 합니다")

        self.eval_resize_size = int(
            getattr(params, "eval_resize_size", None)
            or round(self.crop_size / 0.875)
        )
        if self.eval_resize_size < self.crop_size:
            raise ValueError(
                "eval_resize_size는 crop_size 이상이어야 합니다"
            )

        self.scale = (
            float(getattr(params, "train_scale_min", 0.5)),
            float(getattr(params, "train_scale_max", 1.0)),
        )
        self.ratio = (
            float(getattr(params, "train_ratio_min", 0.75)),
            float(getattr(params, "train_ratio_max", 4.0 / 3.0)),
        )
        if not (0.0 < self.scale[0] <= self.scale[1]):
            raise ValueError(f"학습 크기 범위가 잘못되었습니다: {self.scale}")
        if not (0.0 < self.ratio[0] <= self.ratio[1]):
            raise ValueError(f"학습 종횡비 범위가 잘못되었습니다: {self.ratio}")

        self.mean, self.std = _normalization(params)
        self.interpolation = InterpolationMode.BICUBIC

    @staticmethod
    def _bbox_to_pixels(
        bbox: tuple[float, float, float, float] | None,
        *,
        width: int,
        height: int,
        coordinate_mode: str,
    ) -> torch.Tensor | None:
        if bbox is None:
            return None

        result = torch.tensor(bbox, dtype=torch.float32)
        if coordinate_mode == "normalized":
            result = result * torch.tensor(
                [width, height, width, height],
                dtype=torch.float32,
            )
        elif coordinate_mode != "pixels":
            raise ValueError(
                f"지원하지 않는 bbox_coordinate_mode입니다: {coordinate_mode!r}"
            )

        result[0::2].clamp_(0.0, float(width))
        result[1::2].clamp_(0.0, float(height))
        if result[2] <= result[0] or result[3] <= result[1]:
            return None
        return result

    @staticmethod
    def _crop_bbox(
        bbox: torch.Tensor | None,
        *,
        left: int,
        top: int,
        crop_width: int,
        crop_height: int,
    ) -> torch.Tensor | None:
        if bbox is None:
            return None

        result = bbox.clone()
        result[0::2] -= float(left)
        result[1::2] -= float(top)
        result[0::2].clamp_(0.0, float(crop_width))
        result[1::2].clamp_(0.0, float(crop_height))
        if result[2] <= result[0] or result[3] <= result[1]:
            return None
        return result

    @staticmethod
    def _resize_bbox(
        bbox: torch.Tensor | None,
        *,
        source_width: int,
        source_height: int,
        target_width: int,
        target_height: int,
    ) -> torch.Tensor | None:
        if bbox is None:
            return None

        result = bbox.clone()
        result[0::2] *= float(target_width) / float(source_width)
        result[1::2] *= float(target_height) / float(source_height)
        return result

    @staticmethod
    def _horizontal_flip_bbox(
        bbox: torch.Tensor | None,
        *,
        width: int,
    ) -> torch.Tensor | None:
        if bbox is None:
            return None

        result = bbox.clone()
        x1 = result[0].clone()
        x2 = result[2].clone()
        result[0] = float(width) - x2
        result[2] = float(width) - x1
        return result

    @staticmethod
    def _normalize_bbox(
        bbox: torch.Tensor | None,
        *,
        width: int,
        height: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if bbox is None:
            return (
                torch.zeros(4, dtype=torch.float32),
                torch.tensor(False, dtype=torch.bool),
            )

        result = bbox.clone()
        result[0::2] /= float(width)
        result[1::2] /= float(height)
        result.clamp_(0.0, 1.0)

        valid = bool(result[2] > result[0] and result[3] > result[1])
        if not valid:
            return (
                torch.zeros(4, dtype=torch.float32),
                torch.tensor(False, dtype=torch.bool),
            )
        return result, torch.tensor(True, dtype=torch.bool)

    def __call__(
        self,
        image: Image.Image,
        *,
        bbox: tuple[float, float, float, float] | None,
        bbox_coordinate_mode: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image = image.convert("RGB")
        original_width, original_height = image.size
        transformed_bbox = self._bbox_to_pixels(
            bbox,
            width=original_width,
            height=original_height,
            coordinate_mode=bbox_coordinate_mode,
        )

        if self.training:
            # torchvision은 동일한 매개변수 표본 추출기를 정적 메서드로 제공한다.
            from torchvision.transforms import RandomResizedCrop

            top, left, crop_height, crop_width = RandomResizedCrop.get_params(
                image,
                scale=self.scale,
                ratio=self.ratio,
            )

            image = TF.resized_crop(
                image,
                top=top,
                left=left,
                height=crop_height,
                width=crop_width,
                size=[self.crop_size, self.crop_size],
                interpolation=self.interpolation,
                antialias=True,
            )
            transformed_bbox = self._crop_bbox(
                transformed_bbox,
                left=left,
                top=top,
                crop_width=crop_width,
                crop_height=crop_height,
            )
            transformed_bbox = self._resize_bbox(
                transformed_bbox,
                source_width=crop_width,
                source_height=crop_height,
                target_width=self.crop_size,
                target_height=self.crop_size,
            )

            if random.random() < 0.5:
                image = TF.hflip(image)
                transformed_bbox = self._horizontal_flip_bbox(
                    transformed_bbox,
                    width=self.crop_size,
                )
        else:
            image = TF.resize(
                image,
                size=self.eval_resize_size,
                interpolation=self.interpolation,
                antialias=True,
            )
            resized_width, resized_height = image.size
            transformed_bbox = self._resize_bbox(
                transformed_bbox,
                source_width=original_width,
                source_height=original_height,
                target_width=resized_width,
                target_height=resized_height,
            )

            top = max(0, int(round((resized_height - self.crop_size) / 2.0)))
            left = max(0, int(round((resized_width - self.crop_size) / 2.0)))
            image = TF.center_crop(
                image,
                output_size=[self.crop_size, self.crop_size],
            )
            transformed_bbox = self._crop_bbox(
                transformed_bbox,
                left=left,
                top=top,
                crop_width=self.crop_size,
                crop_height=self.crop_size,
            )

        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(
            image_tensor,
            mean=self.mean,
            std=self.std,
        )
        bbox_tensor, bbox_valid = self._normalize_bbox(
            transformed_bbox,
            width=self.crop_size,
            height=self.crop_size,
        )
        return image_tensor, bbox_tensor, bbox_valid


class FlatImageFolder(Dataset):
    """기존 ``(image, target)`` 형식을 반환하는 ImageFolder 호환 데이터셋."""

    def __init__(
        self,
        base_dataset: datasets.ImageFolder,
        transform: JointImageTransform,
    ) -> None:
        self.base_dataset = base_dataset
        self.transform = transform

        self.classes = base_dataset.classes
        self.class_to_idx = base_dataset.class_to_idx
        self.samples = base_dataset.samples
        self.imgs = base_dataset.imgs
        self.targets = base_dataset.targets
        self.loader = base_dataset.loader

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, target = self.samples[index]
        image = self.loader(path)
        image_tensor, _, _ = self.transform(
            image,
            bbox=None,
            bbox_coordinate_mode="normalized",
        )
        return image_tensor, int(target)


class IndependentImageFolder(Dataset):
    """Taxonomy node의 descendant만 남기고 직접 자식 label로 재매핑한다."""

    def __init__(
        self,
        base_dataset: datasets.ImageFolder,
        transform: JointImageTransform,
        *,
        node,
    ) -> None:
        self.base_dataset = base_dataset
        self.transform = transform
        self.node = node
        self.classes = list(node.child_identifiers)
        self.class_to_idx = dict(node.class_to_idx)
        self.loader = base_dataset.loader

        samples: list[tuple[str, int]] = []
        original_targets: list[int] = []
        for path, original_target in base_dataset.samples:
            local_target = int(node.species_to_child[int(original_target)])
            if local_target < 0:
                continue
            samples.append((path, local_target))
            original_targets.append(int(original_target))

        if not samples:
            raise ValueError(
                f"node {node.display_name}에 속하는 이미지가 없습니다"
            )

        self.samples = samples
        self.imgs = samples
        self.targets = [target for _, target in samples]
        self.original_targets = original_targets

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, target = self.samples[index]
        image = self.loader(path)
        image_tensor, _, _ = self.transform(
            image,
            bbox=None,
            bbox_coordinate_mode="normalized",
        )
        return image_tensor, int(target)


class HierarchicalImageFolder(Dataset):
    """계층 손실에 필요한 모든 라벨을 반환하는 ImageFolder 데이터셋."""

    def __init__(
        self,
        base_dataset: datasets.ImageFolder,
        transform: JointImageTransform,
        *,
        taxonomy: Taxonomy,
        dataset_root: Path,
        split_root: Path,
        metadata: Mapping[str, SampleMetadata],
        identifiability_enabled: bool,
        missing_identifiability_policy: str,
        bbox_coordinate_mode: str,
    ) -> None:
        self.base_dataset = base_dataset
        self.transform = transform
        self.taxonomy = taxonomy
        self.dataset_root = dataset_root.resolve()
        self.split_root = split_root.resolve()
        self.metadata = dict(metadata)
        self.identifiability_enabled = bool(identifiability_enabled)
        self.missing_identifiability_policy = str(
            missing_identifiability_policy
        ).lower()
        self.bbox_coordinate_mode = str(bbox_coordinate_mode).lower()

        if self.missing_identifiability_policy not in {"species", "error"}:
            raise ValueError(
                "missing_identifiability_policy는 'species' 또는 'error'여야 합니다"
            )
        if self.bbox_coordinate_mode not in {"normalized", "pixels"}:
            raise ValueError(
                "bbox_coordinate_mode는 'normalized' 또는 'pixels'여야 합니다"
            )

        self.classes = base_dataset.classes
        self.class_to_idx = base_dataset.class_to_idx
        self.samples = base_dataset.samples
        self.imgs = base_dataset.imgs
        self.targets = base_dataset.targets
        self.loader = base_dataset.loader

    def __len__(self) -> int:
        return len(self.samples)

    def _metadata_for_path(self, image_path: Path) -> SampleMetadata:
        dataset_relative = _relative_path_key(
            image_path.resolve().relative_to(self.dataset_root)
        )
        split_relative = _relative_path_key(
            image_path.resolve().relative_to(self.split_root)
        )

        metadata = self.metadata.get(dataset_relative)
        if metadata is None:
            metadata = self.metadata.get(split_relative)

        if metadata is not None:
            return metadata

        if self.identifiability_enabled and self.missing_identifiability_policy == "error":
            raise KeyError(
                "이미지의 식별 가능성 행이 없습니다. 허용되는 상대 경로 형식은 "
                f"{dataset_relative!r} 또는 {split_relative!r}입니다."
            )

        return SampleMetadata(
            rank_target=RANK_TO_INDEX["species"],
            bbox=None,
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        path_string, species_target = self.samples[index]
        image_path = Path(path_string)
        image = self.loader(path_string)

        metadata = self._metadata_for_path(image_path)
        image_tensor, bbox_tensor, bbox_valid = self.transform(
            image,
            bbox=metadata.bbox,
            bbox_coordinate_mode=self.bbox_coordinate_mode,
        )

        genus_target = self.taxonomy.species_to_genus[species_target]
        family_target = self.taxonomy.genus_to_family[genus_target]
        relative_path = _relative_path_key(
            image_path.resolve().relative_to(self.dataset_root)
        )

        return {
            "image": image_tensor,
            "species_target": torch.tensor(
                species_target,
                dtype=torch.long,
            ),
            "genus_target": torch.tensor(
                genus_target,
                dtype=torch.long,
            ),
            "family_target": torch.tensor(
                family_target,
                dtype=torch.long,
            ),
            "rank_target": torch.tensor(
                metadata.rank_target,
                dtype=torch.long,
            ),
            "bbox": bbox_tensor,
            "bbox_valid": bbox_valid,
            "relative_path": relative_path,
        }


def _build_split_dataset(
    base_dataset: datasets.ImageFolder | None,
    *,
    split_root: Path | None,
    params,
    training: bool,
    taxonomy: Taxonomy | None,
    dataset_root: Path,
    metadata: Mapping[str, SampleMetadata],
    taxonomy_node=None,
) -> Dataset | None:
    if base_dataset is None or split_root is None:
        return None

    transform = JointImageTransform(params, training=training)
    if taxonomy_node is not None:
        return IndependentImageFolder(
            base_dataset,
            transform,
            node=taxonomy_node,
        )
    if taxonomy is None:
        return FlatImageFolder(base_dataset, transform)

    return HierarchicalImageFolder(
        base_dataset,
        transform,
        taxonomy=taxonomy,
        dataset_root=dataset_root,
        split_root=split_root,
        metadata=metadata,
        identifiability_enabled=bool(
            getattr(params, "identifiability_enabled", False)
        ),
        missing_identifiability_policy=str(
            getattr(params, "missing_identifiability_policy", "species")
        ),
        bbox_coordinate_mode=str(
            getattr(params, "bbox_coordinate_mode", "normalized")
        ),
    )


def get_imagefolder(params, mode: str = "splits"):
    """학습·검증·테스트 데이터셋을 만들고 계층 매개변수를 확정한다.

    매개변수
    --------
    params:
        프로젝트의 나머지 부분에서 사용하는 argparse/YAML 네임스페이스.
    mode:
        ``"splits"``는 ``(train, val, test)``를 반환한다. ``"train"``,
        ``"val"``, ``"test"``는 직접 확인할 데이터셋 하나를 반환한다.
    """

    dataset_root = _resolve_dataset_root(params)
    train_split_name = str(getattr(params, "train_split", "train"))
    val_split_name = str(getattr(params, "val_split", "val"))
    test_split_name = str(getattr(params, "test_split", "test"))

    train_root = dataset_root / train_split_name
    val_root = dataset_root / val_split_name
    test_root = dataset_root / test_split_name

    train_base = _load_imagefolder(train_root)
    val_base = _load_imagefolder(val_root) if val_root.is_dir() else None
    test_base = _load_imagefolder(test_root) if test_root.is_dir() else None

    if val_base is not None:
        _assert_same_class_mapping(
            train_base,
            val_base,
            split_name=val_split_name,
        )
    if test_base is not None:
        _assert_same_class_mapping(
            train_base,
            test_base,
            split_name=test_split_name,
        )

    actual_class_num = len(train_base.classes)
    original_taxonomy = bool(
        getattr(params, "original_taxonomy_prompt", False)
    )
    configured_class_num = int(getattr(params, "class_num", 0) or 0)
    if not original_taxonomy and configured_class_num not in (0, actual_class_num):
        raise ValueError(
            f"설정된 class_num={configured_class_num}이지만 학습 분할에는 "
            f"클래스가 {actual_class_num}개 있습니다"
        )

    params.class_num = actual_class_num
    params.class_names = list(train_base.classes)
    params.class_to_idx = dict(train_base.class_to_idx)
    params.resolved_data_path = str(dataset_root)

    hierarchical = bool(getattr(params, "hierarchical_prompt", False))
    taxonomy: Taxonomy | None = None
    taxonomy_node = None

    if hierarchical or original_taxonomy:
        taxonomy_manifest = getattr(params, "taxonomy_manifest", None)
        if taxonomy_manifest in (None, "", "null"):
            raise ValueError(
                "계층 또는 독립 taxonomy 모델에는 taxonomy_manifest가 필요합니다"
            )

        taxonomy = load_taxonomy_manifest(
            taxonomy_manifest,
            train_base.classes,
            class_column=getattr(
                params,
                "taxonomy_class_column",
                None,
            ),
        )

        params.num_genera = taxonomy.num_genera
        params.num_families = taxonomy.num_families
        params.species_to_genus = list(taxonomy.species_to_genus)
        params.genus_to_family = list(taxonomy.genus_to_family)
        params.genus_counts = list(taxonomy.genus_counts)
        params.genus_names = list(taxonomy.genus_names)
        params.family_names = list(taxonomy.family_names)
        params.scientific_names = list(taxonomy.scientific_names)
        params.taxonomy = taxonomy.to_dict()

    if hierarchical:
        expected_prompt_count = taxonomy.prompt_count
        configured_vpt_num = int(getattr(params, "vpt_num", 0) or 0)
        if configured_vpt_num not in (0, expected_prompt_count):
            raise ValueError(
                "계층적 Prompt-CAM은 vpt_num=F+G+C="
                f"{expected_prompt_count}가 필요하지만 {configured_vpt_num}입니다"
            )
        params.vpt_num = expected_prompt_count
    elif original_taxonomy:
        from data.original_taxonomy import resolve_taxonomy_node

        taxonomy_node = resolve_taxonomy_node(
            taxonomy,
            str(getattr(params, "taxonomy_node_rank", "root")),
            getattr(params, "taxonomy_node_name", None),
        )
        if configured_class_num not in (0, taxonomy_node.num_children):
            raise ValueError(
                f"node {taxonomy_node.display_name}의 class_num은 "
                f"{taxonomy_node.num_children}이어야 하지만 {configured_class_num}입니다"
            )
        params.full_class_num = actual_class_num
        params.full_class_names = list(train_base.classes)
        params.class_num = taxonomy_node.num_children
        params.vpt_num = taxonomy_node.num_children
        params.class_names = list(taxonomy_node.child_identifiers)
        params.class_to_idx = dict(taxonomy_node.class_to_idx)
        params.taxonomy_node_id = taxonomy_node.node_id
        params.taxonomy_node = taxonomy_node.to_dict()
    else:
        configured_vpt_num = int(
            getattr(params, "vpt_num", actual_class_num)
            or actual_class_num
        )
        if (
            str(getattr(params, "train_type", "")) == "prompt_cam"
            and configured_vpt_num != actual_class_num
        ):
            raise ValueError(
                "평면 Prompt-CAM은 vpt_num == class_num을 요구합니다. "
                f"현재 vpt_num={configured_vpt_num}, "
                f"class_num={actual_class_num}"
            )
        params.vpt_num = configured_vpt_num

    metadata: dict[str, SampleMetadata] = {}
    if hierarchical and bool(
        getattr(params, "identifiability_enabled", False)
    ):
        metadata = load_identifiability_manifest(
            getattr(params, "identifiability_manifest", None)
        )
    elif hierarchical:
        optional_manifest = getattr(
            params,
            "identifiability_manifest",
            None,
        )
        if optional_manifest not in (None, "", "null"):
            metadata = load_identifiability_manifest(optional_manifest)

    train_dataset = _build_split_dataset(
        train_base,
        split_root=train_root,
        params=params,
        training=True,
        taxonomy=taxonomy,
        dataset_root=dataset_root,
        metadata=metadata,
        taxonomy_node=taxonomy_node,
    )
    val_dataset = _build_split_dataset(
        val_base,
        split_root=val_root if val_base is not None else None,
        params=params,
        training=False,
        taxonomy=taxonomy,
        dataset_root=dataset_root,
        metadata=metadata,
        taxonomy_node=taxonomy_node,
    )
    test_dataset = _build_split_dataset(
        test_base,
        split_root=test_root if test_base is not None else None,
        params=params,
        training=False,
        taxonomy=taxonomy,
        dataset_root=dataset_root,
        metadata=metadata,
        taxonomy_node=taxonomy_node,
    )

    if taxonomy_node is not None:
        params.taxonomy_node_split_counts = {}
        for split_name, dataset in (
            ("train", train_dataset),
            ("val", val_dataset),
            ("test", test_dataset),
        ):
            if dataset is None:
                continue
            counts = [0] * taxonomy_node.num_children
            for target in dataset.targets:
                counts[int(target)] += 1
            params.taxonomy_node_split_counts[split_name] = counts

    normalized_mode = str(mode).lower()
    if normalized_mode in {"splits", "all"}:
        return train_dataset, val_dataset, test_dataset
    if normalized_mode == "train":
        return train_dataset
    if normalized_mode in {"val", "validation"}:
        return val_dataset
    if normalized_mode == "test":
        return test_dataset

    raise ValueError(
        f"지원하지 않는 ImageFolder 모드 {mode!r}입니다. splits/train/val/test를 사용하십시오"
    )
