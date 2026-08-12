#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
식물 객체 hotspot vs 배경 hotspot 가림에 따른 confidence drop을 측정한다.

핵심 아이디어
-------------
각 (image, model, rank)에 대해 사용자가 지정한 두 점

- object point: 식물 객체 위의 red hotspot
- background point: 배경 위의 red hotspot

을 중심으로 동일한 크기의 정사각형 영역을 가리고,
원본 confidence와의 감소량을 비교한다.

지원 모델
---------
1) Flat patch-only Prompt-CAM
   - Species rank
2) Shared hierarchical patch-only Prompt-CAM
   - Family / Genus / Species rank
3) Independent taxonomy patch-only Prompt-CAM (원논문식 node 분리 학습)
   - Family / Genus / Species rank

입력 annotation CSV 형식 (long format)
-------------------------------------
필수 열:
- image_path
- species
- species_index            (없으면 image_path의 부모 폴더명으로 보완 가능)
- model                    (Flat / Shared / Independent)
- rank                     (Family / Genus / Species)
- object_x
- object_y
- background_x
- background_y

선택 열:
- notes
- cam_quality

좌표는 기본적으로 원본 이미지의 pixel 좌표로 해석한다.
--coord-mode normalized 를 쓰면 [0,1] 범위 정규화 좌표로 해석한다.

출력
----
output_dir 아래에 다음 파일 생성:
- occlusion_results.csv          : 각 샘플별 결과
- occlusion_group_summary.csv    : model/rank별 요약
- occlusion_report.md            : 간단한 markdown 보고서

예시
----
python -m evaluation.cam.occlusion template \
  --wide-csv output/evaluation/cam_quality_filled.csv \
  --output-csv output/evaluation/occlusion_template.csv

python -m evaluation.cam.occlusion analyze \
  --annotation-csv output/evaluation/occlusion_points.csv \
  --flat-run-dir output/flat/<run> \
  --shared-run-dir output/shared/<run> \
  --independent-run-root output/independent/runs/<run> \
  --output-dir output/evaluation/cam_occlusion \
  --patch-size 61 \
  --device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

import pandas as pd
import torch
import yaml
from PIL import Image, ImageFilter
from torchvision import datasets
from torchvision.transforms import functional as TF


# ---------------------------------------------------------------------------
# 경로 / import 유틸
# ---------------------------------------------------------------------------


def _resolve(path: str | os.PathLike[str]) -> Path:
    return Path(path).expanduser().resolve()


def _ensure_project_on_syspath(project_root: Path) -> None:
    project_root = project_root.resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


# ---------------------------------------------------------------------------
# 공통 표기
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SharedTaxonomyInfo:
    class_names: tuple[str, ...]
    scientific_names: tuple[str, ...]
    genus_names: tuple[str, ...]
    family_names: tuple[str, ...]
    species_to_genus: tuple[int, ...]
    genus_to_family: tuple[int, ...]

    @property
    def species_to_family(self) -> tuple[int, ...]:
        return tuple(self.genus_to_family[g] for g in self.species_to_genus)


@dataclass(frozen=True)
class FlatTaxonomyInfo:
    class_names: tuple[str, ...]
    genus_names: tuple[str, ...]
    family_names: tuple[str, ...]
    species_to_genus: tuple[int, ...]
    genus_to_family: tuple[int, ...]

    @property
    def species_to_family(self) -> tuple[int, ...]:
        return tuple(self.genus_to_family[g] for g in self.species_to_genus)


@dataclass(frozen=True)
class ScoreResult:
    confidence: float
    logit: float
    predicted_index: int
    predicted_label: str
    extras: dict[str, float]


# ---------------------------------------------------------------------------
# 이미지 가림
# ---------------------------------------------------------------------------


def _convert_point(
    x: float,
    y: float,
    width: int,
    height: int,
    coord_mode: str,
) -> tuple[int, int]:
    if coord_mode == "normalized":
        px = int(round(float(x) * (width - 1)))
        py = int(round(float(y) * (height - 1)))
    elif coord_mode == "pixel":
        px = int(round(float(x)))
        py = int(round(float(y)))
    else:
        raise ValueError(
            f"_convert_point에서는 지원하지 않는 coord_mode: {coord_mode}"
        )

    px = max(0, min(width - 1, px))
    py = max(0, min(height - 1, py))
    return px, py


def _crop_normalized_geometry(
    image: Image.Image,
    *,
    x: float,
    y: float,
    patch_size: int,
    transform,
) -> tuple[int, int, int, int]:
    """
    CAM에 표시된 eval center-crop 좌표를 원본 이미지 좌표로 역변환한다.

    x,y:
        최종 crop(예: 336x336) 기준 [0,1] 좌표.

    patch_size:
        최종 crop 공간에서의 가림 크기(pixel).

    반환:
        원본 이미지 공간의
        (center_x, center_y, patch_width, patch_height)
    """
    if transform is None:
        raise ValueError(
            "crop-normalized 좌표에는 scorer.transform이 필요합니다."
        )

    image = image.convert("RGB")
    original_width, original_height = image.size

    crop_size = int(transform.crop_size)
    eval_resize_size = int(transform.eval_resize_size)

    # 실제 evaluation transform과 동일한 resize를 수행해
    # 정확한 resized geometry를 얻는다.
    resized = TF.resize(
        image,
        size=eval_resize_size,
        interpolation=transform.interpolation,
        antialias=True,
    )

    resized_width, resized_height = resized.size

    top = max(
        0,
        int(round((resized_height - crop_size) / 2.0)),
    )
    left = max(
        0,
        int(round((resized_width - crop_size) / 2.0)),
    )

    # CAM 좌표는 최종 crop 기준.
    crop_x = float(x) * float(crop_size - 1)
    crop_y = float(y) * float(crop_size - 1)

    resized_x = left + crop_x
    resized_y = top + crop_y

    scale_x = (
        float(original_width)
        / float(resized_width)
    )
    scale_y = (
        float(original_height)
        / float(resized_height)
    )

    original_x = int(round(resized_x * scale_x))
    original_y = int(round(resized_y * scale_y))

    # crop에서 동일한 patch_size가 되도록 원본 공간 크기로 환산.
    patch_width = max(
        1,
        int(round(float(patch_size) * scale_x)),
    )
    patch_height = max(
        1,
        int(round(float(patch_size) * scale_y)),
    )

    original_x = max(
        0,
        min(original_width - 1, original_x),
    )
    original_y = max(
        0,
        min(original_height - 1, original_y),
    )

    return (
        original_x,
        original_y,
        patch_width,
        patch_height,
    )


def _occlude_square(
    image: Image.Image,
    *,
    x: float,
    y: float,
    patch_size: int,
    coord_mode: str,
    fill_mode: str,
    transform=None,
) -> Image.Image:
    """
    지정 위치를 가린다.

    pixel:
        원본 이미지 pixel 좌표.

    normalized:
        원본 이미지 기준 [0,1] 좌표.

    crop-normalized:
        모델 evaluation center crop 기준 [0,1] 좌표.
        CAM에서 직접 읽은 좌표에는 이 모드를 사용한다.
    """
    image = image.convert("RGB")
    width, height = image.size

    if coord_mode == "crop-normalized":
        (
            cx,
            cy,
            patch_width,
            patch_height,
        ) = _crop_normalized_geometry(
            image,
            x=x,
            y=y,
            patch_size=patch_size,
            transform=transform,
        )
    else:
        cx, cy = _convert_point(
            x,
            y,
            width,
            height,
            coord_mode,
        )
        patch_width = int(patch_size)
        patch_height = int(patch_size)

    half_w = patch_width // 2
    half_h = patch_height // 2

    left = max(0, cx - half_w)
    upper = max(0, cy - half_h)

    right = min(
        width,
        left + patch_width,
    )
    lower = min(
        height,
        upper + patch_height,
    )

    left = max(
        0,
        right - patch_width,
    )
    upper = max(
        0,
        lower - patch_height,
    )

    result = image.copy()
    patch = result.crop(
        (left, upper, right, lower)
    )

    if fill_mode == "mean":
        tensor = TF.to_tensor(result)
        mean_rgb = tuple(
            int(round(float(v) * 255.0))
            for v in tensor.mean(
                dim=(1, 2)
            ).tolist()
        )

        fill = Image.new(
            "RGB",
            (right - left, lower - upper),
            color=mean_rgb,
        )
        result.paste(
            fill,
            (left, upper, right, lower),
        )

    elif fill_mode == "black":
        fill = Image.new(
            "RGB",
            (right - left, lower - upper),
            color=(0, 0, 0),
        )
        result.paste(
            fill,
            (left, upper, right, lower),
        )

    elif fill_mode == "gray":
        fill = Image.new(
            "RGB",
            (right - left, lower - upper),
            color=(127, 127, 127),
        )
        result.paste(
            fill,
            (left, upper, right, lower),
        )

    elif fill_mode == "blur":
        blurred = patch.filter(
            ImageFilter.GaussianBlur(
                radius=max(
                    3,
                    patch_size // 8,
                )
            )
        )
        result.paste(
            blurred,
            (left, upper, right, lower),
        )

    else:
        raise ValueError(
            f"지원하지 않는 fill_mode: {fill_mode}"
        )

    return result


def _extract_logits_local(output):
    """Flat/Independent Prompt-CAM 출력에서 [B,C] logits를 안전하게 꺼낸다."""
    if isinstance(output, tuple):
        output = output[0]

    if isinstance(output, Mapping):
        for key in (
            "logits",
            "species_logits",
            "output",
            "predictions",
        ):
            if key in output:
                output = output[key]
                break
        else:
            raise KeyError(
                "모델 출력 dict에서 logits를 찾지 못했습니다: "
                f"{sorted(output)}"
            )

    if not torch.is_tensor(output):
        raise TypeError(
            f"모델 출력이 Tensor가 아닙니다: {type(output)!r}"
        )

    if output.ndim == 3 and output.shape[-1] == 1:
        output = output.squeeze(-1)

    if output.ndim != 2:
        raise ValueError(
            f"logits shape은 [B,C]여야 합니다: {tuple(output.shape)}"
        )

    return output



# ---------------------------------------------------------------------------
# Shared model 로더 / scorer
# ---------------------------------------------------------------------------


class SharedModelScorer:
    def __init__(self, project_root: Path, run_dir: Path, device: torch.device):
        _ensure_project_on_syspath(project_root)
        from evaluation import hierarchy as hierarchy_eval
        from data.dataset.imagefolder import JointImageTransform

        self.hierarchy_eval = hierarchy_eval
        self.project_root = project_root.resolve()
        self.run_dir = run_dir.resolve()
        self.device = device

        args_path = self.run_dir / "args.yaml"
        if not args_path.is_file():
            raise FileNotFoundError(f"args.yaml이 없습니다: {args_path}")
        args_data = hierarchy_eval._load_yaml(args_path)
        if not bool(args_data.get("hierarchical_prompt", False)):
            raise ValueError(f"shared run이 hierarchical_prompt=True가 아닙니다: {run_dir}")

        cli_args = SimpleNamespace(batch_size=1, num_workers=0, device=str(device))
        self.params = hierarchy_eval._prepare_params(args_data, self.project_root, self.run_dir, cli_args)
        self.model, _, self.checkpoint_path, _ = hierarchy_eval._load_model(
            self.project_root,
            self.run_dir,
            self.params,
            self.device,
        )
        self.transform = JointImageTransform(self.params, training=False)
        taxonomy = hierarchy_eval._load_taxonomy(self.run_dir, self.params)
        self.taxonomy = SharedTaxonomyInfo(
            class_names=tuple(taxonomy.class_names),
            scientific_names=tuple(taxonomy.scientific_names),
            genus_names=tuple(taxonomy.genus_names),
            family_names=tuple(taxonomy.family_names),
            species_to_genus=tuple(taxonomy.species_to_genus),
            genus_to_family=tuple(taxonomy.genus_to_family),
        )
        self.class_to_index = {name: idx for idx, name in enumerate(self.taxonomy.class_names)}

    def _forward(self, image: Image.Image) -> dict[str, torch.Tensor]:
        tensor, _, _ = self.transform(image, bbox=None, bbox_coordinate_mode="normalized")
        batch = tensor.unsqueeze(0).to(self.device)
        with torch.inference_mode():
            output, _ = self.model(batch, patch_prior=None)
        if not isinstance(output, Mapping):
            raise TypeError("shared 계층 모델 출력이 dict가 아닙니다")
        return {key: value.detach().float().cpu() for key, value in output.items() if torch.is_tensor(value)}

    def score(self, image: Image.Image, *, species_index: int, rank: str) -> ScoreResult:
        out = self._forward(image)
        for required in (
            "family_probabilities",
            "genus_probabilities",
            "species_probabilities",
            "genus_conditional_probabilities",
            "species_conditional_probabilities",
            "rank_logits",
        ):
            if required not in out:
                raise KeyError(f"shared output에 {required!r}가 없습니다")

        species_index = int(species_index)
        genus_index = int(self.taxonomy.species_to_genus[species_index])
        family_index = int(self.taxonomy.species_to_family[species_index])

        family_probs = out["family_probabilities"][0]
        genus_probs = out["genus_probabilities"][0]
        species_probs = out["species_probabilities"][0]
        genus_cond = out["genus_conditional_probabilities"][0]
        species_cond = out["species_conditional_probabilities"][0]
        rank_logits = out["rank_logits"][0]

        predicted_index = int(species_probs.argmax().item())
        extras = {
            "family_conf": float(family_probs[family_index].item()),
            "genus_joint_conf": float(genus_probs[genus_index].item()),
            "species_joint_conf": float(species_probs[species_index].item()),
            "genus_cond_conf": float(genus_cond[genus_index].item()),
            "species_cond_conf": float(species_cond[species_index].item()),
            "rank_logit_species": float(rank_logits[0].item()) if rank_logits.numel() >= 1 else math.nan,
            "rank_logit_genus": float(rank_logits[1].item()) if rank_logits.numel() >= 2 else math.nan,
            "rank_logit_family": float(rank_logits[2].item()) if rank_logits.numel() >= 3 else math.nan,
        }

        rank_norm = str(rank).strip().lower()
        if rank_norm == "family":
            confidence = float(family_probs[family_index].item())
            logit = math.log(max(confidence, 1e-12))
        elif rank_norm == "genus":
            confidence = float(genus_probs[genus_index].item())
            logit = math.log(max(confidence, 1e-12))
        elif rank_norm == "species":
            confidence = float(species_probs[species_index].item())
            logit = math.log(max(confidence, 1e-12))
        else:
            raise ValueError(f"shared에서 지원하지 않는 rank: {rank}")

        return ScoreResult(
            confidence=confidence,
            logit=logit,
            predicted_index=predicted_index,
            predicted_label=self.taxonomy.class_names[predicted_index],
            extras=extras,
        )


# ---------------------------------------------------------------------------
# Flat model 로더 / scorer
# ---------------------------------------------------------------------------


class FlatModelScorer:
    def __init__(self, project_root: Path, run_dir: Path, device: torch.device):
        _ensure_project_on_syspath(project_root)
        from data.dataset.imagefolder import JointImageTransform, load_taxonomy_manifest
        from model.factory import get_model
        from evaluation import independent as ind_eval
        from evaluation import hierarchy as hierarchy_eval

        self.project_root = project_root.resolve()
        self.run_dir = run_dir.resolve()
        self.device = device
        self.ind_eval = ind_eval

        args_path = self.run_dir / "args.yaml"
        if not args_path.is_file():
            raise FileNotFoundError(f"args.yaml이 없습니다: {args_path}")
        args_data = hierarchy_eval._load_yaml(args_path)
        if bool(args_data.get("hierarchical_prompt", False)):
            raise ValueError(f"flat run이어야 하는데 hierarchical_prompt=True입니다: {run_dir}")
        if bool(args_data.get("original_taxonomy_prompt", False)):
            raise ValueError(f"flat run이어야 하는데 original_taxonomy_prompt=True입니다: {run_dir}")

        cli_args = SimpleNamespace(batch_size=1, num_workers=0, device=str(device))
        self.params = hierarchy_eval._prepare_params(args_data, self.project_root, self.run_dir, cli_args)
        self.model, _, _ = get_model(self.params)
        checkpoint_path = self.run_dir / "model.pt"
        checkpoint = hierarchy_eval._torch_load(checkpoint_path)
        state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
        incompatible = self.model.load_state_dict(state, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "flat 체크포인트와 현재 모델 구조가 일치하지 않습니다: "
                f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
            )
        self.model = self.model.to(self.device).eval()
        self.transform = JointImageTransform(self.params, training=False)

        train_root = _resolve(Path(self.params.data_path) / getattr(self.params, "train_split", "train"))
        imagefolder = datasets.ImageFolder(str(train_root))
        taxonomy = load_taxonomy_manifest(
            self.params.taxonomy_manifest,
            imagefolder.classes,
            class_column=getattr(self.params, "taxonomy_class_column", "folder_name"),
        )
        self.taxonomy = FlatTaxonomyInfo(
            class_names=tuple(taxonomy.class_names),
            genus_names=tuple(taxonomy.genus_names),
            family_names=tuple(taxonomy.family_names),
            species_to_genus=tuple(taxonomy.species_to_genus),
            genus_to_family=tuple(taxonomy.genus_to_family),
        )
        self.class_to_index = {name: idx for idx, name in enumerate(self.taxonomy.class_names)}

    def _forward(self, image: Image.Image) -> torch.Tensor:
        tensor, _, _ = self.transform(image, bbox=None, bbox_coordinate_mode="normalized")
        batch = tensor.unsqueeze(0).to(self.device)
        with torch.inference_mode():
            output, _ = self.model(batch, patch_prior=None)
        logits = _extract_logits_local(output).detach().float().cpu()
        return logits[0]

    def score(self, image: Image.Image, *, species_index: int, rank: str) -> ScoreResult:
        if str(rank).strip().lower() != "species":
            raise ValueError("Flat 모델은 species rank만 지원합니다")
        logits = self._forward(image)
        probs = logits.softmax(dim=0)
        species_index = int(species_index)
        predicted_index = int(probs.argmax().item())
        return ScoreResult(
            confidence=float(probs[species_index].item()),
            logit=float(logits[species_index].item()),
            predicted_index=predicted_index,
            predicted_label=self.taxonomy.class_names[predicted_index],
            extras={},
        )


# ---------------------------------------------------------------------------
# Independent(original taxonomy) 로더 / scorer
# ---------------------------------------------------------------------------


class IndependentModelScorer:
    def __init__(self, project_root: Path, run_root: Path, device: torch.device):
        self.project_root = project_root.resolve()
        self.run_root = run_root.resolve()
        self.device = device

        _ensure_project_on_syspath(self.project_root)
        import importlib.util
        from data.dataset.imagefolder import JointImageTransform

        evaluator_path = (
            self.project_root
            / "evaluation"
            / "independent.py"
        )

        if not evaluator_path.is_file():
            raise FileNotFoundError(
                f"루트 taxonomy evaluator가 없습니다: {evaluator_path}"
            )

        spec = importlib.util.spec_from_file_location(
            "promptcam_root_evaluate_original_taxonomy",
            evaluator_path,
        )

        if spec is None or spec.loader is None:
            raise RuntimeError(
                f"evaluator import spec 생성 실패: {evaluator_path}"
            )

        ind_eval = importlib.util.module_from_spec(spec)

        # dataclass가 cls.__module__을 통해 현재 모듈 namespace를
        # 조회하므로 exec_module 전에 반드시 sys.modules에 등록한다.
        sys.modules[spec.name] = ind_eval

        spec.loader.exec_module(ind_eval)

        print(
            "[Independent evaluator]",
            evaluator_path,
        )

        self.project_root = project_root.resolve()
        self.run_root = run_root.resolve()
        self.device = device
        self.ind_eval = ind_eval

        # tree / node spec discovery
        root_config = self.run_root / "configs" / "root.yaml"
        if not root_config.is_file():
            raise FileNotFoundError(f"independent run root 아래 configs/root.yaml이 없습니다: {root_config}")
        root_cfg = ind_eval._load_yaml(root_config)
        taxonomy_manifest = ind_eval._resolve_path(
            root_cfg.get("taxonomy_manifest"),
            project_root=self.project_root,
            base_dir=root_config.parent,
        )
        taxonomy_class_column = str(root_cfg.get("taxonomy_class_column", "folder_name"))
        self.tree = ind_eval._taxonomy_from_manifest(taxonomy_manifest, taxonomy_class_column)

        self.node_specs = ind_eval._discover_node_specs(self.run_root, self.tree)
        self.loaded_nodes = {
            spec.key: ind_eval._load_node(
                spec,
                project_root=self.project_root,
                tree=self.tree,
                device=self.device,
                allow_order_fallback=False,
            )
            for spec in self.node_specs
        }
        ind_eval._validate_preprocessing(list(self.loaded_nodes.values()))
        self.root_node = self.loaded_nodes["root"]
        self.transform = JointImageTransform(self.root_node.params, training=False)
        self.class_to_index = {name: idx for idx, name in enumerate(self.tree.species_names)}
        self.genus_to_index = {name: idx for idx, name in enumerate(self.tree.genera)}
        self.family_to_index = {name: idx for idx, name in enumerate(self.tree.families)}

    def _forward_node(self, loaded_node, image: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        tensor, _, _ = self.transform(image, bbox=None, bbox_coordinate_mode="normalized")
        batch = tensor.unsqueeze(0).to(self.device)
        with torch.inference_mode():
            output, _ = loaded_node.model(batch, patch_prior=None)
        logits = _extract_logits_local(output)[0].detach().float().cpu()
        probs = logits.softmax(dim=0)
        return logits, probs

    def score(self, image: Image.Image, *, species_index: int, rank: str) -> ScoreResult:
        species_index = int(species_index)
        species_name = self.tree.species_names[species_index]
        genus_index = int(self.tree.species_to_genus[species_index])
        family_index = int(self.tree.species_to_family[species_index])
        genus_name = self.tree.genera[genus_index]
        family_name = self.tree.families[family_index]

        root_logits, root_probs = self._forward_node(self.root_node, image)
        family_label_to_local = {label: i for i, label in enumerate(self.root_node.labels)}
        local_family_index = int(family_label_to_local[family_name])
        family_conf = float(root_probs[local_family_index].item())
        family_logit = float(root_logits[local_family_index].item())

        genus_cond_conf = 1.0
        genus_cond_logit = 0.0
        genus_node_key = f"family__{self.ind_eval._slug(family_name)}"
        if genus_node_key in self.loaded_nodes:
            family_node = self.loaded_nodes[genus_node_key]
            genus_logits, genus_probs = self._forward_node(family_node, image)
            genus_label_to_local = {label: i for i, label in enumerate(family_node.labels)}
            local_genus_index = int(genus_label_to_local[genus_name])
            genus_cond_conf = float(genus_probs[local_genus_index].item())
            genus_cond_logit = float(genus_logits[local_genus_index].item())

        species_cond_conf = 1.0
        species_cond_logit = 0.0
        species_node_key = f"genus__{self.ind_eval._slug(genus_name)}"
        if species_node_key in self.loaded_nodes:
            genus_node = self.loaded_nodes[species_node_key]
            species_logits, species_probs = self._forward_node(genus_node, image)
            species_label_to_local = {label: i for i, label in enumerate(genus_node.labels)}
            local_species_index = int(species_label_to_local[species_name])
            species_cond_conf = float(species_probs[local_species_index].item())
            species_cond_logit = float(species_logits[local_species_index].item())

        genus_joint_conf = family_conf * genus_cond_conf
        species_joint_conf = genus_joint_conf * species_cond_conf

        # species MAP prediction (전체 species joint 확률 최대값)
        best_prob = -1.0
        best_species_index = -1
        for candidate_index, candidate_name in enumerate(self.tree.species_names):
            cand_family_index = int(self.tree.species_to_family[candidate_index])
            cand_genus_index = int(self.tree.species_to_genus[candidate_index])
            cand_family_name = self.tree.families[cand_family_index]
            cand_genus_name = self.tree.genera[cand_genus_index]

            cand_local_f = family_label_to_local[cand_family_name]
            cand_f_conf = float(root_probs[cand_local_f].item())

            cand_genus_key = f"family__{self.ind_eval._slug(cand_family_name)}"
            cand_g_conf = 1.0
            if cand_genus_key in self.loaded_nodes:
                node = self.loaded_nodes[cand_genus_key]
                _, cand_genus_probs = self._forward_node(node, image)
                cand_local_g = {label: i for i, label in enumerate(node.labels)}[cand_genus_name]
                cand_g_conf = float(cand_genus_probs[cand_local_g].item())

            cand_species_key = f"genus__{self.ind_eval._slug(cand_genus_name)}"
            cand_s_conf = 1.0
            if cand_species_key in self.loaded_nodes:
                node = self.loaded_nodes[cand_species_key]
                _, cand_species_probs = self._forward_node(node, image)
                cand_local_s = {label: i for i, label in enumerate(node.labels)}[candidate_name]
                cand_s_conf = float(cand_species_probs[cand_local_s].item())

            cand_prob = cand_f_conf * cand_g_conf * cand_s_conf
            if cand_prob > best_prob:
                best_prob = cand_prob
                best_species_index = candidate_index

        extras = {
            "family_conf": family_conf,
            "genus_joint_conf": genus_joint_conf,
            "species_joint_conf": species_joint_conf,
            "genus_cond_conf": genus_cond_conf,
            "species_cond_conf": species_cond_conf,
            "family_logit": family_logit,
            "genus_cond_logit": genus_cond_logit,
            "species_cond_logit": species_cond_logit,
        }

        rank_norm = str(rank).strip().lower()
        if rank_norm == "family":
            confidence = family_conf
            logit = family_logit
        elif rank_norm == "genus":
            confidence = genus_joint_conf
            logit = math.log(max(genus_joint_conf, 1e-12))
        elif rank_norm == "species":
            confidence = species_joint_conf
            logit = math.log(max(species_joint_conf, 1e-12))
        else:
            raise ValueError(f"independent에서 지원하지 않는 rank: {rank}")

        return ScoreResult(
            confidence=confidence,
            logit=logit,
            predicted_index=best_species_index,
            predicted_label=self.tree.species_names[best_species_index],
            extras=extras,
        )


# ---------------------------------------------------------------------------
# Annotation template 생성
# ---------------------------------------------------------------------------


TEMPLATE_MODEL_ROWS = [
    ("Flat", "Species", "flat_species_conf", "flat_species_correct", "flat_species_cam_quality"),
    ("Independent", "Family", "ind_family_conf", "ind_family_correct", "ind_family_cam_quality"),
    ("Independent", "Genus", "ind_genus_joint_conf", "ind_genus_correct", "ind_genus_cam_quality"),
    ("Independent", "Species", "ind_species_joint_conf", "ind_species_correct", "ind_species_cam_quality"),
    ("Shared", "Family", "shared_family_conf", "shared_family_correct", "shared_family_cam_quality"),
    ("Shared", "Genus", "shared_genus_joint_conf", "shared_genus_correct", "shared_genus_cam_quality"),
    ("Shared", "Species", "shared_species_joint_conf", "shared_species_correct", "shared_species_cam_quality"),
]


def run_template(args: argparse.Namespace) -> None:
    wide_csv = _resolve(args.wide_csv)
    out_csv = _resolve(args.output_csv)
    df = pd.read_csv(wide_csv)

    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        for model, rank, conf_col, correct_col, quality_col in TEMPLATE_MODEL_ROWS:
            if conf_col not in row.index:
                continue
            conf = row.get(conf_col)
            if pd.isna(conf):
                continue
            correct = row.get(correct_col, True)
            if args.only_correct and bool(correct) is False:
                continue
            rows.append(
                {
                    "image_path": row.get("image_path"),
                    "species": row.get("species"),
                    "species_index": row.get("species_index"),
                    "model": model,
                    "rank": rank,
                    "confidence": conf,
                    "correct": correct,
                    "cam_quality": row.get(quality_col),
                    "object_x": "",
                    "object_y": "",
                    "background_x": "",
                    "background_y": "",
                    "notes": row.get("notes", ""),
                }
            )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[저장] annotation template: {out_csv}")
    print(f"rows={len(rows)}")


# ---------------------------------------------------------------------------
# 핵심 분석
# ---------------------------------------------------------------------------


def _species_from_row(row: Mapping[str, Any]) -> str:
    species = str(row.get("species", "")).strip()
    if species:
        return species
    image_path = Path(str(row["image_path"])).expanduser()
    return image_path.parent.name



def _species_index_from_row(row: Mapping[str, Any], class_to_index: Mapping[str, int]) -> int:
    value = row.get("species_index", None)
    if value is not None and str(value).strip() != "" and not pd.isna(value):
        return int(value)
    species = _species_from_row(row)
    if species not in class_to_index:
        raise KeyError(f"species={species!r}가 모델 taxonomy에 없습니다")
    return int(class_to_index[species])



def _get_scorer(model_name: str, scorers: Mapping[str, Any]) -> Any:
    key = str(model_name).strip().lower()
    if key == "flat":
        scorer = scorers.get("flat")
    elif key == "shared":
        scorer = scorers.get("shared")
    elif key == "independent":
        scorer = scorers.get("independent")
    else:
        raise ValueError(f"지원하지 않는 model: {model_name}")
    if scorer is None:
        raise ValueError(f"{model_name} scorer가 준비되지 않았습니다")
    return scorer



def _safe_float(value: Any) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return math.nan
    return float(value)



def run_analyze(args: argparse.Namespace) -> None:
    project_root = _resolve(args.project_root)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    scorers: dict[str, Any] = {"flat": None, "shared": None, "independent": None}
    if args.flat_run_dir:
        print("[로드] Flat scorer")
        scorers["flat"] = FlatModelScorer(project_root, _resolve(args.flat_run_dir), device)
    if args.shared_run_dir:
        print("[로드] Shared scorer")
        scorers["shared"] = SharedModelScorer(project_root, _resolve(args.shared_run_dir), device)
    if args.independent_run_root:
        print("[로드] Independent scorer")
        scorers["independent"] = IndependentModelScorer(project_root, _resolve(args.independent_run_root), device)

    annotation_csv = _resolve(args.annotation_csv)
    df = pd.read_csv(annotation_csv)
    required = [
        "image_path", "model", "rank", "object_x", "object_y", "background_x", "background_y"
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"annotation CSV에 필요한 열이 없습니다: {missing}")

    results: list[dict[str, Any]] = []
    skipped_rows = 0

    for row_idx, row in df.iterrows():
        coordinate_columns = (
            "object_x",
            "object_y",
            "background_x",
            "background_y",
        )

        if any(
            pd.isna(row[column])
            or str(row[column]).strip() == ""
            for column in coordinate_columns
        ):
            skipped_rows += 1
            print(
                f"[SKIP {row_idx + 1}/{len(df)}] "
                "object/background 좌표가 모두 채워지지 않았습니다."
            )
            continue

        model_name = str(row["model"]).strip()
        rank = str(row["rank"]).strip()
        scorer = _get_scorer(model_name, scorers)
        image_path = _resolve(row["image_path"])
        image = Image.open(image_path).convert("RGB")

        species_index = _species_index_from_row(row, scorer.class_to_index)
        species_name = _species_from_row(row)

        original = scorer.score(image, species_index=species_index, rank=rank)

        object_image = _occlude_square(
            image,
            x=float(row["object_x"]),
            y=float(row["object_y"]),
            patch_size=int(args.patch_size),
            coord_mode=args.coord_mode,
            fill_mode=args.fill_mode,
            transform=getattr(scorer, "transform", None),
        )
        background_image = _occlude_square(
            image,
            x=float(row["background_x"]),
            y=float(row["background_y"]),
            patch_size=int(args.patch_size),
            coord_mode=args.coord_mode,
            fill_mode=args.fill_mode,
            transform=getattr(scorer, "transform", None),
        )

        object_masked = scorer.score(object_image, species_index=species_index, rank=rank)
        background_masked = scorer.score(background_image, species_index=species_index, rank=rank)

        delta_object = original.confidence - object_masked.confidence
        delta_background = original.confidence - background_masked.confidence
        delta_gap = delta_object - delta_background

        delta_object_logit = original.logit - object_masked.logit
        delta_background_logit = original.logit - background_masked.logit
        delta_gap_logit = delta_object_logit - delta_background_logit

        record = {
            "row_index": int(row_idx),
            "image_path": str(image_path),
            "species": species_name,
            "species_index": species_index,
            "model": model_name,
            "rank": rank,
            "cam_quality": row.get("cam_quality", math.nan),
            "object_x": _safe_float(row["object_x"]),
            "object_y": _safe_float(row["object_y"]),
            "background_x": _safe_float(row["background_x"]),
            "background_y": _safe_float(row["background_y"]),
            "original_confidence": original.confidence,
            "object_masked_confidence": object_masked.confidence,
            "background_masked_confidence": background_masked.confidence,
            "delta_object": delta_object,
            "delta_background": delta_background,
            "delta_gap": delta_gap,
            "original_logit": original.logit,
            "object_masked_logit": object_masked.logit,
            "background_masked_logit": background_masked.logit,
            "delta_object_logit": delta_object_logit,
            "delta_background_logit": delta_background_logit,
            "delta_gap_logit": delta_gap_logit,
            "original_predicted_label": original.predicted_label,
            "object_masked_predicted_label": object_masked.predicted_label,
            "background_masked_predicted_label": background_masked.predicted_label,
            "original_predicted_index": original.predicted_index,
            "object_masked_predicted_index": object_masked.predicted_index,
            "background_masked_predicted_index": background_masked.predicted_index,
            "original_correct": int(original.predicted_index == species_index),
            "object_masked_correct": int(object_masked.predicted_index == species_index),
            "background_masked_correct": int(background_masked.predicted_index == species_index),
            "notes": row.get("notes", ""),
        }
        # extras는 prefix를 붙여 저장
        for prefix, score in (
            ("orig", original),
            ("obj", object_masked),
            ("bg", background_masked),
        ):
            for key, value in score.extras.items():
                record[f"{prefix}_{key}"] = value

        results.append(record)
        print(
            f"[{row_idx+1}/{len(df)}] {model_name}/{rank} | {species_name} | "
            f"orig={original.confidence:.6f} obj_drop={delta_object:.6f} bg_drop={delta_background:.6f} gap={delta_gap:.6f}"
        )

    if not results:
        raise RuntimeError(
            "분석 가능한 행이 없습니다. "
            "object/background 좌표를 확인하세요."
        )

    print(
        f"\n분석 행={len(results)}, "
        f"좌표 미입력 skip={skipped_rows}"
    )

    out_dir = _resolve(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result_df = pd.DataFrame(results)
    result_csv = out_dir / "occlusion_results.csv"
    result_df.to_csv(result_csv, index=False)

    group = result_df.groupby(["model", "rank"], dropna=False)
    summary = group.agg(
        n=("delta_gap", "size"),
        mean_original_confidence=("original_confidence", "mean"),
        mean_delta_object=("delta_object", "mean"),
        mean_delta_background=("delta_background", "mean"),
        mean_delta_gap=("delta_gap", "mean"),
        median_delta_gap=("delta_gap", "median"),
        std_delta_gap=("delta_gap", "std"),
        object_flip_rate=("object_masked_correct", lambda s: 1.0 - float(pd.Series(s).mean())),
        background_flip_rate=("background_masked_correct", lambda s: 1.0 - float(pd.Series(s).mean())),
        mean_delta_object_logit=("delta_object_logit", "mean"),
        mean_delta_background_logit=("delta_background_logit", "mean"),
        mean_delta_gap_logit=("delta_gap_logit", "mean"),
    ).reset_index()
    summary_csv = out_dir / "occlusion_group_summary.csv"
    summary.to_csv(summary_csv, index=False)

    report_path = out_dir / "occlusion_report.md"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# CAM object-vs-background occlusion analysis\n\n")
        f.write(f"- annotation_csv: `{annotation_csv}`\n")
        f.write(f"- patch_size: {args.patch_size}\n")
        f.write(f"- coord_mode: {args.coord_mode}\n")
        f.write(f"- fill_mode: {args.fill_mode}\n")
        f.write(f"- device: {device}\n\n")
        f.write("## Group summary\n\n")
        f.write(summary.to_markdown(index=False))
        f.write("\n\n")
        f.write("## Interpretation rule\n\n")
        f.write("- `delta_object = original_confidence - object_masked_confidence`\n")
        f.write("- `delta_background = original_confidence - background_masked_confidence`\n")
        f.write("- `delta_gap = delta_object - delta_background`\n")
        f.write("\n`delta_gap > 0`이면 객체(red hotspot on plant)를 가렸을 때 confidence가 더 크게 감소했음을 뜻하므로, 해당 모델/랭크가 배경보다 식물 객체 근거를 더 많이 사용한다고 해석할 수 있다.\n")

    print("\n===== GROUP SUMMARY =====")
    print(summary.to_string(index=False))
    print(f"\n[저장] results : {result_csv}")
    print(f"[저장] summary : {summary_csv}")
    print(f"[저장] report  : {report_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="식물 hotspot vs 배경 hotspot 가림에 따른 confidence drop 분석"
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    p_template = sub.add_parser("template", help="wide annotation CSV를 long-format 좌표 template로 변환")
    p_template.add_argument("--wide-csv", required=True, help="cam_quality_confidence_annotations_filled.csv")
    p_template.add_argument("--output-csv", required=True, help="저장할 template CSV")
    p_template.add_argument("--only-correct", action="store_true", help="정답 샘플만 template에 포함")
    p_template.set_defaults(func=run_template)

    p_analyze = sub.add_parser("analyze", help="좌표가 채워진 annotation CSV로 occlusion 분석")
    p_analyze.add_argument("--project-root", default=".", help="PromptCAM-SnapMix 프로젝트 루트")
    p_analyze.add_argument("--annotation-csv", required=True, help="long-format annotation CSV")
    p_analyze.add_argument("--flat-run-dir", default=None, help="flat model run dir")
    p_analyze.add_argument("--shared-run-dir", default=None, help="shared hierarchical run dir")
    p_analyze.add_argument("--independent-run-root", default=None, help="original taxonomy independent run root")
    p_analyze.add_argument("--output-dir", required=True, help="결과 저장 디렉터리")
    p_analyze.add_argument("--patch-size", type=int, default=61, help="가릴 정사각형 한 변 길이(pixel)")
    p_analyze.add_argument(
        "--coord-mode",
        choices=["pixel", "normalized", "crop-normalized"],
        default="pixel",
    )
    p_analyze.add_argument("--fill-mode", choices=["mean", "gray", "black", "blur"], default="mean")
    p_analyze.add_argument("--device", default="auto", help="cuda / cpu / auto")
    p_analyze.set_defaults(func=run_analyze)
    return parser



def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
