"""Prompt-CAM의 종·속 CAM 의미 질량을 이용해 계층적 Semantic SnapMix를 수행한다."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# 여러 attention head를 하나의 SPM으로 줄이는 방식을 제한한다.
HeadReduction = Literal["mean", "max"]


@dataclass(frozen=True)
class Box:
    """복사 영역의 좌상단·우하단 좌표를 보관하는 불변 사각형이다."""
    y1: int
    x1: int
    y2: int
    x2: int

    @property
    def height(self) -> int:
        """사각형의 음수가 아닌 세로 길이를 반환한다."""
        return max(0, self.y2 - self.y1)

    @property
    def width(self) -> int:
        """사각형의 음수가 아닌 가로 길이를 반환한다."""
        return max(0, self.x2 - self.x1)

    @property
    def area(self) -> int:
        """사각형의 픽셀 면적을 반환한다."""
        return self.height * self.width


@dataclass
class SnapMixBatch:
    """혼합 이미지와 종·속 soft target 계산에 필요한 모든 결과를 묶는다."""
    images: torch.Tensor
    target_a: torch.Tensor
    target_b: torch.Tensor
    weight_a: torch.Tensor
    weight_b: torch.Tensor
    permutation: torch.Tensor
    destination_box: Optional[Box]
    source_box: Optional[Box]
    applied: bool
    applied_mask: Optional[torch.Tensor] = None
    genus_target_a: Optional[torch.Tensor] = None
    genus_target_b: Optional[torch.Tensor] = None
    genus_weight_a: Optional[torch.Tensor] = None
    genus_weight_b: Optional[torch.Tensor] = None
    family_target_a: Optional[torch.Tensor] = None
    family_target_b: Optional[torch.Tensor] = None
    family_weight_a: Optional[torch.Tensor] = None
    family_weight_b: Optional[torch.Tensor] = None


def eta_for_epoch(epoch: int, start_epoch: int, end_epoch: int) -> float:
    """지정한 시작·종료 epoch 사이에서 CAM 사용 비율 eta를 0에서 1로 선형 증가시킨다."""
    if end_epoch < start_epoch:
        raise ValueError(
            f"eta_end_epoch({end_epoch})는 eta_start_epoch({start_epoch}) 이상이어야 합니다"
        )
    if end_epoch == start_epoch:
        return 1.0 if epoch >= end_epoch else 0.0
    return float(np.clip((epoch - start_epoch) / (end_epoch - start_epoch), 0.0, 1.0))


def _normalize_mass(maps: torch.Tensor, eps: float) -> torch.Tensor:
    """각 표본의 비음수 map 합이 1이 되도록 정규화하고 비정상 map은 균등분포로 대체한다."""
    if maps.ndim != 3:
        raise ValueError(f"맵 형태는 [B,H,W]여야 하지만 {tuple(maps.shape)}입니다")
    # CAM을 확률 질량으로 해석하므로 음수 값을 제거하고 float 정밀도로 계산한다.
    maps = maps.float().clamp_min(0.0)
    sums = maps.sum(dim=(1, 2), keepdim=True)
    normalized = maps / sums.clamp_min(eps)
    invalid = (~torch.isfinite(sums)) | (sums <= eps)
    if invalid.any():
        uniform = torch.full_like(normalized, 1.0 / (maps.shape[1] * maps.shape[2]))
        normalized = torch.where(invalid.expand_as(normalized), uniform, normalized)
    return normalized


def _pixel_spm(
    patch_map: torch.Tensor,
    image_size: Tuple[int, int],
    eta: float,
    eps: float,
) -> torch.Tensor:
    """patch CAM을 픽셀 해상도로 보간하고 eta에 따라 균등 질량과 혼합한다."""
    batch_size = patch_map.shape[0]
    image_h, image_w = image_size
    patch_map = _normalize_mass(patch_map, eps)
    # patch grid의 질량을 실제 이미지 해상도로 부드럽게 보간한다.
    pixel_map = F.interpolate(
        patch_map.unsqueeze(1),
        size=(image_h, image_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)
    pixel_map = _normalize_mass(pixel_map, eps)
    # eta=0에서는 면적 기반 CutMix, eta=1에서는 CAM 기반 SnapMix가 된다.
    uniform = torch.full_like(pixel_map, 1.0 / (image_h * image_w))
    return _normalize_mass((1.0 - eta) * uniform + eta * pixel_map, eps)


def uniform_spm(images: torch.Tensor) -> torch.Tensor:
    """이미지마다 합이 1인 균등 픽셀 의미 질량 맵을 만든다."""
    if images.ndim != 4:
        raise ValueError(f"이미지 형태는 [B,C,H,W]여야 하지만 {tuple(images.shape)}입니다")
    batch_size, _, height, width = images.shape
    return torch.full(
        (batch_size, height, width),
        1.0 / (height * width),
        device=images.device,
        dtype=torch.float32,
    )

def should_apply_snapmix(
    batch_size: int,
    probability: float,
    eligible_mask: Optional[torch.Tensor] = None,
) -> bool:
    """허용 표본 수와 적용 확률을 확인해 이번 배치의 혼합 여부를 한 번만 추첨한다."""
    if batch_size < 0:
        raise ValueError("batch_size는 음수가 아니어야 합니다")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability는 [0,1] 범위여야 합니다")
    eligible_count = batch_size
    if eligible_mask is not None:
        if eligible_mask.shape != (batch_size,):
            raise ValueError("eligible_mask에는 이미지마다 항목이 하나씩 있어야 합니다")
        eligible_count = int(eligible_mask.to(dtype=torch.bool).sum().item())
    return eligible_count >= 2 and float(np.random.random()) < probability


def unchanged_snapmix_batch(
    images: torch.Tensor,
    targets: torch.Tensor,
    *,
    genus_targets: Optional[torch.Tensor] = None,
    family_targets: Optional[torch.Tensor] = None,
) -> SnapMixBatch:
    """증강을 생략한 원본 배치와 항등 target 가중치를 만든다."""
    if images.ndim != 4:
        raise ValueError(
            f"이미지 형태는 [B,C,H,W]여야 하지만 {tuple(images.shape)}입니다"
        )
    batch_size = images.shape[0]
    if targets.shape != (batch_size,):
        raise ValueError("targets에는 이미지마다 항목이 하나씩 있어야 합니다")
    if genus_targets is not None and genus_targets.shape != (batch_size,):
        raise ValueError(
            "genus_targets에는 이미지마다 항목이 하나씩 있어야 합니다"
        )
    if family_targets is not None and family_targets.shape != (batch_size,):
        raise ValueError(
            "family_targets에는 이미지마다 항목이 하나씩 있어야 합니다"
        )

    device = images.device
    permutation = torch.arange(batch_size, device=device)
    weight_a = torch.ones(batch_size, device=device, dtype=images.dtype)
    weight_b = torch.zeros(batch_size, device=device, dtype=images.dtype)

    return SnapMixBatch(
        images=images,
        target_a=targets,
        target_b=targets.clone(),
        weight_a=weight_a,
        weight_b=weight_b,
        permutation=permutation,
        destination_box=None,
        source_box=None,
        applied=False,
        applied_mask=torch.zeros(
            batch_size,
            dtype=torch.bool,
            device=device,
        ),
        genus_target_a=genus_targets,
        genus_target_b=(
            genus_targets.clone()
            if genus_targets is not None
            else None
        ),
        genus_weight_a=(
            torch.ones_like(weight_a)
            if genus_targets is not None
            else None
        ),
        genus_weight_b=(
            torch.zeros_like(weight_b)
            if genus_targets is not None
            else None
        ),
        family_target_a=family_targets,
        family_target_b=(
            family_targets.clone()
            if family_targets is not None
            else None
        ),
        family_weight_a=(
            torch.ones_like(weight_a)
            if family_targets is not None
            else None
        ),
        family_weight_b=(
            torch.zeros_like(weight_b)
            if family_targets is not None
            else None
        ),
    )


def _unpack_teacher_output(teacher_output):
    """teacher 반환값을 모델 출력과 attention의 공통 두 항 형식으로 맞춘다."""
    if isinstance(teacher_output, tuple) and len(teacher_output) == 2:
        return teacher_output
    return teacher_output, None


def _squeeze_promptcam_logits(logits: torch.Tensor) -> torch.Tensor:
    """Prompt-CAM 로짓을 표준 [B,C] 텐서로 변환한다."""
    if logits.ndim == 3 and logits.shape[-1] == 1:
        return logits.squeeze(-1)
    if logits.ndim != 2:
        raise ValueError(
            f"Prompt-CAM 로짓은 [B,C] 또는 [B,C,1]이어야 하지만 {tuple(logits.shape)}입니다"
        )
    return logits


@torch.no_grad()
def hierarchical_promptcam_spm(
    teacher: torch.nn.Module,
    images: torch.Tensor,
    species_targets: torch.Tensor,
    genus_targets: torch.Tensor,
    family_targets: torch.Tensor,
    *,
    patch_size: int | Tuple[int, int],
    eta: float,
    eps: float = 1e-8,
    patch_prior: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """정답 종·속·과 CAM에서 픽셀 단위 SPM을 생성한다."""

    if not 0.0 <= eta <= 1.0:
        raise ValueError(f"eta는 [0,1] 범위여야 하지만 {eta}입니다")
    if images.ndim != 4:
        raise ValueError(
            f"이미지 형태는 [B,C,H,W]여야 하지만 {tuple(images.shape)}입니다"
        )

    batch_size, _, image_h, image_w = images.shape
    expected = (batch_size,)
    if (
        species_targets.shape != expected
        or genus_targets.shape != expected
        or family_targets.shape != expected
    ):
        raise ValueError(
            "species_targets, genus_targets, family_targets에는 "
            "이미지마다 항목이 하나씩 있어야 합니다"
        )

    teacher.eval()
    if patch_prior is None:
        teacher_result = teacher(images)
    else:
        teacher_result = teacher(
            images,
            patch_prior=patch_prior,
        )

    output, _ = _unpack_teacher_output(teacher_result)
    if not isinstance(output, dict):
        raise ValueError(
            "계층적 Prompt-CAM 교사 모델은 딕셔너리 출력을 반환해야 합니다"
        )

    species_cam = output.get("species_cam")
    genus_cam = output.get("genus_cam")
    family_cam = output.get("family_cam")
    if species_cam is None or genus_cam is None or family_cam is None:
        raise ValueError(
            "계층 출력에는 species_cam, genus_cam, family_cam이 모두 필요합니다"
        )

    patch_h, patch_w = (
        (patch_size, patch_size)
        if isinstance(patch_size, int)
        else patch_size
    )
    if image_h % patch_h != 0 or image_w % patch_w != 0:
        raise ValueError(
            f"입력 크기 {(image_h, image_w)}는 패치 크기 "
            f"{(patch_h, patch_w)}로 나누어떨어져야 합니다"
        )

    grid_h = image_h // patch_h
    grid_w = image_w // patch_w
    patch_count = grid_h * grid_w
    for level, cam in (
        ("species", species_cam),
        ("genus", genus_cam),
        ("family", family_cam),
    ):
        if cam.shape[-1] != patch_count:
            raise ValueError(
                f"{level} CAM 패치 수 {cam.shape[-1]}가 기대값 "
                f"{patch_count}와 다릅니다"
            )

    batch_index = torch.arange(
        batch_size,
        device=images.device,
    )
    species_patch = species_cam[
        batch_index,
        species_targets,
    ].reshape(batch_size, grid_h, grid_w)
    genus_patch = genus_cam[
        batch_index,
        genus_targets,
    ].reshape(batch_size, grid_h, grid_w)
    family_patch = family_cam[
        batch_index,
        family_targets,
    ].reshape(batch_size, grid_h, grid_w)

    return (
        _pixel_spm(
            species_patch,
            (image_h, image_w),
            eta,
            eps,
        ),
        _pixel_spm(
            genus_patch,
            (image_h, image_w),
            eta,
            eps,
        ),
        _pixel_spm(
            family_patch,
            (image_h, image_w),
            eta,
            eps,
        ),
        output,
    )


@torch.no_grad()
def promptcam_spm(
    teacher: torch.nn.Module,
    images: torch.Tensor,
    targets: torch.Tensor,
    *,
    prompt_count: int,
    prefix_token_count: int,
    patch_size: int | Tuple[int, int],
    eta: float,
    head_reduction: HeadReduction = "mean",
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """비계층 Prompt-CAM 출력에서 정답 클래스의 픽셀 의미 확률 map을 생성한다."""

    if not 0.0 <= eta <= 1.0:
        raise ValueError(f"eta는 [0,1] 범위여야 하지만 {eta}입니다")
    if images.ndim != 4:
        raise ValueError(f"이미지 형태는 [B,C,H,W]여야 하지만 {tuple(images.shape)}입니다")
    batch_size, _, image_h, image_w = images.shape
    patch_h, patch_w = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
    if image_h % patch_h != 0 or image_w % patch_w != 0:
        raise ValueError("입력 크기는 패치 크기로 나누어떨어져야 합니다")
    grid_h, grid_w = image_h // patch_h, image_w // patch_w
    patch_count = grid_h * grid_w

    teacher.eval()
    output, attention = _unpack_teacher_output(teacher(images))
    if isinstance(output, dict):
        logits = output["species_logits"]
        cam = output["species_cam"]
        batch_index = torch.arange(batch_size, device=images.device)
        patch_map = cam[batch_index, targets].reshape(batch_size, grid_h, grid_w)
        return _pixel_spm(patch_map, (image_h, image_w), eta, eps), logits

    logits = _squeeze_promptcam_logits(output)
    if attention is None or attention.ndim != 4:
        raise ValueError("Prompt-CAM은 마지막 계층 어텐션 [B,heads,tokens,tokens]을 반환해야 합니다")
    batch_index = torch.arange(
        batch_size,
        device=attention.device,
    )

    if attention.shape[2:] == (
        prompt_count,
        patch_count,
    ):
        # PatchOnlyPromptDecoder attention: [B,H,C,P]
        patch_attention = attention[
            batch_index,
            :,
            targets,
            :,
        ].float()
    else:
        # 기존 Prompt-CAM self-attention: [B,H,N,N]
        patch_start = (
            prompt_count
            + prefix_token_count
        )
        patch_end = (
            patch_start
            + patch_count
        )
        patch_attention = attention[
            batch_index,
            :,
            targets,
            patch_start:patch_end,
        ].float()
    if head_reduction == "mean":
        patch_map = patch_attention.mean(dim=1)
    elif head_reduction == "max":
        patch_map = patch_attention.max(dim=1).values
    else:
        raise ValueError(f"지원하지 않는 헤드 축약 방식입니다: {head_reduction}")
    patch_map = patch_map.reshape(batch_size, grid_h, grid_w)
    return _pixel_spm(patch_map, (image_h, image_w), eta, eps), logits


def random_box(height: int, width: int, lam: float) -> Box:
    """Beta 표본으로 결정된 면적 비율을 따르는 임의의 잘린 사각형을 만든다."""
    if height <= 0 or width <= 0:
        raise ValueError("높이와 너비는 양수여야 합니다")
    lam = float(np.clip(lam, 0.0, 1.0))
    # lam은 유지 면적 비율이므로 잘라낼 한 변의 비율은 sqrt(1-lam)이다.
    cut_ratio = float(np.sqrt(1.0 - lam))
    cut_h = int(round(height * cut_ratio))
    cut_w = int(round(width * cut_ratio))
    center_y = int(np.random.randint(0, height))
    center_x = int(np.random.randint(0, width))
    y1 = int(np.clip(center_y - cut_h // 2, 0, height))
    x1 = int(np.clip(center_x - cut_w // 2, 0, width))
    y2 = int(np.clip(center_y + (cut_h + 1) // 2, 0, height))
    x2 = int(np.clip(center_x + (cut_w + 1) // 2, 0, width))
    return Box(y1=y1, x1=x1, y2=y2, x2=x2)


def _normalized_pair_weights(
    retained: torch.Tensor,
    inserted: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """남은 의미 질량과 삽입 의미 질량을 합이 1인 두 혼합 가중치로 바꾼다."""
    denominator = retained + inserted
    fallback = denominator <= eps
    weight_a = retained / denominator.clamp_min(eps)
    weight_b = inserted / denominator.clamp_min(eps)
    if fallback.any():
        weight_a = torch.where(fallback, torch.ones_like(weight_a), weight_a)
        weight_b = torch.where(fallback, torch.zeros_like(weight_b), weight_b)
    return weight_a, weight_b


def semantic_snapmix(
    images: torch.Tensor,
    targets: torch.Tensor,
    spm: torch.Tensor,
    *,
    probability: float,
    beta: float,
    eps: float = 1e-8,
    genus_targets: Optional[torch.Tensor] = None,
    genus_spm: Optional[torch.Tensor] = None,
    family_targets: Optional[torch.Tensor] = None,
    family_spm: Optional[torch.Tensor] = None,
    eligible_mask: Optional[torch.Tensor] = None,
    apply: Optional[bool] = None,
) -> SnapMixBatch:
    """종·속·과 의미 질량을 사용해 계층별 soft target을 만든다."""

    if images.ndim != 4:
        raise ValueError(
            f"이미지 형태는 [B,C,H,W]여야 하지만 {tuple(images.shape)}입니다"
        )

    batch_size, _, height, width = images.shape
    expected_target_shape = (batch_size,)
    expected_spm_shape = (batch_size, height, width)

    if targets.shape != expected_target_shape or spm.shape != expected_spm_shape:
        raise ValueError("targets/SPM 형태가 이미지와 일치하지 않습니다")
    if genus_targets is not None and genus_targets.shape != expected_target_shape:
        raise ValueError(
            "genus_targets에는 이미지마다 항목이 하나씩 있어야 합니다"
        )
    if genus_spm is not None and genus_spm.shape != expected_spm_shape:
        raise ValueError("genus_spm 형태가 이미지와 일치하지 않습니다")
    if family_targets is not None and family_targets.shape != expected_target_shape:
        raise ValueError(
            "family_targets에는 이미지마다 항목이 하나씩 있어야 합니다"
        )
    if family_spm is not None and family_spm.shape != expected_spm_shape:
        raise ValueError("family_spm 형태가 이미지와 일치하지 않습니다")
    if not 0.0 <= probability <= 1.0 or beta <= 0:
        raise ValueError(
            "probability는 [0,1] 범위이고 beta는 양수여야 합니다"
        )

    device = images.device
    if eligible_mask is None:
        eligible_mask = torch.ones(
            batch_size,
            dtype=torch.bool,
            device=device,
        )
    else:
        eligible_mask = eligible_mask.to(
            device=device,
            dtype=torch.bool,
        )

    eligible_indices = torch.nonzero(
        eligible_mask,
        as_tuple=False,
    ).flatten()

    permutation = torch.arange(batch_size, device=device)
    target_b = targets.clone()
    weight_a = torch.ones(
        batch_size,
        device=device,
        dtype=images.dtype,
    )
    weight_b = torch.zeros_like(weight_a)
    applied_mask = torch.zeros(
        batch_size,
        dtype=torch.bool,
        device=device,
    )

    genus_target_b = (
        genus_targets.clone()
        if genus_targets is not None
        else None
    )
    genus_weight_a = (
        torch.ones_like(weight_a)
        if genus_targets is not None
        else None
    )
    genus_weight_b = (
        torch.zeros_like(weight_b)
        if genus_targets is not None
        else None
    )

    family_target_b = (
        family_targets.clone()
        if family_targets is not None
        else None
    )
    family_weight_a = (
        torch.ones_like(weight_a)
        if family_targets is not None
        else None
    )
    family_weight_b = (
        torch.zeros_like(weight_b)
        if family_targets is not None
        else None
    )

    def unchanged(
        destination_box: Optional[Box] = None,
        source_box: Optional[Box] = None,
    ) -> SnapMixBatch:
        unchanged_batch = unchanged_snapmix_batch(
            images,
            targets,
            genus_targets=genus_targets,
            family_targets=family_targets,
        )
        unchanged_batch.destination_box = destination_box
        unchanged_batch.source_box = source_box
        return unchanged_batch

    if apply is None:
        apply = should_apply_snapmix(
            batch_size,
            probability,
            eligible_mask,
        )
    if not apply or eligible_indices.numel() < 2:
        return unchanged()

    order = eligible_indices[
        torch.randperm(
            eligible_indices.numel(),
            device=device,
        )
    ]
    sources = torch.roll(order, shifts=1, dims=0)
    permutation[order] = sources

    target_b = targets[permutation]
    if genus_targets is not None:
        genus_target_b = genus_targets[permutation]
    if family_targets is not None:
        family_target_b = family_targets[permutation]

    destination_box = random_box(
        height,
        width,
        float(np.random.beta(beta, beta)),
    )
    source_box = random_box(
        height,
        width,
        float(np.random.beta(beta, beta)),
    )
    if destination_box.area == 0 or source_box.area == 0:
        return unchanged(destination_box, source_box)

    mixed = images.clone()
    source_crop = images[
        permutation[eligible_indices],
        :,
        source_box.y1:source_box.y2,
        source_box.x1:source_box.x2,
    ].clone()
    source_crop = F.interpolate(
        source_crop,
        size=(destination_box.height, destination_box.width),
        mode="bilinear",
        align_corners=False,
    )
    mixed[
        eligible_indices,
        :,
        destination_box.y1:destination_box.y2,
        destination_box.x1:destination_box.x2,
    ] = source_crop

    def weights_from_map(
        level_spm: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        donor_spm = level_spm[permutation]
        removed = level_spm[
            :,
            destination_box.y1:destination_box.y2,
            destination_box.x1:destination_box.x2,
        ].sum(dim=(1, 2)).clamp(0.0, 1.0)
        inserted = donor_spm[
            :,
            source_box.y1:source_box.y2,
            source_box.x1:source_box.x2,
        ].sum(dim=(1, 2)).clamp(0.0, 1.0)
        return _normalized_pair_weights(
            1.0 - removed,
            inserted,
            eps,
        )

    species_a, species_b = weights_from_map(spm)
    weight_a[eligible_indices] = species_a[eligible_indices].to(images.dtype)
    weight_b[eligible_indices] = species_b[eligible_indices].to(images.dtype)
    applied_mask[eligible_indices] = True

    if genus_targets is not None:
        map_for_genus = genus_spm if genus_spm is not None else spm
        genus_a, genus_b = weights_from_map(map_for_genus)
        same_genus = genus_targets.eq(genus_target_b)
        genus_a = torch.where(
            same_genus,
            torch.ones_like(genus_a),
            genus_a,
        )
        genus_b = torch.where(
            same_genus,
            torch.zeros_like(genus_b),
            genus_b,
        )
        genus_weight_a[eligible_indices] = genus_a[eligible_indices].to(
            images.dtype
        )
        genus_weight_b[eligible_indices] = genus_b[eligible_indices].to(
            images.dtype
        )

    if family_targets is not None:
        map_for_family = family_spm if family_spm is not None else spm
        family_a, family_b = weights_from_map(map_for_family)
        same_family = family_targets.eq(family_target_b)
        family_a = torch.where(
            same_family,
            torch.ones_like(family_a),
            family_a,
        )
        family_b = torch.where(
            same_family,
            torch.zeros_like(family_b),
            family_b,
        )
        family_weight_a[eligible_indices] = family_a[eligible_indices].to(
            images.dtype
        )
        family_weight_b[eligible_indices] = family_b[eligible_indices].to(
            images.dtype
        )

    return SnapMixBatch(
        images=mixed,
        target_a=targets,
        target_b=target_b,
        weight_a=weight_a,
        weight_b=weight_b,
        permutation=permutation,
        destination_box=destination_box,
        source_box=source_box,
        applied=True,
        applied_mask=applied_mask,
        genus_target_a=genus_targets,
        genus_target_b=genus_target_b,
        genus_weight_a=genus_weight_a,
        genus_weight_b=genus_weight_b,
        family_target_a=family_targets,
        family_target_b=family_target_b,
        family_weight_a=family_weight_a,
        family_weight_b=family_weight_b,
    )
