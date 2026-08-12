"""Node-conditional taxonomy 경로 손실과 CAM 제약을 계산한다."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.semantic_snapmix import SnapMixBatch


RANK_SPECIES = 0
RANK_GENUS = 1
RANK_FAMILY = 2
RANK_UNIDENTIFIABLE = 3


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """boolean mask가 참인 표본만 평균하고 유효 표본이 없으면 0을 반환한다."""
    mask = mask.to(dtype=torch.bool)
    if not bool(mask.any()):
        return values.new_zeros(())
    return values[mask].mean()


class HierarchicalTaxonomicCriterion(nn.Module):
    """P(f|x)P(g|f,x)P(c|g,x)의 경로 음의 로그우도를 계산한다."""

    def __init__(self, params) -> None:
        super().__init__()

        self.num_species = int(params.class_num)
        self.num_genera = int(params.num_genera)
        self.num_families = int(params.num_families)

        self.register_buffer(
            "species_to_genus",
            torch.as_tensor(
                params.species_to_genus,
                dtype=torch.long,
            ),
        )
        self.register_buffer(
            "genus_to_family",
            torch.as_tensor(
                params.genus_to_family,
                dtype=torch.long,
            ),
        )

        self.lambda_family_path = float(
            getattr(params, "loss_family_path_weight", 1.0)
        )
        self.lambda_genus_path = float(
            getattr(params, "loss_genus_path_weight", 1.0)
        )
        self.lambda_species_path = float(
            getattr(params, "loss_species_path_weight", 1.0)
        )
        self.lambda_rank = float(
            getattr(params, "loss_rank_weight", 1.0)
        )
        self.lambda_localization = float(
            getattr(params, "loss_localization_weight", 0.0)
        )
        self.lambda_localization_genus = float(
            getattr(params, "loss_localization_genus_weight", 1.0)
        )
        self.lambda_center_species = float(
            getattr(params, "loss_center_species_weight", 1e-3)
        )
        self.lambda_center_genus = float(
            getattr(params, "loss_center_genus_weight", 1e-3)
        )

        self.rank_enabled = bool(
            getattr(params, "identifiability_enabled", False)
        )
        self.bbox_attention_gate = bool(
            getattr(params, "bbox_attention_gate", False)
        )
        self.eps = float(getattr(params, "spm_eps", 1e-8))

        patch_size = int(getattr(params, "patch_size", 1))
        crop_size = int(getattr(params, "crop_size", 0))
        if crop_size <= 0 or crop_size % patch_size != 0:
            raise ValueError(
                "crop_size는 양수이며 patch_size로 나누어떨어져야 합니다"
            )
        self.grid_h = crop_size // patch_size
        self.grid_w = crop_size // patch_size

    def _bbox_patch_overlap(self, bbox: torch.Tensor) -> torch.Tensor:
        """정규화 bbox와 각 patch의 교집합 비율을 [B,P] tensor로 계산한다."""
        device, dtype = bbox.device, bbox.dtype
        x_left = (
            torch.arange(self.grid_w, device=device, dtype=dtype)
            / self.grid_w
        )
        x_right = (
            torch.arange(self.grid_w, device=device, dtype=dtype) + 1
        ) / self.grid_w
        y_top = (
            torch.arange(self.grid_h, device=device, dtype=dtype)
            / self.grid_h
        )
        y_bottom = (
            torch.arange(self.grid_h, device=device, dtype=dtype) + 1
        ) / self.grid_h

        overlap_x = (
            torch.minimum(bbox[:, None, 2], x_right[None, :])
            - torch.maximum(bbox[:, None, 0], x_left[None, :])
        ).clamp_min(0.0)
        overlap_y = (
            torch.minimum(bbox[:, None, 3], y_bottom[None, :])
            - torch.maximum(bbox[:, None, 1], y_top[None, :])
        ).clamp_min(0.0)

        overlap = overlap_y[:, :, None] * overlap_x[:, None, :]
        patch_area = 1.0 / (self.grid_h * self.grid_w)
        return (
            overlap / patch_area
        ).clamp(0.0, 1.0).reshape(bbox.shape[0], -1)

    def build_patch_prior(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        excluded_mask: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """유효 bbox가 있는 표본에만 attention용 patch prior를 구성한다."""
        if not self.bbox_attention_gate:
            return None

        valid = batch["bbox_valid"].bool().clone()
        if excluded_mask is not None:
            valid &= ~excluded_mask.to(
                device=valid.device,
                dtype=torch.bool,
            )

        if not bool(valid.any()):
            return None

        overlap = self._bbox_patch_overlap(batch["bbox"].float())
        prior = torch.ones_like(overlap)
        prior[valid] = overlap[valid]
        return prior

    def _localization_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        eligible: torch.Tensor,
    ) -> torch.Tensor:
        """정답 species/genus CAM이 bbox 내부에 두는 질량의 NLL을 계산한다."""
        batch_size = batch["species_target"].shape[0]

        if self.lambda_localization <= 0.0:
            return outputs["species_logits"].new_zeros(batch_size)

        bbox_valid = batch["bbox_valid"].bool() & eligible
        if not bool(bbox_valid.any()):
            return outputs["species_logits"].new_zeros(batch_size)

        patch_mask = self._bbox_patch_overlap(batch["bbox"].float())
        indices = torch.arange(
            batch_size,
            device=patch_mask.device,
        )

        species_cam = outputs["species_cam"][
            indices,
            batch["species_target"],
        ]
        genus_cam = outputs["genus_cam"][
            indices,
            batch["genus_target"],
        ]

        species_mass = (
            species_cam * patch_mask
        ).sum(dim=-1).clamp_min(self.eps)
        genus_mass = (
            genus_cam * patch_mask
        ).sum(dim=-1).clamp_min(self.eps)

        values = (
            -species_mass.log()
            - self.lambda_localization_genus * genus_mass.log()
        )
        return torch.where(
            bbox_valid,
            values,
            torch.zeros_like(values),
        )

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        *,
        mix: Optional[SnapMixBatch] = None,
        regularization: Optional[Dict[str, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """식별 가능 수준에 맞는 taxonomy path NLL을 선택한다."""

        species_target = batch["species_target"]
        genus_target = batch["genus_target"]
        family_target = batch["family_target"]
        rank_target = batch["rank_target"]
        batch_size = species_target.shape[0]

        applied_mask = (
            mix.applied_mask.bool()
            if mix is not None and mix.applied_mask is not None
            else torch.zeros(
                batch_size,
                dtype=torch.bool,
                device=species_target.device,
            )
        )
        species_mask = rank_target.eq(RANK_SPECIES)
        genus_mask = rank_target.eq(RANK_GENUS)
        family_mask = rank_target.eq(RANK_FAMILY)

        indices = torch.arange(
            batch_size,
            device=species_target.device,
        )

        family_nll = -outputs["family_log_probabilities"][
            indices,
            family_target,
        ]
        genus_nll = -outputs[
            "genus_conditional_log_probabilities"
        ][
            indices,
            genus_target,
        ]
        species_nll = -outputs[
            "species_conditional_log_probabilities"
        ][
            indices,
            species_target,
        ]

        species_objective = (
            self.lambda_family_path * family_nll
            + self.lambda_genus_path * genus_nll
            + self.lambda_species_path * species_nll
        )
        genus_objective = (
            self.lambda_family_path * family_nll
            + self.lambda_genus_path * genus_nll
        )
        family_objective = self.lambda_family_path * family_nll

        taxonomic = torch.zeros_like(family_nll)
        taxonomic = torch.where(
            species_mask,
            species_objective,
            taxonomic,
        )
        taxonomic = torch.where(
            genus_mask,
            genus_objective,
            taxonomic,
        )
        taxonomic = torch.where(
            family_mask,
            family_objective,
            taxonomic,
        )

        mixed_objective = torch.zeros_like(taxonomic)
        if mix is not None and bool(applied_mask.any()):
            genus_target_b = mix.genus_target_b
            if genus_target_b is None:
                genus_target_b = self.species_to_genus.index_select(
                    0,
                    mix.target_b,
                )

            family_target_b = mix.family_target_b
            if family_target_b is None:
                family_target_b = self.genus_to_family.index_select(
                    0,
                    genus_target_b,
                )

            family_b_nll = -outputs["family_log_probabilities"][
                indices,
                family_target_b,
            ]
            genus_b_nll = -outputs[
                "genus_conditional_log_probabilities"
            ][
                indices,
                genus_target_b,
            ]
            species_b_nll = -outputs[
                "species_conditional_log_probabilities"
            ][
                indices,
                mix.target_b,
            ]

            species_a_weight = mix.weight_a.to(species_nll.dtype)
            species_b_weight = mix.weight_b.to(species_nll.dtype)

            genus_a_weight = (
                mix.genus_weight_a.to(genus_nll.dtype)
                if mix.genus_weight_a is not None
                else species_a_weight
            )
            genus_b_weight = (
                mix.genus_weight_b.to(genus_nll.dtype)
                if mix.genus_weight_b is not None
                else species_b_weight
            )

            family_a_weight = (
                mix.family_weight_a.to(family_nll.dtype)
                if mix.family_weight_a is not None
                else genus_a_weight
            )
            family_b_weight = (
                mix.family_weight_b.to(family_nll.dtype)
                if mix.family_weight_b is not None
                else genus_b_weight
            )

            mixed_objective = (
                self.lambda_family_path
                * (
                    family_a_weight * family_nll
                    + family_b_weight * family_b_nll
                )
                + self.lambda_genus_path
                * (
                    genus_a_weight * genus_nll
                    + genus_b_weight * genus_b_nll
                )
                + self.lambda_species_path
                * (
                    species_a_weight * species_nll
                    + species_b_weight * species_b_nll
                )
            )

            taxonomic = torch.where(
                applied_mask,
                mixed_objective,
                taxonomic,
            )

        rank_loss = F.cross_entropy(
            outputs["rank_logits"],
            rank_target,
            reduction="none",
        )
        rank_eligible = (
            ~applied_mask
            if self.rank_enabled
            else torch.zeros_like(species_mask)
        )

        localization_eligible = species_mask & ~applied_mask
        localization = self._localization_loss(
            outputs,
            batch,
            localization_eligible,
        )

        total = taxonomic.mean()

        if self.rank_enabled:
            total = total + self.lambda_rank * _masked_mean(
                rank_loss,
                rank_eligible,
            )

        if self.lambda_localization > 0.0:
            bbox_mask = (
                batch["bbox_valid"].bool()
                & localization_eligible
            )
            total = total + self.lambda_localization * _masked_mean(
                localization,
                bbox_mask,
            )

        regularization = regularization or {}
        center_species = regularization.get(
            "center_species",
            total.new_zeros(()),
        )
        center_genus = regularization.get(
            "center_genus",
            total.new_zeros(()),
        )

        total = (
            total
            + self.lambda_center_species * center_species
            + self.lambda_center_genus * center_genus
        )

        metrics = {
            "loss_total": total.detach(),
            "loss_taxonomic": taxonomic.mean().detach(),
            "loss_family_path": _masked_mean(
                family_nll,
                (species_mask | genus_mask | family_mask) & ~applied_mask,
            ).detach(),
            "loss_genus_path": _masked_mean(
                genus_nll,
                (species_mask | genus_mask) & ~applied_mask,
            ).detach(),
            "loss_species_path": _masked_mean(
                species_nll,
                species_mask & ~applied_mask,
            ).detach(),
            "loss_snapmix_path": _masked_mean(
                mixed_objective,
                applied_mask,
            ).detach(),
            # 기존 로그 이름과의 임시 호환 alias.
            "loss_leaf": _masked_mean(
                species_nll,
                species_mask & ~applied_mask,
            ).detach(),
            "loss_genus": _masked_mean(
                genus_nll,
                (species_mask | genus_mask) & ~applied_mask,
            ).detach(),
            "loss_family": _masked_mean(
                family_nll,
                species_mask | genus_mask | family_mask,
            ).detach(),
            "loss_rank": _masked_mean(
                rank_loss,
                rank_eligible,
            ).detach(),
            "loss_localization": _masked_mean(
                localization,
                batch["bbox_valid"].bool() & localization_eligible,
            ).detach(),
            "loss_center_species": center_species.detach(),
            "loss_center_genus": center_genus.detach(),
        }

        return total, metrics
