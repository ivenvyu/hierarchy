"""과 기본 벡터, 속 잔차, 종 잔차로 계층 프롬프트 토큰을 구성하고 정규화 손실을 계산한다."""

from __future__ import annotations

from functools import reduce
from operator import mul
from typing import Dict, Tuple
import math

import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair


class VPT(nn.Module):
    """Transformer 각 활성 층의 과·속·종 계층 prompt 파라미터를 관리한다."""

    def __init__(self, params, depth, patch_size, embed_dim):
        """taxonomy mapping과 층별 family base, genus residual, species residual을 초기화한다."""
        super().__init__()
        self.params = params
        self.depth = depth
        self.embed_dim = int(embed_dim)
        if params.train_type == "prompt_cam":
            # Prompt-CAM은 활성화된 모든 디코더 계층에서 클래스/분류 체계 프롬프트가 필요하다.
            prompt_layer = params.vpt_layer if params.vpt_layer else depth
        elif params.vpt_mode == "shallow":
            prompt_layer = 1
        elif params.vpt_mode == "deep":
            prompt_layer = params.vpt_layer if params.vpt_layer else depth
        else:
            raise ValueError("VPT에는 vpt_mode 또는 train_type='prompt_cam'이 필요합니다")
        self.prompt_layer = int(prompt_layer)
        val = math.sqrt(6.0 / float(3 * reduce(mul, _pair(patch_size), 1) + embed_dim))
        self.hierarchical = bool(getattr(params, "hierarchical_prompt", False))

        if self.hierarchical:
            species_to_genus = torch.as_tensor(params.species_to_genus, dtype=torch.long)
            genus_to_family = torch.as_tensor(params.genus_to_family, dtype=torch.long)
            self.num_species = int(params.class_num)
            self.num_genera = int(params.num_genera)
            self.num_families = int(params.num_families)
            if species_to_genus.shape != (self.num_species,):
                raise ValueError("species_to_genus 형태가 class_num과 일치하지 않습니다")
            if genus_to_family.shape != (self.num_genera,):
                raise ValueError("genus_to_family 형태가 num_genera와 일치하지 않습니다")
            self.register_buffer("species_to_genus", species_to_genus, persistent=True)
            self.register_buffer("genus_to_family", genus_to_family, persistent=True)

            self.family_base = nn.Parameter(
                torch.empty(self.prompt_layer, self.num_families, embed_dim)
            )
            self.genus_residual = nn.Parameter(
                torch.empty(self.prompt_layer, self.num_genera, embed_dim)
            )
            self.species_residual = nn.Parameter(
                torch.empty(self.prompt_layer, self.num_species, embed_dim)
            )
            for parameter in (self.family_base, self.genus_residual, self.species_residual):
                nn.init.uniform_(parameter, -val, val)
            self.prompt_count = (
                self.num_families
                + self.num_genera
                + self.num_species
            )
            if int(params.vpt_num) != self.prompt_count:
                raise ValueError(
                    f"계층 프롬프트 수는 F+G+C={self.prompt_count}여야 하지만 "
                    f"{params.vpt_num}입니다"
                )
        else:
            self.prompt_embeddings = nn.Parameter(
                torch.zeros(self.prompt_layer, params.vpt_num, embed_dim)
            )
            nn.init.uniform_(self.prompt_embeddings.data, -val, val)
            self.prompt_count = int(params.vpt_num)

        self.prompt_dropout = nn.Dropout(params.vpt_dropout)

    def _prompt_index(self, block_index: int) -> int | None:
        """backbone block 인덱스를 활성 prompt 파라미터 인덱스로 변환한다."""
        index = int(block_index)
        if self.params.vpt_layer:
            index = index - (self.depth - int(self.params.vpt_layer))
            if index < 0:
                return None
        if index >= self.prompt_layer:
            return None
        return index

    def _components_from_prompt_index(
        self,
        prompt_index: int,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """prompt-local index로 과·속·종 계층 표현을 구성한다.

        ``prompt_index``는 전체 backbone block 번호가 아니라
        활성 prompt parameter 내부의 0부터 시작하는 index이다.
        """
        if not self.hierarchical:
            raise RuntimeError(
                "_components_from_prompt_index()는 계층 프롬프트에만 정의되어 있습니다"
            )

        index = int(prompt_index)

        if index < 0 or index >= self.prompt_layer:
            raise IndexError(
                f"프롬프트 인덱스 {index}이(가) 범위를 벗어났습니다: "
                f"[0, {self.prompt_layer})"
            )

        family = self.family_base[index]
        genus_residual = self.genus_residual[index]
        species_residual = self.species_residual[index]

        genus = (
            family.index_select(0, self.genus_to_family)
            + genus_residual
        )

        species = (
            genus.index_select(0, self.species_to_genus)
            + species_residual
        )

        return (
            family,
            genus_residual,
            species_residual,
            genus,
            species,
        )

    def components(
        self,
        block_index: int,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """전체 backbone block 번호에 해당하는 계층 prompt를 반환한다."""
        prompt_index = self._prompt_index(block_index)

        if prompt_index is None:
            raise IndexError(
                f"블록 {block_index}에서 활성화된 프롬프트가 없습니다"
            )

        return self._components_from_prompt_index(prompt_index)

    def retrieve_prompt(self, index, batch_size):
        """고유 과·속·종 토큰을 결합해 batch 크기만큼 복제한다."""
        prompt_index = self._prompt_index(index)
        if prompt_index is None:
            return None
        if self.hierarchical:
            family, _, _, genus, species = self.components(index)
            # 토큰 순서는 항상 [family, genus, species]로 고정한다.
            prompt = torch.cat([family, genus, species], dim=0)
        else:
            prompt = self.prompt_embeddings[prompt_index]
        return self.prompt_dropout(prompt).unsqueeze(0).expand(batch_size, -1, -1)

    def center_losses(self) -> Dict[str, torch.Tensor]:
        """같은 속의 종 잔차와 같은 과의 속 잔차 평균이 0이 되도록 중심화 손실을 계산한다."""
        if not self.hierarchical:
            zero = next(self.parameters()).new_zeros(())
            return {"species": zero, "genus": zero}

        species_terms = []
        for genus_index in range(self.num_genera):
            members = torch.nonzero(self.species_to_genus == genus_index, as_tuple=False).flatten()
            if members.numel() > 1:
                mean = self.species_residual.index_select(1, members).mean(dim=1)
                species_terms.append(mean.square().sum(dim=-1).mean())
        genus_terms = []
        for family_index in range(self.num_families):
            members = torch.nonzero(self.genus_to_family == family_index, as_tuple=False).flatten()
            if members.numel() > 1:
                mean = self.genus_residual.index_select(1, members).mean(dim=1)
                genus_terms.append(mean.square().sum(dim=-1).mean())
        zero = self.family_base.new_zeros(())
        return {
            "species": torch.stack(species_terms).mean() if species_terms else zero,
            "genus": torch.stack(genus_terms).mean() if genus_terms else zero,
        }
