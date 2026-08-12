"""공유 과·속·종 prompt를 node별 Prompt-CAM 분류로 연결한다."""

from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalPromptDecoder(nn.Module):
    """표현은 계층적으로 공유하고 확률은 taxonomy node별로 정규화한다.

    prompt 표현은 VPT에서 다음처럼 만들어진다.

    A_f                      : family prompt
    U_g = A_{phi(g)} + R_g   : genus prompt
    V_c = U_{gamma(c)} + S_c : species prompt

    이 decoder는 각 수준의 prompt가 patch를 직접 읽도록 만든 뒤
    P(f|x), P(g|f,x), P(c|g,x)를 계산한다.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        species_to_genus: list[int],
        genus_to_family: list[int],
        genus_counts: list[int],
        *,
        eps: float = 1e-8,
        rank_classes: int = 4,
    ) -> None:
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim은 num_heads로 나누어떨어져야 합니다")

        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.eps = float(eps)

        species_to_genus_tensor = torch.as_tensor(
            species_to_genus,
            dtype=torch.long,
        )
        genus_to_family_tensor = torch.as_tensor(
            genus_to_family,
            dtype=torch.long,
        )
        genus_counts_tensor = torch.as_tensor(
            genus_counts,
            dtype=torch.long,
        )

        self.num_species = int(species_to_genus_tensor.numel())
        self.num_genera = int(genus_to_family_tensor.numel())
        self.num_families = int(genus_to_family_tensor.max().item()) + 1

        if species_to_genus_tensor.shape != (self.num_species,):
            raise ValueError("species_to_genus 형태가 잘못되었습니다")
        if genus_to_family_tensor.shape != (self.num_genera,):
            raise ValueError("genus_to_family 형태가 잘못되었습니다")
        if genus_counts_tensor.shape != (self.num_genera,):
            raise ValueError("genus_counts 형태가 num_genera와 일치하지 않습니다")
        if species_to_genus_tensor.min().item() < 0:
            raise ValueError("species_to_genus에는 음수 인덱스를 사용할 수 없습니다")
        if species_to_genus_tensor.max().item() >= self.num_genera:
            raise ValueError("species_to_genus가 num_genera 범위를 벗어났습니다")
        if genus_to_family_tensor.min().item() < 0:
            raise ValueError("genus_to_family에는 음수 인덱스를 사용할 수 없습니다")

        computed_genus_counts = torch.bincount(
            species_to_genus_tensor,
            minlength=self.num_genera,
        )
        if not torch.equal(computed_genus_counts, genus_counts_tensor):
            raise ValueError(
                "genus_counts가 species_to_genus에서 계산한 값과 일치하지 않습니다"
            )

        family_counts_tensor = torch.bincount(
            genus_to_family_tensor,
            minlength=self.num_families,
        )
        if bool(family_counts_tensor.eq(0).any()):
            raise ValueError("자식 genus가 없는 family가 존재합니다")

        species_to_family_tensor = genus_to_family_tensor.index_select(
            0,
            species_to_genus_tensor,
        )

        self.register_buffer(
            "species_to_genus",
            species_to_genus_tensor,
            persistent=True,
        )
        self.register_buffer(
            "genus_to_family",
            genus_to_family_tensor,
            persistent=True,
        )
        self.register_buffer(
            "species_to_family",
            species_to_family_tensor,
            persistent=True,
        )
        self.register_buffer(
            "genus_counts",
            genus_counts_tensor,
            persistent=True,
        )
        self.register_buffer(
            "family_counts",
            family_counts_tensor,
            persistent=True,
        )

        self.family_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.family_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.family_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.family_out = nn.Linear(embed_dim, embed_dim, bias=False)
        self.family_norm = nn.LayerNorm(embed_dim)

        self.genus_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.genus_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.genus_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.genus_out = nn.Linear(embed_dim, embed_dim, bias=False)
        self.genus_norm = nn.LayerNorm(embed_dim)

        self.species_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.species_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.species_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.species_out = nn.Linear(embed_dim, embed_dim, bias=False)
        self.species_norm = nn.LayerNorm(embed_dim)

        # 각 수준의 head 결합 비율은 simplex 위에서 학습한다.
        self.family_head_logits = nn.Parameter(torch.zeros(num_heads))
        self.genus_head_logits = nn.Parameter(torch.zeros(num_heads))
        self.species_head_logits = nn.Parameter(torch.zeros(num_heads))

        # 동일 taxonomy level의 모든 node child가 하나의 scalar head를 공유한다.
        self.family_score = nn.Linear(embed_dim, 1, bias=False)
        self.genus_score = nn.Linear(embed_dim, 1, bias=False)
        self.species_score = nn.Linear(embed_dim, 1, bias=False)

        self.rank_head = nn.Linear(embed_dim, rank_classes)

    def _split_heads(
        self,
        tensor: torch.Tensor,
        token_count: int,
    ) -> torch.Tensor:
        batch = tensor.shape[0]
        return tensor.reshape(
            batch,
            token_count,
            self.num_heads,
            self.head_dim,
        ).permute(0, 2, 1, 3)

    def _cross_attention(
        self,
        query_tokens: torch.Tensor,
        patch_tokens: torch.Tensor,
        *,
        q_proj: nn.Linear,
        k_proj: nn.Linear,
        v_proj: nn.Linear,
        output_proj: nn.Linear,
        norm: nn.LayerNorm,
        head_weight: torch.Tensor,
        energy_bias: torch.Tensor | None = None,
        blur_head_indices: Sequence[int] | None = None,
        blur_query_indices: Sequence[int] | int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """query-to-patch attention과 image-conditioned prompt를 계산한다."""

        token_count = query_tokens.shape[1]
        patch_count = patch_tokens.shape[1]

        q = self._split_heads(
            q_proj(query_tokens),
            token_count,
        )
        k = self._split_heads(
            k_proj(patch_tokens),
            patch_count,
        )
        v = self._split_heads(
            v_proj(patch_tokens),
            patch_count,
        )

        energy = torch.matmul(
            q,
            k.transpose(-2, -1),
        ) * self.scale

        if energy_bias is not None:
            energy = energy + energy_bias

        if blur_head_indices:
            if blur_query_indices is None:
                raise ValueError(
                    "blur_head_indices를 설정하면 blur_query_indices가 필요합니다"
                )

            head_indices = sorted({int(index) for index in blur_head_indices})
            invalid_heads = [
                index
                for index in head_indices
                if not 0 <= index < self.num_heads
            ]
            if invalid_heads:
                raise IndexError(
                    f"잘못된 헤드 인덱스 {invalid_heads}; num_heads={self.num_heads}"
                )

            if isinstance(blur_query_indices, int):
                query_indices = [int(blur_query_indices)]
            else:
                query_indices = sorted(
                    {int(index) for index in blur_query_indices}
                )

            if not query_indices:
                raise ValueError("blur_query_indices에는 질의가 하나 이상 있어야 합니다")

            invalid_queries = [
                index
                for index in query_indices
                if not 0 <= index < token_count
            ]
            if invalid_queries:
                raise IndexError(
                    f"잘못된 질의 인덱스 {invalid_queries}; 질의 수={token_count}"
                )

            energy = energy.clone()
            head_tensor = torch.as_tensor(
                head_indices,
                device=energy.device,
                dtype=torch.long,
            )

            # 선택한 query/head의 patch energy를 0으로 만들어 uniform attention으로 둔다.
            for query_index in query_indices:
                energy[
                    :,
                    head_tensor,
                    query_index,
                    :,
                ] = 0

        attention = energy.softmax(dim=-1)
        pooled = torch.matmul(attention, v)

        if head_weight.shape != (self.num_heads,):
            raise ValueError(
                f"head_weight는 [H]=({self.num_heads},)여야 하지만 "
                f"{tuple(head_weight.shape)}입니다"
            )

        # 초기 uniform head weight에서 feature scale이 기존 concat과 같도록 H를 곱한다.
        scaled_head_weight = head_weight.to(
            device=pooled.device,
            dtype=pooled.dtype,
        ).view(
            1,
            self.num_heads,
            1,
            1,
        ) * self.num_heads
        pooled = pooled * scaled_head_weight

        pooled = pooled.permute(
            0,
            2,
            1,
            3,
        ).reshape(
            query_tokens.shape[0],
            token_count,
            self.embed_dim,
        )

        updated = norm(
            output_proj(pooled)
        )
        return updated, attention

    @staticmethod
    def _group_log_softmax(
        logits: torch.Tensor,
        child_to_parent: torch.Tensor,
        num_parents: int,
    ) -> torch.Tensor:
        """각 parent의 child 집합 안에서 독립적으로 log-softmax한다."""

        if logits.ndim != 2:
            raise ValueError(
                f"logits는 [B,K]여야 하지만 {tuple(logits.shape)}입니다"
            )
        if child_to_parent.shape != (logits.shape[1],):
            raise ValueError(
                "child_to_parent 길이가 logits의 child 수와 일치하지 않습니다"
            )

        result = torch.empty_like(logits)

        for parent_index in range(int(num_parents)):
            members = torch.nonzero(
                child_to_parent.eq(parent_index),
                as_tuple=False,
            ).flatten()

            if members.numel() == 0:
                raise RuntimeError(
                    f"parent {parent_index}에 child가 없습니다"
                )

            local_logits = logits.index_select(1, members)

            if members.numel() == 1:
                # 자식이 하나면 조건부 확률은 구조적으로 1이므로 log-probability는 0이다.
                local_log_probabilities = torch.zeros_like(local_logits)
            else:
                working_logits = (
                    local_logits.float()
                    if local_logits.dtype in {torch.float16, torch.bfloat16}
                    else local_logits
                )
                local_log_probabilities = F.log_softmax(
                    working_logits,
                    dim=1,
                ).to(dtype=logits.dtype)

            result = result.index_copy(
                1,
                members,
                local_log_probabilities,
            )

        return result

    def forward(
        self,
        family_tokens: torch.Tensor,
        genus_tokens: torch.Tensor,
        species_tokens: torch.Tensor,
        patch_tokens: torch.Tensor,
        global_feature: torch.Tensor,
        patch_prior: torch.Tensor | None = None,
        blur_family_heads: Sequence[int] | None = None,
        target_family: int | None = None,
        blur_genus_heads: Sequence[int] | None = None,
        target_genus: int | None = None,
        blur_species_heads: Sequence[int] | None = None,
        target_species: int | None = None,
    ) -> Dict[str, torch.Tensor]:
        """node별 CAM, 조건부 확률, 전역 경로 확률을 계산한다."""

        if family_tokens.shape[1] != self.num_families:
            raise ValueError(
                f"과 토큰 {self.num_families}개가 필요하지만 "
                f"{family_tokens.shape[1]}개입니다"
            )
        if genus_tokens.shape[1] != self.num_genera:
            raise ValueError(
                f"속 토큰 {self.num_genera}개가 필요하지만 "
                f"{genus_tokens.shape[1]}개입니다"
            )
        if species_tokens.shape[1] != self.num_species:
            raise ValueError(
                f"종 토큰 {self.num_species}개가 필요하지만 "
                f"{species_tokens.shape[1]}개입니다"
            )

        spatial_bias = None
        if patch_prior is not None:
            expected_shape = (
                patch_tokens.shape[0],
                patch_tokens.shape[1],
            )
            if patch_prior.shape != expected_shape:
                raise ValueError(
                    "patch_prior는 [B,P]여야 합니다: "
                    f"expected={expected_shape}, got={tuple(patch_prior.shape)}"
                )
            spatial_bias = patch_prior.to(
                device=patch_tokens.device,
                dtype=patch_tokens.dtype,
            ).clamp_min(self.eps).log()[
                :,
                None,
                None,
                :,
            ]

        family_head_weight = self.family_head_logits.softmax(dim=0)
        genus_head_weight = self.genus_head_logits.softmax(dim=0)
        species_head_weight = self.species_head_logits.softmax(dim=0)

        family_query_indices = None
        if blur_family_heads:
            if target_family is None:
                raise ValueError(
                    "blur_family_heads를 설정하면 target_family가 필요합니다"
                )
            family_query_indices = [int(target_family)]

        genus_query_indices = None
        if blur_genus_heads:
            if target_genus is None:
                raise ValueError(
                    "blur_genus_heads를 설정하면 target_genus가 필요합니다"
                )
            genus_query_indices = [int(target_genus)]

        species_query_indices = None
        if blur_species_heads:
            if target_species is None:
                raise ValueError(
                    "blur_species_heads를 설정하면 target_species가 필요합니다"
                )
            species_query_indices = [int(target_species)]

        family_updated, family_attention = self._cross_attention(
            family_tokens,
            patch_tokens,
            q_proj=self.family_q,
            k_proj=self.family_k,
            v_proj=self.family_v,
            output_proj=self.family_out,
            norm=self.family_norm,
            head_weight=family_head_weight,
            energy_bias=spatial_bias,
            blur_head_indices=blur_family_heads,
            blur_query_indices=family_query_indices,
        )
        genus_updated, genus_attention = self._cross_attention(
            genus_tokens,
            patch_tokens,
            q_proj=self.genus_q,
            k_proj=self.genus_k,
            v_proj=self.genus_v,
            output_proj=self.genus_out,
            norm=self.genus_norm,
            head_weight=genus_head_weight,
            energy_bias=spatial_bias,
            blur_head_indices=blur_genus_heads,
            blur_query_indices=genus_query_indices,
        )
        species_updated, species_attention = self._cross_attention(
            species_tokens,
            patch_tokens,
            q_proj=self.species_q,
            k_proj=self.species_k,
            v_proj=self.species_v,
            output_proj=self.species_out,
            norm=self.species_norm,
            head_weight=species_head_weight,
            energy_bias=spatial_bias,
            blur_head_indices=blur_species_heads,
            blur_query_indices=species_query_indices,
        )

        family_cam = torch.einsum(
            "h,bhfp->bfp",
            family_head_weight,
            family_attention,
        )
        genus_cam = torch.einsum(
            "h,bhgp->bgp",
            genus_head_weight,
            genus_attention,
        )
        species_cam = torch.einsum(
            "h,bhcp->bcp",
            species_head_weight,
            species_attention,
        )

        # child가 하나뿐인 node에는 구분 문제 자체가 없다. 그 경우 하위 CAM을
        # 임의의 미학습 CAM으로 두지 않고 부모 node CAM으로 정의한다.
        genus_has_siblings = self.family_counts.index_select(
            0,
            self.genus_to_family,
        ).gt(1)
        parent_family_cam = family_cam.index_select(
            1,
            self.genus_to_family,
        )
        genus_cam = torch.where(
            genus_has_siblings.view(1, -1, 1),
            genus_cam,
            parent_family_cam,
        )

        parent_family_attention = family_attention.index_select(
            2,
            self.genus_to_family,
        )
        genus_attention = torch.where(
            genus_has_siblings.view(1, 1, -1, 1),
            genus_attention,
            parent_family_attention,
        )

        species_has_siblings = self.genus_counts.index_select(
            0,
            self.species_to_genus,
        ).gt(1)
        parent_genus_cam = genus_cam.index_select(
            1,
            self.species_to_genus,
        )
        species_cam = torch.where(
            species_has_siblings.view(1, -1, 1),
            species_cam,
            parent_genus_cam,
        )

        parent_genus_attention = genus_attention.index_select(
            2,
            self.species_to_genus,
        )
        species_attention = torch.where(
            species_has_siblings.view(1, 1, -1, 1),
            species_attention,
            parent_genus_attention,
        )

        family_node_logits = self.family_score(
            family_updated
        ).squeeze(-1)
        genus_node_logits = self.genus_score(
            genus_updated
        ).squeeze(-1)
        species_node_logits = self.species_score(
            species_updated
        ).squeeze(-1)

        family_log_probabilities = F.log_softmax(
            family_node_logits.float(),
            dim=1,
        ).to(dtype=family_node_logits.dtype)
        genus_conditional_log_probabilities = self._group_log_softmax(
            genus_node_logits,
            self.genus_to_family,
            self.num_families,
        )
        species_conditional_log_probabilities = self._group_log_softmax(
            species_node_logits,
            self.species_to_genus,
            self.num_genera,
        )

        genus_log_probabilities = (
            family_log_probabilities.index_select(
                1,
                self.genus_to_family,
            )
            + genus_conditional_log_probabilities
        )
        species_log_probabilities = (
            family_log_probabilities.index_select(
                1,
                self.species_to_family,
            )
            + genus_conditional_log_probabilities.index_select(
                1,
                self.species_to_genus,
            )
            + species_conditional_log_probabilities
        )

        family_probabilities = family_log_probabilities.exp()
        genus_conditional_probabilities = (
            genus_conditional_log_probabilities.exp()
        )
        species_conditional_probabilities = (
            species_conditional_log_probabilities.exp()
        )
        genus_probabilities = genus_log_probabilities.exp()
        species_probabilities = species_log_probabilities.exp()

        rank_logits = self.rank_head(global_feature)

        return {
            # 기존 trainer/시각화 코드와의 호환 alias. 값은 raw logit이 아니라
            # taxonomy 전체에서 정규화된 log-probability이다.
            "species_logits": species_log_probabilities,
            "genus_logits": genus_log_probabilities,
            "family_logits": family_node_logits,
            "species_log_probabilities": species_log_probabilities,
            "genus_log_probabilities": genus_log_probabilities,
            "family_log_probabilities": family_log_probabilities,
            "species_node_logits": species_node_logits,
            "genus_node_logits": genus_node_logits,
            "family_node_logits": family_node_logits,
            "species_conditional_log_probabilities": (
                species_conditional_log_probabilities
            ),
            "genus_conditional_log_probabilities": (
                genus_conditional_log_probabilities
            ),
            "species_probabilities": species_probabilities,
            "genus_probabilities": genus_probabilities,
            "family_probabilities": family_probabilities,
            "species_conditional_probabilities": (
                species_conditional_probabilities
            ),
            "genus_conditional_probabilities": (
                genus_conditional_probabilities
            ),
            "species_cam": species_cam,
            "genus_cam": genus_cam,
            "family_cam": family_cam,
            "species_attention_heads": species_attention,
            "genus_attention_heads": genus_attention,
            "family_attention_heads": family_attention,

            "family_head_weights": family_head_weight,
            "genus_head_weights": genus_head_weight,
            "species_head_weights": species_head_weight,

            "species_contrast_defined": species_has_siblings,
            "genus_contrast_defined": genus_has_siblings,
            "rank_logits": rank_logits,
            "species_prediction": species_log_probabilities.argmax(dim=1),
            "genus_prediction": genus_log_probabilities.argmax(dim=1),
            "family_prediction": family_log_probabilities.argmax(dim=1),
            "rank_prediction": rank_logits.argmax(dim=1),
            "species_features": species_updated,
            "genus_features": genus_updated,
            "family_features": family_updated,
        }



class PatchOnlyPromptDecoder(nn.Module):
    """비계층 Prompt-CAM을 residual 없는 prompt-to-patch 경로로 분류한다."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
    ) -> None:
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                "embed_dim은 num_heads로 나누어떨어져야 합니다"
            )

        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(
            embed_dim,
            embed_dim,
            bias=False,
        )
        self.k = nn.Linear(
            embed_dim,
            embed_dim,
            bias=False,
        )
        self.v = nn.Linear(
            embed_dim,
            embed_dim,
            bias=False,
        )
        self.out = nn.Linear(
            embed_dim,
            embed_dim,
            bias=False,
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.head_logits = nn.Parameter(
            torch.zeros(num_heads)
        )
        self.score = nn.Linear(
            embed_dim,
            1,
            bias=False,
        )

    def _split_heads(
        self,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, token_count, _ = tensor.shape

        return tensor.reshape(
            batch_size,
            token_count,
            self.num_heads,
            self.head_dim,
        ).permute(
            0,
            2,
            1,
            3,
        )

    def forward(
        self,
        prompt_tokens: torch.Tensor,
        patch_tokens: torch.Tensor,
        *,
        blur_head_indices: Sequence[int] | None = None,
        target_prompt: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if prompt_tokens.ndim != 3:
            raise ValueError(
                "prompt_tokens는 [B,C,D]여야 합니다"
            )
        if patch_tokens.ndim != 3:
            raise ValueError(
                "patch_tokens는 [B,P,D]여야 합니다"
            )
        if prompt_tokens.shape[0] != patch_tokens.shape[0]:
            raise ValueError(
                "prompt와 patch의 batch 크기가 다릅니다"
            )
        if prompt_tokens.shape[-1] != self.embed_dim:
            raise ValueError(
                "prompt embed_dim이 decoder와 다릅니다"
            )
        if patch_tokens.shape[-1] != self.embed_dim:
            raise ValueError(
                "patch embed_dim이 decoder와 다릅니다"
            )
        if patch_tokens.shape[1] == 0:
            raise ValueError(
                "patch-only decoder에는 patch token이 필요합니다"
            )

        q = self._split_heads(
            self.q(prompt_tokens)
        )
        k = self._split_heads(
            self.k(patch_tokens)
        )
        v = self._split_heads(
            self.v(patch_tokens)
        )

        energy = torch.matmul(
            q,
            k.transpose(-2, -1),
        ) * self.scale

        if blur_head_indices:
            if target_prompt is None:
                raise ValueError(
                    "blur할 때 target_prompt가 필요합니다"
                )

            prompt_index = int(target_prompt)
            if not 0 <= prompt_index < prompt_tokens.shape[1]:
                raise IndexError(
                    f"target_prompt={prompt_index}이 범위를 벗어났습니다"
                )

            head_indices = sorted(
                {int(index) for index in blur_head_indices}
            )
            invalid_heads = [
                index
                for index in head_indices
                if not 0 <= index < self.num_heads
            ]
            if invalid_heads:
                raise IndexError(
                    f"잘못된 head index: {invalid_heads}"
                )

            energy = energy.clone()
            head_tensor = torch.as_tensor(
                head_indices,
                device=energy.device,
                dtype=torch.long,
            )

            # 원논문 greedy pruning과 같은 uniform-attention intervention.
            energy[
                :,
                head_tensor,
                prompt_index,
                :,
            ] = 0

        attention = energy.softmax(dim=-1)
        pooled = torch.matmul(
            attention,
            v,
        )

        head_weight = self.head_logits.softmax(
            dim=0
        ).to(
            device=pooled.device,
            dtype=pooled.dtype,
        ).view(
            1,
            self.num_heads,
            1,
            1,
        ) * self.num_heads

        pooled = pooled * head_weight
        pooled = pooled.permute(
            0,
            2,
            1,
            3,
        ).reshape(
            prompt_tokens.shape[0],
            prompt_tokens.shape[1],
            self.embed_dim,
        )

        # 핵심: prompt_tokens residual을 절대 더하지 않는다.
        updated = self.norm(
            self.out(pooled)
        )

        logits = self.score(
            updated
        ).squeeze(-1)

        return logits, attention
