"""프롬프트 토큰의 계층적 attention mask를 적용하는 multi-head self-attention을 구현한다."""

from torch.jit import Final
import torch.nn as nn
from timm.layers import use_fused_attn
import torch
import torch.nn.functional as F
from typing import Tuple


def _prompt_is_active(params, block_idx: int) -> bool:
    """현재 block이 설정된 VPT 활성 구간에 포함되는지 판단한다."""

    if getattr(params, 'train_type', None) not in {'vpt', 'prompt_cam'}:
        return False
    configured = getattr(params, 'vpt_layer', None)
    if configured in (None, 0, '0', '', 'null'):
        return True
    depth = int(getattr(params, 'model_depth', 0))
    active_layers = int(configured)
    if depth <= 0:
        raise ValueError('계층 어텐션을 사용하기 전에 model_depth를 기록해야 합니다')
    if not 1 <= active_layers <= depth:
        raise ValueError(f'vpt_layer는 [1,{depth}] 범위여야 하지만 {active_layers}입니다')
    return int(block_idx) >= depth - active_layers


class AttentionPETL(nn.Module):
    """계층 prompt mask를 지원하는 multi-head self-attention 계층이다."""
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
            params=None,
    ) -> None:
        """객체가 사용할 입력 설정과 내부 상태를 초기화한다."""
        super().__init__()
        assert dim % num_heads == 0, 'dim은 num_heads로 나누어떨어져야 합니다'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        ############# 추가 모듈 #############
        self.params = params
        ############# 추가 모듈 끝 #############

    def forward(self, x: torch.Tensor, block_idx, blur_head_lst=[],target_cls=-1) -> Tuple[torch.Tensor,torch.Tensor]:
        """입력 tensor를 계층의 순전파 연산에 통과시켜 결과를 반환한다."""
        B, N, C = x.shape
        ############# 추가 모듈 #############
        qkv = self.qkv(x)
        ############# 추가 모듈 끝 #############

        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        ############# 추가 모듈 #############
        prompt_active = _prompt_is_active(self.params, block_idx)
        hierarchical_active = (
            prompt_active and bool(getattr(self.params, 'hierarchical_prompt', False))
        )
        if hierarchical_active:
            family_count = int(getattr(self.params, 'num_families', 0))
            genus_count = int(getattr(self.params, 'num_genera', 0))
            species_count = int(getattr(self.params, 'class_num', 0))

            family_end = family_count
            genus_end = family_end + genus_count
            prompt_end = genus_end + species_count

            if (
                family_count <= 0
                or genus_count <= 0
                or species_count <= 0
                or N < prompt_end
            ):
                raise ValueError(
                    '계층 프롬프트 배치가 잘못되었습니다: '
                    f'N={N}, F={family_count}, G={genus_count}, C={species_count}'
                )

            negative_infinity = torch.finfo(attn.dtype).min

            # family query는 하위 genus/species prompt key를 읽지 않는다.
            attn[
                :,
                :,
                0:family_end,
                family_end:prompt_end,
            ] = negative_infinity

            # genus query는 하위 species prompt key를 읽지 않는다.
            attn[
                :,
                :,
                family_end:genus_end,
                genus_end:prompt_end,
            ] = negative_infinity

        if len(blur_head_lst) != 0:
            if not prompt_active:
                raise ValueError('프롬프트가 비활성인 블록에서는 프롬프트 헤드를 흐릴 수 없습니다')
            query_index = int(target_cls)
            if hierarchical_active and query_index >= 0:
                query_index += (
                    int(getattr(self.params, 'num_families', 0))
                    + int(getattr(self.params, 'num_genera', 0))
                )
            attn[:, blur_head_lst, query_index, :] = 0

        ############# 추가 모듈 끝 #############

        # key token 축으로 정규화해 각 query의 attention 합을 1로 만든다.
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        proj = self.proj(x)
        x = self.proj_drop(proj)
        return x,attn
