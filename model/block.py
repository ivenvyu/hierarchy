"""attention과 MLP를 residual 연결로 조합한 Prompt-CAM Transformer block을 정의한다."""

import torch.nn as nn
from timm.layers import DropPath
from timm.models.vision_transformer import LayerScale
from timm.layers.trace_utils import _assert
import torch.nn.functional as F
import torch
from typing import Any, Callable, Dict, Optional, Set, Tuple, Type, Union, List
from model.mlp import MlpPETL
from model.attention import AttentionPETL

class BlockPETL(nn.Module):
    """self-attention과 MLP residual branch를 결합한 Transformer block이다."""
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = MlpPETL,
            params=None
    ) -> None:
        """객체가 사용할 입력 설정과 내부 상태를 초기화한다."""
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = AttentionPETL(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            ############# 추가 모듈 #############
            params=params
            ############# 추가 모듈 끝 #############
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        ############# 추가 모듈 #############
        self.params = params
        ############# 추가 모듈 끝 #############

    def forward(self, x: torch.Tensor, idx, blur_head_lst=[], target_cls=-1) -> Tuple[torch.Tensor,torch.Tensor]:
        """입력 tensor를 계층의 순전파 연산에 통과시켜 결과를 반환한다."""
        output, attn_map = self.attn(self.norm1(x), idx , blur_head_lst=blur_head_lst, target_cls=target_cls)
        x = x + self.drop_path1(self.ls1(output))
        # feed-forward MLP 출력을 두 번째 residual branch에 더한다.
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x,attn_map
