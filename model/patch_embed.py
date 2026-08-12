"""입력 이미지를 고정 크기 patch token 시퀀스로 투영한다."""

from typing import Callable, List, Optional, Tuple, Union

import torch
from torch import nn as nn
import torch.nn.functional as F

from timm.layers.format import Format, nchw_to
from timm.layers.helpers import to_2tuple
from timm.layers.trace_utils import _assert



class PatchEmbedPETL(nn.Module):
    """이미지를 convolution으로 patch embedding 시퀀스로 변환한다."""
    output_fmt: Format
    dynamic_img_pad: torch.jit.Final[bool]

    def __init__(
            self,
            img_size: Optional[int] = 224,
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            norm_layer: Optional[Callable] = None,
            flatten: bool = True,
            output_fmt: Optional[str] = None,
            bias: bool = True,
            strict_img_size: bool = True,
            dynamic_img_pad: bool = False,
            params = None
    ):
        """객체가 사용할 입력 설정과 내부 상태를 초기화한다."""
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        if img_size is not None:
            self.img_size = to_2tuple(img_size)
            self.grid_size = tuple([s // p for s, p in zip(self.img_size, self.patch_size)])
            self.num_patches = self.grid_size[0] * self.grid_size[1]
        else:
            self.img_size = None
            self.grid_size = None
            self.num_patches = None

        if output_fmt is not None:
            self.flatten = False
            self.output_fmt = Format(output_fmt)
        else:
            # 공간 차원을 평탄화하고 채널을 마지막으로 옮긴다. 이전 버전 호환용이다.
            self.flatten = flatten
            self.output_fmt = Format.NCHW
        self.strict_img_size = strict_img_size
        self.dynamic_img_pad = dynamic_img_pad

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        ############# 추가 모듈 #############
        self.params = params
        self.norm_layer = norm_layer
        ############# 추가 모듈 끝 #############

    def forward(self, x):
        """입력 tensor를 계층의 순전파 연산에 통과시켜 결과를 반환한다."""
        B, C, H, W = x.shape
        if self.img_size is not None:
            if self.strict_img_size:
                _assert(H == self.img_size[0], f"입력 높이({H})가 모델 설정({self.img_size[0]})과 다릅니다.")
                _assert(W == self.img_size[1], f"입력 너비({W})가 모델 설정({self.img_size[1]})과 다릅니다.")
            elif not self.dynamic_img_pad:
                _assert(
                    H % self.patch_size[0] == 0,
                    f"입력 높이({H})는 패치 크기({self.patch_size[0]})로 나누어떨어져야 합니다."
                )
                _assert(
                    W % self.patch_size[1] == 0,
                    f"입력 너비({W})는 패치 크기({self.patch_size[1]})로 나누어떨어져야 합니다."
                )
        if self.dynamic_img_pad:
            pad_h = (self.patch_size[0] - H % self.patch_size[0]) % self.patch_size[0]
            pad_w = (self.patch_size[1] - W % self.patch_size[1]) % self.patch_size[1]
            x = F.pad(x, (0, pad_w, 0, pad_h))
        # kernel=stride=patch_size인 convolution으로 서로 겹치지 않는 patch embedding을 만든다.
        x = self.proj(x)
        # Transformer 입력 형식이 필요한 경우 [B,C,H,W]를 [B,P,C]로 평탄화한다.
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        elif self.output_fmt != Format.NCHW:
            x = nchw_to(x, self.output_fmt)
        x = self.norm(x)
        return x
