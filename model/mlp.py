"""Transformer block 내부의 feed-forward MLP 계층을 구현한다."""

from torch import nn as nn
from timm.layers.helpers import to_2tuple
from functools import partial


# timm MLP의 연산 순서를 유지하면서 Prompt-CAM Transformer block에서 재사용한다.
class MlpPETL(nn.Module):
    """두 선형 변환과 활성화·dropout으로 구성된 feed-forward network다."""
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False
    ):
        """객체가 사용할 입력 설정과 내부 상태를 초기화한다."""
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])


    def forward(self, x):
        """입력 tensor를 계층의 순전파 연산에 통과시켜 결과를 반환한다."""
        h = self.fc1(x)
        x = self.act(h)
        x = self.drop1(x)
        x = self.norm(x)
        h = self.fc2(x)
        x = self.drop2(h)
        return x