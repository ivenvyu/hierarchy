"""VPT 계층 프롬프트와 hierarchical decoder를 결합한 Vision Transformer 본체를 구현한다."""

from functools import partial
from typing import Any, Callable, Dict, Optional, Sequence, Set, Tuple, Type, Union, List

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.jit import Final
from timm.layers import LayerNorm, DropPath, AttentionPoolLatent, RmsNorm, PatchDropout, SwiGLUPacked, \
    trunc_normal_, lecun_normal_, resample_patch_embed, resample_abs_pos_embed, use_fused_attn, \
    get_act_layer, get_norm_layer, LayerType
from timm.models._builder import build_model_with_cfg
from timm.models._manipulate import named_apply, adapt_input_conv
from timm.models._registry import generate_default_cfgs, register_model, register_model_deprecations
from timm.models.vision_transformer import VisionTransformer
from timm.models.vision_transformer import LayerScale, init_weights_vit_timm, get_init_weights_vit, \
    _load_weights, checkpoint_filter_fn
## PETL을 위해 추가
from utils.setup_logging import get_logger
from model.block import BlockPETL
from model.patch_embed import PatchEmbedPETL
from model.mlp import MlpPETL
from model.vpt import VPT
from model.hierarchical_prompt import (
    HierarchicalPromptDecoder,
    PatchOnlyPromptDecoder,
)

logger = get_logger("Prompt_CAM")


class VisionTransformerPETL(VisionTransformer):
    """pretrained ViT에 VPT prompt와 계층 decoder를 추가한 전체 모델이다."""
    dynamic_img_size: Final[bool]

    def __init__(
            self,
            img_size: Union[int, Tuple[int, int]] = 224,
            patch_size: Union[int, Tuple[int, int]] = 16,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: Literal['', 'avg', 'token', 'map'] = 'token',
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            init_values: Optional[float] = None,
            class_token: bool = True,
            no_embed_class: bool = False,
            reg_tokens: int = 0,
            pre_norm: bool = False,
            fc_norm: Optional[bool] = None,
            dynamic_img_size: bool = False,
            dynamic_img_pad: bool = False,
            drop_rate: float = 0.,
            pos_drop_rate: float = 0.,
            patch_drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            weight_init: Literal['skip', 'jax', 'jax_nlhb', 'moco', ''] = '',
            embed_layer: Callable = PatchEmbedPETL,
            norm_layer: Optional[LayerType] = None,
            act_layer: Optional[LayerType] = None,
            block_fn: Type[nn.Module] = BlockPETL,
            mlp_layer: Type[nn.Module] = MlpPETL,
            params=None
    ) -> None:
        """patch embedding, Transformer block, prompt manager, 계층 decoder를 초기화한다."""
        super().__init__()
        assert global_pool in ('', 'avg', 'token', 'map')
        assert class_token or global_pool != 'token'
        use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
        norm_layer = get_norm_layer(norm_layer) or partial(nn.LayerNorm, eps=1e-6)
        act_layer = get_act_layer(act_layer) or nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # 다른 모델과 일관된 속성 이름을 유지한다.
        self.num_prefix_tokens = 1 if class_token else 0
        self.num_prefix_tokens += reg_tokens
        self.num_reg_tokens = reg_tokens
        self.has_class_token = class_token
        self.no_embed_class = no_embed_class  # 레지스터를 포함한 접두 토큰 위치에는 임베딩하지 않는다.
        self.dynamic_img_size = dynamic_img_size
        self.grad_checkpointing = False

        embed_args = {}
        if dynamic_img_size:
            # 평탄화는 위치 임베딩 적용 뒤로 미룬다.
            embed_args.update(dict(strict_img_size=False, output_fmt='NHWC'))
        # 이미지를 [B,P,D] patch token으로 바꾸는 backbone 입력 계층을 생성한다.
        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,  # CLIP처럼 사전 정규화를 쓰면 편향을 끈다.
            dynamic_img_pad=dynamic_img_pad,
            params=params,
            **embed_args,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        self.reg_token = nn.Parameter(torch.zeros(1, reg_tokens, embed_dim)) if reg_tokens else None
        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * .02)
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
            )
        else:
            self.patch_drop = nn.Identity()
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # 확률적 깊이 감쇠 규칙

        ############# 추가 모듈 시작 #############
        self.patch_size = patch_size
        self.params = params
        # 어텐션 모듈은 공유 params 객체를 받는다. vpt_layer < depth일 때
        # 프롬프트 활성 계층과 앞쪽 백본 전용 계층을 구분하도록 실제 인코더
        # 깊이를 기록한다.
        self.params.model_depth = int(depth)
        if self.params.train_type in ['vpt','prompt_cam']:
            self.vpt = VPT(params, depth, patch_size, embed_dim)
        ############# 추가 모듈 끝 #############

        self.blocks = nn.Sequential(*[
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                init_values=init_values,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                mlp_layer=mlp_layer,
                ############# 추가 모듈 시작 #############
                params=params
                ############# 추가 모듈 끝 #############
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

        # 분류기 헤드
        if global_pool == 'map':
            self.attn_pool = AttentionPoolLatent(
                self.embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
            )
        else:
            self.attn_pool = None
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.num_heads= num_heads

        ############# 추가 모듈 시작 #############
        if self.params.train_type == 'vpt' or self.params.train_type == 'linear':
            self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        elif self.params.train_type == 'prompt_cam':
            if bool(getattr(self.params, 'hierarchical_prompt', False)):
                self.head = nn.Identity()
                self.hierarchical_head = HierarchicalPromptDecoder(
                    embed_dim=self.embed_dim,
                    num_heads=num_heads,
                    species_to_genus=list(self.params.species_to_genus),
                    genus_to_family=list(self.params.genus_to_family),
                    genus_counts=list(self.params.genus_counts),
                    eps=float(getattr(self.params, 'spm_eps', 1e-8)),
                )
            else:
                if bool(
                    getattr(
                        self.params,
                        "prompt_patch_only_head",
                        False,
                    )
                ):
                    self.head = nn.Identity()
                    self.prompt_patch_head = PatchOnlyPromptDecoder(
                        embed_dim=self.embed_dim,
                        num_heads=num_heads,
                    )
                else:
                    self.head = nn.Linear(
                        self.embed_dim,
                        1,
                    )
        ############# 추가 모듈 끝 #############


        if weight_init != 'skip':
            self.init_weights(weight_init)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path: str, prefix: str = '') -> None:
        """지정한 checkpoint 형식에 맞춰 pretrained backbone 가중치를 불러온다."""
        _load_weights_PETL(self, checkpoint_path, prefix)

    def forward_features(self, x: torch.Tensor, blur_head_lst=[], target_cls=-1) -> Tuple[torch.Tensor,torch.Tensor]:
        """patch·prefix·prompt token을 Transformer에 통과시켜 마지막 attention과 token feature를 반환한다."""
        pcam_outputs = None
        attn_map = None
        # 이미지 patch를 embedding하고 prefix token 및 위치 embedding을 결합한다.
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        # 일반 실행과 gradient checkpointing 실행이 동일한 prompt 삽입·제거
        # 경로를 사용하도록 하나의 block loop로 통합한다.
        for idx, block in enumerate(self.blocks):
            prompt = None

            if self.params.train_type in {"vpt", "prompt_cam"}:
                prompt = self.vpt.retrieve_prompt(
                    idx,
                    x.shape[0],
                )

                if prompt is not None:
                    x = torch.cat(
                        [prompt, x],
                        dim=1,
                    )

            is_last_block = idx == len(self.blocks) - 1

            if self.grad_checkpointing and not torch.jit.is_scripting():
                if is_last_block:

                    def run_block(
                        tensor: torch.Tensor,
                        current_block=block,
                        current_idx=idx,
                    ):
                        """마지막 block을 checkpointing하면서 attention map도 반환한다."""
                        return current_block(
                            tensor,
                            current_idx,
                            blur_head_lst=blur_head_lst,
                            target_cls=target_cls,
                        )

                else:

                    def run_block(
                        tensor: torch.Tensor,
                        current_block=block,
                        current_idx=idx,
                    ):
                        """중간 block을 checkpointing하여 실행한다."""
                        return current_block(
                            tensor,
                            current_idx,
                        )

                x, current_attn = torch.utils.checkpoint.checkpoint(
                    run_block,
                    x,
                    use_reentrant=False,
                )

            else:
                if is_last_block:
                    x, current_attn = block(
                        x,
                        idx,
                        blur_head_lst=blur_head_lst,
                        target_cls=target_cls,
                    )
                else:
                    x, current_attn = block(
                        x,
                        idx,
                    )

            if is_last_block:
                attn_map = current_attn

            # Prompt-CAM은 최종 활성 계층의 프롬프트 출력을 분류기에 전달하기 위해
            # prompt가 포함된 출력을 저장한 뒤, backbone 경로에서는 prompt를 제거한다.
            if (
                self.params.train_type == "prompt_cam"
                and prompt is not None
            ):
                pcam_outputs = x
                x = x[:, self.vpt.prompt_count:, :]

            elif (
                getattr(self.params, "vpt_mode", None)
                and prompt is not None
            ):
                x = x[:, self.vpt.prompt_count:, :]

        if self.params.train_type == "prompt_cam":
            if pcam_outputs is None:
                raise RuntimeError(
                    "Prompt-CAM이 활성 프롬프트 계층 없이 끝났습니다. "
                    "vpt_layer와 모델 깊이를 확인하십시오."
                )

            # 마지막 활성 layer의 prompt와 backbone token을 decoder로 전달한다.
            x = pcam_outputs
        x = self.norm(x)
        return x,attn_map

    def forward_head(
        self,
        x: torch.Tensor,
        pre_logits: bool = False,
        patch_prior: torch.Tensor | None = None,
        blur_head_lst=None,
        target_cls: int = -1,
        target_level: str = "species",
    ) -> torch.Tensor:
        """Transformer token을 기존 또는 계층 분류 출력으로 변환한다."""
        if self.params.train_type == "prompt_cam":
            if bool(
                getattr(
                    self.params,
                    "hierarchical_prompt",
                    False,
                )
            ):
                family_count = int(self.params.num_families)
                genus_count = int(self.params.num_genera)
                species_count = int(self.params.class_num)

                family_end = family_count
                genus_end = family_end + genus_count
                prompt_count = genus_end + species_count

                family_tokens = x[:, 0:family_end]
                genus_tokens = x[:, family_end:genus_end]
                species_tokens = x[:, genus_end:prompt_count]
                backbone_tokens = x[:, prompt_count:]
                patch_tokens = backbone_tokens[
                    :,
                    self.num_prefix_tokens:,
                ]

                if patch_tokens.shape[1] == 0:
                    raise ValueError(
                        "계층적 Prompt-CAM에는 패치 토큰이 필요합니다"
                    )

                if self.num_prefix_tokens > 0:
                    global_feature = backbone_tokens[:, 0]
                else:
                    global_feature = patch_tokens.mean(dim=1)

                blur_heads = (
                    []
                    if blur_head_lst is None
                    else list(blur_head_lst)
                )
                level = str(target_level).lower()
                if level not in {"species", "genus", "family"}:
                    raise ValueError(
                        "target_level은 {'species','genus','family'} 중 하나여야 "
                        f"하지만 {target_level!r}입니다"
                    )

                blur_family_heads = None
                target_family = None
                blur_genus_heads = None
                target_genus = None
                blur_species_heads = None
                target_species = None

                if blur_heads:
                    target_index = int(target_cls)

                    if level == "species":
                        if not 0 <= target_index < species_count:
                            raise IndexError(
                                f"대상 종 {target_index}이(가) 범위를 벗어났습니다: "
                                f"[0,{species_count})"
                            )
                        blur_species_heads = blur_heads
                        target_species = target_index

                    elif level == "genus":
                        if not 0 <= target_index < genus_count:
                            raise IndexError(
                                f"대상 속 {target_index}이(가) 범위를 벗어났습니다: "
                                f"[0,{genus_count})"
                            )
                        blur_genus_heads = blur_heads
                        target_genus = target_index

                    else:
                        if not 0 <= target_index < family_count:
                            raise IndexError(
                                f"대상 과 {target_index}이(가) 범위를 벗어났습니다: "
                                f"[0,{family_count})"
                            )
                        blur_family_heads = blur_heads
                        target_family = target_index

                return self.hierarchical_head(
                    family_tokens,
                    genus_tokens,
                    species_tokens,
                    patch_tokens,
                    global_feature,
                    patch_prior=patch_prior,
                    blur_family_heads=blur_family_heads,
                    target_family=target_family,
                    blur_genus_heads=blur_genus_heads,
                    target_genus=target_genus,
                    blur_species_heads=blur_species_heads,
                    target_species=target_species,
                )

            if bool(
                getattr(
                    self.params,
                    "prompt_patch_only_head",
                    False,
                )
            ):
                prompt_count = int(
                    self.params.vpt_num
                )

                prompt_tokens = x[
                    :,
                    :prompt_count,
                ]
                backbone_tokens = x[
                    :,
                    prompt_count:,
                ]
                patch_tokens = backbone_tokens[
                    :,
                    self.num_prefix_tokens:,
                ]

                if patch_tokens.shape[1] == 0:
                    raise ValueError(
                        "patch-only Prompt-CAM에는 patch token이 필요합니다"
                    )

                blur_heads = (
                    []
                    if blur_head_lst is None
                    else list(blur_head_lst)
                )
                target_prompt = (
                    int(target_cls)
                    if blur_heads
                    else None
                )

                return self.prompt_patch_head(
                    prompt_tokens,
                    patch_tokens,
                    blur_head_indices=blur_heads,
                    target_prompt=target_prompt,
                )

            output_feature = x[
                :,
                : self.params.vpt_num,
            ]
        else:
            if self.attn_pool is not None:
                output_feature = self.attn_pool(x)
            elif self.global_pool == "avg":
                output_feature = x[
                    :,
                    self.num_prefix_tokens:,
                ].mean(dim=1)
            elif self.global_pool:
                output_feature = x[:, 0]
            else:
                output_feature = x

            output_feature = self.fc_norm(output_feature)
            output_feature = self.head_drop(output_feature)

        return (
            output_feature
            if pre_logits
            else self.head(output_feature)
        )



    def forward(
        self,
        x: torch.Tensor,
        blur_head_lst=None,
        target_cls=-1,
        patch_prior: torch.Tensor | None = None,
        target_level: str = "species",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """이미지 forward와 계층별 Prompt-CAM intervention을 수행한다."""
        blur_head_lst = (
            []
            if blur_head_lst is None
            else list(blur_head_lst)
        )

        hierarchical_active = bool(
            self.params.train_type == "prompt_cam"
            and getattr(
                self.params,
                "hierarchical_prompt",
                False,
            )
        )

        if not hierarchical_active and str(target_level).lower() != "species":
            raise ValueError(
                "속/과 Prompt-CAM에는 hierarchical_prompt=True가 필요합니다"
            )

        patch_only_active = bool(
            self.params.train_type == "prompt_cam"
            and not hierarchical_active
            and getattr(
                self.params,
                "prompt_patch_only_head",
                False,
            )
        )

        attn_maps = None
        if self.params.vis_attn:
            if hierarchical_active or patch_only_active:
                # blur는 각각의 전용 decoder 안에서 처리한다.
                x, attn_maps = self.forward_features(x)
            else:
                x, attn_maps = self.forward_features(
                    x,
                    blur_head_lst=blur_head_lst,
                    target_cls=target_cls,
                )
        else:
            x, _ = self.forward_features(x)

        head_output = self.forward_head(
            x,
            patch_prior=patch_prior,
            blur_head_lst=(
                blur_head_lst
                if (
                    hierarchical_active
                    or patch_only_active
                )
                else None
            ),
            target_cls=int(target_cls),
            target_level=str(target_level),
        )

        if patch_only_active:
            logits, decoder_attention = head_output
            return logits, decoder_attention

        return head_output, attn_maps



    def hierarchical_regularization(self) -> Dict[str, torch.Tensor]:
        """VPT 중심화 손실을 반환한다."""
        if not bool(getattr(self.params, 'hierarchical_prompt', False)):
            zero = next(self.parameters()).new_zeros(())
            return {'center_species': zero, 'center_genus': zero}
        centers = self.vpt.center_losses()
        return {
            'center_species': centers['species'],
            'center_genus': centers['genus'],
        }

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None):
        """새 클래스 수와 pooling 방식에 맞춰 최종 분류 head를 재생성한다."""
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg', 'avgmax', 'max', 'token', 'map')
            if global_pool == 'map' and self.attn_pool is None:
                assert False, "현재 reset_classifier()에서는 어텐션 풀링을 추가할 수 없습니다."
            elif global_pool != 'map' and self.attn_pool is not None:
                self.attn_pool = None  # 어텐션 풀링을 제거한다.
            self.global_pool = global_pool
        ############# 추가 모듈 #############
        if self.params.train_type == 'vpt' or self.params.train_type == 'linear':
            self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        elif bool(getattr(self.params, 'hierarchical_prompt', False)):
            if num_classes != int(self.params.class_num):
                raise ValueError(
                    f'계층 분류기를 {num_classes}개 클래스로 재설정할 수 없습니다. 분류 체계에는 종이 {self.params.class_num}개 있습니다'
                )
            self.head = nn.Identity()
        else:
            self.head = (
                nn.Identity()
                if bool(
                    getattr(
                        self.params,
                        "prompt_patch_only_head",
                        False,
                    )
                )
                else nn.Linear(
                    self.embed_dim,
                    1,
                )
            )
        ############# 추가 모듈 끝 #############
        # 원본 구현
        # self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()


@torch.no_grad()
def _load_weights_PETL(model: VisionTransformerPETL, checkpoint_path: str, prefix: str = ''):
    """checkpoint 확장자에 따라 npz 또는 PyTorch 로더를 선택한다."""
    if checkpoint_path.endswith('.npz'):
        _load_weights(model, checkpoint_path, prefix)
    elif checkpoint_path.endswith('.pth') or checkpoint_path.endswith('.bin'):
        _load_weights_pth(model, checkpoint_path, checkpoint_filter_fn)


def _load_weights_pth(model, checkpoint_path, filter_fn=checkpoint_filter_fn):
    """PyTorch checkpoint의 state_dict를 정리해 모델에 비엄격 로드한다."""
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    if filter_fn is not None:
        state_dict = filter_fn(state_dict, model)
    if 'head.weight' in state_dict:
        state_dict.pop('head.weight', None)
    if 'head.bias' in state_dict:
        state_dict.pop('head.bias', None)
    model.load_state_dict(state_dict, strict=False)


def _create_vision_transformer_petl(variant: str, pretrained: bool = False, **kwargs):
    """timm 설정과 프로젝트 인자를 합쳐 지정 variant의 PETL ViT를 생성한다."""
    if kwargs.get('features_only', None):
        raise RuntimeError('Vision Transformer 모델에는 features_only가 구현되어 있지 않습니다.')

    if 'flexi' in variant:
        # Google FlexiViT 사전 학습 모델은 이중 선형 패치/임베딩 보간에 더 잘 맞고,
        # 다른 사전 학습 모델은 앤티앨리어싱이 적용된 이중 삼차 보간이 더 적합하다.
        _filter_fn = partial(checkpoint_filter_fn, interpolation='bilinear', antialias=False)
    else:
        _filter_fn = checkpoint_filter_fn

    # 풀링이 꺼지면 어텐션 풀(현재 SigLIP 전용) 매개변수를 제거한다.
    strict = True
    if 'siglip' in variant and kwargs.get('global_pool', None) != 'map':
        strict = False

    return build_model_with_cfg(
        VisionTransformerPETL,
        variant,
        pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_strict=strict,
        **kwargs,
    )


@register_model
def vit_base_patch14_dinov2_petl(pretrained: bool = False, **kwargs):
    """DINOv2 ViT-B/14 구조의 Prompt-CAM 모델을 생성한다."""
    model_args = dict(patch_size=14, embed_dim=768, depth=12, num_heads=12, init_values=1e-5, img_size=224)
    model = _create_vision_transformer_petl(
        'vit_base_patch14_dinov2', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def vit_base_patch16_dino_petl(pretrained: bool = False, **kwargs):
    """DINO ViT-B/16 구조의 Prompt-CAM 모델을 생성한다."""
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
    model = _create_vision_transformer_petl(
        'vit_base_patch16_224.dino', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def vit_base_patch16_224_in21k_petl(pretrained=False, **kwargs):
    """ImageNet-21K ViT-B/16 구조의 Prompt-CAM 모델을 생성한다."""
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
    model = _create_vision_transformer_petl(
        'vit_base_patch16_224_in21k', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_base_patch16_clip_224_petl(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """CLIP ViT-B/16 구조의 Prompt-CAM 모델을 생성한다."""
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, pre_norm=True, norm_layer=nn.LayerNorm,
                      act_layer='quick_gelu')
    model = _create_vision_transformer_petl(
        'vit_base_patch16_clip_quickgelu_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model
