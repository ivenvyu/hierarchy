"""학습과 평가에서 공유하는 Prompt-CAM 모델을 생성한다."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import timm
import torch
import torch.distributed as dist

# 이 import는 PETL ViT 변형을 timm에 등록한다.
from model.vision_transformer import VisionTransformerPETL  # noqa: F401
from utils.log_utils import log_model_info
from utils.setup_logging import get_logger


logger = get_logger("Prompt_CAM")
TUNE_MODULES = ("vpt", "head")


def is_main_process() -> bool:
    """현재 분산 순위가 0인지 반환한다."""
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def _checkpoint_path(value, *, field: str) -> Path:
    """선택적 체크포인트 입력을 빈 값과 유효 경로로 정규화한다."""
    if value in (None, "", "null"):
        raise ValueError(
            f"{field} 값을 명시해야 합니다. 절대·상대 경로를 모두 사용할 수 있으며 "
            "통합 코드는 특정 장비의 체크포인트 위치를 가정하지 않습니다."
        )
    path = Path(str(value)).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{field} 경로가 존재하지 않습니다: {path}")
    return path


def _load_optional_promptcam_checkpoint(
    model: torch.nn.Module,
    value,
    *,
    strict: bool = False,
) -> None:
    """지정된 외부 Prompt-CAM state_dict를 현재 모델에 불러온다.

    전이 초기화에는 ``strict=False``를 허용하지만, 평가 checkpoint 복원은
    ``strict=True``를 사용해 일부 가중치만 로드되는 무효 평가를 차단한다.
    """
    if value in (None, "", "null"):
        return
    path = _checkpoint_path(value, field="promptcam_checkpoint")
    payload = torch.load(path, map_location="cpu")
    state = payload.get("model_state_dict", payload.get("state_dict", payload))
    incompatible = model.load_state_dict(state, strict=bool(strict))
    if is_main_process():
        logger.info(
            "%s에서 Prompt-CAM 초기값을 불러왔습니다(strict=%s, 누락=%d, 예상 외=%d)",
            path,
            bool(strict),
            len(incompatible.missing_keys),
            len(incompatible.unexpected_keys),
        )


def get_model(params, visualize: bool = False):
    """백본을 만들고 사전 학습 및 선택적 실험 체크포인트를 적용한다."""
    params.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if is_main_process():
        logger.info("사용 장치: %s", params.device)

    if params.train_type == "prompt_cam":
        hierarchical = bool(getattr(params, "hierarchical_prompt", False))
        original_taxonomy = bool(
            getattr(
                params,
                "original_taxonomy_prompt",
                False,
            )
        )
        patch_only = bool(
            getattr(
                params,
                "prompt_patch_only_head",
                False,
            )
        )

        if hierarchical and patch_only:
            raise ValueError(
                "hierarchical_prompt는 전용 계층 decoder를 사용하므로 "
                "prompt_patch_only_head와 동시에 활성화할 수 없습니다"
            )

        if hierarchical and original_taxonomy:
            raise ValueError(
                "hierarchical_prompt와 original_taxonomy_prompt를 동시에 "
                "활성화할 수 없습니다"
            )
        if original_taxonomy and int(params.class_num) < 2:
            raise ValueError(
                "원논문식 taxonomy node 모델은 데이터 로더가 확정한 2개 이상의 "
                "node-local class가 필요합니다. get_loader()를 먼저 실행하십시오."
            )

        if hierarchical:
            expected = (
                int(params.num_families)
                + int(params.num_genera)
                + int(params.class_num)
            )
            if int(params.vpt_num) != expected:
                raise ValueError(
                    f"계층적 Prompt-CAM은 vpt_num=F+G+C={expected}가 필요하지만 "
                    f"{params.vpt_num}입니다"
                )
        elif int(params.vpt_num) != int(params.class_num):
            raise ValueError(
                f"Prompt-CAM은 vpt_num == class_num을 요구합니다. 현재 값: "
                f"vpt_num={params.vpt_num}, class_num={params.class_num}"
            )
        # 계층 의미 맵은 전용 디코더에서 나오므로 원시 백본 어텐션은 시각화가
        # 요청한 경우에만 유지한다.
        params.vis_attn = bool(getattr(params, "vis_attn", False))

    model = get_base_model(params, visualize=visualize)
    _load_optional_promptcam_checkpoint(
        model,
        getattr(params, "promptcam_checkpoint", None),
        strict=bool(getattr(params, "promptcam_checkpoint_strict", False)),
    )

    tune_parameters = []
    if getattr(params, "debug", False) and is_main_process():
        logger.info("학습 가능한 매개변수:")

    for name, parameter in model.named_parameters():
        trainable = any(module_name in name for module_name in TUNE_MODULES)
        parameter.requires_grad = trainable
        if trainable:
            tune_parameters.append(parameter)
            if getattr(params, "debug", False) and is_main_process():
                logger.info("\t%s, %d, %s", name, parameter.numel(), tuple(parameter.shape))

    model_grad_params_no_head = log_model_info(model, logger)
    model = model.to(params.device)
    return model, tune_parameters, model_grad_params_no_head


def get_base_model(params, visualize: bool = False):
    """설정의 모델 계열과 사전 학습 이름을 실제 생성 함수에 연결한다."""
    model_name = str(params.pretrained_weights)
    img_size = int(params.crop_size)
    common = dict(
        drop_path_rate=float(params.drop_path_rate),
        pretrained=False,
        img_size=img_size,
        params=params,
    )

    if model_name == "vit_base_patch16_224_in21k":
        params.patch_size = 16
        model = timm.create_model("vit_base_patch16_224_in21k_petl", **common)
    elif model_name == "vit_base_mae":
        params.patch_size = 16
        model = timm.create_model("vit_base_patch16_224_in21k_petl", **common)
    elif model_name == "vit_base_patch14_dinov2":
        params.patch_size = 14
        model = timm.create_model("vit_base_patch14_dinov2_petl", **common)
    elif model_name == "vit_base_patch16_dino":
        params.patch_size = 16
        model = timm.create_model("vit_base_patch16_dino_petl", **common)
    else:
        raise NotImplementedError(
            f"통합 의미 기반 SnapMix는 현재 ViT/MAE/DINO/DINOv2 Prompt-CAM 백본만 지원합니다: {model_name}"
        )

    if img_size % int(params.patch_size) != 0:
        raise ValueError(
            f"crop_size={img_size}는 patch_size={params.patch_size}로 나누어떨어져야 합니다"
        )

    if not visualize and bool(getattr(params, "load_pretrained_backbone", True)):
        checkpoint = _checkpoint_path(
            getattr(params, "pretrained_checkpoint", None),
            field="pretrained_checkpoint",
        )
        model.load_pretrained(str(checkpoint))
        if is_main_process():
            logger.info("고정 백본 체크포인트를 불러왔습니다: %s", checkpoint)

    model.reset_classifier(int(params.class_num))
    return model
