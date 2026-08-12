"""Prompt-CAM 논리로 이미지별 top-K species attention heads를 시각화한다."""

from __future__ import annotations

import json
import os
from typing import Any

import torch
from timm.utils import accuracy, get_outdir

from data.loader import get_loader
from model.factory import get_model
from utils.log_utils import logging_env_setup
from utils.misc import AverageMeter
from utils.setup_logging import get_logger


logger = get_logger("Prompt_CAM")


def create_overlay_images(*args, **kwargs):
    """실제 시각화 시점에만 OpenCV 기반 함수를 불러온다."""
    from evaluation.cam.utils import create_overlay_images as implementation

    return implementation(*args, **kwargs)


def combine_images(*args, **kwargs):
    """원본과 head별 overlay를 가로로 결합한다."""
    from evaluation.cam.utils import combine_images as implementation

    return implementation(*args, **kwargs)


def _core_model(model):
    """분산 wrapper가 있으면 내부 모델을 반환한다."""
    return model.module if hasattr(model, "module") else model


def _unpack_batch(batch, device):
    """dict/tuple batch에서 이미지와 species target을 추출한다."""
    if isinstance(batch, dict):
        return (
            batch["image"].to(device),
            batch["species_target"].to(device),
        )

    images, targets = batch
    return images.to(device), targets.to(device)


def _species_logits(output: Any) -> torch.Tensor:
    """모델 출력에서 species logits [B,C]를 추출한다."""
    if isinstance(output, dict):
        return output["species_logits"]

    return output.squeeze(-1)


def _species_head_maps(
    output,
    attention,
    target_class: int,
    params,
    *,
    sample_index: int,
) -> torch.Tensor:
    """분류에 사용되는 class-specific head map을 [1,H,P]로 반환한다."""
    target_class = int(target_class)
    sample_index = int(sample_index)

    # Shared hierarchical patch-only 모델
    if isinstance(output, dict):
        if "species_attention_heads" not in output:
            raise KeyError(
                "계층 출력에 species_attention_heads가 없습니다."
            )

        maps = output["species_attention_heads"]

        if maps.ndim != 4:
            raise ValueError(
                "species_attention_heads는 [B,H,C,P]여야 하지만 "
                f"{tuple(maps.shape)}입니다."
            )

        if not 0 <= sample_index < maps.shape[0]:
            raise IndexError(
                f"sample_index={sample_index}, "
                f"batch_size={maps.shape[0]}"
            )

        if not 0 <= target_class < maps.shape[2]:
            raise IndexError(
                f"target_class={target_class}, "
                f"class_count={maps.shape[2]}"
            )

        return maps[
            sample_index : sample_index + 1,
            :,
            target_class,
            :,
        ]

    if attention is None:
        raise RuntimeError(
            "모델이 attention을 반환하지 않았습니다."
        )

    if attention.ndim != 4:
        raise ValueError(
            "attention은 4차원이어야 하지만 "
            f"{tuple(attention.shape)}입니다."
        )

    if not 0 <= sample_index < attention.shape[0]:
        raise IndexError(
            f"sample_index={sample_index}, "
            f"batch_size={attention.shape[0]}"
        )

    # Flat / independent patch-only decoder:
    # [B, H, C, P]
    #
    # 마지막 두 차원의 크기가 다르므로 기존 self-attention
    # [B, H, T, T] 형식과 구분할 수 있다.
    if attention.shape[-2] != attention.shape[-1]:
        if not 0 <= target_class < attention.shape[2]:
            raise IndexError(
                f"target_class={target_class}, "
                f"class_count={attention.shape[2]}"
            )

        return attention[
            sample_index : sample_index + 1,
            :,
            target_class,
            :,
        ]

    # 기존 self-attention Prompt-CAM:
    # [B, H, T, T]
    patch_start = int(params.vpt_num) + 1

    if not 0 <= target_class < attention.shape[2]:
        raise IndexError(
            f"target_class={target_class}, "
            f"query_count={attention.shape[2]}"
        )

    if patch_start >= attention.shape[-1]:
        raise ValueError(
            f"patch_start={patch_start}, "
            f"token_count={attention.shape[-1]}"
        )

    return attention[
        sample_index : sample_index + 1,
        :,
        target_class,
        patch_start:,
    ]


def _head_count(model) -> int:
    """현재 Prompt-CAM attention 모듈의 head 수를 반환한다."""
    core = _core_model(model)

    if bool(
        getattr(
            core.params,
            "hierarchical_prompt",
            False,
        )
    ):
        return int(core.hierarchical_head.num_heads)

    return int(core.num_heads)


def _greedy_promptcam_top_heads(
    model,
    image: torch.Tensor,
    target_class: int,
    top_traits: int,
) -> dict[str, Any]:
    """Prompt-CAM의 누적 uniform-blur greedy pruning을 수행한다."""
    target_class = int(target_class)
    top_traits = int(top_traits)
    num_heads = _head_count(model)

    if not 1 <= top_traits <= num_heads:
        raise ValueError(
            f"top_traits는 [1,{num_heads}] 범위여야 하지만 {top_traits}입니다"
        )

    remaining_heads = list(range(num_heads))
    blurred_heads: list[int] = []
    pruning_steps: list[dict[str, Any]] = []

    while len(remaining_heads) > top_traits:
        candidate_probabilities: dict[int, float] = {}

        for candidate_head in remaining_heads:
            candidate_output, _ = model(
                image,
                blur_head_lst=(
                    blurred_heads + [candidate_head]
                ),
                target_cls=target_class,
            )

            probability = torch.softmax(
                _species_logits(candidate_output).float(),
                dim=-1,
            )[0, target_class]

            candidate_probabilities[candidate_head] = float(
                probability.item()
            )

        least_important = max(
            remaining_heads,
            key=lambda head: (
                candidate_probabilities[head],
                -head,
            ),
        )

        pruning_steps.append(
            {
                "step": len(pruning_steps) + 1,
                "removed_head_zero_based": least_important,
                "removed_head_one_based": least_important + 1,
                "target_probability_after_blur": (
                    candidate_probabilities[least_important]
                ),
                "candidate_probabilities": {
                    str(head + 1): candidate_probabilities[head]
                    for head in remaining_heads
                },
            }
        )

        blurred_heads.append(least_important)
        remaining_heads.remove(least_important)

    return {
        "selected_heads": remaining_heads,
        "blurred_heads": blurred_heads,
        "pruning_steps": pruning_steps,
    }


def _save_selection(
    folder: str,
    selection: dict[str, Any],
    target_class: int,
    predicted_class: int,
) -> None:
    """선택 head 집합과 각 greedy 제거 단계를 JSON으로 저장한다."""
    os.makedirs(folder, exist_ok=True)

    payload = {
        "method": (
            "Prompt-CAM 누적 탐욕적 균등 어텐션 가지치기"
        ),
        "attention_source": (
            "종 분류에 사용한 계층적 종 어텐션"
        ),
        "target_class": int(target_class),
        "predicted_class": int(predicted_class),
        "selected_heads_zero_based": selection[
            "selected_heads"
        ],
        "selected_heads_one_based": [
            head + 1
            for head in selection["selected_heads"]
        ],
        "blurred_heads_zero_based": selection[
            "blurred_heads"
        ],
        "blurred_heads_one_based": [
            head + 1
            for head in selection["blurred_heads"]
        ],
        "pruning_steps": selection["pruning_steps"],
    }

    with open(
        os.path.join(
            folder,
            "promptcam_top_heads.json",
        ),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
        )


def _visualize_correct_sample(
    model,
    image: torch.Tensor,
    target_class: int,
    predicted_class: int,
    sample_number: int,
    params,
) -> None:
    """한 올바른 예측에 대해 논문식 top-K head를 저장한다."""
    selection = _greedy_promptcam_top_heads(
        model,
        image,
        target_class=target_class,
        top_traits=int(params.top_traits),
    )

    pruned_output, pruned_attention = model(
        image,
        blur_head_lst=selection["blurred_heads"],
        target_cls=target_class,
    )

    all_maps = _species_head_maps(
        pruned_output,
        pruned_attention,
        target_class,
        params,
        sample_index=0,
    )

    selected_indices = torch.as_tensor(
        selection["selected_heads"],
        device=all_maps.device,
        dtype=torch.long,
    )
    top_maps = all_maps.index_select(
        1,
        selected_indices,
    )

    if top_maps.shape[1] != int(params.top_traits):
        raise RuntimeError(
            f"선택 헤드 {params.top_traits}개가 필요하지만 {top_maps.shape[1]}개입니다"
        )

    folder = os.path.join(
        params.output_dir,
        f"img_{sample_number}",
    )

    create_overlay_images(
        image,
        _core_model(model).patch_size,
        top_maps,
        folder,
    )
    _save_selection(
        folder,
        selection,
        target_class=target_class,
        predicted_class=predicted_class,
    )
    combine_images(
        path=folder,
        pred_class=predicted_class,
    )

    logger.info(
        "이미지 %d의 Prompt-CAM 상위 헤드(1부터 시작): %s",
        sample_number,
        [
            head + 1
            for head in selection["selected_heads"]
        ],
    )


def basic_vis(params):
    """올바르게 분류된 지정 class 표본의 top-K heads를 생성한다."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    requested_samples = int(params.nmbr_samples)
    target_class = int(params.vis_cls)

    if requested_samples <= 0:
        raise ValueError(
            f"nmbr_samples는 양수여야 하지만 {requested_samples}입니다"
        )

    dataset_name = params.data.split("-")[0]
    output_dir = os.path.join(
        params.vis_outdir,
        params.model,
        dataset_name,
        f"class_{target_class}",
        f"top_traits_{params.top_traits}",
    )

    params.output_dir = get_outdir(output_dir)
    logging_env_setup(params)

    _, _, test_loader = get_loader(params, logger)
    model, _, _ = get_model(params, visualize=True)

    checkpoint = torch.load(
        params.checkpoint,
        map_location="cpu",
    )
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    model.to(params.device).eval()

    saved_count = 0
    skipped_misclassified = 0

    with torch.no_grad():
        for batch in test_loader:
            images, targets = _unpack_batch(
                batch,
                params.device,
            )

            matching_indices = torch.nonzero(
                targets.eq(target_class),
                as_tuple=False,
            ).flatten()

            if matching_indices.numel() == 0:
                continue

            batch_output, _ = model(images)
            predictions = _species_logits(
                batch_output
            ).argmax(dim=1)

            for index_tensor in matching_indices:
                batch_index = int(index_tensor.item())
                predicted_class = int(
                    predictions[batch_index].item()
                )

                if predicted_class != target_class:
                    skipped_misclassified += 1
                    continue

                selected_image = images[
                    batch_index : batch_index + 1
                ]
                saved_count += 1

                _visualize_correct_sample(
                    model,
                    selected_image,
                    target_class=target_class,
                    predicted_class=predicted_class,
                    sample_number=saved_count,
                    params=params,
                )

                if saved_count >= requested_samples:
                    break

            if saved_count >= requested_samples:
                break

    logger.info(
        "정확히 분류한 표본 %d개를 저장하고 오분류 표본 %d개를 건너뛰었습니다",
        saved_count,
        skipped_misclassified,
    )

    if saved_count < requested_samples:
        logger.warning(
            "정확히 분류한 표본을 %d개만 찾았습니다(클래스 %d)",
            saved_count,
            target_class,
        )

    top1 = AverageMeter()
    with torch.no_grad():
        for batch in test_loader:
            images, targets = _unpack_batch(
                batch,
                params.device,
            )
            output, _ = model(images)
            logits = _species_logits(output)
            acc1, _ = accuracy(
                logits,
                targets,
                topk=(
                    1,
                    min(5, logits.shape[1]),
                ),
            )
            top1.update(
                acc1.item(),
                images.shape[0],
            )

    logger.info(
        "평가: 평균 종 top-1 정확도: %.2f",
        top1.avg,
    )


def prune_attn_heads(
    model,
    inputs,
    target,
    prediction,
    smpl_count,
    params,
):
    """기존 호출부 호환용 단일 표본 pruning 함수."""
    target_class = int(target[0].item())
    predicted_class = int(prediction)

    if predicted_class != target_class:
        raise ValueError(
            "Prompt-CAM 상위 헤드 가지치기에는 정확히 분류된 표본이 필요합니다"
        )

    _visualize_correct_sample(
        model,
        inputs,
        target_class=target_class,
        predicted_class=predicted_class,
        sample_number=int(smpl_count),
        params=params,
    )
