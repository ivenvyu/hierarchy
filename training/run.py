"""분산 실행 환경을 초기화하고 데이터·모델·Trainer를 연결하는 실험 진입 로직이다."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from timm.utils import get_outdir

from data.loader import get_loader
from model.factory import get_model
from training.trainer import Trainer
from utils.log_utils import logging_env_setup
from utils.misc import method_name
from utils.setup_logging import get_logger


logger = get_logger("Prompt_CAM")


def init_distributed() -> bool:
    """환경 변수에서 분산 rank 정보를 읽고 process group을 초기화한다."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        return True
    return False


def _json_safe(value):
    """로그 저장을 위해 객체를 JSON 직렬화 가능한 값으로 재귀 변환한다."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def train(params, train_loader, val_loader, test_loader):
    """모델과 Trainer를 구성해 분류 학습 루프를 실행한다."""
    model, tune_parameters, model_grad_params_no_head = get_model(params)
    model_grad_params_total = sum(
        parameter.numel()
        for parameter in tune_parameters
    )
    model = model.to(params.device)

    if params.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[params.local_rank],
            output_device=params.local_rank,
            find_unused_parameters=True,
        )

    trainer = Trainer(model, tune_parameters, params)
    train_metrics, best_eval_metrics, final_metrics = trainer.train_classifier(
        train_loader, val_loader, test_loader
    )
    return (
        train_metrics,
        best_eval_metrics,
        final_metrics,
        model_grad_params_no_head,
        model_grad_params_total,
        trainer.model,
    )


def basic_run(params):
    """seed, 로그, loader를 초기화하고 단일 또는 분산 학습을 시작한다."""
    params.distributed = init_distributed()
    params.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    dataset_name = str(params.data).split("-")[0]
    data_name = Path(str(params.data_path)).name or dataset_name
    method = method_name(params)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_root = Path(str(getattr(params, "output_root", "./output"))).expanduser()
    output_dir = output_root / str(params.pretrained_weights) / dataset_name / method / data_name / timestamp
    params.output_dir = get_outdir(str(output_dir))
    logging_env_setup(params)

    logger.info("다음 경로에서 데이터셋을 불러옵니다: %s", params.data_path)
    # 데이터 구성이 확정된 뒤 taxonomy 정보가 params에 기록되고 모델이 이를 사용한다.
    train_loader, val_loader, test_loader = get_loader(params, logger)

    # 로더가 클래스 순서를 검증하고 기록한 뒤 확정된 설정을 저장한다.
    if (not params.distributed) or dist.get_rank() == 0:
        resolved = _json_safe(dict(vars(params)))
        with open(Path(params.output_dir) / "args.yaml", "w", encoding="utf-8") as handle:
            yaml.safe_dump(resolved, handle, sort_keys=True, allow_unicode=True)
        with open(Path(params.output_dir) / "class_to_idx.json", "w", encoding="utf-8") as handle:
            json.dump(resolved.get("class_to_idx", {}), handle, indent=2, ensure_ascii=False)
        if resolved.get("taxonomy"):
            with open(Path(params.output_dir) / "taxonomy.json", "w", encoding="utf-8") as handle:
                json.dump(resolved["taxonomy"], handle, indent=2, ensure_ascii=False)
        if resolved.get("taxonomy_node"):
            with open(Path(params.output_dir) / "taxonomy_node.json", "w", encoding="utf-8") as handle:
                json.dump(resolved["taxonomy_node"], handle, indent=2, ensure_ascii=False)

    (
        train_metrics,
        best_eval_metrics,
        final_metrics,
        trainable_count_no_head,
        trainable_count_total,
        _,
    ) = train(
        params, train_loader, val_loader, test_loader
    )

    if (not params.distributed) or dist.get_rank() == 0:
        result = {
            "train_metrics_last_epoch": dict(train_metrics),
            "best_validation_metrics": dict(best_eval_metrics),
            "final_test_metrics": dict(final_metrics),
            "trainable_parameters": trainable_count_total,
            "trainable_parameters_without_head": trainable_count_no_head,
            "selection_metric": str(getattr(params, "selection_metric", "macro_f1")),
        }
        with open(Path(params.output_dir) / "final_result.json", "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
        logger.info("최종 결과: %s", result)
    return final_metrics
