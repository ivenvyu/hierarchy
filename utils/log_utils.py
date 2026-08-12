"""실행 환경과 모델 파라미터 정보를 수집해 로그로 기록한다."""

import os
import pprint
import sys
from collections import defaultdict
import torch

from utils.distributed import get_rank, get_world_size
from utils.setup_logging import setup_logging

def is_main_process():
    """현재 distributed rank가 0인지 반환한다."""
    return get_rank() == 0


def get_env_module():
    """가능한 경우 PyTorch 환경 수집 모듈을 동적으로 불러온다."""
    var_name = "ENV_MODULE"
    return var_name, os.environ.get(var_name, "<not set>")


def collect_torch_env() -> str:
    """PyTorch가 제공하는 실행 환경 진단 문자열을 수집한다."""
    try:
        import torch.__config__

        return torch.__config__.show()
    except ImportError:
        # 이전 PyTorch 버전과의 호환성을 유지한다.
        from torch.utils.collect_env import get_pretty_env_info

        return get_pretty_env_info()


def collect_env_info():
    """Python, PyTorch, CUDA와 주요 패키지 버전을 하나의 보고서로 만든다."""
    data = []
    data.append(("Python", sys.version.replace("\n", "")))
    data.append(get_env_module())
    data.append(("PyTorch", torch.__version__))
    data.append(("PyTorch 디버그 빌드", torch.version.debug))

    has_cuda = torch.cuda.is_available()
    data.append(("CUDA 사용 가능", has_cuda))
    if has_cuda:
        data.append(
            (
                "CUDA_VISIBLE_DEVICES",
                os.environ.get(
                    "CUDA_VISIBLE_DEVICES",
                    "<미설정: 보이는 모든 CUDA 장치 사용>",
                ),
            )
        )
        devices = defaultdict(list)
        for k in range(torch.cuda.device_count()):
            devices[torch.cuda.get_device_name(k)].append(str(k))
        for name, devids in devices.items():
            data.append(("GPU " + ",".join(devids), name))

    width = max(len(str(label)) for label, _ in data)
    env_str = "\n".join(
        f"{label!s:<{width}}  {value}"
        for label, value in data
    ) + "\n"
    env_str += collect_torch_env()
    return env_str


def logging_env_setup(params) -> None:
    """출력 logger를 설정하고 실행 환경 정보를 기록한다."""
    logger = setup_logging(
        params.gpu_num, get_world_size(), params.output_dir, name="Prompt_CAM")

    if not is_main_process():
        return

    # 환경, 명령줄 인자와 설정의 기본 정보를 기록한다.
    rank = get_rank()
    logger.info(
        f"현재 프로세스 순위: {rank}. 전체 크기: {get_world_size()}")
    logger.info("실행 환경 정보:\n" + collect_env_info())


    # 설정을 표시한다.
    logger.info("학습 설정:")
    logger.info(pprint.pformat(params))

def log_model_info(model, logger, verbose=False):
    """전체·학습 가능 파라미터 수와 선택적 모델 구조를 로그에 남긴다."""
    model_total_params = sum(p.numel() for p in model.parameters())
    model_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_grad_params_no_head = sum(
        p.numel() for n, p in model.named_parameters()
        if p.requires_grad and 'head' not in n
    )

    if is_main_process():
        if verbose:
            logger.info(f"분류 모델:\n{model}")

        logger.info(
            "전체 매개변수: {0}\t 경사 매개변수: {1}\t 헤드 제외 경사 매개변수: {2}".format(
                model_total_params, model_grad_params, model_grad_params_no_head
            )
        )
        logger.info(f"전체 조정 비율:{(model_grad_params/model_total_params*100):.2f} %")
        logger.info(f"헤드 제외 조정 비율:{(model_grad_params_no_head/model_total_params*100):.2f} %")

        logger.info("미세 조정 매개변수:")
        for n, p in model.named_parameters():
            if p.requires_grad:
                logger.info(n)

    return model_grad_params_no_head
