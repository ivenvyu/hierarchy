"""설정값으로 PyTorch 최적화기를 생성한다."""

import torch.optim as optim
from typing import Any, Callable, Iterable, List, Tuple, Optional

from utils.setup_logging import get_logger

logger = get_logger("Prompt_CAM")


# 지원하는 최적화기를 명시적으로 분기하며, 그 밖의 값은 SGD 설정으로 처리한다.
def make_optimizer(tune_parameters, params):
    """설정된 최적화기 이름에 따라 학습 파라미터용 최적화기를 생성한다."""
    if params.optimizer == 'adam':
        optimizer = optim.Adam(
            tune_parameters,
            lr=params.lr,
            weight_decay=params.wd,
        )

    elif params.optimizer == 'adamw':
        optimizer = optim.AdamW(
            tune_parameters,
            lr=params.lr,
            weight_decay=params.wd,
        )
    else:
        optimizer = optim.SGD(
            tune_parameters,
            lr=params.lr,
            weight_decay=params.wd,
            momentum=params.momentum,
        )
    return optimizer
