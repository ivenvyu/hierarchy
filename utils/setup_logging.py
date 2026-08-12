#!/usr/bin/env python3

"""분산 학습에서도 중복 없이 파일·콘솔 로그를 남기도록 logger를 구성한다."""

import builtins
import functools
import logging
import sys
import os
from .distributed import is_master_process

# 로그에 파일명과 줄 번호를 표시한다.
_FORMAT = "[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s"


def _suppress_print():
    """주 프로세스가 아닌 곳에서 print 출력을 막는 대체 함수를 설치한다."""

    def print_pass(*objects, sep=" ", end="\n", file=sys.stdout, flush=False):
        """print_pass 작업에 필요한 입력을 처리하고 계산 결과를 반환한다."""
        pass

    builtins.print = print_pass


# 열린 파일 객체를 캐시해 같은 파일명을 사용하는 여러 `setup_logger` 호출이
# 하나의 파일에 안전하게 기록하도록 한다.
@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    """동일 파일에 대한 line-buffered 로그 스트림을 재사용한다."""
    return open(filename, "a", encoding="utf-8", buffering=1)


@functools.lru_cache()  # 여러 번 호출해도 핸들러가 중복되지 않게 한다.  # noqa
def setup_logging(
    num_gpu, num_shards, output="", name="Prompt_CAM", color=True):
    """분산 순위를 고려해 콘솔과 파일 핸들러를 설정한다."""
    # 주 프로세스에서만 로깅을 활성화한다.
    if is_master_process(num_gpu):
        # 다른 모듈 등이 설정한 기존 로깅 구성이 영향을 주지 않도록 루트
        # 로거를 비운다.
        logging.root.handlers = []
        # 로깅을 설정한다.
        logging.basicConfig(
            level=logging.INFO, format=_FORMAT, stream=sys.stdout
        )
    else:
        _suppress_print()

    if name is None:
        name = __name__
    logger = logging.getLogger(name)
    # 남아 있는 핸들러를 제거한다.
    logger.handlers.clear()

    logger.setLevel(logging.INFO)
    logger.propagate = False

    plain_formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(name)s: %(lineno)4d: %(message)s",
        datefmt="%m/%d %H:%M:%S",
    )
    formatter = plain_formatter

    if is_master_process(num_gpu):
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    if is_master_process(num_gpu * num_shards):
        if len(output) > 0:
            if output.endswith(".txt") or output.endswith(".log"):
                filename = output
            else:
                filename = os.path.join(output, "logs.txt")

            os.makedirs(os.path.dirname(filename), exist_ok=True)

            fh = logging.StreamHandler(_cached_log_stream(filename))
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(plain_formatter)
            logger.addHandler(fh)
    return logger


def setup_single_logging(name, output=""):
    """단일 프로세스 실행용 간단한 콘솔·파일 로거를 설정한다."""
    # 주 프로세스에서만 로깅을 활성화하고 다른 모듈의 기존 설정이 영향을 주지
    # 않도록 루트 로거를 비운다.
    logging.root.handlers = []
    # 로깅을 설정한다.
    logging.basicConfig(
        level=logging.INFO, format=_FORMAT, stream=sys.stdout
    )

    if len(name) == 0:
        name = __name__
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    plain_formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(name)s: %(lineno)4d: %(message)s",
        datefmt="%m/%d %H:%M:%S",
    )
    formatter = plain_formatter

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if len(output) > 0:
        if output.endswith(".txt") or output.endswith(".log"):
            filename = output
        else:
            filename = os.path.join(output, "logs.txt")

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        fh = logging.StreamHandler(_cached_log_stream(filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(plain_formatter)
        logger.addHandler(fh)

    return logger


def get_logger(name):
    """프로젝트에서 설정한 이름의 로거를 반환한다."""
    return logging.getLogger(name)
