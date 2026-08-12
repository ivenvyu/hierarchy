#!/usr/bin/env python3

"""PyTorch distributed 실행의 rank 조회, process group 관리, tensor 통신을 제공한다."""

import torch
import torch.distributed as dist
_LOCAL_PROCESS_GROUP = None


def get_world_size() -> int:
    """초기화된 process group의 전체 process 수를 반환한다."""
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    """현재 process의 전역 rank를 반환한다."""
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_master_process(num_gpus=8):
    """is_master_process 작업에 필요한 입력을 처리하고 계산 결과를 반환한다."""
    if torch.distributed.is_initialized():
        return dist.get_rank() % num_gpus == 0
    else:
        return True


def run(
    local_rank,
    num_proc,
    func,
    init_method,
    shard_id,
    num_shards,
    backend,
    cfg,
    args,
):
    """한 worker의 device와 process group을 설정한 뒤 사용자 함수를 실행한다."""
    # 프로세스 그룹을 초기화한다.
    # 샤드 식별자 = get_rank()
    world_size = num_proc * num_shards
    rank = shard_id * num_proc + local_rank

    try:
        torch.distributed.init_process_group(
            backend=backend,
            init_method=init_method,
            world_size=world_size,
            rank=rank,
        )
    except Exception as e:
        raise e

    # 각 worker를 하나의 local GPU에 고정한 뒤 process group을 초기화한다.
    torch.cuda.set_device(local_rank)
    func(cfg, args)


def destroy_process_group():
    """초기화된 distributed process group을 안전하게 종료한다."""
    torch.distributed.destroy_process_group()


def scaled_all_reduce(cfg, tensors):
    """tensor 목록을 all-reduce하고 선택적으로 world size로 나눈다."""
    # 축소 연산을 대기열에 넣는다.
    reductions = []
    for tensor in tensors:
        reduction = torch.distributed.all_reduce(tensor, async_op=True)
        reductions.append(reduction)
    # 축소 연산이 끝날 때까지 기다린다.
    for reduction in reductions:
        reduction.wait()
    # 결과의 크기를 조정한다.
    for tensor in tensors:
        tensor.mul_(1.0 / cfg.NUM_GPUS / cfg.NUM_SHARDS)
    return tensors


def cat_all_gather(tensors):
    """모든 rank의 tensor를 모아 첫 번째 차원으로 연결한다."""
    tensors_gather = [
        torch.ones_like(tensors)
        for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensors, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def local_cat_all_gather(tensors):
    """같은 node의 local rank tensor만 모아 연결한다."""
    tensors_gather = [
        torch.ones_like(tensors)
        for _ in range(get_local_size())
    ]
    torch.distributed.all_gather(
        tensors_gather,
        tensors,
        async_op=False,
        group=_LOCAL_PROCESS_GROUP,
    )
    output = torch.cat(tensors_gather, dim=0)
    return output


def get_local_size():
    """현재 node에서 실행 중인 local process 수를 반환한다."""
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size(group=_LOCAL_PROCESS_GROUP)


def get_local_rank():
    """현재 process의 node 내부 local rank를 반환한다."""
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    assert _LOCAL_PROCESS_GROUP is not None
    return dist.get_rank(group=_LOCAL_PROCESS_GROUP)
