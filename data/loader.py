"""학습과 평가에서 공유하는 데이터 로더를 생성한다."""

import torch
from torch.utils.data.distributed import DistributedSampler


def get_dataset(data, params, logger):
    """선택한 데이터셋 모듈만 불러와 train·validation·test Dataset을 생성한다."""
    if not data.startswith("imagefolder"):
        raise ValueError(
            f"데이터셋 '{data}'은(는) 지원하지 않습니다. ImageFolder 데이터셋을 사용하세요"
        )

    from data.dataset.imagefolder import get_imagefolder

    logger.info("범용 ImageFolder 데이터를 불러옵니다...")
    dataset_train, dataset_val, dataset_test = get_imagefolder(
        params,
        mode="splits",
    )

    return (
        dataset_train,
        dataset_val,
        dataset_test,
    )


def get_loader(params, logger):
    """설정된 데이터셋을 만든 뒤 split별 DataLoader를 반환한다."""
    data_name = (
        params.test_data
        if getattr(params, "test_data", None)
        else params.data
    )

    dataset_train, dataset_val, dataset_test = get_dataset(
        data_name,
        params,
        logger,
    )

    if isinstance(dataset_train, list):
        train_loader = []
        val_loader = []
        test_loader = []

        for index in range(len(dataset_train)):
            current_val = (
                dataset_val[index]
                if dataset_val is not None
                else None
            )

            current_train_loader, current_val_loader, current_test_loader = (
                gen_loader(
                    params,
                    dataset_train[index],
                    current_val,
                    None,
                )
            )

            train_loader.append(
                current_train_loader
            )
            val_loader.append(
                current_val_loader
            )
            test_loader.append(
                current_test_loader
            )

    else:
        train_loader, val_loader, test_loader = gen_loader(
            params,
            dataset_train,
            dataset_val,
            dataset_test,
        )

    logger.info("데이터 로더 설정을 마쳤습니다")

    return (
        train_loader,
        val_loader,
        test_loader,
    )


def gen_loader(
    params,
    dataset_train,
    dataset_val,
    dataset_test,
):
    """분산 sampler와 batch 설정을 반영해 split별 DataLoader를 생성한다."""
    train_loader = None
    val_loader = None
    test_loader = None

    debug = bool(
        getattr(
            params,
            "debug",
            False,
        )
    )

    num_workers = int(
        getattr(
            params,
            "num_workers",
            1 if debug else 4,
        )
    )

    distributed = bool(
        getattr(
            params,
            "distributed",
            False,
        )
    )

    # -------- 학습 --------
    if dataset_train is not None:
        if distributed:
            train_sampler = DistributedSampler(
                dataset_train,
                shuffle=True,
            )
            shuffle = False
        else:
            train_sampler = None
            shuffle = True

        train_loader = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=params.batch_size,
            shuffle=shuffle,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=bool(
                getattr(
                    params,
                    "drop_last",
                    True,
                )
            ),
            persistent_workers=num_workers > 0,
        )

    # -------- 검증 --------
    if dataset_val is not None:
        if distributed:
            val_sampler = DistributedSampler(
                dataset_val,
                shuffle=False,
            )
        else:
            val_sampler = None

        val_loader = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=params.test_batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )

    # -------- 테스트 --------
    if dataset_test is not None:
        if distributed:
            test_sampler = DistributedSampler(
                dataset_test,
                shuffle=False,
            )
        else:
            test_sampler = None

        test_loader = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=params.test_batch_size,
            shuffle=False,
            sampler=test_sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )

    return (
        train_loader,
        val_loader,
        test_loader,
    )
