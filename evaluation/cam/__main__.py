"""종 및 계층 Prompt-CAM 시각화 명령줄 인터페이스."""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence


_CLI_FIELDS = {
    "command",
    "config",
    "checkpoint",
    "vis_cls",
    "nmbr_samples",
    "top_traits",
    "vis_outdir",
    "random_seed",
    "gpu_num",
}


def _add_common_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_samples: int,
    default_output: str,
) -> None:
    parser.add_argument("--config", required=True, help="학습 설정 YAML")
    parser.add_argument("--checkpoint", required=True, help="model.pt 경로")
    parser.add_argument(
        "--vis-cls",
        "--vis_cls",
        dest="vis_cls",
        required=True,
        type=int,
        help="class_to_idx.json 기준 0-based 종 index",
    )
    parser.add_argument(
        "--num-samples",
        "--nmbr-samples",
        "--nmbr_samples",
        dest="nmbr_samples",
        type=int,
        default=default_samples,
        help="생성할 정답 표본 수",
    )
    parser.add_argument(
        "--top-traits",
        "--top_traits",
        dest="top_traits",
        type=int,
        default=4,
        help="표본별로 표시할 주요 attention head 수",
    )
    parser.add_argument(
        "--output-dir",
        "--vis-outdir",
        "--vis_outdir",
        dest="vis_outdir",
        default=default_output,
        help="시각화 결과 디렉터리",
    )
    parser.add_argument(
        "--random-seed",
        "--random_seed",
        dest="random_seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--gpu-num",
        "--gpu_num",
        dest="gpu_num",
        type=int,
        default=1,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.cam",
        description="학습된 Prompt-CAM checkpoint의 attention을 시각화합니다.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    species = subparsers.add_parser(
        "species",
        help="Flat Prompt-CAM의 종 attention 시각화",
    )
    _add_common_arguments(
        species,
        default_samples=10,
        default_output="output/cam/species",
    )

    hierarchy = subparsers.add_parser(
        "hierarchy",
        help="Shared 모델의 과·속·종 attention 비교",
    )
    _add_common_arguments(
        hierarchy,
        default_samples=5,
        default_output="output/cam/hierarchy",
    )
    return parser


def _apply_yaml(args: argparse.Namespace) -> None:
    # 무거운 모델 의존성은 실제 실행 시점까지 불러오지 않는다.
    from utils.misc import load_yaml

    yaml_config = load_yaml(args.config)
    for key, value in yaml_config.items():
        if key not in _CLI_FIELDS:
            setattr(args, key, value)


def run(args: argparse.Namespace) -> None:
    from utils.misc import set_seed
    from utils.setup_logging import get_logger

    _apply_yaml(args)
    args.vis_attn = True
    set_seed(args.random_seed)

    if args.command == "species":
        from evaluation.cam.species import basic_vis

        visualizer = basic_vis
    else:
        from evaluation.cam.hierarchy import basic_hierarchy_vis

        visualizer = basic_hierarchy_vis

    started = time.time()
    visualizer(args)
    get_logger("Prompt_CAM").info(
        "----------- 전체 실행 시간: %.4f분 -----------",
        (time.time() - started) / 60.0,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
