"""YAML 설정으로 Prompt-CAM 학습을 시작하는 명령줄 진입점."""

import argparse
from training.run import basic_run
from utils.setup_logging import get_logger
from utils.misc import set_seed,load_yaml,override_args_with_yaml
import time

logger = get_logger("Prompt_CAM")

def str2bool(value):
    """명령줄의 다양한 참·거짓 문자열을 Python bool로 변환한다."""
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"참·거짓 값이 필요하지만 다음 값이 입력되었습니다: {value}")



def main():
    """명령줄 인자를 해석하고 이 스크립트의 전체 작업을 실행한다."""
    args = setup_parser().parse_args()

    # YAML 값을 명령줄 기본값 위에 반영해 한 파일로 실험을 재현할 수 있게 한다.
    if args.config:
        yaml_config = load_yaml(args.config)
        if yaml_config:
            args = override_args_with_yaml(args, yaml_config)

    # 데이터 순서와 초기화를 가능한 범위에서 재현하도록 모든 난수원을 고정한다.
    set_seed(args.random_seed)
    start = time.time()
    args.vis_attn = bool(getattr(args, "vis_attn", False))
    # 데이터·모델·Trainer 구성과 학습 루프는 training.run에 위임한다.
    basic_run(args)
    end = time.time()
    logger.info(f'----------- 전체 실행 시간: {(end - start) / 60}분 -----------')


def setup_parser():
    """프로그램에서 사용할 명령줄 인자 parser를 구성한다."""
    # --- 실험 YAML과 실행 모드 ---
    parser = argparse.ArgumentParser(description='Prompt_CAM')

    ######################## 사전 학습 모델 #########################
    parser.add_argument('--pretrained_weights', type=str, default='vit_base_patch16_224_in21k',
                        choices=['vit_base_patch16_224_in21k', 'vit_base_mae', 'vit_base_patch14_dinov2','vit_base_patch16_dino',
                                 'vit_base_patch16_clip_224'],
                        help='사전 학습 가중치 이름')
    parser.add_argument('--drop_path_rate', default=0.1,
                        type=float,
                        help='드롭 경로 비율(기본값: %(default)s)')
    parser.add_argument('--model', type=str, default='dinov2', choices=['vit', 'dino', 'dinov2'],
                        help='사전 학습 모델 이름')
    parser.add_argument('--pretrained_checkpoint', type=str, default=None,
                        help='고정 백본 체크포인트의 명시적 로컬 경로')
    parser.add_argument('--promptcam_checkpoint', type=str, default=None,
                        help='선택적 Prompt-CAM 프롬프트/헤드 초기화 체크포인트')
    parser.add_argument('--load_pretrained_backbone', type=str2bool, default=True,
                        help='프롬프트 학습 전에 지정한 pretrained_checkpoint 불러오기')

    parser.add_argument('--train_type', type=str, default='vpt', choices=['vpt', 'prompt_cam', 'linear'],
                        help='학습 방식')

    ######################## 최적화기와 스케줄러 #########################
    parser.add_argument('--optimizer', default='sgd', choices=['sgd', 'adam', 'adamw'],
                        help='최적화기(기본값: %(default)s)')
    parser.add_argument('--lr', default=0.005,
                        type=float,
                        help='학습률(기본값: %(default)s)')
    parser.add_argument('--epoch', default=100,
                        type=int,
                        help='전체 에폭 수(기본값: %(default)s)')
    parser.add_argument('--warmup_epoch', default=20,
                        type=int,
                        help='스케줄러 준비 에폭 수(기본값: %(default)s)')
    parser.add_argument('--lr_min', type=float, default=1e-5,
                        help='스케줄러의 최소 학습률(기본값: %(default)s)')
    parser.add_argument('--warmup_lr_init', type=float, default=1e-6,
                        help='준비 구간 초기 학습률(기본값: %(default)s)')
    parser.add_argument('--batch_size', default=16,
                        type=int,
                        help='배치 크기(기본값: %(default)s)')
    parser.add_argument('--test_batch_size', default=32,
                        type=int,
                        help='테스트 배치 크기(기본값: %(default)s)')
    parser.add_argument('--wd', type=float, default=0.001,
                        help='가중치 감쇠(기본값: %(default)s)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD 모멘텀(기본값: %(default)s)')
    parser.add_argument('--early_patience', type=int, default=101,
                        help='조기 종료 대기 에폭 수(기본값: %(default)s)')

    ######################## 데이터 #########################
    parser.add_argument('--data', default="processed_vtab-dtd",
                        help='데이터 이름(기본값: %(default)s)')
    parser.add_argument('--data_path', default="data_folder/vtab_processed",
                        help='데이터셋 경로(기본값: %(default)s)')
    parser.add_argument('--crop_size', default=224,
                        type=int,
                        help='입력 이미지 자르기 크기(기본값: %(default)s)')
    parser.add_argument('--class_num', default=0, type=int,
                        help='클래스 수. ImageFolder 모드에서는 폴더 수와 일치하는지 검사한다')
    parser.add_argument('--train_split', default='train', type=str)
    parser.add_argument('--val_split', default='val', type=str)
    parser.add_argument('--test_split', default='test', type=str)
    parser.add_argument('--normalization', default='inception', choices=['inception', 'imagenet'])
    parser.add_argument('--normalization_mean', nargs=3, type=float, default=None)
    parser.add_argument('--normalization_std', nargs=3, type=float, default=None)
    parser.add_argument('--train_scale_min', default=0.5, type=float)
    parser.add_argument('--train_scale_max', default=1.0, type=float)
    parser.add_argument('--train_ratio_min', default=0.75, type=float)
    parser.add_argument('--train_ratio_max', default=4.0 / 3.0, type=float)
    parser.add_argument('--eval_resize_size', default=None, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--drop_last', default=True, type=str2bool)
    parser.add_argument('--final_run', action='store_false',
                        help='final_run이 참이면 train+val을, 거짓이면 train만 학습에 사용한다')
    parser.add_argument('--normalized', action='store_false',
                        help='ImageNet 평균과 분산으로 이미지를 정규화할지 여부')

    ######################## VPT #########################
    parser.add_argument('--vpt_mode', type=str, default=None, choices=['deep', 'shallow'],
                        help='VPT 방식: deep 또는 shallow')
    parser.add_argument('--vpt_num', default=10, type=int,
                        help='프롬프트 수(기본값: %(default)s)')
    parser.add_argument('--vpt_layer', default=None, type=int,
                        help='마지막 계층부터 프롬프트를 추가할 계층 수(기본값: %(default)s)')
    parser.add_argument('--vpt_dropout', default=0.1, type=float,
                        help='deep 방식의 VPT 드롭아웃 비율(기본값: %(default)s)')

    ######################## 계층적 Prompt-CAM #########################
    parser.add_argument('--hierarchical_prompt', type=str2bool, default=False,
                        help='과·속·종 잔차 프롬프트와 속 게이트 종 CAM 사용')
    parser.add_argument('--original_taxonomy_prompt', type=str2bool, default=False,
                        help='Prompt-CAM 원논문처럼 taxonomy 내부 node마다 독립 flat Prompt-CAM 학습')
    parser.add_argument('--prompt_patch_only_head', type=str2bool, default=False,
                        help='비계층 Prompt-CAM을 residual 없는 prompt-to-patch decoder로 분류')
    parser.add_argument('--taxonomy_node_rank', type=str, default='root',
                        choices=['root', 'family', 'genus'],
                        help='원논문식 Prompt-CAM을 학습할 taxonomy node 수준')
    parser.add_argument('--taxonomy_node_name', type=str, default=None,
                        help='family/genus node 이름. root node에서는 사용하지 않음')
    parser.add_argument('--taxonomy_experiment_run_id', type=str, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument('--taxonomy_manifest', type=str, default=None,
                        help='scientific_name, genus, family와 클래스 별칭이 있는 CSV')
    parser.add_argument('--taxonomy_class_column', type=str, default=None,
                        help='ImageFolder 이름 대응에 사용할 선택적 분류 체계 CSV 열')
    parser.add_argument('--identifiability_enabled', type=str2bool, default=False,
                        help='종·속·과·식별 불가 등급 헤드 학습')
    parser.add_argument('--identifiability_manifest', type=str, default=None,
                        help='relative_path와 identifiable_rank를 담은 이미지별 CSV')
    parser.add_argument('--missing_identifiability_policy', type=str, default='species',
                        choices=['species', 'error'])
    parser.add_argument('--bbox_coordinate_mode', type=str, default='normalized',
                        choices=['normalized', 'pixels'])
    parser.add_argument('--bbox_attention_gate', type=str2bool, default=False,
                        help='속·종 패치 에너지에 로그 bbox 중첩 사전확률 추가')
    parser.add_argument('--taxonomy_objective', type=str, default='node_conditional',
                        choices=['node_conditional'],
                        help='taxonomy 경로 확률 목적함수')
    parser.add_argument('--loss_family_path_weight', type=float, default=1.0)
    parser.add_argument('--loss_genus_path_weight', type=float, default=1.0)
    parser.add_argument('--loss_species_path_weight', type=float, default=1.0)
    parser.add_argument('--loss_rank_weight', type=float, default=1.0)
    parser.add_argument('--loss_localization_weight', type=float, default=0.0)
    parser.add_argument('--loss_localization_genus_weight', type=float, default=1.0)
    parser.add_argument('--loss_center_species_weight', type=float, default=0.001)
    parser.add_argument('--loss_center_genus_weight', type=float, default=0.001)

    ######################## 의미 기반 SnapMix #########################
    parser.add_argument('--semantic_snapmix', type=str2bool, default=False,
                        help='정답 Prompt-CAM 어텐션을 SnapMix SPM으로 사용')
    parser.add_argument('--mix_probability', default=0.5, type=float)
    parser.add_argument('--mix_beta', default=1.0, type=float)
    parser.add_argument('--eta_start_epoch', default=5, type=int)
    parser.add_argument('--eta_end_epoch', default=15, type=int)
    parser.add_argument('--ema_decay', default=0.999, type=float)
    parser.add_argument('--spm_head_reduction', default='mean', choices=['mean', 'max'])
    parser.add_argument('--spm_eps', default=1e-8, type=float)
    parser.add_argument('--amp_dtype', default='none', choices=['none', 'float16', 'bfloat16'])
    parser.add_argument('--selection_metric', default='macro_f1',
                        choices=['top1', 'balanced_accuracy', 'macro_f1', 'genus_accuracy', 'family_accuracy', 'rank_accuracy'])
    parser.add_argument('--resume', default=None, type=str,
                        help='학생·EMA 교사·최적화기·스케줄러가 포함된 체크포인트에서 재개')
    parser.add_argument('--vis_attn', type=str2bool, default=False)

    ######################## 전체 미세 조정 #########################
    parser.add_argument('--full', action='store_true',
                        help='전체 미세 조정 활성화 여부')

    ######################## 기타 #########################
    parser.add_argument('--gpu_num', default=1,
                        type=int,
                        help='GPU 수(기본값: %(default)s)')
    parser.add_argument('--debug', action='store_false',
                        help='추가 정보를 표시하는 디버그 모드(기본값: %(default)s)')
    parser.add_argument('--random_seed', default=42,
                        type=int,
                        help='난수 시드(기본값: %(default)s)')
    parser.add_argument('--eval_freq', default=10,
                        type=int,
                        help='평가 주기(에폭, 기본값: %(default)s)')
    parser.add_argument('--store_ckp', action='store_true',
                        help='체크포인트 저장 여부')
    parser.add_argument('--output_root', default='./output', type=str,
                        help='시간표시가 붙은 실험 결과의 최상위 디렉터리')
    parser.add_argument('--final_acc_hp', action='store_false',
                        help='참이면 전체 에폭의 최고 정확도, 거짓이면 마지막 에폭 정확도로 초매개변수를 선택한다')

    ######################## YAML 설정 #########################
    parser.add_argument('--config', type=str, default=None, help='YAML 설정 파일 경로')

    return parser


# 모듈 import 시에는 실행하지 않고 직접 호출된 경우에만 학습을 시작한다.
if __name__ == '__main__':
    main()
