"""seed, YAML, 성능 측정, 평균 추적, 조기 종료 등 공통 실행 도구를 제공한다."""

import torch
import numpy as np
import random
import time
import yaml
import re

class AverageMeter(object):
    """현재값, 누적합, 표본수, 평균을 온라인으로 추적한다."""
    def __init__(self, name=None, fmt=':f'):
        """객체가 사용할 입력 설정과 내부 상태를 초기화한다."""
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        """누적 통계와 카운터를 초기 상태로 되돌린다."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """새 관측값을 누적 통계에 반영한다."""
        self.val = val
        # 표본 수 n을 가중치로 사용해 누적 평균을 정확히 갱신한다.
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        """현재 상태를 로그에 표시할 문자열로 변환한다."""
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def method_name(params):
    """설정값을 조합해 실험 결과 디렉터리에 사용할 방법 이름을 만든다."""
    name = ''
    if params.train_type == 'prompt_cam':
        if bool(getattr(params, 'original_taxonomy_prompt', False)):
            node_id = getattr(params, 'taxonomy_node_id', None)
            if not node_id:
                rank = str(getattr(params, 'taxonomy_node_rank', 'node')).strip().lower()
                node_name = getattr(params, 'taxonomy_node_name', None)
                node_id = rank if rank == 'root' or node_name in (None, '', 'null') else f'{rank}_{node_name}'
            node_id = re.sub(r'[^0-9A-Za-z._-]+', '-', str(node_id)).strip('-').lower()
            name += 'original_taxonomy_prompt_cam_' + node_id + '_'
        elif bool(getattr(params, 'hierarchical_prompt', False)):
            name += 'hierarchical_prompt_cam_'
        else:
            name += 'pcam_' + params.train_type + '_' + str(params.vpt_num) + '_'
    elif params.vpt_mode:
        name += 'vpt_' + params.vpt_mode + '_' + str(params.vpt_num) + '_' + str(params.vpt_layer) + '_'
    if name == '':
        name += 'linear_'
    name += params.optimizer
    if bool(getattr(params, "semantic_snapmix", False)):
        name += "_semantic_snapmix"
    return name


def set_seed(random_seed=42):
    """Python, NumPy, PyTorch 난수 시드를 고정해 실행 재현성을 높인다."""
    np.random.seed(random_seed)
    random.seed(random_seed)
    # CPU와 모든 CUDA device의 난수 상태를 같은 seed로 맞춘다.
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@torch.no_grad()
def throughput(model,img_size=224,bs=1):
    """더미 입력을 반복 실행해 모델의 초당 이미지 처리량을 측정한다."""
    with torch.no_grad():
        x = torch.randn(bs, 3, img_size, img_size).cuda()
        batch_size=x.shape[0]
        # model=create_model('vit_base_patch16_224_in21k', checkpoint_path='./ViT-B_16.npz', drop_path_rate=0.1)
        model.eval()
        for i in range(50):
            model(x)
        torch.cuda.synchronize()
        print("처리량은 30회 평균입니다")
        tic1 = time.time()
        for i in range(30):
            model(x)
        torch.cuda.synchronize()
        tic2 = time.time()
        print(f"배치 크기 {batch_size}, 처리량 {30 * batch_size / (tic2 - tic1)}")
        MB = 1024.0 * 1024.0
        print('메모리:', torch.cuda.max_memory_allocated() / MB)

def load_yaml(path):
    """YAML 파일을 읽어 Python 딕셔너리로 반환한다."""
    with open(path, 'r') as stream:
        try:
            return yaml.load(stream, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)

def override_args_with_yaml(args, yaml_config):
    """YAML의 키–값을 argparse namespace에 덮어쓴다."""
    for key, value in yaml_config.items():
        if hasattr(args, key):
            setattr(args, key, value)
    return args

class EarlyStop:
    """검증 지표가 개선되지 않는 epoch 수를 세어 조기 종료를 결정한다."""
    def __init__(self, patience=1, min_delta=0):
        """객체가 사용할 입력 설정과 내부 상태를 초기화한다."""
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.max_metrics = None

    def early_stop(self, eval_metrics):
        """현재 검증 지표를 반영하고 patience 초과 여부를 반환한다."""
        if self.max_metrics is None:
            self.max_metrics = eval_metrics
        if eval_metrics['top1'] > self.max_metrics['top1']:
            self.max_metrics = eval_metrics
            self.counter = 0
            return False, True
        elif eval_metrics['top1'] < (self.max_metrics['top1'] - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True, False
        return False, False
