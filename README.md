# PromptCAM-SnapMix

Wild-30 식물 이미지에서 Prompt-CAM을 과·속·종 계층으로 확장하고 Semantic SnapMix를 적용한 학습·평가 코드입니다. **Hierarchy가 제안 모델이며 Flat과 Independent는 비교 baseline입니다.**

| 역할 | 모델 | 설명 | 설정 |
|---|---|---|---|
| 제안 모델 | Hierarchy | 과·속·종을 한 모델에서 공동 학습 | `configs/hierarchy.yaml` |
| Baseline | Flat | 30개 종을 직접 분류 | `configs/flat.yaml` |
| Baseline | Independent | taxonomy node별로 7개 모델 학습 | `configs/independent.yaml` |

데이터, DINOv2 가중치, 학습 checkpoint와 결과물은 저장소에 포함하지 않습니다.

## 1. 설치

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 2. 데이터와 백본

Wild-30 고정 분할은 train 9,000장, val 1,500장, test 1,500장입니다.

```bash
# 매니페스트 검사
python data/restore_wild30_dataset.py --validate-only

# 이미지 복원
python data/restore_wild30_dataset.py
```

이미지는 `data/dataset/imagefolder/{train,val,test}/<class>/`에 저장됩니다.

DINOv2 ViT-B/14 가중치를 다음 위치에 받습니다.

```bash
mkdir -p pretrained_weights
curl -L https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth -o pretrained_weights/dinov2_vitb14_pretrain.pth
```

```text
경로: pretrained_weights/dinov2_vitb14_pretrain.pth
SHA-256: 0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73
```

## 3. Hierarchy 학습

```bash
python train.py --config configs/hierarchy.yaml
```

비교 baseline이 필요한 경우에만 추가로 학습합니다.

```bash
# Flat baseline
python train.py --config configs/flat.yaml

# Independent baseline 전체 7개 node
python -m training.independent --base-config configs/independent.yaml --run-id independent-v1

# Independent 실행 명령만 확인
python -m training.independent --base-config configs/independent.yaml --run-id independent-v1 --dry-run
```

각 실행에는 설정과 mapping, best checkpoint, 최종 지표가 저장됩니다.

```text
args.yaml  class_to_idx.json  taxonomy.json  model.pt  last.pt  final_result.json
```

## 4. Hierarchy checkpoint 평가

평가에는 `model.pt`만이 아니라 같은 실행의 `args.yaml`과 taxonomy/class mapping도 필요합니다.

```bash
# 실행 디렉터리 지정
python -m evaluation.hierarchy --project-root . --run-dir "<hierarchy-run-dir>" --device cuda --batch-size 16 --num-workers 4

# 최신 실행 자동 탐색
python -m evaluation.hierarchy --project-root . --search-root output/shared --device cuda
```

결과는 summary JSON, 표본별 예측 CSV, class별 지표와 confusion matrix로 저장됩니다.

### Baseline 평가

```bash
# Independent run 평가
python -m evaluation.independent --run-id independent-v1 --split test --device cuda --batch-size 16 --num-workers 4

# Independent 저용량 GPU용 순차 평가
python -m evaluation.checkpoints --training-summary "output/independent/runs/independent-v1/training_summary.json" --split test --device cuda --amp-dtype none --output "output/evaluations/independent_test.json"
```

Flat은 별도 `evaluation.flat` 명령이 없습니다. 학습 종료 시 best `model.pt`로 test split을 평가하고 `<flat-run-dir>/final_result.json`의 `final_test_metrics`에 기록합니다.

완료된 실행 경로 찾기:

```bash
python -m evaluation.tools.find_runs --project-root .
```

## 5. CAM 생성

`--vis-cls`는 `class_to_idx.json` 기준의 0-based 종 index입니다.

```bash
# 제안 모델: Hierarchy 과·속·종 CAM
python -m evaluation.cam hierarchy --config configs/hierarchy.yaml --checkpoint "<hierarchy-run-dir>/model.pt" --vis-cls 0 --num-samples 5 --output-dir "output/cam/hierarchy"

# Baseline: Flat 종 CAM
python -m evaluation.cam species --config configs/flat.yaml --checkpoint "<flat-run-dir>/model.pt" --vis-cls 0 --num-samples 5 --output-dir "output/cam/flat"
```

Independent CAM과 모델 간 비교는 입력 경로가 많으므로 도움말을 확인합니다.

```bash
python -m evaluation.cam.visualize_independent --help
python -m evaluation.cam.compare --help
python -m evaluation.cam.compare_official --help
```

## 6. 추가 평가

```bash
python -m evaluation.hsc --help
python -m evaluation.cam.faithfulness --help
python -m evaluation.cam.quality --help
python -m evaluation.cam.occlusion --help
python -m evaluation.deletion.compare --help
python -m evaluation.metrics.bootstrap --help
```

## 구조

```text
configs/       실험 설정
data/          데이터 복원, loader, taxonomy
evaluation/    모델 평가, CAM, deletion, 통계
model/         Prompt-CAM과 계층 모델
training/      학습, loss, Semantic SnapMix
utils/         로깅과 공용 함수
train.py       학습 진입점
```

데이터, 가중치, checkpoint와 생성 결과는 `.gitignore`로 제외됩니다. 코드는 [`LICENSE`](LICENSE)를 따르며 이미지별 라이선스와 attribution은 `data/manifests/wild30_frozen.csv`에 기록되어 있습니다.
