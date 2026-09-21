# 이미지 트랙 앙상블(Fusion) 방식 비교

이진 판별 3개 모델(DINOv3, F3Net, U-Net)의 점수를 결합하는 방식을 **음성 트랙과 같은 프로토콜**로 비교하기 위한 코드입니다. 논문 음성 절(3.4 결합 방법 / 4.5 앙상블 성능)에 대응하는 이미지 절의 실험을 만드는 것이 목적이며, 성능 개선이 목적이 아닙니다.

`fit_fusion.py`는 [heimdall-vox/fusion/fit_fusion.py](../../heimdall-vox/fusion/fit_fusion.py)의 보정·결합 방식·K-Fold·EER 및 fold 대응 비교를 그대로 쓰고(같은 seed에서 EER이 동일함을 확인), `--track image`와 아래 두 가지를 추가했습니다. 음성과 이미지의 비교 조건이 같다는 점이 논문에서 중요하기 때문입니다.

- **정확도 표**: 방식별 임계값 0.5 정확도(out-of-fold, 평균/표준편차/최저·최고 fold). 이미지 절이 정확도를 보고해 왔기 때문에 추가했습니다.
- **`single_*` 행**: 결합 없이 보정만 한 단일 모델의 EER·정확도. 앙상블이 단일 모델보다 나은지 같은 표에서 볼 수 있습니다. ("simple_mean 대비" 열은 단순 평균 기준이라, 단일 모델 행에서 `0/5`는 매 fold 단순 평균보다 나쁘다는 뜻입니다.)

## 비교하는 방식

| 방식 | 파라미터 | 비고 |
| --- | --- | --- |
| 단순 평균 | 0 | 기준선 |
| 다수결 | 0 | 기준선. 점수가 0, 1/3, 2/3, 1 네 값뿐이라 EER이 거칠게 나옴 |
| 고정 가중 평균 | 0 | 학습 분할의 모델별 EER 역수 가중, 기준선 |
| Soft Voting | 1 | `softmax(a·ACC)`, a는 학습 분할 EER 최소화로 탐색 |
| 로지스틱 회귀 | 4 | L2 정규화, C ∈ {0.001, 0.01, 0.1, 1, 10} |

모든 방식 앞에 모델별 Platt scaling을 공통 적용합니다(보정 효과가 방법 간 비교를 교란하지 않도록).

## 실행 순서

세 모델 모두 **Test 세트(10만 장)** 를 같은 이미지로 채점해서 `path,label,raw_score` CSV를 만듭니다. `raw_score`는 sigmoid/softmax 이전 logit입니다.

**1. 특징 재추출 (DINOv3, U-Net)** — 이번에 `extract_features.py`와 `preprocess.py`가 `paths`를 함께 저장하도록 수정됐습니다. 기존 Test 특징 파일에는 `paths`가 없어서 Test 세트만 다시 추출해야 합니다(학습 특징·체크포인트는 그대로 사용 가능). F3Net은 `.pt` 파일 경로 자체가 키가 되므로 재전처리가 필요 없습니다.

`extract_features.py`는 `--num_workers`(기본 8)로 이미지 로딩을 병렬화합니다. 워커가 없으면 메인 프로세스가 이미지를 한 장씩 디코딩해서 GPU가 대부분 놀게 됩니다. 출력 경로(`--output`)는 실행한 디렉토리 기준 상대경로이니, `binary/DINOv3 Code`에서 실행하지 않았다면 `score_test.py`의 `--input`에 실제 저장 위치를 주세요.

**2. 모델별 점수 저장**

```bash
# DINOv3 (binary/DINOv3 Code 에서)
python score_test.py --model heimdall_dinov3_mlp.pth --input ./features/tests_v3_18.pt \
    --crop 5crop --out ../../fusion/scores/image_dinov3.csv

# F3Net (binary/F3Net Code 에서)
python score_test.py --dataset_root ./feature --weights_path best_model.pth \
    --out ../../fusion/scores/image_f3net.csv

# U-Net (binary/U-Net in sdv1.5 + K-NN or MLP code 에서) — MLP 헤드
python score_test.py --input test.pt --model unet_mlp_model.pth --model_type mlp \
    --out ../../fusion/scores/image_unet.csv
```

**3. K-Fold 비교** (`fusion/`에서)

```bash
python fit_fusion.py --track image \
    --scores scores/image_dinov3.csv scores/image_f3net.csv scores/image_unet.csv \
    --names DINOv3 F3Net U-Net --n_folds 5
```

결과 표를 보고 최종 방식을 정한 뒤, `--final_method soft_voting --out image_fusion.json`을 붙여 Test 전체로 재학습한 운용 파라미터를 저장합니다.

## 평가 프로토콜 (음성 절 4.5.1과 동일)

- 앙상블 학습·평가는 Test 세트만 사용합니다. 베이스 모델의 Train은 물론 체크포인트 선택에 쓴 Val(학습 데이터 안의 검증 분할)도 쓰지 않습니다.
- Test 세트 내부 계층적 5-fold 교차검증. 보고하는 수치는 전부 out-of-fold입니다.
- 평균 EER과 함께 fold별 대응 비교("simple_mean보다 나은 fold 수")를 봅니다. 5/5일 때만 일관된 개선으로 봅니다.
- 전체 Test로 재학습한 파라미터로 같은 Test를 재평가한 수치는 성능으로 보고하지 않습니다.

## 조인 키

세 모델의 경로 표기가 달라서(원본 `D:/.../test/00_BDD/a.jpg`, F3Net `feature/test/00_BDD/a.pt`) `test/` 이후의 확장자 없는 상대경로(`00_BDD/a`)를 키로 씁니다. Test 폴더가 평면 18개(`00_BDD` … `19_SDXL`)이든 그룹 폴더가 있는 2단 구조(`00_Real/…`, `01_AI-T2I/…`)이든 같은 키가 나오도록 그룹 폴더는 키에서 뺍니다. 라벨은 그룹 폴더가 있으면 그것으로(`00_Real`=0, `01_AI-T2I`=1), 없으면 폴더명 앞 숫자(0~9 Real, 10~ AI)로 정합니다. 2단 구조에서는 AI 폴더도 `00_BigGAN`~`09_SDXL`로 번호가 Real과 겹치므로 숫자 규칙만 쓰면 AI가 Real로 잘못 라벨링됩니다. 각 `score_test.py`가 끝에 `real=…, AI=…` 개수를 출력하니 각각 5만 개인지 확인하세요. 세 모델이 서로 다른 이미지를 채점했으면 `fit_fusion.py`가 공통 샘플 수와 함께 경고를 출력합니다.

## 알아둘 점

- **U-Net은 MLP 헤드를 사용합니다.** 2-class logit 차(`z_AI - z_Real`)가 그대로 raw_score가 되어 별도 변환이 없습니다. (`score_test.py`는 K-NN도 지원하지만 — 이 경우 점수가 이웃 비율이라 이산적이어서 Laplace 평활 후 logit으로 변환 — 이번 비교에는 쓰지 않습니다.)
- **Soft Voting이 한 모델로 쏠릴 수 있습니다.** 모델 간 정확도 차이가 크면 지수 파라미터 a가 탐색 상한(500)까지 올라가 특정 fold에서 한 모델의 가중치가 사실상 1이 됩니다(음성 절의 가창 트랙에서 나온 현상과 같음). `fit_fusion.py`가 fold별 a와 가중치를 출력하니 표와 함께 확인하세요.
- **기존 이미지 절의 가중치와 다른 값이 나옵니다.** 현재 README/논문의 가중치(DINOv3 34.95 · F3Net 46.28 · U-Net 18.77)는 학습 데이터 Val 정확도 기반이고, 여기서 fit하는 Soft Voting 가중치는 Test 세트 기반입니다. 논문에 둘을 함께 싣는다면 산정 기준이 다르다고 명시해야 합니다.
- 기대 EER이 매우 낮아서(정확도 99.8% 수준) 방식 간 차이가 작을 수 있습니다. fold 대응 비교 열이 표준편차보다 신뢰할 만한 지표입니다.
