#!/bin/bash
# =============================================================================
# 원격 PC 이미지 실험 일괄 실행 스크립트
#
# 아래 세 작업을 순서대로, 이미 끝난 단계는 건너뛰며 실행한다.
#   A  F3Net 다중 분류 정확도            -> 논문 표 17
#   B  이진 앙상블 5-fold 비교            -> 논문 6.5절
#   C  다중 분류 앙상블 비교              -> 논문 6.4절
# 자세한 배경·문제 해결법은 [원격 PC] 이미지 실험 실행 가이드.md 참고.
#
# 사용법:
#   1) 아래 "설정" 값을 원격 PC에 맞게 고친다.
#   2) 이 파일을 heimdall-image 저장소 최상위(이 파일이 있는 위치)에서 실행한다.
#        bash run_remote_experiments.sh 2>&1 | tee run_log_$(date +%Y%m%d_%H%M%S).txt
#   3) 중간에 실패하면 원인을 고친 뒤 그냥 다시 실행한다. 이미 끝난 단계는
#      결과 파일이 있으면 자동으로 건너뛴다(재실행해도 안전).
# =============================================================================
set -e   # 각 단계 실패 시 즉시 중단 (원인 파악 전 뒷 단계로 넘어가지 않기 위함)

# ------------------------------- 설정 (원격 PC에 맞게 수정) -------------------------------
DATA_ROOT="D:/Heimdall_Image_Dataset"     # train/, test/ 가 들어있는 최상위 폴더
PY="./.venv/Scripts/python.exe"           # 각 하위 폴더의 가상환경 python (run_all.sh와 동일)
# -----------------------------------------------------------------------------------------

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
echo "작업 루트: $ROOT"
echo "데이터 루트: $DATA_ROOT"
echo

# ---- 0. 사전 점검: 이번에 새로 추가한 스크립트들이 다 있는지 ----
REQUIRED=(
  "binary/DINOv3 Code/score_test.py"
  "binary/F3Net Code/score_test.py"
  "binary/U-Net in sdv1.5 + K-NN or MLP code/score_test.py"
  "multiple/DINOv3 Code/score_test_multi.py"
  "multiple/F3Net Code/score_test_multi.py"
  "multiple/U-Net in sdv1.5 + K-NN or MLP code/score_test_multi.py"
  "fusion/fit_fusion.py"
  "fusion/fuse_multiclass.py"
)
missing=0
for f in "${REQUIRED[@]}"; do
  [ -f "$f" ] || { echo "  [누락] $f"; missing=1; }
done
if [ "$missing" = 1 ]; then
  echo "필요한 스크립트가 없습니다. heimdall-image 저장소를 최신으로 받은 뒤(git pull) 다시 실행하세요."
  exit 1
fi
mkdir -p fusion/scores
echo "사전 점검 통과"
echo

# ---- 가중치 존재 확인 (없으면 그 단계에서 바로 실패하니 미리 안내만) ----
check_weight() { [ -f "$1" ] || echo "  [주의] 가중치 없음: $1 (해당 단계 실패할 수 있음)"; }
check_weight "multiple/F3Net Code/best_model.pth"
check_weight "binary/DINOv3 Code/models/heimdall_dinov3_mlp.pth"
check_weight "binary/F3Net Code/best_model.pth"
check_weight "binary/U-Net in sdv1.5 + K-NN or MLP code/unet_mlp_model.pth"
check_weight "multiple/DINOv3 Code/models/dinov3_multiclass.pth"
check_weight "multiple/U-Net in sdv1.5 + K-NN or MLP code/unet_mlp_model_multi.pth"
echo

# preprocessing.py는 train/test를 함께 처리하므로, test 전용 임시본을 만들어 쓰고
# 끝나면 원본을 그대로 복원한다(원본 파일을 건드리지 않기 위함).
run_test_only_preprocess() {
  local dir="$1"; local extra_args="$2"
  ( cd "$dir"
    cp preprocessing.py preprocessing.py.orig
    sed -i "s/for mode in \['train', 'test'\]:/for mode in ['test']:/" preprocessing.py
    trap 'mv preprocessing.py.orig preprocessing.py' EXIT
    eval "$PY preprocessing.py $extra_args"
  )
}

# =============================================================================
echo "=============================================================="
echo " A. F3Net 다중 분류 정확도"
echo "=============================================================="
if [ -f "fusion/scores/multi_f3net.csv" ]; then
  echo "  이미 완료 (fusion/scores/multi_f3net.csv 존재) — 건너뜀"
else
  if [ -d "multiple/F3Net Code/feature/test" ]; then
    echo "  전처리본 있음 — 건너뜀"
  elif [ -d "binary/F3Net Code/feature/test" ]; then
    echo "  binary 쪽 전처리본을 재사용합니다 (전처리 계산이 이진·다중 동일)"
  else
    echo "  전처리 시작 (AI 이미지 5만 장, 수십 분 소요될 수 있음)"
    run_test_only_preprocess "multiple/F3Net Code" "--data_root \"$DATA_ROOT\""
  fi
  ( cd "multiple/F3Net Code"
    SRC="./feature"
    [ -d "$SRC/test" ] || SRC="../../binary/F3Net Code/feature"
    $PY score_test_multi.py --dataset_root "$SRC" --weights_path best_model.pth \
        --out ../../fusion/scores/multi_f3net.csv | tee f3net_multiclass_result.txt
  )
fi
echo

# =============================================================================
echo "=============================================================="
echo " B-1 + C-1. DINOv3 (이진 + 다중 점수, 특징 1회 추출)"
echo "=============================================================="
DINO_FEAT="binary/DINOv3 Code/features/test_paths.pt"
if [ -f "$DINO_FEAT" ]; then
  echo "  특징 이미 추출됨 — 건너뜀"
else
  ( cd "binary/DINOv3 Code"
    $PY extract_features.py --data_dir "$DATA_ROOT/test" --output ./features/test_paths.pt \
        --crop 5crop --num_workers 8
  )
fi
if [ -f "fusion/scores/image_dinov3.csv" ]; then
  echo "  B-1 이진 점수 이미 있음 — 건너뜀"
else
  ( cd "binary/DINOv3 Code"
    $PY score_test.py --model heimdall_dinov3_mlp.pth --input ./features/test_paths.pt \
        --crop 5crop --out ../../fusion/scores/image_dinov3.csv
  )
fi
if [ -f "fusion/scores/multi_dinov3.csv" ]; then
  echo "  C-1 다중 점수 이미 있음 — 건너뜀"
else
  ( cd "multiple/DINOv3 Code"
    $PY score_test_multi.py --model dinov3_multiclass.pth \
        --input "../../binary/DINOv3 Code/features/test_paths.pt" \
        --out ../../fusion/scores/multi_dinov3.csv
  )
fi
echo

# =============================================================================
echo "=============================================================="
echo " B-2. F3Net 이진 점수"
echo "=============================================================="
if [ -f "fusion/scores/image_f3net.csv" ]; then
  echo "  이미 완료 — 건너뜀"
else
  if [ -d "binary/F3Net Code/feature/test" ]; then
    echo "  전처리본 있음 — 건너뜀"
  else
    echo "  전처리 시작 (실제+AI 10만 장, 수십 분 소요될 수 있음)"
    run_test_only_preprocess "binary/F3Net Code" ""
  fi
  ( cd "binary/F3Net Code"
    $PY score_test.py --dataset_root ./feature --weights_path best_model.pth \
        --out ../../fusion/scores/image_f3net.csv
  )
fi
echo

# =============================================================================
echo "=============================================================="
echo " B-3 + C-2. U-Net (이진 + 다중 점수, 특징 1회 추출) — 가장 오래 걸림"
echo "=============================================================="
UNET_FEAT="binary/U-Net in sdv1.5 + K-NN or MLP code/test_paths.pt"
if [ -f "$UNET_FEAT" ]; then
  echo "  특징 이미 추출됨 — 건너뜀"
else
  ( cd "binary/U-Net in sdv1.5 + K-NN or MLP code"
    $PY preprocess.py --data_dir "$DATA_ROOT/test" --output test_paths
  )
fi
if [ -f "fusion/scores/image_unet.csv" ]; then
  echo "  B-3 이진 점수 이미 있음 — 건너뜀"
else
  ( cd "binary/U-Net in sdv1.5 + K-NN or MLP code"
    $PY score_test.py --input test_paths.pt --model unet_mlp_model.pth --model_type mlp \
        --out ../../fusion/scores/image_unet.csv
  )
fi
if [ -f "fusion/scores/multi_unet.csv" ]; then
  echo "  C-2 다중 점수 이미 있음 — 건너뜀"
else
  ( cd "multiple/U-Net in sdv1.5 + K-NN or MLP code"
    $PY score_test_multi.py --model unet_mlp_model_multi.pth \
        --input "../../binary/U-Net in sdv1.5 + K-NN or MLP code/test_paths.pt" \
        --out ../../fusion/scores/multi_unet.csv
  )
fi
echo

# =============================================================================
echo "=============================================================="
echo " B-5. 이진 앙상블 5-fold 비교"
echo "=============================================================="
( cd fusion
  $PY fit_fusion.py --track image \
      --scores scores/image_dinov3.csv scores/image_f3net.csv scores/image_unet.csv \
      --names DINOv3 F3Net U-Net --n_folds 5 | tee image_fusion_kfold.txt
)
echo

# =============================================================================
echo "=============================================================="
echo " C-3. 다중 분류 앙상블 비교"
echo "=============================================================="
( cd fusion
  $PY fuse_multiclass.py \
      --scores scores/multi_dinov3.csv scores/multi_f3net.csv scores/multi_unet.csv \
      --names DINOv3 F3Net U-Net --out multi_fusion.json | tee multi_fusion_result.txt
)
echo

# =============================================================================
echo "=============================================================="
echo " 전부 완료. 아래 파일을 논문 작성자에게 전달하세요."
echo "=============================================================="
echo "  multiple/F3Net Code/f3net_multiclass_result.txt   -> 표 17 F3Net 값"
echo "  fusion/image_fusion_kfold.txt                     -> 6.5절 이진 앙상블"
echo "  fusion/multi_fusion_result.txt                    -> 6.4절 다중 분류 앙상블 (+ 표 17 앙상블 값)"
echo "  fusion/multi_fusion.json                          -> 위와 동일 내용의 구조화 데이터"
echo "  fusion/scores/*.csv (6개)                          -> 재계산·검증용 원본 점수"
