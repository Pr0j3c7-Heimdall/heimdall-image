#!/bin/bash

# 1. 경로 설정 (사용자 환경에 맞게 수정)
DATA_ROOT="D:/Heimdall_Image_Dataset/train"
FEATURE_DIR="./features"
MODEL_DIR="./models"
TRAIN_FEAT="$FEATURE_DIR/train_v3.pt"
TEST_FEAT="$FEATURE_DIR/test_v3.pt"
MODEL_NAME="heimdall_dinov3_mlp"

# 폴더 생성
mkdir -p $FEATURE_DIR
mkdir -p $MODEL_DIR

echo "=========================================================="
echo " [Step 1/4] Extracting Features for Training Set"
echo "=========================================================="
# ./.venv/Scripts/python.exe extract_features.py --data_dir "D:\Heimdall_Image_Dataset\train" --output "./features/train_v3.pt" --crop random --batch_size 128

echo "=========================================================="
echo " [Step 2/4] Extracting Features for Test Set"
echo "=========================================================="
# ./.venv/Scripts/python.exe extract_features.py --data_dir "D:\Heimdall_Image_Dataset\test" --output "./features/tests_v3_18.pt" --crop 5crop --batch_size 128

echo "=========================================================="
echo " [Step 3/4] Training MLP Model (Binary Classification)"
echo "=========================================================="
# epochs는 테스트를 위해 짧게 설정하거나 필요에 따라 조절하세요.
# ./.venv/Scripts/python.exe train.py --input "$TRAIN_FEAT" --output "$MODEL_NAME" --epochs 1000 --batch_size 256 --crop random

echo "=========================================================="
echo " [Step 4/4] Final Evaluation with Probability & Threshold"
echo "=========================================================="
# 임계값 0.5로 먼저 테스트
./.venv/Scripts/python.exe test.py --input "./features/tests_v3_18.pt" --model "heimdall_dinov3_mlp.pth" --threshold 0.95 --crop 5crop

echo "=========================================================="
echo " Pipeline Finished! Check 'models/' and 'cm_result.png'"
echo "=========================================================="