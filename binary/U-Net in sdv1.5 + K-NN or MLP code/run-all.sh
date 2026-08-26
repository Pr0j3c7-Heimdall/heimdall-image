#!/bin/bash

# 에러 발생 시 즉시 스크립트 실행 중단
set -e

# ==========================================
# ⚙️ 환경 설정 (경로 및 하이퍼파라미터)
# ==========================================
# 데이터셋 최상위 폴더 경로 (본인 환경에 맞게 수정하세요)
DATA_ROOT="D:/Heimdall_Image_Dataset"

# 사용할 모델 타입 ('knn' 또는 'mlp')
MODEL_TYPE="knn"

# 모델 저장 이름
MODEL_SAVE_NAME="unet_${MODEL_TYPE}_model.pth"

# ==========================================

echo "=========================================="
echo "🚀 U-Net 파이프라인 자동 실행을 시작합니다!"
echo "=========================================="

echo -e "\n[Step 1] 학습 데이터(Train) 전처리 및 .pt 파일 생성 중..."
# Train 폴더 경로를 읽어 train.pt 생성
# python preprocess.py --data_dir "$DATA_ROOT/train" --output "train" --batch_size 24 --num_workers 8

echo -e "\n[Step 2] 평가 데이터(Test) 전처리 및 .pt 파일 생성 중..."
# Test 폴더 경로를 읽어 test.pt 생성
# python preprocess.py --data_dir "$DATA_ROOT/test" --output "test" --batch_size 24 --num_workers 8

echo -e "\n[Step 3] $MODEL_TYPE 모델 학습 시작..."
# train.pt를 이용해 모델 학습 후 .pth 파일로 저장
python train.py --input "train.pt" --model "$MODEL_TYPE" --output "$MODEL_SAVE_NAME"

echo -e "\n[Step 4] $MODEL_TYPE 모델 평가(테스트) 시작..."
# test.pt와 학습된 모델을 이용해 18개 폴더별 리포트 및 혼동 행렬 출력
python test.py --input "test.pt" --model "$MODEL_SAVE_NAME" --model_type "$MODEL_TYPE" --threshold 0.5

echo -e "\n=========================================="
echo "🎉 모든 파이프라인 실행이 성공적으로 완료되었습니다!"
echo "=========================================="