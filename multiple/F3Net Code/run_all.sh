#!/bin/bash
set -e

# [설정] 원본 이미지 데이터셋의 최상위 루트 경로를 적어주세요.
# (train과 test 폴더가 들어있는 그 상위 폴더입니다)
DATA_ROOT="D:/Heimdall_Image_Dataset"
SAVE_ROOT="./feature"

echo "=================================================="
echo " [1/3] 주파수 특징(.pt) 전처리 시작 (train/test 01_AI-T2I 타겟)"
echo "=================================================="
# ./.venv/Scripts/python.exe preprocessing.py --data_root "$DATA_ROOT" --save_root "$SAVE_ROOT"

echo -e "\n=================================================="
echo " [2/3] Dual-Stream ConvNeXt 다중 분류 학습 시작"
echo "=================================================="
./.venv/Scripts/python.exe train.py

echo -e "\n=================================================="
echo " [3/3] 다중 분류 모델 평가 및 혼동행렬 시각화"
echo "=================================================="
./.venv/Scripts/python.exe test.py

echo -e "\n=================================================="
echo " 🎉 모든 파이프라인 실행이 완료되었습니다!"
echo "=================================================="