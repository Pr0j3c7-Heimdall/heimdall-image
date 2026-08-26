# 학습 데이터셋 추출
./.venv/Scripts/python.exe extract_features.py --data_dir "D:/Heimdall_Image_Dataset/train/01_AI-T2I" --output features/train_ai_multi.pt

# 테스트 데이터셋 추출
./.venv/Scripts/python.exe extract_features.py --data_dir "D:/Heimdall_Image_Dataset/test/AI-T2I" --output features/test_ai_multi.pt

./.venv/Scripts/python.exe train.py --input "features/train_ai_multi.pt" --output "dinov3_multiclass.pth" --epochs 1000 --crop random

./.venv/Scripts/python.exe test.py --model "dinov3_multiclass.pth" --input "features/test_ai_multi.pt" --crop 5crop