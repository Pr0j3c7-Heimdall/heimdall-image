import torch
import argparse
import os
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# 우리가 만든 모듈 import
from utils import CustomDataset
from models import DualStreamConvNeXt

def main():
    parser = argparse.ArgumentParser(description='DualStreamConvNeXt Testing')
    parser.add_argument('--dataset_root', default='./feature', type=str, help='전처리된 데이터 폴더 경로')
    parser.add_argument('--weights_path', default='best_model.pth', type=str, help='학습된 모델 파일 경로')
    parser.add_argument('--batch_size', default=64, type=int) 
    parser.add_argument('--gpu_ids', default='0', type=str)
    parser.add_argument('--threshold', type=float, default=0.5, help='AI 판별 임계값') # 임계값 설정 추가
    
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Loading Test Data from {args.dataset_root}/test ...")
    
    try:
        # test 모드이므로 augment=False
        test_dataset = CustomDataset(
            dataset_root=args.dataset_root, 
            mode='test', 
            size=256, 
            augment=False
        )
    except Exception as e:
        print(f"데이터 로드 실패: {e}")
        return
    
    # shuffle=False 필수 (경로 매핑을 위해 순서 유지)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, 
        num_workers=4, pin_memory=True
    )
    
    print(f"Total Test Images: {len(test_dataset)}")

    print("Loading Model...")
    model = DualStreamConvNeXt(num_classes=1).to(DEVICE)
    
    if os.path.exists(args.weights_path):
        model.load_state_dict(torch.load(args.weights_path, map_location=DEVICE))
        print(">>> Weights loaded successfully.")
    else:
        print(f"[Error] Weights file not found: {args.weights_path}")
        return

    model.eval()
    y_true, y_pred_probs = [], []
    
    print(f"Starting Evaluation (Threshold: {args.threshold})...")
    with torch.no_grad():
        for rgb, lfs, fad, label in tqdm(test_loader):
            rgb, lfs, fad = rgb.to(DEVICE), lfs.to(DEVICE), fad.to(DEVICE)
            
            output = model(rgb, lfs, fad)
            # 확률값 도출
            probs = torch.sigmoid(output).cpu().numpy().flatten() 
            
            y_pred_probs.extend(probs)
            y_true.extend(label.numpy())

    # ==========================================
    # 폴더별 결과 매핑 및 분석
    # ==========================================
    class_stats = {}
    y_preds_binary = [] # Threshold 적용된 최종 예측값 (CM용)

    # 순서가 보장되므로 zip처럼 순회 가능
    for i, pt_path in enumerate(test_dataset.image_paths):
        # OS 상관없이 경로 분리 (예: feature/test/00_BDD/img.pt -> 00_BDD 추출)
        norm_path = os.path.normpath(pt_path)
        parts = norm_path.split(os.sep)
        
        try:
            test_idx = parts.index('test')
            folder_name = parts[test_idx + 1] # 'test' 폴더 바로 다음 디렉토리명
        except ValueError:
            folder_name = "Unknown"
        
        prob = y_pred_probs[i]
        true_label = int(y_true[i])
        
        # 임계값(Threshold)에 따른 예측
        pred_label = 1 if prob >= args.threshold else 0
        y_preds_binary.append(pred_label)

        if folder_name not in class_stats:
            class_stats[folder_name] = {'total': 0, 'correct': 0, 'sum_fake_prob': 0.0}
        
        class_stats[folder_name]['total'] += 1
        class_stats[folder_name]['sum_fake_prob'] += prob
        
        if pred_label == true_label:
            class_stats[folder_name]['correct'] += 1

    # ==========================================
    # 리포트 출력
    # ==========================================
    print("\n" + "="*105)
    print(f"{'Group':<10} | {'Subfolder Name':<30} | {'Count':<6} | {'Acc (%)':<10} | {'Avg AI Prob (%)':<15}")
    print("-" * 105)
    
    total_correct = 0
    total_count = len(test_dataset)
    
    # 폴더명 순으로 정렬하여 깔끔하게 출력
    for folder_name in sorted(class_stats.keys()):
        stats = class_stats[folder_name]
        if stats['total'] == 0: continue
        
        acc = (stats['correct'] / stats['total']) * 100
        avg_ai_prob = (stats['sum_fake_prob'] / stats['total']) * 100
        
        # 폴더명 앞 숫자로 Real/AI 그룹 구분 (0~9: REAL, 10~: AI)
        try:
            prefix_num = int(folder_name.split('_')[0])
            group = "REAL" if prefix_num < 10 else "AI-T2I"
        except:
            group = "UNKNOWN"
        
        print(f"{group:<10} | {folder_name:<30} | {stats['total']:<6} | {acc:8.2f} | {avg_ai_prob:13.2f}%")
        
        total_correct += stats['correct']

    print("-" * 105)
    overall_acc = (total_correct / total_count) * 100 if total_count > 0 else 0
    
    try:
        overall_auc = roc_auc_score(y_true, y_pred_probs)
    except:
        overall_auc = 0.0
        
    print(f"OVERALL ACCURACY: {overall_acc:.2f}% (Total: {total_count})")
    print(f"OVERALL AUC     : {overall_auc:.4f}")
    print("="*105)

    # ==========================================
    # Confusion Matrix 저장
    # ==========================================
    cm = confusion_matrix(y_true, y_preds_binary)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real(0)', 'AI(1)'], 
                yticklabels=['Real(0)', 'AI(1)'])
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - ConvNeXt_DualStream (Thr:{args.threshold})')
    
    # 가중치 파일명을 따서 저장 (예: best_model -> cm_best_model.png)
    model_name = os.path.basename(args.weights_path).split('.')[0]
    save_filename = f"cm_{model_name}.png"
    plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n[Info] 혼동행렬 이미지가 '{save_filename}' 이름으로 저장되었습니다!")

if __name__ == '__main__':
    main()