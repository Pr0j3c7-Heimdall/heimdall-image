import torch
import argparse
import os
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from utils import CustomDataset
from models import DualStreamConvNeXt

def main():
    parser = argparse.ArgumentParser(description='DualStreamConvNeXt Multi-class Testing')
    parser.add_argument('--dataset_root', default='./feature', type=str)
    parser.add_argument('--weights_path', default='best_model.pth', type=str)
    parser.add_argument('--batch_size', default=64, type=int) 
    parser.add_argument('--gpu_ids', default='0', type=str)
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Loading Test Data from {args.dataset_root}/test ...")
    
    test_dataset = CustomDataset(dataset_root=args.dataset_root, mode='test', size=256, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Total Test Images: {len(test_dataset)}")

    print("Loading Model...")
    model = DualStreamConvNeXt(num_classes=10).to(DEVICE)
    
    if os.path.exists(args.weights_path):
        model.load_state_dict(torch.load(args.weights_path, map_location=DEVICE))
        print(">>> Weights loaded successfully.")
    else:
        print(f"[Error] Weights file not found: {args.weights_path}")
        return

    model.eval()
    y_true, y_pred = [], []
    class_stats = {}
    
    print("Starting Multi-class Evaluation...")
    with torch.no_grad():
        for (rgb, lfs, fad, label), pt_path in tqdm(zip(test_loader, test_dataset.image_paths), total=len(test_loader)):
            rgb, lfs, fad = rgb.to(DEVICE), lfs.to(DEVICE), fad.to(DEVICE)
            
            output = model(rgb, lfs, fad)
            preds = torch.argmax(output, dim=1).cpu().numpy()
            targets = label.numpy()
            
            y_pred.extend(preds)
            y_true.extend(targets)

    # 폴더 이름 추출 및 매핑
    for i, pt_path in enumerate(test_dataset.image_paths):
        norm_path = os.path.normpath(pt_path)
        parts = norm_path.split(os.sep)
        
        try:
            test_idx = parts.index('test')
            # 'feature/test/01_AI-T2I/10_BigGAN/...' 구조이므로 인덱스 +2가 AI 모델 폴더명
            folder_name = parts[test_idx + 2] 
        except ValueError:
            folder_name = "Unknown"
        
        true_label = int(y_true[i])
        pred_label = int(y_pred[i])

        if folder_name not in class_stats:
            class_stats[folder_name] = {'total': 0, 'correct': 0}
        
        class_stats[folder_name]['total'] += 1
        if pred_label == true_label:
            class_stats[folder_name]['correct'] += 1

    # 리포트 출력
    print("\n" + "="*80)
    print(f"{'Class Index':<12} | {'AI Model Name':<30} | {'Count':<6} | {'Acc (%)':<10}")
    print("-" * 80)
    
    total_correct = 0
    total_count = len(test_dataset)
    
    # 클래스 이름을 추출하기 위해 정렬된 폴더명 리스트 생성
    class_names = []
    
    for folder_name in sorted(class_stats.keys()):
        stats = class_stats[folder_name]
        if stats['total'] == 0: continue
        
        acc = (stats['correct'] / stats['total']) * 100
        print(f"{len(class_names):<12} | {folder_name:<30} | {stats['total']:<6} | {acc:8.2f}%")
        
        total_correct += stats['correct']
        
        # 라벨 처리를 위해 앞의 숫자 자르기 (예: '10_BigGAN' -> 'BigGAN')
        clean_name = folder_name.split('_', 1)[-1] if '_' in folder_name else folder_name
        class_names.append(clean_name)

    print("-" * 80)
    overall_acc = (total_correct / total_count) * 100 if total_count > 0 else 0
    print(f"OVERALL ACCURACY: {overall_acc:.2f}% (Total: {total_count})")
    print("="*80)

    # Confusion Matrix 저장
    cm = confusion_matrix(y_true, y_pred, labels=range(10))
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix - ConvNeXt_DualStream (Multi-class)')
    plt.xticks(rotation=45, ha='right')
    
    model_name = os.path.basename(args.weights_path).split('.')[0]
    save_filename = f"cm_{model_name}_multi.png"
    plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n[Info] 혼동행렬 이미지가 '{save_filename}' 이름으로 저장되었습니다!")

if __name__ == '__main__':
    main()