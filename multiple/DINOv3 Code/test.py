import os
import argparse
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from mlp_model import DINOMlpClassifier
import utils
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

CONFIG = {
    'input_dim': 1024,
    'num_classes': 10, # 다중 분류 10개
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

def main():
    parser = argparse.ArgumentParser(description='Heimdall Detailed Tester (Multi-class)')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--crop', type=str, default='5crop', choices=['center', '5crop'])
    parser.add_argument('--batch_size', type=int, default=128)
    # 다중 분류이므로 threshold 인자 제거
    args = parser.parse_args()

    # 데이터 로드
    data = torch.load(args.input)
    X, y, classes = data['X'], data['y'], data['classes']
    
    model_path = os.path.join('models', args.model)
    model = DINOMlpClassifier(CONFIG['input_dim'], CONFIG['num_classes']).to(CONFIG['device'])
    model.load_state_dict(torch.load(model_path))
    model.eval()

    loader = DataLoader(TensorDataset(X, y), batch_size=args.batch_size, shuffle=False)

    class_stats = {i: {'total': 0, 'correct': 0} for i in range(len(classes))}

    print(f"\n[Testing] Model: {args.model} | Crop: {args.crop}")

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Evaluating"):
            inputs = inputs.to(CONFIG['device'])
            
            if args.crop == '5crop' and inputs.dim() == 3:
                b, n, c = inputs.shape
                outputs = model(inputs.view(b * n, c))
                outputs = outputs.view(b, n, -1).mean(dim=1)
            else:
                outputs = model(inputs)

            # argmax로 가장 확률이 높은 클래스 인덱스 도출
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            targets_np = targets.cpu().numpy()
            
            for i in range(targets.size(0)):
                target_idx = targets_np[i]
                pred_idx = preds[i]
                
                class_stats[target_idx]['total'] += 1
                
                if pred_idx == target_idx:
                    class_stats[target_idx]['correct'] += 1
                
                all_targets.append(target_idx)
                all_preds.append(pred_idx)

    # 결과 출력
    print("\n" + "="*80)
    print(f"{'Class Index':<12} | {'Class Name':<30} | {'Count':<6} | {'Acc (%)':<10}")
    print("-" * 80)
    
    total_correct, total_count = 0, 0
    
    for i, cls_name in enumerate(classes):
        stats = class_stats[i]
        if stats['total'] == 0: continue
        
        acc = (stats['correct'] / stats['total']) * 100
        
        print(f"{i:<12} | {cls_name:<30} | {stats['total']:<6} | {acc:8.2f}%")
        
        total_correct += stats['correct']
        total_count += stats['total']

    print("-" * 80)
    print(f"OVERALL ACCURACY: {(total_correct / total_count * 100):.2f}% (Total: {total_count})")
    print("="*80)

    # --- 혼동행렬(Confusion Matrix) 생성 및 저장 ---
    cm = confusion_matrix(all_targets, all_preds, labels=range(len(classes)))
    
    plt.figure(figsize=(10, 8))
    
    # 클래스 이름에서 앞의 숫자(예: '10_BigGAN' -> 'BigGAN')를 잘라내서 라벨을 깔끔하게 표시
    clean_labels = [name.split('_', 1)[-1] if '_' in name else name for name in classes]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=clean_labels, 
                yticklabels=clean_labels)
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Multi-class Confusion Matrix - {args.model}')
    plt.xticks(rotation=45, ha='right')
    
    save_filename = f"cm_{args.model.split('.')[0]}_multi.png"
    plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n[Info] 혼동행렬 이미지가 '{save_filename}' 이름으로 저장되었습니다!")

if __name__ == '__main__':
    main()