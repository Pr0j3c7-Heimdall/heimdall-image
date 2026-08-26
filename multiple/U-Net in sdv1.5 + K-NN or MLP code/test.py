import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from model import get_model

def test(input_pt, model_path, model_type='knn'):
    print(f"Loading test data from {input_pt}...")
    data = torch.load(input_pt)
    X_test, y_test, folders_test = data['X'], data['y'].numpy(), data['folders']
    
    # 동적 클래스 세팅
    unique_labels = np.unique(y_test)
    classes = data.get('classes', [str(i) for i in range(len(unique_labels))])
        
    print(f"Loading {model_type} model from {model_path}...")
    model = get_model(model_type, input_dim=X_test.shape[1], num_classes=len(classes))
    model.load(model_path)
    
    print(f"Running Multiclass inference...")
    
    probs = model.predict_proba(X_test)
    # 다중 분류: 가장 높은 확률을 가진 클래스를 예측값으로 사용
    preds = np.argmax(probs, axis=1)

    class_stats = {}
    for i in range(len(y_test)):
        folder = folders_test[i]
        pred = preds[i]
        true_label = int(y_test[i])
        
        if folder not in class_stats:
            class_stats[folder] = {'total': 0, 'correct': 0}
            
        class_stats[folder]['total'] += 1
        
        if pred == true_label:
            class_stats[folder]['correct'] += 1
            
    print("\n" + "="*80)
    print(f"{'Class Name':<30} | {'Count':<6} | {'Acc (%)':<10}")
    print("-" * 80)
    
    total_correct = 0
    total_count = len(y_test)
    
    for folder_name in sorted(class_stats.keys()):
        stats = class_stats[folder_name]
        if stats['total'] == 0: continue
        
        acc = (stats['correct'] / stats['total']) * 100
        print(f"{folder_name:<30} | {stats['total']:<6} | {acc:8.2f}%")
        total_correct += stats['correct']

    print("-" * 80)
    overall_acc = (total_correct / total_count) * 100 if total_count > 0 else 0
    print(f"OVERALL ACCURACY: {overall_acc:.2f}% (Total: {total_count})")
    print("="*80)
    
    # 클래스명 클렌징 (예: '10_BigGAN' -> 'BigGAN')
    clean_labels = [name.split('_', 1)[-1] if '_' in name else name for name in classes]

    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=clean_labels, 
                yticklabels=clean_labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix - Multiclass')
    plt.xticks(rotation=45, ha='right')
    
    model_name = os.path.basename(model_path).split('.')[0]
    save_filename = f"cm_{model_name}_multi.png"
    plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n[Info] 혼동행렬 이미지가 '{save_filename}' 이름으로 저장되었습니다!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='test.pt', help='평가용 .pt 파일 경로')
    parser.add_argument('--model', type=str, default='unet_model.pth')
    parser.add_argument('--model_type', type=str, default='knn', choices=['knn', 'mlp', 'mlp-320', 'mlp-640'])
    args = parser.parse_args()
    
    test(args.input, args.model, args.model_type)