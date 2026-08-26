import os
import argparse
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from model import get_model

def test(input_pt, model_path, model_type='knn', threshold=0.5):
    print(f"Loading test data from {input_pt}...")
    data = torch.load(input_pt)
    X_test, y_test, folders_test = data['X'], data['y'].numpy(), data['folders']
        
    print(f"Loading {model_type} model from {model_path}...")
    model = get_model(model_type, input_dim=X_test.shape[1])
    model.load(model_path)
    
    print(f"Running inference (Threshold: {threshold})...")
    
    probs = model.predict_proba(X_test)
    preds = (probs >= threshold).astype(int)

    class_stats = {}
    for i in range(len(y_test)):
        folder = folders_test[i]
        prob = probs[i]
        pred = preds[i]
        true_label = int(y_test[i])
        
        if folder not in class_stats:
            class_stats[folder] = {'total': 0, 'correct': 0, 'sum_fake_prob': 0.0}
            
        class_stats[folder]['total'] += 1
        class_stats[folder]['sum_fake_prob'] += prob
        
        if pred == true_label:
            class_stats[folder]['correct'] += 1
            
    print("\n" + "="*105)
    print(f"{'Group':<10} | {'Subfolder Name':<30} | {'Count':<6} | {'Acc (%)':<10} | {'Avg AI Prob (%)':<15}")
    print("-" * 105)
    
    total_correct = 0
    total_count = len(y_test)
    
    for folder_name in sorted(class_stats.keys()):
        stats = class_stats[folder_name]
        if stats['total'] == 0: continue
        
        acc = (stats['correct'] / stats['total']) * 100
        avg_ai_prob = (stats['sum_fake_prob'] / stats['total']) * 100
        
        try:
            prefix_num = int(folder_name.split('_')[0])
            group = "REAL" if prefix_num < 10 else "AI-T2I"
        except:
            group = "UNKNOWN"
            
        print(f"{group:<10} | {folder_name:<30} | {stats['total']:<6} | {acc:8.2f} | {avg_ai_prob:13.2f}%")
        total_correct += stats['correct']

    print("-" * 105)
    overall_acc = (total_correct / total_count) * 100 if total_count > 0 else 0
    print(f"OVERALL ACCURACY: {overall_acc:.2f}% (Total: {total_count})")
    print("="*105)
    
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real(0)', 'AI(1)'], 
                yticklabels=['Real(0)', 'AI(1)'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix (Thr:{threshold})')
    
    model_name = os.path.basename(model_path).split('.')[0]
    save_filename = f"cm_{model_name}.png"
    plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n[Info] 혼동행렬 이미지가 '{save_filename}' 이름으로 저장되었습니다!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='test.pt', help='평가용 .pt 파일 경로')
    parser.add_argument('--model', type=str, default='unet_model.pth')
    parser.add_argument('--model_type', type=str, default='knn', choices=['knn', 'mlp', 'mlp-320', 'mlp-640'])
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()
    
    test(args.input, args.model, args.model_type, args.threshold)