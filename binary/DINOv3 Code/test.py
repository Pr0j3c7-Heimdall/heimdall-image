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
    'num_classes': 2, 
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

def main():
    parser = argparse.ArgumentParser(description='Heimdall Detailed Tester')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--crop', type=str, default='5crop', choices=['center', '5crop'])
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()

    # 데이터 로드
    data = torch.load(args.input)
    X, y, classes = data['X'], data['y'], data['classes']
    
    model_path = os.path.join('models', args.model)
    model = DINOMlpClassifier(CONFIG['input_dim'], CONFIG['num_classes']).to(CONFIG['device'])
    model.load_state_dict(torch.load(model_path))
    model.eval()

    loader = DataLoader(TensorDataset(X, y), batch_size=args.batch_size, shuffle=False)

    class_stats = {i: {'total': 0, 'correct': 0, 'sum_fake_prob': 0.0} for i in range(len(classes))}

    print(f"\n[Testing] Model: {args.model} | Crop: {args.crop} | Threshold: {args.threshold}")

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Evaluating"):
            inputs = inputs.to(CONFIG['device'])
            
            # Inference (5crop 차원 변환 처리)
            if args.crop == '5crop' and inputs.dim() == 3:
                b, n, c = inputs.shape
                outputs = model(inputs.view(b * n, c))
                outputs = outputs.view(b, n, -1).mean(dim=1)
            else:
                outputs = model(inputs)

            # AI일 확률(fake_probs) 및 예측값 도출
            probs = torch.softmax(outputs, dim=1)
            fake_probs = probs[:, 1].cpu().numpy()
            preds = (fake_probs >= args.threshold).astype(int)
            
            for i in range(targets.size(0)):
                idx = targets[i].item()
                folder_name = classes[idx]
                
                # 핵심 수정: 폴더 이름 앞의 숫자를 추출하여 Real/AI 구분
                # 예: '00_BDD' -> 0 (Real), '10_BigGAN' -> 10 (AI)
                prefix_num = int(folder_name.split('_')[0])
                target_binary = 0 if prefix_num < 10 else 1
                
                class_stats[idx]['total'] += 1
                class_stats[idx]['sum_fake_prob'] += fake_probs[i]
                
                # 예측값과 실제 정답이 같으면 correct 증가
                if preds[i] == target_binary:
                    class_stats[idx]['correct'] += 1
                
                all_targets.append(target_binary)
                all_preds.append(preds[i])

    # 결과 출력
    print("\n" + "="*105)
    print(f"{'Group':<10} | {'Subfolder Name':<30} | {'Count':<6} | {'Acc (%)':<10} | {'Avg AI Prob (%)':<15}")
    print("-" * 105)
    
    total_correct, total_count = 0, 0
    
    for i, cls_name in enumerate(classes):
        stats = class_stats[i]
        if stats['total'] == 0: continue
        
        # 각 폴더별 정확도 및 평균 AI 확률 계산
        acc = (stats['correct'] / stats['total']) * 100
        avg_ai_prob = (stats['sum_fake_prob'] / stats['total']) * 100
        
        # 출력 그룹 명시 (00~07은 REAL, 10~19는 AI-T2I)
        prefix_num = int(cls_name.split('_')[0])
        group = "REAL" if prefix_num < 10 else "AI-T2I"
        
        print(f"{group:<10} | {cls_name:<30} | {stats['total']:<6} | {acc:8.2f} | {avg_ai_prob:13.2f}%")
        
        total_correct += stats['correct']
        total_count += stats['total']

    print("-" * 105)
    print(f"OVERALL ACCURACY: {(total_correct / total_count * 100):.2f}% (Total: {total_count})")
    print("="*105)

    # --- [추가] 혼동행렬(Confusion Matrix) 생성 및 저장 ---
    cm = confusion_matrix(all_targets, all_preds)
    
    plt.figure(figsize=(6, 5))
    # annot=True로 숫자 표시, fmt='d'로 정수 포맷, 파란색 테마(Blues) 사용
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real(0)', 'AI(1)'], 
                yticklabels=['Real(0)', 'AI(1)'])
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - {args.model}')
    
    # 모델 이름을 활용하여 이미지 파일 이름 설정 (예: model.pth -> cm_model.png)
    save_filename = f"cm_{args.model.split('.')[0]}.png"
    plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n[Info] 혼동행렬 이미지가 '{save_filename}' 이름으로 저장되었습니다!")

if __name__ == '__main__':
    main()