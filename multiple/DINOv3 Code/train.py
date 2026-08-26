import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import utils
from mlp_model import DINOMlpClassifier

# Configuration
CONFIG = {
    'input_dim': 1024, 
    'num_classes': 10,  # 다중 분류를 위해 10으로 변경
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'backbone': 'dinov3_vitl16' 
}

def main():
    parser = argparse.ArgumentParser(description='DINOv3 MLP Multi-class Trainer')
    parser.add_argument('--res', type=int, default=224)
    parser.add_argument('--crop', type=str, default='random', choices=['center', 'random'])
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--output', type=str, default=None, help='Output model filename (e.g. my_model.pth)')
    parser.add_argument('--input', type=str, required=True, help='Path to input feature file (.pt)')
    args = parser.parse_args()

    print(f"Start Multi-class Training | Res: {args.res} | Crop: {args.crop}")
    
    feature_file = args.input
    X_full, y_full = utils.load_features_from_file(feature_file)
    
    if X_full is None: 
        return

    # [수정] 이진 분류 변환 로직 제거, 추출된 라벨(0~9) 그대로 사용
    y_multi = y_full.long()

    # 데이터 불균형 확인 및 가중치 계산
    unique_classes = torch.unique(y_multi).cpu().numpy()
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_multi.cpu().numpy())
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(CONFIG['device'])
    
    print(f"Data Distribution: {len(unique_classes)} Classes detected")
    print(f"Class Weights: {class_weights.cpu().numpy()}")

    # Split Train/Validation (9:1)
    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_multi, test_size=0.1, random_state=42, stratify=y_multi
    )
    print(f"Train: {len(y_train)} | Valid: {len(y_val)}")

    # DataLoader
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val, y_val), batch_size=args.batch_size, shuffle=False)

    # Initialize Model
    model = DINOMlpClassifier(CONFIG['input_dim'], CONFIG['num_classes']).to(CONFIG['device'])
    
    # 모델 이름 결정
    if args.output:
        model_name = args.output
        if model_name.endswith('.pth'):
            model_name = model_name[:-4]
    else:
        model_name = utils.get_model_name(model, CONFIG['backbone'], args.res, args.crop) + "_multiclass"
    
    print(f"Model will be saved as: models/{model_name}.pth")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    best_val_acc = 0

    # Training Loop
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        
        for X, y in train_loader:
            X, y = X.to(CONFIG['device']), y.to(CONFIG['device'])
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        val_acc = utils.evaluate_model(model, val_loader, CONFIG['device'], args.crop)
        
        # Save Best Model
        is_best = False
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            is_best = True
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), f"models/{model_name}.pth")
        
        if is_best or epoch % 10 == 0:
            mark = "*" if is_best else ""
            avg_loss = train_loss / len(train_loader)
            print(f"Ep {epoch:03d} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2f}% {mark}")

    print(f"Training Finished. Best Acc: {best_val_acc:.2f}%")
    print(f"Saved to: models/{model_name}.pth")

if __name__ == '__main__':
    main()