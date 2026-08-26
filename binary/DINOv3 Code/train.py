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
    'input_dim': 1024, # ViT-Large 모델이므로 1024 유지
    'num_classes': 2,  # Binary classification (Real vs Fake)
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'backbone': 'dinov3_vitl16' # [수정] dinov2 -> dinov3로 이름 변경
}

def main():
    parser = argparse.ArgumentParser(description='DINOv2 MLP Binary Trainer')
    parser.add_argument('--res', type=int, default=224)
    parser.add_argument('--crop', type=str, default='random', choices=['center', 'random'])
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--merge', action='store_true', help='Merge specific Fake classes before binary conversion')
    parser.add_argument('--output', type=str, default=None, help='Output model filename (e.g. my_model.pth)')
    
    # [추가] 입력 파일 경로를 직접 지정하는 옵션 (필수값으로 변경)
    parser.add_argument('--input', type=str, required=True, help='Path to input feature file (.pt)')
    
    args = parser.parse_args()

    print(f"Start Training | Res: {args.res} | Crop: {args.crop}")
    
    # [수정] 입력받은 파일 경로 사용
    feature_file = args.input
    X_full, y_full = utils.load_features_from_file(feature_file)
    
    if X_full is None: 
        return

    # [단계 1] 라벨 병합 (옵션)
    if args.merge:
        y_full = utils.merge_binary_experiment_classes(y_full)

    # [단계 2] 이진 분류용 라벨 변환 (Real=0 -> 0, Fake=1~11 -> 1)
    print("Converting labels to Binary (Real vs Fake)...")
    y_binary = (y_full > 0).long()

    # 데이터 불균형 확인 및 가중치 계산
    unique_classes = torch.unique(y_binary).cpu().numpy()
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_binary.cpu().numpy())
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(CONFIG['device'])
    
    print(f"Data Distribution: Real(0) vs Fake(1)")
    print(f"Class Weights: {class_weights.cpu().numpy()}")

    # Split Train/Validation (9:1)
    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_binary, test_size=0.1, random_state=42, stratify=y_binary
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
        model_name = utils.get_model_name(model, CONFIG['backbone'], args.res, args.crop) + "_binary"
    
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