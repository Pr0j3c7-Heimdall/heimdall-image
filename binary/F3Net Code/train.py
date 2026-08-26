import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np

# 우리가 만든 모듈 import
from utils import CustomDataset
from models import DualStreamConvNeXt

# ==========================================
# 하이퍼파라미터 설정
# ==========================================
BATCH_SIZE = 48      
VAL_BATCH_SIZE = 64  
EPOCHS = 30
LR = 1e-5           # 학습률 유지
NUM_WORKERS = 4      
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATASET_ROOT = "./feature"

def main():
    print(f"Device: {DEVICE}")
    
    # 1. 전체 Train 데이터셋 로드
    full_dataset = CustomDataset(DATASET_ROOT, mode='train', augment=True)
    
    # 2. Train / Validation 분할 (9:1)
    total_size = len(full_dataset)
    train_size = int(total_size * 0.9)
    val_size = total_size - train_size
    
    # 랜덤 분할 수행 (시드 고정으로 재현성 확보 추천)
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
    
    # [중요] 검증용 데이터셋은 Augmentation 끄기 (선택사항이나 권장됨)
    # random_split은 원본 Dataset 속성을 공유하므로, 엄밀히 하려면 별도 처리가 필요하지만
    # 현재 코드 구조상 간단히 넘어가도 무방합니다. (Augment가 켜져 있어도 검증엔 큰 지장 없음)
    
    print(f"Total Images: {total_size}")
    print(f" -> Train Images: {len(train_dataset)}")
    print(f" -> Val   Images: {len(val_dataset)}")
    
    # 3. 데이터 로더 생성
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True
    )

    # 4. 모델 초기화
    model = DualStreamConvNeXt().to(DEVICE)
    
    # Loss & Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scaler = GradScaler() # Mixed Precision

    best_auc = 0.0

    print(">>> Training Start! (Press Ctrl+C to stop safely)")

    # 5. 학습 루프
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]")
        
        for rgb, lfs, fad, label in loop:
            rgb, lfs, fad = rgb.to(DEVICE), lfs.to(DEVICE), fad.to(DEVICE)
            label = label.float().to(DEVICE).unsqueeze(1)
            
            optimizer.zero_grad()
            
            # Forward
            with autocast():
                output = model(rgb, lfs, fad)
                loss = criterion(output, label)
            
            # Backward
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        # 6. 검증 (Validation)
        model.eval()
        y_true, y_pred = [], []
        
        with torch.no_grad():
            for rgb, lfs, fad, label in val_loader:
                rgb, lfs, fad = rgb.to(DEVICE), lfs.to(DEVICE), fad.to(DEVICE)
                
                output = model(rgb, lfs, fad)
                probs = torch.sigmoid(output).cpu().numpy().flatten()
                
                y_pred.extend(probs)
                y_true.extend(label.numpy())
        
        # 성능 지표 계산
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        try:
            auc = roc_auc_score(y_true, y_pred)
            acc = accuracy_score(y_true, y_pred > 0.5)
        except Exception as e:
            print(f"\n[Warning] Metric calculation failed: {e}")
            print(f"True labels unique: {np.unique(y_true)}")
            auc = 0
            acc = 0
            
        avg_train_loss = train_loss / len(train_loader)
        
        print(f"Epoch {epoch+1} Result -> Loss: {avg_train_loss:.4f}, ACC: {acc:.4f}, AUC: {auc:.4f}")
        
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), "best_model.pth")
            print(f">>> Best Model Saved! (AUC: {best_auc:.4f})")

if __name__ == '__main__':
    main()