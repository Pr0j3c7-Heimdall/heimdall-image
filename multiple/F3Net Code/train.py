import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import accuracy_score
import numpy as np

from utils import CustomDataset
from models import DualStreamConvNeXt

BATCH_SIZE = 32
VAL_BATCH_SIZE = 64  
EPOCHS = 30
LR = 1e-5           
NUM_WORKERS = 12
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATASET_ROOT = "./feature"

def main():
    print(f"Device: {DEVICE}")
    
    full_dataset = CustomDataset(DATASET_ROOT, mode='train', augment=True)
    
    total_size = len(full_dataset)
    train_size = int(total_size * 0.9)
    val_size = total_size - train_size
    
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
    
    print(f"Total Images: {total_size}")
    print(f" -> Train Images: {len(train_dataset)}")
    print(f" -> Val   Images: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)

    # 10개 클래스로 초기화
    model = DualStreamConvNeXt(num_classes=10).to(DEVICE)
    
    # [수정] 다중 분류 Loss로 변경
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scaler = GradScaler() 

    best_acc = 0.0

    print(">>> Training Start! (Press Ctrl+C to stop safely)")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]")
        
        for rgb, lfs, fad, label in loop:
            rgb, lfs, fad = rgb.to(DEVICE), lfs.to(DEVICE), fad.to(DEVICE)
            # [수정] CrossEntropy는 Long 타입의 1D 텐서를 받습니다.
            label = label.long().to(DEVICE)
            
            optimizer.zero_grad()
            
            with autocast():
                output = model(rgb, lfs, fad)
                loss = criterion(output, label)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        model.eval()
        y_true, y_pred = [], []
        
        with torch.no_grad():
            for rgb, lfs, fad, label in val_loader:
                rgb, lfs, fad = rgb.to(DEVICE), lfs.to(DEVICE), fad.to(DEVICE)
                
                output = model(rgb, lfs, fad)
                # [수정] 확률 대신 argmax로 예측 클래스 도출
                preds = torch.argmax(output, dim=1).cpu().numpy()
                
                y_pred.extend(preds)
                y_true.extend(label.numpy())
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # [수정] 다중 분류 메트릭
        acc = accuracy_score(y_true, y_pred)
        avg_train_loss = train_loss / len(train_loader)
        
        print(f"Epoch {epoch+1} Result -> Loss: {avg_train_loss:.4f}, ACC: {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "best_model.pth")
            print(f">>> Best Model Saved! (ACC: {best_acc:.4f})")

if __name__ == '__main__':
    main()