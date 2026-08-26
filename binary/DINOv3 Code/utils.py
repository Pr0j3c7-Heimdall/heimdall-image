import torch
import os
from mlp_model import DINOMlpClassifier

def get_model_name(model, backbone, res, crop):
    prefix = f"{backbone}_{res}_{crop}"
    if isinstance(model, DINOMlpClassifier):
        return f"{prefix}_mlp"
    return f"{prefix}_unknown"

def load_features_from_file(file_path):
    if not os.path.exists(file_path):
        print(f"Error: Feature file not found ({file_path})")
        return None, None
        
    print(f"Loading features: {file_path}")
    data = torch.load(file_path)
    return data['X'], data['y']

def evaluate_model(model, dataloader, device, crop_type):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            
            # 5-crop handling
            if crop_type == '5crop' and X.dim() == 3:
                b, n, c = X.shape
                out = model(X.view(b*n, c))
                out = out.view(b, n, -1).mean(dim=1)
            else:
                out = model(X)
            
            # Binary prediction (assuming model output is 2 classes)
            _, predicted = torch.max(out.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            
    return 100 * correct / total

# 라벨 병합 로직
def merge_binary_experiment_classes(y):
    """
    00_real (0) ~ 11_sdxl (11) 총 12개 클래스를 10개로 병합
    
    [변경 규칙]
    0~6: 유지 (Real + 기타 Fake)
    7(MJv5), 8(MJv6) -> 7 (Midjourney Unified)
    9(Nano), 10(NanoPro) -> 8 (NanoBanana Unified)
    11(SDXL) -> 9 (Index Shift)
    """
    print("[Info] Merging classes enabled (12 -> 10 classes)...")
    new_y = y.clone()
    
    # 1. SDXL (11 -> 9)
    new_y[y == 11] = 9
    
    # 2. NanoBanana (9, 10 -> 8)
    new_y[(y == 9) | (y == 10)] = 8
    
    # 3. Midjourney (7, 8 -> 7)
    new_y[(y == 7) | (y == 8)] = 7
    
    return new_y

# 병합된 클래스 이름
def get_merged_binary_class_names():
    return [
        "00_real",
        "01_adm",
        "02_biggan",
        "03_dalle3",
        "04_flux",
        "05_gpt-image-1",
        "06_imagen4",
        "07_midjourney_unified",
        "08_nano-banana_unified",
        "09_stable-diffusion-xl"
    ]