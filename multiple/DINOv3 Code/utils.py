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