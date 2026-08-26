import argparse
import torch
from model import get_model

def train(input_pt, model_type='knn', output_model='model.pth'):
    print(f"Loading features from {input_pt}...")
    data = torch.load(input_pt)
    X, y = data['X'], data['y']
    
    input_dim = X.shape[1]
    print(f"Feature Dimension: {input_dim} | Dataset Size: {len(X)} samples")
    
    print(f"Initializing and Training {model_type}...")
    model = get_model(model_type, input_dim=input_dim)
    
    model.fit(X, y)

    model.save(output_model)
    print(f"Model successfully saved to {output_model}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # [수정] 파일 2개 대신 단일 .pt 파일 경로를 받습니다.
    parser.add_argument('--input', type=str, default='train.pt', help='학습용 .pt 파일 경로')
    parser.add_argument('--model', type=str, default='knn', choices=['knn', 'mlp', 'mlp-320', 'mlp-640'])
    parser.add_argument('--output', type=str, default='unet_model.pth')
    args = parser.parse_args()
    
    train(args.input, args.model, args.output)