import os
import argparse
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel

# Argument Parsing
parser = argparse.ArgumentParser(description='DINOv3 Feature Extractor (Hugging Face)')
parser.add_argument('--data_dir', type=str, required=True, help='Dataset directory path')
parser.add_argument('--output', type=str, required=True, help='Output .pt file path')
parser.add_argument('--res', type=int, default=224, help='Input resolution')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--crop', type=str, default='center', choices=['center', 'random', '5crop'])
args = parser.parse_args()

def get_transform(crop_type, res):
    norm = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    base_transform = [T.Resize(256, interpolation=3)]

    if crop_type == 'center':
        base_transform.append(T.CenterCrop(res))
        base_transform.append(T.ToTensor())
        base_transform.append(norm)
        return T.Compose(base_transform)
        
    elif crop_type == 'random':
        base_transform.append(T.RandomCrop(res))
        base_transform.append(T.ToTensor())
        base_transform.append(norm)
        return T.Compose(base_transform)
        
    elif crop_type == '5crop':
        return T.Compose([
            T.Resize(256, interpolation=3),
            T.FiveCrop(res),
            T.Lambda(lambda crops: torch.stack([norm(T.ToTensor()(crop)) for crop in crops]))
        ])

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # [수정] Hugging Face 모델 로드
    model_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    print(f"Loading DINOv3 model: {model_name}")
    
    try:
        model = AutoModel.from_pretrained(model_name).to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please install transformers: pip install transformers")
        return

    model.eval()

    transform = get_transform(args.crop, args.res)

    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory not found ({args.data_dir})")
        return

    classes = sorted([d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))])
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    
    print("Class Mapping:")
    for cls_name, idx in class_to_idx.items():
        print(f"  [{idx}] {cls_name}")

    image_paths = []
    labels = []
    
    # 지원하는 이미지 확장자
    valid_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.tif')

    for cls_name in classes:
        cls_dir = os.path.join(args.data_dir, cls_name)
        
        # os.walk를 사용하여 하위 폴더의 모든 파일을 재귀적으로 탐색합니다.
        for root, dirs, files in os.walk(cls_dir):
            for fname in sorted(files):
                if fname.lower().endswith(valid_extensions):
                    image_paths.append(os.path.join(root, fname))
                    labels.append(class_to_idx[cls_name])

    print(f"Total Images Found: {len(image_paths)}")

    all_features = []
    all_labels = []

    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), args.batch_size), desc=f"Extracting ({args.crop})"):
            batch_paths = image_paths[i : i + args.batch_size]
            batch_labels = labels[i : i + args.batch_size]

            batch_imgs = []
            valid_labels = []

            for path, label in zip(batch_paths, batch_labels):
                try:
                    img = Image.open(path).convert('RGB')
                    img = transform(img)
                    batch_imgs.append(img)
                    valid_labels.append(label)
                except Exception as e:
                    print(f"Skipping error image: {path} ({e})\n")

            if not batch_imgs:
                continue

            # [수정] 모델 출력 처리 (Hugging Face 방식)
            if args.crop == '5crop':
                input_tensor = torch.stack(batch_imgs).to(device)
                b, n, c, h, w = input_tensor.shape
                
                outputs = model(input_tensor.view(-1, c, h, w))
                features = outputs.last_hidden_state[:, 0]  # CLS token
                
                features = features.view(b, n, -1)
            else:
                input_tensor = torch.stack(batch_imgs).to(device)
                
                outputs = model(input_tensor)
                features = outputs.last_hidden_state[:, 0]  # CLS token

            all_features.append(features.cpu())
            all_labels.extend(valid_labels)

    if len(all_features) > 0:
        X = torch.cat(all_features, dim=0)
        y = torch.tensor(all_labels)

        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        torch.save({'X': X, 'y': y, 'classes': classes}, args.output)
        print(f"Saved features to: {args.output}")
        print(f"Shape: X={X.shape}, y={y.shape}")
    else:
        print("No features extracted.")

if __name__ == '__main__':
    main()