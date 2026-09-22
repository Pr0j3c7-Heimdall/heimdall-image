import os
import sys
import argparse
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel

# Argument Parsing
parser = argparse.ArgumentParser(description='DINOv3 Feature Extractor (Hugging Face)')
parser.add_argument('--data_dir', type=str, required=True, help='Dataset directory path')
parser.add_argument('--output', type=str, required=True, help='Output .pt file path')
parser.add_argument('--res', type=int, default=224, help='Input resolution')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_workers', type=int, default=8, help='이미지 로딩·전처리 워커 수 (0이면 메인 프로세스에서 순차 처리)')
parser.add_argument('--crop', type=str, default='center', choices=['center', 'random', '5crop'])
args = parser.parse_args()

class StackNormalizedCrops:
    """FiveCrop 결과(PIL 5장)를 정규화해 (5, C, H, W)로 쌓는다. lambda 대신 클래스로 둬서 DataLoader 워커(spawn)에서도 pickle 가능"""
    def __init__(self, norm):
        self.norm = norm

    def __call__(self, crops):
        return torch.stack([self.norm(T.ToTensor()(crop)) for crop in crops])


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
            StackNormalizedCrops(norm)
        ])

class ImageListDataset(Dataset):
    """읽기에 실패한 이미지는 None을 돌려주고 collate에서 제외한다(기존 '건너뛰기' 동작 유지)."""
    def __init__(self, paths, labels, transform):
        self.paths, self.labels, self.transform = paths, labels, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            img = self.transform(Image.open(path).convert('RGB'))
        except Exception as e:
            print(f"Skipping error image: {path} ({e})\n")
            return None
        return img, self.labels[idx], path


def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    imgs, labels, paths = zip(*batch)
    return torch.stack(imgs), list(labels), list(paths)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # [수정] Hugging Face 모델 로드
    model_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    print(f"Loading DINOv3 model: {model_name}")
    
    try:
        model = AutoModel.from_pretrained(model_name).to(device)
    except Exception as e:
        # [수정] 원래는 여기서 조용히 return해서 exit code가 0이 됐다. 그러면
        # run_remote_experiments.sh의 `set -e`가 이 실패를 못 잡고 다음 단계로
        # 넘어가, 출력 파일이 없다는 훨씬 헷갈리는 오류로 뒤늦게 실패했다.
        # sys.exit(1)로 바꿔서 여기서 바로, 명확하게 멈추게 한다.
        print(f"Error loading model: {e}", file=sys.stderr)
        print("gated repo 오류면: https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m 에서 접근 승인 후"
              " HF_TOKEN 환경변수를 설정하세요. 그 외 오류면 pip install -U transformers 를 확인하세요.",
              file=sys.stderr)
        sys.exit(1)

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
    all_paths = []

    loader = DataLoader(
        ImageListDataset(image_paths, labels, transform),
        batch_size=args.batch_size, shuffle=False,  # 순서 유지 필수 (paths/labels와 특징이 대응)
        num_workers=args.num_workers, collate_fn=collate_skip_none,
    )

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Extracting ({args.crop})"):
            if batch is None:
                continue
            input_batch, valid_labels, valid_paths = batch

            # [수정] 모델 출력 처리 (Hugging Face 방식)
            if args.crop == '5crop':
                input_tensor = input_batch.to(device)
                b, n, c, h, w = input_tensor.shape
                
                outputs = model(input_tensor.view(-1, c, h, w))
                features = outputs.last_hidden_state[:, 0]  # CLS token
                
                features = features.view(b, n, -1)
            else:
                input_tensor = input_batch.to(device)
                
                outputs = model(input_tensor)
                features = outputs.last_hidden_state[:, 0]  # CLS token

            all_features.append(features.cpu())
            all_labels.extend(valid_labels)
            all_paths.extend(valid_paths)

    if len(all_features) > 0:
        X = torch.cat(all_features, dim=0)
        y = torch.tensor(all_labels)

        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        torch.save({'X': X, 'y': y, 'classes': classes, 'paths': all_paths}, args.output)
        print(f"Saved features to: {args.output}")
        print(f"Shape: X={X.shape}, y={y.shape}")
    else:
        print("No features extracted.")

if __name__ == '__main__':
    main()