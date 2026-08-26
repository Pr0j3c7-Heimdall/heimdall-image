import os
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils import FeatureExtractor, get_transform
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, folders):
        self.image_paths = image_paths
        self.labels = labels
        self.folders = folders
        self.transform = get_transform()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.labels[idx]
        folder = self.folders[idx]
        try:
            img = Image.open(path).convert('RGB')
            img_tensor = self.transform(img)
            return img_tensor, label, folder, True
        except Exception:
            return torch.zeros((3, 512, 512)), label, folder, False

def process_dataset(data_dir, output_name, batch_size=8, num_workers=4, device='cuda'):
    extractor = FeatureExtractor(device=device)
    
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found ({data_dir})")
        return

    valid_exts = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif')
    
    image_paths, labels_list, folders_list = [], [], []
    
    print(f"디렉토리 탐색 시작: {data_dir}")
    for root, _, files in os.walk(data_dir):
        # 현재 파일들이 들어있는 가장 하위 폴더의 이름 (예: '00_BDD', '10_BigGAN')
        folder_name = os.path.basename(root)
        
        for f in files:
            if f.lower().endswith(valid_exts):
                # 1. Train 구조: 경로에 '00_Real'이나 '01_AI-T2I'가 포함되어 있는 경우 확실하게 라벨링
                if '00_Real' in root:
                    label = 0
                elif '01_AI-T2I' in root:
                    label = 1
                # 2. Test 구조: 상위 폴더 없이 바로 나열된 경우 폴더명 앞의 숫자로 판별
                else:
                    try:
                        prefix_num = int(folder_name.split('_')[0])
                        label = 0 if prefix_num < 10 else 1
                    except ValueError:
                        continue # 폴더명이 숫자로 시작하지 않으면 스킵
                
                image_paths.append(os.path.join(root, f))
                labels_list.append(label)
                folders_list.append(folder_name)
    
    print(f"총 {len(image_paths)}장의 이미지 특징 추출 준비 (배치: {batch_size}, 워커: {num_workers})...")
    
    dataset = ImageDataset(image_paths, labels_list, folders_list)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    
    all_features, all_labels, all_folders = [], [], []
    
    for batch_imgs, batch_labels, batch_folders, batch_valid in tqdm(loader):
        valid_idx = batch_valid.bool()
        if not valid_idx.any(): continue
            
        valid_imgs = batch_imgs[valid_idx]
        batch_features = extractor.extract_features_batch(valid_imgs)
        
        all_features.append(batch_features)
        all_labels.extend(batch_labels[valid_idx].tolist())
        
        valid_folders = [f for i, f in enumerate(batch_folders) if valid_idx[i]]
        all_folders.extend(valid_folders)
            
    # [핵심] 텐서 결합 및 .pt 딕셔너리로 저장
    X = torch.cat(all_features, dim=0)
    y = torch.tensor(all_labels, dtype=torch.long)
    
    save_path = f"{output_name}.pt"
    torch.save({'X': X, 'y': y, 'folders': all_folders}, save_path)
    print(f"Saved: {save_path} (Features shape: {X.shape})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output', type=str, default='train', help='저장할 .pt 파일 이름 (확장자 제외)')
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--num_workers', type=int, default=12)
    args = parser.parse_args()
    
    process_dataset(args.data_dir, args.output, args.batch_size, args.num_workers)