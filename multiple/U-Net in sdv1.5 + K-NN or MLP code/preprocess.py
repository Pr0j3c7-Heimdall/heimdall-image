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
    
    # 클래스명 추출 및 매핑 (최하위 폴더 이름 기준)
    class_set = set()
    for root, dirs, files in os.walk(data_dir):
        if not dirs:  # 하위 폴더가 없는 최하위 폴더인 경우
            class_set.add(os.path.basename(root))
            
    classes = sorted(list(class_set))
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    
    print(f"발견된 클래스 ({len(classes)}개):")
    for cls_name, idx in class_to_idx.items():
        print(f" - [{idx}] {cls_name}")

    image_paths, labels_list, folders_list = [], [], []
    
    print(f"\n디렉토리 탐색 시작: {data_dir}")
    for root, dirs, files in os.walk(data_dir):
        if not dirs:
            folder_name = os.path.basename(root)
            label = class_to_idx[folder_name]
            
            for f in files:
                if f.lower().endswith(valid_exts):
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
            
    # 텐서 결합 및 .pt 딕셔너리로 저장 (classes 메타데이터 포함)
    X = torch.cat(all_features, dim=0)
    y = torch.tensor(all_labels, dtype=torch.long)
    
    save_path = f"{output_name}.pt"
    torch.save({'X': X, 'y': y, 'folders': all_folders, 'classes': classes}, save_path)
    print(f"Saved: {save_path} (Features shape: {X.shape})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output', type=str, default='train', help='저장할 .pt 파일 이름 (확장자 제외)')
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--num_workers', type=int, default=12)
    args = parser.parse_args()
    
    process_dataset(args.data_dir, args.output, args.batch_size, args.num_workers)