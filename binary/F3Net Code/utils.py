import os
import glob
import torch
from torch.utils import data
import torchvision.transforms.functional as TF
import random

class CustomDataset(data.Dataset):
    def __init__(self, dataset_root, mode='train', size=256, augment=True):
        self.root = os.path.join(dataset_root, mode) # 예: Processed_Dataset/train
        self.size = size
        self.augment = augment
        self.mode = mode
        
        if not os.path.exists(self.root):
            print(f"[에러] 경로가 존재하지 않습니다: {self.root}")
            self.image_paths = []
            return

        # 1. 재귀적으로 모든 .pt 파일을 찾습니다. (preprocessing.py가 .pt로 저장함)
        search_path = os.path.join(self.root, '**', '*.pt')
        self.image_paths = glob.glob(search_path, recursive=True)
        
        print(f"[{mode}] 총 .pt 파일 수: {len(self.image_paths)}")

    def __getitem__(self, index):
        # 1. .pt 파일 로드 (여기 안에 rgb, lfs, fad, label이 다 들어있음!)
        pt_path = self.image_paths[index]
        data_dict = torch.load(pt_path)
        
        # 2. Tensor 가져오기 (preprocessing.py에서 uint8 numpy로 넘긴 것을 받아옴)
        rgb = data_dict['rgb'] # (H, W, 3), cv2로 읽었으므로 BGR
        lfs = data_dict['lfs'] # (H, W)
        fad = data_dict['fad'] # (H, W)
        label = data_dict['label']
        
        # 3. 차원 변경 및 실수형(Float) 변환 (0~255 -> 0.0~1.0)
        # BGR을 RGB로 바꾸고, (H, W, C) -> (C, H, W)로 변경
        rgb = rgb[:, :, [2, 1, 0]].permute(2, 0, 1).float() / 255.0
        
        # 흑백 이미지들은 채널 차원(1) 추가: (H, W) -> (1, H, W)
        lfs = lfs.unsqueeze(0).float() / 255.0
        fad = fad.unsqueeze(0).float() / 255.0

        # 4. Augmentation (네가 작성한 TF 로직 적용!)
        if self.augment:
            # Random Horizontal Flip
            if random.random() > 0.5:
                rgb = TF.hflip(rgb)
                lfs = TF.hflip(lfs)
                fad = TF.hflip(fad)
            
            # Random Rotation
            if random.random() > 0.5:
                angle = random.uniform(-10, 10)
                rgb = TF.rotate(rgb, angle)
                lfs = TF.rotate(lfs, angle)
                fad = TF.rotate(fad, angle)

        # 5. Normalize (ImageNet Mean/Std 및 [-1, 1] 정규화)
        rgb = TF.normalize(rgb, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        lfs = (lfs - 0.5) / 0.5
        fad = (fad - 0.5) / 0.5

        return rgb, lfs, fad, label

    def __len__(self):
        return len(self.image_paths)