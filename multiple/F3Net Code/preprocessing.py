import os
import cv2
import numpy as np
import scipy.fftpack as fftpack
from multiprocessing import Pool
import glob
import torch
from tqdm import tqdm
import argparse

IMG_SIZE = 256

def dct_2d(x):
    return fftpack.dct(fftpack.dct(x.T, norm='ortho').T, norm='ortho')

def calculate_lfs(img_gray, window_size=10, stride=2):
    dct = dct_2d(img_gray)
    log_spectrum = np.log(np.abs(dct) + 1e-12)
    return log_spectrum

def calculate_fad(img_gray):
    f = np.fft.fft2(img_gray)
    fshift = np.fft.fftshift(f)
    rows, cols = img_gray.shape
    crow, ccol = rows//2 , cols//2
    
    mask_size = 30
    fshift[crow-mask_size:crow+mask_size, ccol-mask_size:ccol+mask_size] = 0
    
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    return np.abs(img_back)

def process_image(task):
    file_path, save_path_base, label = task
    try:
        save_file = save_path_base + ".pt"
        if os.path.exists(save_file):
            return

        img = cv2.imread(file_path)
        if img is None: return
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        lfs = calculate_lfs(gray)
        fad = calculate_fad(gray)
        
        lfs = cv2.normalize(lfs, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        fad = cv2.normalize(fad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        data_dict = {
            'rgb': torch.from_numpy(img),
            'lfs': torch.from_numpy(lfs),
            'fad': torch.from_numpy(fad),
            'label': label
        }
        
        torch.save(data_dict, save_file)
        
    except Exception as e:
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True, help="예: D:/Heimdall_Image_Dataset")
    parser.add_argument('--save_root', type=str, default="./feature")
    args = parser.parse_args()

    tasks = []
    
    for mode in ['train', 'test']:
        # [수정] 다중 분류 타겟인 01_AI-T2I 폴더로 경로 고정
        mode_path = os.path.join(args.data_root, mode, '01_AI-T2I')
        if not os.path.exists(mode_path): 
            print(f"[경고] 경로를 찾을 수 없습니다: {mode_path}")
            continue
        
        # 알파벳 순으로 폴더 정렬하여 0~9 라벨 고정
        classes = sorted([f for f in os.listdir(mode_path) if os.path.isdir(os.path.join(mode_path, f))])
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        
        print(f"\n[{mode}] 클래스 매핑 결과:")
        for cls, idx in class_to_idx.items():
            print(f" - {idx}: {cls}")

        for cls_name in classes:
            folder_path = os.path.join(mode_path, cls_name)
            label = class_to_idx[cls_name]
            
            valid_extensions = ('*.jpg', '*.png', '*.jpeg', '*.webp', '*.bmp')
            files = []
            for ext in valid_extensions:
                files.extend(glob.glob(os.path.join(folder_path, "**", ext), recursive=True))
            
            for f in files:
                rel_path = os.path.relpath(f, args.data_root)
                save_path_base = os.path.join(args.save_root, os.path.splitext(rel_path)[0])
                os.makedirs(os.path.dirname(save_path_base), exist_ok=True)
                
                tasks.append((f, save_path_base, label))

    print(f"\n총 {len(tasks)}장의 다중 분류 이미지를 .pt 형식으로 전처리합니다...")
    
    num_workers = max(1, os.cpu_count() - 2)
    with Pool(processes=num_workers) as pool:
        list(tqdm(pool.imap_unordered(process_image, tasks), total=len(tasks), desc="특징 추출"))
        
    print("전처리 완료! feature 폴더를 확인하세요.")