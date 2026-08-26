import os
import cv2
import numpy as np
import scipy.fftpack as fftpack
from multiprocessing import Pool
import glob
import torch  # [추가] PyTorch import
from tqdm import tqdm

# 설정 (올려준 사진 기반 경로)
DATA_ROOT = r"D:/Heimdall_Image_Dataset"
SAVE_ROOT = "./feature"
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
    img_back = np.abs(img_back)
    return img_back

# [수정] 개별 이미지 처리 함수 (task 튜플을 받아서 .pt로 저장)
def process_image(task):
    file_path, save_path_base, label = task
    try:
        # 이미 처리된 파일이면 패스 (이어하기 기능)
        save_file = save_path_base + ".pt"
        if os.path.exists(save_file):
            return

        img = cv2.imread(file_path)
        if img is None: return
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        lfs = calculate_lfs(gray)
        fad = calculate_fad(gray)
        
        # 0~255 uint8 변환 (용량 최적화)
        lfs = cv2.normalize(lfs, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        fad = cv2.normalize(fad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # [핵심] Numpy 배열을 PyTorch Tensor로 변환 후 저장
        # 이미지 형태: (H, W, C), 흑백은 (H, W) 그대로 저장 (나중에 DataLoader에서 차원 변경)
        data_dict = {
            'rgb': torch.from_numpy(img),
            'lfs': torch.from_numpy(lfs),
            'fad': torch.from_numpy(fad),
            'label': label
        }
        
        torch.save(data_dict, save_file)
        
    except Exception as e:
        pass # 에러 로그가 tqdm 바를 망치지 않도록 pass

if __name__ == '__main__':
    tasks = []
    
    # Train, Test 폴더 순회
    for mode in ['train', 'test']:
        mode_path = os.path.join(DATA_ROOT, mode)
        if not os.path.exists(mode_path): continue
        
        subfolders = [f for f in os.listdir(mode_path) if os.path.isdir(os.path.join(mode_path, f))]
        
        for folder in subfolders:
            folder_path = os.path.join(mode_path, folder)
            
            # 올려준 디렉토리 기준 라벨링
            if mode == 'train':
                label = 0 if 'Real' in folder else 1
            else:
                prefix_num = int(folder.split('_')[0])
                label = 0 if prefix_num < 10 else 1
            
            # 지원하는 이미지 확장자 모두 탐색
            valid_extensions = ('*.jpg', '*.png', '*.jpeg', '*.webp', '*.bmp')
            files = []
            for ext in valid_extensions:
                files.extend(glob.glob(os.path.join(folder_path, "**", ext), recursive=True))
            
            for f in files:
                rel_path = os.path.relpath(f, DATA_ROOT)
                save_path_base = os.path.join(SAVE_ROOT, os.path.splitext(rel_path)[0])
                os.makedirs(os.path.dirname(save_path_base), exist_ok=True)
                
                tasks.append((f, save_path_base, label))

    print(f"총 {len(tasks)}장의 이미지를 PyTorch .pt 형식으로 전처리합니다...")
    
    # 병렬 처리 (윈도우 환경 안정성을 위해 여유 코어 2개 확보)
    num_workers = max(1, os.cpu_count() - 2)
    with Pool(processes=num_workers) as pool:
        list(tqdm(pool.imap_unordered(process_image, tasks), total=len(tasks), desc="특징 추출 및 .pt 저장"))
        
    print("전처리 완료! feature 폴더를 확인하세요.")