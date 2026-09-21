"""
F3Net(DualStreamConvNeXt)의 Test 샘플별 raw_score(logit)를 CSV(path,label,raw_score)로 저장한다.
세 모델(DINOv3/F3Net/U-Net)의 CSV를 fusion/fit_fusion.py로 결합하기 위한 입력이다 (fusion/README.md 참고).

test.py와 같은 추론을 하되 sigmoid 이전의 1-뉴런 출력(logit)을 그대로 저장한다.
조인 키는 전처리된 .pt의 경로(feature/test/00_BDD/a.pt)에서 'test/' 이후 확장자를 뗀 값(00_BDD/a)이며,
원본 이미지 경로에서 만든 DINOv3/U-Net의 키와 같은 형태가 된다.

Usage:
    python score_test.py --dataset_root ./feature --weights_path best_model.pth \
        --out ../../fusion/scores/image_f3net.csv
"""
import argparse
import csv
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import DualStreamConvNeXt
from utils import CustomDataset


def sample_key(path):
    """모델마다 다른 경로 표기(원본 이미지 / F3Net의 .pt)를 'test/' 이후의 확장자 없는 상대경로로
    통일해 조인 키로 쓴다. 예) feature/test/00_BDD/a.pt -> 00_BDD/a"""
    parts = path.replace('\\', '/').split('/')
    if 'test' not in parts:
        raise ValueError("경로에 'test' 폴더가 없어 조인 키를 만들 수 없습니다: {}".format(path))
    idx = len(parts) - 1 - parts[::-1].index('test')
    rel = '/'.join(parts[idx + 1:])
    return os.path.splitext(rel)[0]


def label_from_key(key):
    """폴더명 앞 숫자 기준 (0~9: Real, 10~: AI) — test.py와 동일한 규칙"""
    return 0 if int(key.split('/')[0].split('_')[0]) < 10 else 1


def main():
    parser = argparse.ArgumentParser(description='F3Net Test 샘플별 logit CSV 저장')
    parser.add_argument('--dataset_root', default='./feature', type=str, help='전처리된 데이터 폴더 경로')
    parser.add_argument('--weights_path', default='best_model.pth', type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--gpu_ids', default='0', type=str)
    parser.add_argument('--out', type=str, required=True)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = CustomDataset(dataset_root=args.dataset_root, mode='test', size=256, augment=False)
    if len(dataset) == 0:
        raise RuntimeError('테스트 데이터가 없습니다: {}/test'.format(args.dataset_root))
    # shuffle=False 필수 — dataset.image_paths 순서와 출력 순서가 일치해야 한다
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = DualStreamConvNeXt(num_classes=1).to(device)
    model.load_state_dict(torch.load(args.weights_path, map_location=device))
    model.eval()

    raw_scores = []
    with torch.no_grad():
        for rgb, lfs, fad, _ in tqdm(loader, desc='Scoring'):
            output = model(rgb.to(device), lfs.to(device), fad.to(device))
            raw_scores.extend(output.cpu().numpy().flatten().tolist())

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'label', 'raw_score'])
        for path, score in zip(dataset.image_paths, raw_scores):
            key = sample_key(path)
            writer.writerow([key, label_from_key(key), score])
    print('Saved {} rows -> {}'.format(len(raw_scores), args.out))


if __name__ == '__main__':
    main()
