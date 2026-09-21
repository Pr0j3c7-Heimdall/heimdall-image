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


_GROUP_DIRS = {'00_Real': 0, '01_AI-T2I': 1}


def _rel_parts(path):
    """경로에서 마지막 'test' 폴더 이후의 구성요소. 예) D:/.../test/00_BDD/a.jpg -> ['00_BDD', 'a.jpg']"""
    parts = path.replace('\\', '/').split('/')
    if 'test' not in parts:
        raise ValueError("경로에 'test' 폴더가 없어 조인 키를 만들 수 없습니다: {}".format(path))
    return parts[len(parts) - parts[::-1].index('test'):]


def sample_key(path):
    """모델마다 다른 경로 표기(원본 이미지 / F3Net의 .pt)와 폴더 구조(그룹 폴더 00_Real/01_AI-T2I 유무)를
    통일해 조인 키로 쓴다: 'test/' 이후 상대경로에서 확장자를 떼고, 맨 앞 그룹 폴더가 있으면 제거한다.
    예) D:/.../test/00_BDD/a.jpg, D:/.../test/00_Real/00_BDD/a.jpg, feature/test/00_BDD/a.pt -> 00_BDD/a"""
    rel = _rel_parts(path)
    if rel[0] in _GROUP_DIRS and len(rel) > 2:  # 그룹 폴더 바로 아래에 파일만 있는 구조는 그룹 폴더를 유지(키 충돌 방지)
        rel = rel[1:]
    return os.path.splitext('/'.join(rel))[0]


def sample_label(path):
    """그룹 폴더(00_Real=0, 01_AI-T2I=1)가 있으면 그것으로, 없으면 폴더명 앞 숫자(0~9 Real, 10~ AI)로 정한다.
    Train 구조는 AI 폴더도 00~09로 번호가 겹치므로 그룹 폴더가 있을 때 숫자 규칙을 쓰면 안 된다."""
    first = _rel_parts(path)[0]
    if first in _GROUP_DIRS:
        return _GROUP_DIRS[first]
    return 0 if int(first.split('_')[0]) < 10 else 1


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

    labels = [sample_label(p) for p in dataset.image_paths]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'label', 'raw_score'])
        for path, label, score in zip(dataset.image_paths, labels, raw_scores):
            writer.writerow([sample_key(path), label, score])
    print('Saved {} rows (real={}, AI={}) -> {}'.format(len(raw_scores), labels.count(0), labels.count(1), args.out))


if __name__ == '__main__':
    main()
