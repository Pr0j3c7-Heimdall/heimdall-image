'''
F3Net(DualStreamConvNeXt) 다중 분류의 Test 샘플별 10-class 확률을 CSV(key,label,p0..p9)로 저장한다.
다중 분류 앙상블(fusion/fuse_multiclass.py)의 입력을 만드는 용도이며, 개별 정확도도 함께 출력하므로
test.py 대신 이것만 돌려도 F3Net 다중 분류 정확도를 얻을 수 있다.

입력은 preprocessing.py가 만든 .pt 폴더다. 전처리 계산은 이진·다중이 동일하므로
multiple/F3Net Code/feature 가 없으면 binary/F3Net Code/feature 를 --dataset_root로 넣어도 된다
(Real .pt는 경로로 걸러내고, 저장된 label 대신 경로의 폴더명으로 클래스를 정한다).

Usage (multiple/F3Net Code 에서):
    python score_test_multi.py --dataset_root ./feature --weights_path best_model.pth \
        --out ../../fusion/scores/multi_f3net.csv
'''
import os
import csv
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import CustomDataset
from models import DualStreamConvNeXt

CLASSES = ['BigGAN', 'Dalle-3', 'Flux-1.1-pro', 'Glide', 'GPT-image-1',
           'Imagen-4.0', 'Midjourney-V6', 'Nano-Banana-Family', 'SD3.5', 'SDXL']
_CLASS_IDX = {c.lower(): i for i, c in enumerate(CLASSES)}
_GROUPS = {'00_Real': 0, '01_AI-T2I': 1, 'AI-T2I': 1}


def parse_path(path):
    """Test 이미지(.pt) 경로 -> (AI 여부, 클래스 번호, 조인 키). DINOv3/U-Net 스크립트와 같은 규칙."""
    parts = path.replace('\\', '/').split('/')
    rel = parts[len(parts) - parts[::-1].index('test'):]
    if rel[0] in _GROUPS and len(rel) > 2:
        is_ai, cls_dir, rest = _GROUPS[rel[0]] == 1, rel[1], rel[2:]
    else:
        cls_dir, rest = rel[0], rel[1:]
        is_ai = int(cls_dir.split('_')[0]) >= 10
    if not is_ai:
        return False, None, None
    head, _, tail = cls_dir.partition('_')
    name = tail if head.isdigit() else cls_dir
    if name.lower() not in _CLASS_IDX:
        raise ValueError("알 수 없는 생성 모델 폴더: {} (경로: {})".format(cls_dir, path))
    idx = _CLASS_IDX[name.lower()]
    return True, idx, CLASSES[idx] + '/' + os.path.splitext('/'.join(rest))[0]


def main():
    parser = argparse.ArgumentParser(description='F3Net 다중 분류 샘플별 확률 CSV 저장')
    parser.add_argument('--dataset_root', default='./feature', type=str, help='전처리된 .pt 폴더 (그 아래 test/)')
    parser.add_argument('--weights_path', default='best_model.pth', type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--out', type=str, required=True)
    args = parser.parse_args()

    ds = CustomDataset(dataset_root=args.dataset_root, mode='test', size=256, augment=False)
    parsed = [(p,) + parse_path(p) for p in ds.image_paths]
    parsed = [x for x in parsed if x[1]]                 # AI만
    ds.image_paths = [x[0] for x in parsed]               # __getitem__/__len__이 이 목록을 쓴다
    labels = [x[2] for x in parsed]
    keys = [x[3] for x in parsed]
    print(f"AI 이미지 {len(ds.image_paths)}장")
    if len(ds.image_paths) < 49000:
        print(f"[경고] AI 이미지가 {len(ds.image_paths)}장뿐입니다(정상은 50,000장). "
              "전처리가 덜 끝난 채로 실행됐을 수 있습니다 — 결과를 논문에 쓰기 전에 원인을 확인하세요.")
    if not ds.image_paths:
        raise SystemExit("AI .pt를 찾지 못했습니다. --dataset_root 아래 test/ 구조를 확인하세요.")

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DualStreamConvNeXt(num_classes=len(CLASSES)).to(device)
    model.load_state_dict(torch.load(args.weights_path, map_location=device))
    model.eval()

    probs = []
    with torch.no_grad():
        for rgb, lfs, fad, _ in tqdm(loader, desc='Scoring'):
            out = model(rgb.to(device), lfs.to(device), fad.to(device))
            probs.append(torch.softmax(out.float(), dim=1).cpu())
    probs = torch.cat(probs).numpy()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['key', 'label'] + [f'p{i}' for i in range(len(CLASSES))])
        for k, y, p in zip(keys, labels, probs):
            w.writerow([k, y] + [f'{v:.6f}' for v in p])

    acc = (probs.argmax(1) == np.array(labels)).mean() * 100
    print(f"개별 정확도: {acc:.3f}%   <- 논문 표의 F3Net 다중 분류 값")
    for i, c in enumerate(CLASSES):
        m = np.array(labels) == i
        if m.any():
            print(f"  {c:<20s} {m.sum():6d}장  {(probs[m].argmax(1) == i).mean() * 100:6.2f}%")
    print(f"Saved {len(keys)} rows -> {args.out}")


if __name__ == '__main__':
    main()
