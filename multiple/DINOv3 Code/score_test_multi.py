'''
DINOv3 + MLP 다중 분류 헤드의 Test 샘플별 10-class 확률을 CSV(key,label,p0..p9)로 저장한다.
다중 분류 앙상블(fusion/fuse_multiclass.py)의 입력을 만드는 용도다.

입력 .pt는 binary/DINOv3 Code/extract_features.py(수정본)로 만든 'paths' 포함 파일이면 된다.
이진 앙상블용으로 Test 전체(Real+AI)를 --crop 5crop으로 뽑은 파일을 그대로 넣으면 AI 이미지만 골라 쓴다.
원래 다중 분류 평가(95.17 %)는 center crop 특징으로 했으므로 기본값은 --crop center이며,
5crop 특징이 들어오면 FiveCrop의 5번째(center) crop을 꺼내 쓴다(Resize 256 → CenterCrop 224로 동일).

Usage (multiple/DINOv3 Code 에서):
    python score_test_multi.py --model dinov3_multiclass.pth \
        --input "../../binary/DINOv3 Code/features/test_paths.pt" \
        --out ../../fusion/scores/multi_dinov3.csv
'''
import os
import csv
import argparse
import torch
import numpy as np
from mlp_model import DINOMlpClassifier

# 학습 시 class_to_idx = sorted(train/01_AI-T2I 폴더명) = 00_BigGAN … 09_SDXL 순서
CLASSES = ['BigGAN', 'Dalle-3', 'Flux-1.1-pro', 'Glide', 'GPT-image-1',
           'Imagen-4.0', 'Midjourney-V6', 'Nano-Banana-Family', 'SD3.5', 'SDXL']
_CLASS_IDX = {c.lower(): i for i, c in enumerate(CLASSES)}
_GROUPS = {'00_Real': 0, '01_AI-T2I': 1, 'AI-T2I': 1}


def parse_path(path):
    """Test 이미지 경로 -> (AI 여부, 클래스 번호, 조인 키).
    폴더 구조(1단 00_BDD…19_SDXL / 2단 00_Real·01_AI-T2I)와 번호 접두사(00_ / 10_)가 달라도
    같은 이미지면 같은 키가 나오도록 키를 '클래스명/파일명(확장자 제외)'로 만든다."""
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
    parser = argparse.ArgumentParser(description='DINOv3 다중 분류 샘플별 확률 CSV 저장')
    parser.add_argument('--model', type=str, required=True, help="models/ 아래 .pth 파일명")
    parser.add_argument('--input', type=str, required=True, help="'paths'가 들어 있는 test 특징 .pt")
    parser.add_argument('--crop', type=str, default='center', choices=['center', '5crop'])
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--out', type=str, required=True)
    args = parser.parse_args()

    data = torch.load(args.input)
    if 'paths' not in data:
        raise KeyError("입력 .pt에 'paths'가 없습니다. 수정된 extract_features.py로 test 특징을 다시 추출하세요.")
    X, paths = data['X'], data['paths']

    keep, labels, keys = [], [], []
    for i, p in enumerate(paths):
        is_ai, idx, key = parse_path(p)
        if is_ai:
            keep.append(i); labels.append(idx); keys.append(key)
    X = X[keep]
    print(f"AI 이미지 {len(keep)}장 선택 (전체 {len(paths)}장 중)")

    if X.dim() == 3 and args.crop == 'center':
        X = X[:, 4, :]   # torchvision FiveCrop 순서: tl, tr, bl, br, center

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DINOMlpClassifier(1024, len(CLASSES)).to(device)
    model.load_state_dict(torch.load(os.path.join('models', args.model), map_location=device))
    model.eval()

    probs = []
    with torch.no_grad():
        for s in range(0, len(X), args.batch_size):
            xb = X[s:s + args.batch_size].float().to(device)
            if xb.dim() == 3:   # 5crop 평균
                b, n, c = xb.shape
                out = model(xb.view(b * n, c)).view(b, n, -1).mean(dim=1)
            else:
                out = model(xb)
            probs.append(torch.softmax(out, dim=1).cpu())
    probs = torch.cat(probs).numpy()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['key', 'label'] + [f'p{i}' for i in range(len(CLASSES))])
        for k, y, p in zip(keys, labels, probs):
            w.writerow([k, y] + [f'{v:.6f}' for v in p])

    acc = (probs.argmax(1) == np.array(labels)).mean() * 100
    print(f"개별 정확도: {acc:.3f}%  (참고: 기존 보고값 95.17 %, center crop 기준)")
    print(f"Saved {len(keys)} rows -> {args.out}")


if __name__ == '__main__':
    main()
