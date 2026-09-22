'''
U-Net(SD v1.5) 특징 + MLP 다중 분류 헤드의 Test 샘플별 10-class 확률을 CSV(key,label,p0..p9)로 저장한다.
다중 분류 앙상블(fusion/fuse_multiclass.py)의 입력을 만드는 용도다.

특징 추출기(utils.py)는 이진·다중이 동일하므로, 이진 앙상블용으로 binary 폴더에서
preprocess.py(수정본)로 뽑은 Test 전체 'paths' 포함 .pt를 그대로 넣으면 AI 이미지만 골라 쓴다.

Usage (multiple/U-Net in sdv1.5 + K-NN or MLP code 에서):
    python score_test_multi.py --model unet_mlp_model_multi.pth \
        --input "../../binary/U-Net in sdv1.5 + K-NN or MLP code/test_paths.pt" \
        --out ../../fusion/scores/multi_unet.csv
'''
import os
import csv
import argparse
import torch
import torch.nn as nn
import numpy as np

CLASSES = ['BigGAN', 'Dalle-3', 'Flux-1.1-pro', 'Glide', 'GPT-image-1',
           'Imagen-4.0', 'Midjourney-V6', 'Nano-Banana-Family', 'SD3.5', 'SDXL']
_CLASS_IDX = {c.lower(): i for i, c in enumerate(CLASSES)}
_GROUPS = {'00_Real': 0, '01_AI-T2I': 1, 'AI-T2I': 1}


def parse_path(path):
    """Test 이미지 경로 -> (AI 여부, 클래스 번호, 조인 키). DINOv3/F3Net 스크립트와 같은 규칙."""
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


def load_mlp(path, device):
    """TorchMLP(net = Linear → ReLU → Linear)의 state_dict에서 차원을 읽어 그대로 복원한다."""
    sd = torch.load(path, map_location=device)
    in_dim, hid = sd['net.0.weight'].shape[1], sd['net.0.weight'].shape[0]
    out_dim = sd['net.2.weight'].shape[0]
    net = nn.Sequential(nn.Linear(in_dim, hid), nn.ReLU(), nn.Linear(hid, out_dim)).to(device)
    net.load_state_dict({k.replace('net.', '', 1): v for k, v in sd.items()})
    return net.eval(), out_dim


def main():
    parser = argparse.ArgumentParser(description='U-Net 다중 분류 샘플별 확률 CSV 저장')
    parser.add_argument('--input', type=str, required=True, help="'paths'가 들어 있는 test 특징 .pt")
    parser.add_argument('--model', type=str, default='unet_mlp_model_multi.pth')
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--out', type=str, required=True)
    args = parser.parse_args()

    data = torch.load(args.input)
    if 'paths' not in data:
        raise KeyError("입력 .pt에 'paths'가 없습니다. 수정된 preprocess.py로 test 특징을 다시 추출하세요.")
    X, paths = data['X'], data['paths']

    keep, labels, keys = [], [], []
    for i, p in enumerate(paths):
        is_ai, idx, key = parse_path(p)
        if is_ai:
            keep.append(i); labels.append(idx); keys.append(key)
    X = X[keep].float()
    print(f"AI 이미지 {len(keep)}장 선택 (전체 {len(paths)}장 중)")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net, out_dim = load_mlp(args.model, device)
    if out_dim != len(CLASSES):
        raise ValueError(f"출력 차원이 {out_dim}입니다. 다중 분류(10-class) 가중치가 맞는지 확인하세요.")

    probs = []
    with torch.no_grad():
        for s in range(0, len(X), args.batch_size):
            probs.append(torch.softmax(net(X[s:s + args.batch_size].to(device)), dim=1).cpu())
    probs = torch.cat(probs).numpy()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['key', 'label'] + [f'p{i}' for i in range(len(CLASSES))])
        for k, y, p in zip(keys, labels, probs):
            w.writerow([k, y] + [f'{v:.6f}' for v in p])

    acc = (probs.argmax(1) == np.array(labels)).mean() * 100
    print(f"개별 정확도: {acc:.3f}%  (참고: 기존 보고값 93.36 %)")
    print(f"Saved {len(keys)} rows -> {args.out}")


if __name__ == '__main__':
    main()
