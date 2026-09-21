"""
DINOv3 + MLP 헤드의 Test 샘플별 raw_score(logit)를 CSV(path,label,raw_score)로 저장한다.
세 모델(DINOv3/F3Net/U-Net)의 CSV를 fusion/fit_fusion.py로 결합하기 위한 입력이다 (fusion/README.md 참고).

test.py와 같은 추론(5crop은 crop별 logit을 평균)을 하되, softmax 이전 값을 그대로 저장한다.
2-class 모델이므로 raw_score = logit(AI) - logit(Real). Platt scaling은 sigmoid 이전 logit 단계에서
해야 하므로 확률이 아닌 logit을 저장한다.

입력 .pt는 이 저장소의 extract_features.py로 만든 것이어야 한다('paths' 키 필요 — 예전 버전으로
추출한 파일에는 없으므로 테스트셋 특징을 다시 추출해야 한다).

Usage:
    python score_test.py --model heimdall_dinov3_mlp.pth --input ./features/tests_v3_18.pt \
        --crop 5crop --out ../../fusion/scores/image_dinov3.csv
"""
import argparse
import csv
import os

import torch
from tqdm import tqdm

from mlp_model import DINOMlpClassifier

CONFIG = {
    'input_dim': 1024,
    'num_classes': 2,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}


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
    parser = argparse.ArgumentParser(description='DINOv3 Test 샘플별 logit CSV 저장')
    parser.add_argument('--model', type=str, required=True, help="models/ 아래 .pth 파일명")
    parser.add_argument('--input', type=str, required=True, help="extract_features.py가 만든 test .pt")
    parser.add_argument('--crop', type=str, default='5crop', choices=['center', '5crop'])
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--out', type=str, required=True)
    args = parser.parse_args()

    data = torch.load(args.input)
    if 'paths' not in data:
        raise KeyError("입력 .pt에 'paths'가 없습니다. 수정된 extract_features.py로 test 특징을 다시 추출하세요.")
    X, paths = data['X'], data['paths']
    if len(paths) != len(X):
        raise ValueError("paths({})와 X({}) 개수가 다릅니다".format(len(paths), len(X)))

    model = DINOMlpClassifier(CONFIG['input_dim'], CONFIG['num_classes']).to(CONFIG['device'])
    model.load_state_dict(torch.load(os.path.join('models', args.model), map_location=CONFIG['device']))
    model.eval()

    raw_scores = []
    with torch.no_grad():
        for i in tqdm(range(0, len(X), args.batch_size), desc='Scoring'):
            inputs = X[i:i + args.batch_size].to(CONFIG['device'])
            if args.crop == '5crop' and inputs.dim() == 3:
                b, n, c = inputs.shape
                outputs = model(inputs.view(b * n, c)).view(b, n, -1).mean(dim=1)
            else:
                outputs = model(inputs)
            raw_scores.extend((outputs[:, 1] - outputs[:, 0]).cpu().tolist())

    labels = [sample_label(p) for p in paths]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'label', 'raw_score'])
        for path, label, score in zip(paths, labels, raw_scores):
            writer.writerow([sample_key(path), label, score])
    print('Saved {} rows (real={}, AI={}) -> {}'.format(len(paths), labels.count(0), labels.count(1), args.out))


if __name__ == '__main__':
    main()
