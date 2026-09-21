"""
U-Net(SD v1.5) 특징 + K-NN 또는 MLP 헤드의 Test 샘플별 raw_score(logit)를 CSV(path,label,raw_score)로 저장한다.
세 모델(DINOv3/F3Net/U-Net)의 CSV를 fusion/fit_fusion.py로 결합하기 위한 입력이다 (fusion/README.md 참고).

- MLP: 2-class logit이므로 raw_score = logit(AI) - logit(Real).
- K-NN: 이웃 k개의 AI 비율(k+1가지 이산값)만 나오고 logit이 없다. 0/1 극단값에서 logit이 발산하지 않도록
  Laplace 평활 p' = (p*k + 0.5) / (k + 1)을 적용한 뒤 logit(p')를 raw_score로 쓴다.
  점수가 이산적이라 Platt scaling과 결합 방법 간 비교에서 동점이 많아질 수 있다.

입력 .pt는 이 저장소의 preprocess.py로 만든 것이어야 한다('paths' 키 필요 — 예전 버전으로
추출한 파일에는 없으므로 테스트셋 특징을 다시 추출해야 한다).

Usage:
    python score_test.py --input test.pt --model unet_knn_model.pth --model_type knn \
        --out ../../fusion/scores/image_unet.csv
"""
import argparse
import csv
import math
import os

import torch

from model import get_model


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


def knn_logits(model, X):
    """TorchKNN.predict_proba의 AI 비율 p에 Laplace 평활을 적용한 logit"""
    probs = model.predict_proba(X)
    k = model.k
    smoothed = (probs * k + 0.5) / (k + 1)
    return [math.log(p / (1.0 - p)) for p in smoothed.tolist()]


def mlp_logits(model, X, batch_size=4096):
    model.eval()
    scores = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = X[i:i + batch_size].clone().detach().float().to(model.device)
            out = model.net(batch)
            scores.extend((out[:, 1] - out[:, 0]).cpu().tolist())
    return scores


def main():
    parser = argparse.ArgumentParser(description='U-Net Test 샘플별 logit CSV 저장')
    parser.add_argument('--input', type=str, default='test.pt', help='preprocess.py가 만든 test .pt')
    parser.add_argument('--model', type=str, default='unet_model.pth')
    parser.add_argument('--model_type', type=str, default='knn', choices=['knn', 'mlp', 'mlp-320', 'mlp-640'])
    parser.add_argument('--out', type=str, required=True)
    args = parser.parse_args()

    data = torch.load(args.input)
    if 'paths' not in data:
        raise KeyError("입력 .pt에 'paths'가 없습니다. 수정된 preprocess.py로 test 특징을 다시 추출하세요.")
    X, paths = data['X'], data['paths']
    if len(paths) != len(X):
        raise ValueError("paths({})와 X({}) 개수가 다릅니다".format(len(paths), len(X)))

    model = get_model(args.model_type, input_dim=X.shape[1])
    model.load(args.model)

    raw_scores = knn_logits(model, X) if args.model_type == 'knn' else mlp_logits(model, X)

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
