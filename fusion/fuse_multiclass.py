"""
다중 분류(생성 모델 추정) 앙상블 평가.

세 모델의 샘플별 10-class 확률 CSV(multiple/*/score_test_multi.py 출력)를 키로 조인한 뒤,
선행 이미지 프레임워크와 같은 방식 — 모델별 점수 벡터를 정규화(softmax 확률)해 합산하고
총점이 가장 높은 생성 모델을 고른다 — 으로 앙상블 정확도를 계산한다.

이 결합은 학습하는 파라미터가 없으므로(가중치 고정) 이진 앙상블처럼 K-Fold가 필요 없고,
Test의 AI 이미지 전체로 한 번 평가한다. 개별 모델 정확도와 같은 조건(실제 AI 5만 장 전체,
이진 판정 결과와 무관)이라 표에 나란히 실을 수 있다.

numpy + 표준 csv만 쓴다.

Usage (fusion/ 에서):
    python fuse_multiclass.py \
        --scores scores/multi_dinov3.csv scores/multi_f3net.csv scores/multi_unet.csv \
        --names DINOv3 F3Net U-Net --out multi_fusion.json
"""
import argparse
import csv
import json

import numpy as np

CLASSES = ['BigGAN', 'Dalle-3', 'Flux-1.1-pro', 'Glide', 'GPT-image-1',
           'Imagen-4.0', 'Midjourney-V6', 'Nano-Banana-Family', 'SD3.5', 'SDXL']


def read_scores(path):
    rows = {}
    with open(path, newline='') as f:
        r = csv.reader(f)
        header = next(r)
        n = len(header) - 2
        for row in r:
            rows[row[0]] = (int(row[1]), np.array([float(v) for v in row[2:2 + n]]))
    return rows


def per_class(y, pred):
    return [float((pred[y == i] == i).mean() * 100) if (y == i).any() else float('nan') for i in range(len(CLASSES))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scores', nargs='+', required=True)
    ap.add_argument('--names', nargs='+', required=True)
    ap.add_argument('--weights', nargs='+', type=float, default=None,
                    help='모델별 가중치(생략하면 균등 = 정규화 합산)')
    ap.add_argument('--out', type=str, default=None, help='결과 JSON 경로')
    args = ap.parse_args()
    assert len(args.scores) == len(args.names), '--scores와 --names 개수가 다릅니다'

    tables = [read_scores(p) for p in args.scores]
    common = sorted(set.intersection(*[set(t) for t in tables]))
    for name, t in zip(args.names, tables):
        print(f'  {name:8s}: {len(t):6d}장')
    print(f'  공통    : {len(common):6d}장')
    if any(len(t) != len(common) for t in tables):
        print('  [경고] 모델마다 채점한 이미지가 다릅니다. 공통 샘플만으로 평가합니다.')
    if not common:
        raise SystemExit('공통 샘플이 없습니다. 세 CSV의 key 형식을 확인하세요.')

    y = np.array([tables[0][k][0] for k in common])
    for name, t in zip(args.names[1:], tables[1:]):
        bad = sum(t[k][0] != tables[0][k][0] for k in common)
        if bad:
            raise SystemExit(f'{name}의 라벨이 {args.names[0]}와 {bad}건 다릅니다. 클래스 순서를 확인하세요.')
    P = [np.stack([t[k][1] for k in common]) for t in tables]

    w = np.ones(len(P)) if args.weights is None else np.array(args.weights, dtype=float)
    ens = sum(wi * p for wi, p in zip(w, P))
    results = {}
    for name, p in zip(args.names, P):
        pred = p.argmax(1)
        results[name] = {'acc': float((pred == y).mean() * 100), 'per_class': per_class(y, pred)}
    ens_pred = ens.argmax(1)
    ens_name = 'Ensemble'
    results[ens_name] = {'acc': float((ens_pred == y).mean() * 100), 'per_class': per_class(y, ens_pred)}

    cols = args.names + [ens_name]
    print('\n' + '=' * (22 + 11 * len(cols)))
    print(f"{'생성 모델':<22s}" + ''.join(f'{c:>11s}' for c in cols))
    print('-' * (22 + 11 * len(cols)))
    for i, c in enumerate(CLASSES):
        print(f'{c:<22s}' + ''.join(f"{results[m]['per_class'][i]:10.2f}%" for m in cols))
    print('-' * (22 + 11 * len(cols)))
    print(f"{'전체':<22s}" + ''.join(f"{results[m]['acc']:10.2f}%" for m in cols))
    print('=' * (22 + 11 * len(cols)))
    print(f'가중치: {dict(zip(args.names, w.tolist()))}')

    cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for t, p in zip(y, ens_pred):
        cm[t, p] += 1
    pairs = sorted(((cm[i, j] + cm[j, i], CLASSES[i], CLASSES[j])
                    for i in range(len(CLASSES)) for j in range(i + 1, len(CLASSES))), reverse=True)[:5]
    print('\n앙상블 주요 혼동 쌍 (양방향 합):')
    for n, a, b in pairs:
        print(f'  {a} <-> {b}: {n}건')

    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump({'classes': CLASSES, 'n_samples': len(common), 'weights': dict(zip(args.names, w.tolist())),
                       'results': results, 'ensemble_confusion_matrix': cm.tolist(),
                       'top_confusions': [{'pair': [a, b], 'count': int(n)} for n, a, b in pairs]},
                      f, ensure_ascii=False, indent=2)
        print(f'\nSaved -> {args.out}')


if __name__ == '__main__':
    main()
