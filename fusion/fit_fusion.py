"""
이미지 이진 판별 3개 모델(DINOv3/F3Net/U-Net)의 raw_score CSV(각 모델 디렉토리의 score_test.py 출력)를
입력받아 fusion 방식들을 같은 조건에서 학습·비교한다. heimdall-vox/fusion/fit_fusion.py와 같은
절차(보정, 결합 방식, K-Fold, EER 및 fold 대응 비교)를 그대로 쓰고(논문 음성 절과 같은 프로토콜로
비교하기 위해), 이미지 절이 보고하는 정확도(임계값 0.5, out-of-fold)와 단일 모델 참고 행을 추가했다:

  1. 모델별 확률 보정(Platt scaling) — raw_score(softmax/sigmoid 이전 logit)를 보정된
     확률로 변환. 파라미터 2개(a, b)뿐이라 과적합 위험이 거의 없음.
  2. baseline 계산: 단순 평균 / 다수결 / 고정 가중 평균(개별 모델 EER 역수 가중) /
     soft_voting(팀 T2I 이미지 판별 논문 방식 — softmax(a*ACC) 가중, train fold로 a 탐색)
  3. 정규화 로지스틱 회귀를 여러 C(정규화 강도)로 스윕
  4. 전부 Test pool 안에서 K-Fold 교차검증으로 fit+eval — Train/Val 재사용 없음
     (fusion/README.md "평가 프로토콜" 참고)
  5. baseline 대비 로지스틱 회귀가 유의미하게 나은지 확인, fold 간 EER 편차도 같이 리포트
     (편차가 크면 "이 데이터 양으론 학습 기반 fusion이 불안정하다"는 신호)
  6. 마지막으로 Test pool 전체로 최종 배포용 파라미터를 다시 fit해서 저장

K-Fold 비교에서 어떤 방식이 이겼는지는 사람이 보고 판단한다 — 이 스크립트가 자동으로
"최선"을 골라 저장하지 않고, 모든 후보의 결과를 다 보여준 뒤 --final_method로 명시적으로
고른 방식만 최종 파라미터로 저장한다.

pandas 없이 표준 csv 모듈 + numpy만 쓴다 (이 정도 조인/집계엔 pandas가 과함).

Usage:
    python fit_fusion.py --track image \
        --scores scores/image_dinov3.csv scores/image_f3net.csv scores/image_unet.csv \
        --names DINOv3 F3Net U-Net \
        --n_folds 5 --out image_fusion.json --final_method soft_voting
"""
import argparse
import csv
import json

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq, minimize_scalar
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold

DEFAULT_C_VALUES = [0.001, 0.01, 0.1, 1.0, 10.0]


def get_args():
    parser = argparse.ArgumentParser(description="이미지 모델 3개 fusion 학습/검증 (fusion/README.md 참고)")
    parser.add_argument("--track", type=str, required=True, choices=["image", "speech", "singing"])
    parser.add_argument("--scores", type=str, nargs="+", required=True,
                        help="score_test.py가 만든 CSV 경로들 (path,label,raw_score), 모델 3개 순서대로")
    parser.add_argument("--names", type=str, nargs="+", required=True,
                        help="--scores와 같은 순서의 모델 이름 (리포트/JSON 키로 사용)")
    parser.add_argument("--c_values", type=float, nargs="+", default=DEFAULT_C_VALUES,
                        help="로지스틱 회귀 정규화 강도(C) 스윕 범위. 기본값보다 더 키우거나 좁혀서 재검색할 때 사용")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out", type=str, default=None,
                        help="최종 fusion 파라미터를 저장할 JSON 경로. --final_method 지정 시에만 필요")
    parser.add_argument("--final_method", type=str, default=None,
                        help='K-Fold 결과 보고 사람이 고른 최종 방식. "simple_mean", "fixed_weighted", '
                             '"soft_voting", 또는 "logreg_C<값>"(예: logreg_C1.0). 생략하면 K-Fold 비교만 하고 저장 안 함')
    return parser.parse_args()


def compute_eer(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    return float(eer)


def compute_accuracy(y_true, y_score, threshold=0.5):
    """임계값 기준 정확도. 점수는 모두 [0,1] 확률 형태다(다수결은 AI 투표 비율이라 3모델 기준 2표 이상이면 AI)."""
    return float(np.mean((np.asarray(y_score) >= threshold).astype(int) == y_true))


def load_scores(paths, names):
    """각 CSV(path,label,raw_score)를 path 기준으로 inner join.
    -> (paths: list[str], y: (N,) int array, X_raw: (N, n_models) float array)"""
    per_model = []  # list of dict: path -> (label, raw_score)
    for path in paths:
        d = {}
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                d[row["path"]] = (int(row["label"]), float(row["raw_score"]))
        per_model.append(d)

    common_paths = set(per_model[0].keys())
    for d in per_model[1:]:
        common_paths &= set(d.keys())
    common_paths = sorted(common_paths)

    n_individual = [len(d) for d in per_model]
    if any(len(common_paths) < n for n in n_individual):
        print("WARNING: 모델 간 path가 완전히 일치하지 않습니다 (공통={}, 개별={}) "
              "— 세 모델을 같은 Test 리스트로 채점했는지 확인하세요".format(len(common_paths), n_individual))

    labels_per_model = np.array([[per_model[i][p][0] for i in range(len(names))] for p in common_paths])
    if not np.all(labels_per_model == labels_per_model[:, [0]]):
        raise ValueError("같은 path인데 모델별 label이 다릅니다 — 리스트 파일이 서로 다른 것 같습니다")

    y = labels_per_model[:, 0]
    X_raw = np.array([[per_model[i][p][1] for i in range(len(names))] for p in common_paths])
    return common_paths, y, X_raw


class PlattScaler:
    """1D 로지스틱 회귀로 raw_score(logit) -> 보정된 확률. 파라미터 2개(a, b)뿐."""

    def __init__(self):
        self.lr = LogisticRegression(C=1e6, max_iter=1000)  # 사실상 무정규화, 파라미터 2개라 위험 낮음

    def fit(self, raw_scores, labels):
        self.lr.fit(raw_scores.reshape(-1, 1), labels)
        return self

    def transform(self, raw_scores):
        return self.lr.predict_proba(raw_scores.reshape(-1, 1))[:, 1]


def calibrate_all(X_train_raw, y_train, X_test_raw, n_models):
    scalers = [PlattScaler().fit(X_train_raw[:, i], y_train) for i in range(n_models)]
    X_train_cal = np.stack([s.transform(X_train_raw[:, i]) for i, s in enumerate(scalers)], axis=1)
    X_test_cal = np.stack([s.transform(X_test_raw[:, i]) for i, s in enumerate(scalers)], axis=1)
    return X_train_cal, X_test_cal, scalers


def fixed_weights_from_train(X_train_cal, y_train, n_models):
    """train fold 안에서만 각 모델 EER을 재서 가중치 산출 (test fold 정보 사용 안 함).
    train fold가 작으면(특히 가창) EER이 정확히 0으로 나올 수 있어 1/EER이 0-나누기가
    되므로, 최소값을 깔아서 방지한다 — 데이터가 적을수록 이런 극단치가 나오기 쉽다는
    사실 자체가 fixed_weighted 방식의 한계를 보여주는 신호이기도 하다."""
    eers = np.clip([compute_eer(y_train, X_train_cal[:, i]) for i in range(n_models)], 1e-3, None)
    weights = 1.0 / eers
    return weights / weights.sum()


def soft_voting_weights(X_train_cal, y_train, n_models):
    """지수함수(softmax) 기반 Soft Voting — 팀 T2I 이미지 판별 논문(V장, 식1~4)의 앙상블 방식.
    가중치 = softmax(a * ACC_model), train fold accuracy로 모델별 ACC를 구하고, train fold EER을
    최소화하는 지수 파라미터 a를 탐색한다 (논문은 grid search로 a=72.8을 찾았고, 여기선
    minimize_scalar로 동일한 걸 더 정밀하게 탐색). test fold는 가중치 선택에 전혀 관여하지 않음."""
    acc = np.array([np.mean((X_train_cal[:, i] > 0.5).astype(int) == y_train) for i in range(n_models)])

    def eer_for_a(a):
        w = softmax(a * acc)
        return compute_eer(y_train, (X_train_cal * w).sum(axis=1))

    res = minimize_scalar(eer_for_a, bounds=(0.0, 500.0), method="bounded")
    best_a = float(res.x)
    return softmax(best_a * acc), best_a


def run_kfold(X_raw, y, model_names, n_folds, seed, c_values):
    n_models = len(model_names)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    results = {"simple_mean": [], "majority_vote": [], "fixed_weighted": [], "soft_voting": []}
    for c in c_values:
        results["logreg_C{}".format(c)] = []
    for name in model_names:
        results["single_{}".format(name)] = []  # 참고: 결합 없이 단일 모델(보정 후) 그대로
    accuracies = {k: [] for k in results}  # 같은 fold/같은 점수로 계산한 임계값 0.5 정확도
    soft_voting_log = []  # fold별 (a, weights) — 리포트용

    def record(name, y_true, score):
        results[name].append(compute_eer(y_true, score))
        accuracies[name].append(compute_accuracy(y_true, score))

    for train_idx, test_idx in skf.split(X_raw, y):
        X_train_raw, X_test_raw = X_raw[train_idx], X_raw[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_train_cal, X_test_cal, _ = calibrate_all(X_train_raw, y_train, X_test_raw, n_models)

        record("simple_mean", y_test, X_test_cal.mean(axis=1))
        record("majority_vote", y_test, (X_test_cal > 0.5).mean(axis=1))
        for i, name in enumerate(model_names):
            record("single_{}".format(name), y_test, X_test_cal[:, i])

        w = fixed_weights_from_train(X_train_cal, y_train, n_models)
        record("fixed_weighted", y_test, (X_test_cal * w).sum(axis=1))

        w_sv, a_sv = soft_voting_weights(X_train_cal, y_train, n_models)
        record("soft_voting", y_test, (X_test_cal * w_sv).sum(axis=1))
        soft_voting_log.append((a_sv, w_sv))

        for c in c_values:
            lr = LogisticRegression(C=c, max_iter=1000)  # penalty 기본값 l2
            lr.fit(X_train_cal, y_train)
            record("logreg_C{}".format(c), y_test, lr.predict_proba(X_test_cal)[:, 1])

    return results, accuracies, soft_voting_log


def print_accuracy_report(accuracies):
    """임계값 0.5 정확도(%) — EER 표와 같은 fold·같은 점수에서 계산한 out-of-fold 값"""
    n_folds = len(next(iter(accuracies.values())))
    print("\n=== 정확도 (임계값 0.5, out-of-fold, fold {}개, 단위 %) ===".format(n_folds))
    print("{:<20} {:>10} {:>10} {:>10} {:>10}".format("방식", "평균 ACC", "fold 표준편차", "최저 fold", "최고 fold"))
    for name, values in accuracies.items():
        arr = np.array(values) * 100
        print("{:<20} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.3f}".format(
            name, arr.mean(), arr.std(), arr.min(), arr.max()))
    print("(single_* 행은 결합 없이 보정된 단일 모델. 임계값 0.5는 Platt 보정 확률 기준이며 두 오류 비용이 같다는 가정)")


def print_report(results, model_names, soft_voting_log=None):
    n_folds = len(next(iter(results.values())))
    baseline = np.array(results["simple_mean"])

    print("\n=== K-Fold 결과 (fold {}개) ===".format(n_folds))
    print("{:<20} {:>10} {:>10} {:>14} {:>10}".format(
        "방식", "평균 EER", "fold 표준편차", "simple_mean 대비", "부호 일관"))
    for name, values in results.items():
        arr = np.array(values)
        if name == "simple_mean":
            win_str, sign_str = "-", "-"
        else:
            # fold별 직접 비교(paired): 평균/표준편차가 따로 보면 애매해도, 매 fold 같은
            # train/test 분할이라 "이 방식이 simple_mean보다 나은 fold가 몇 개인지"가
            # 훨씬 직접적인 신호다. 전부(또는 대부분) 한쪽으로 쏠려 있으면 표준편차가
            # 커도 신뢰할 만하고, fold마다 승패가 뒤집히면 그게 진짜 "불안정하다"는 신호.
            diff = baseline - arr  # 양수면 이 방식이 simple_mean보다 좋음(EER 더 낮음)
            wins = int((diff > 0).sum())
            win_str = "{}/{} fold".format(wins, n_folds)
            sign_str = "일관" if wins in (0, n_folds) else ("우세" if wins >= n_folds - 1 or wins <= 1 else "혼재")
        print("{:<20} {:>10.4f} {:>10.4f} {:>14} {:>10}".format(
            name, arr.mean(), arr.std(), win_str, sign_str))
    print()
    print("'simple_mean 대비' 열이 n/n(전부 일관) 또는 그에 가까우면, 평균 차이가 fold 표준편차보다")
    print("작아 보여도 실제로는 신뢰할 만한 신호일 가능성이 높습니다. '혼재'면 fold마다 승패가")
    print("뒤집힌다는 뜻이라 — 그 방식은 이 데이터 양으로는 진짜 불안정한 것으로 봐야 합니다.")
    print("이걸 보고 사람이 --final_method로 최종 방식을 고르세요.")

    if soft_voting_log:
        print("\n=== soft_voting 상세 (fold별 탐색된 지수 파라미터 a, 모델별 가중치) ===")
        for i, (a, w) in enumerate(soft_voting_log):
            w_str = ", ".join("{}={:.4f}".format(n, wi) for n, wi in zip(model_names, w))
            print("fold {}: a={:.2f}  {}".format(i + 1, a, w_str))


def fit_final(X_raw, y, model_names, method):
    n_models = len(model_names)
    scalers = [PlattScaler().fit(X_raw[:, i], y) for i in range(n_models)]
    X_cal = np.stack([s.transform(X_raw[:, i]) for i, s in enumerate(scalers)], axis=1)

    calibration = [{"a": float(s.lr.coef_[0][0]), "b": float(s.lr.intercept_[0])} for s in scalers]

    if method == "simple_mean":
        return {"method": "simple_mean", "calibration": calibration}
    if method == "fixed_weighted":
        w = fixed_weights_from_train(X_cal, y, n_models)
        return {"method": "fixed_weighted", "calibration": calibration,
                "weights": dict(zip(model_names, w.tolist()))}
    if method == "soft_voting":
        w, a = soft_voting_weights(X_cal, y, n_models)
        return {"method": "soft_voting", "calibration": calibration,
                "exponent_a": a, "weights": dict(zip(model_names, w.tolist()))}
    if method.startswith("logreg_C"):
        c = float(method[len("logreg_C"):])
        lr = LogisticRegression(C=c, max_iter=1000)  # penalty 기본값 l2
        lr.fit(X_cal, y)
        return {"method": method, "calibration": calibration,
                "logreg_coef": dict(zip(model_names, lr.coef_[0].tolist())),
                "logreg_intercept": float(lr.intercept_[0])}
    raise ValueError("unknown method: {}".format(method))


def main():
    args = get_args()
    if len(args.scores) != len(args.names):
        raise ValueError("--scores와 --names 개수가 다릅니다")
    if args.final_method and not args.out:
        raise ValueError("--final_method를 지정하려면 --out도 함께 지정해야 합니다")

    paths, y, X_raw = load_scores(args.scores, args.names)
    print("track={}, 샘플 수={}, 모델={}".format(args.track, len(paths), args.names))
    print("클래스 분포: real(0)={}, fake(1)={}".format(int((y == 0).sum()), int((y == 1).sum())))

    results, accuracies, soft_voting_log = run_kfold(X_raw, y, args.names, args.n_folds, args.seed, args.c_values)
    print_report(results, args.names, soft_voting_log)
    print_accuracy_report(accuracies)

    if args.final_method:
        final = fit_final(X_raw, y, args.names, args.final_method)
        final["track"] = args.track
        final["models"] = args.names
        final["n_samples"] = int(len(paths))
        with open(args.out, "w") as f:
            json.dump(final, f, indent=2, ensure_ascii=False)
        print("\n최종 방식 '{}'으로 전체 Test pool에 재학습해서 저장 -> {}".format(args.final_method, args.out))
    else:
        print("\n--final_method 지정 안 됨 — K-Fold 비교만 하고 저장은 생략했습니다.")


if __name__ == "__main__":
    main()
