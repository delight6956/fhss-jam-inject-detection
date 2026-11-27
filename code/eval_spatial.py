#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_spatial.py — 2채널 공간 특징의 세션 단위 교차검증 (논문 7.9절, <표 7-9>)

입력 : build_spatial.py 가 만든 특징 CSV (시간·RF·개선 시간·공간 S1~S4·정규화 공간 4개, enroll 열 포함)
절차 : 기준 등록 구간(enroll=1; 세션별 정상 캡처 앞부분)을 평가에서 빼고, 남은 윈도우에서 특징 집합 6개를
       같은 조건으로 비교한다. 결측 대체(공간 표준편차는 상한값 π rad·20 dB) → 학습 집합 기준 z-점수 →
       학습 집합 클래스 균형 → MLP(16, ReLU) → 세션 단위 교차검증(LOSO) × 시드.
       정확도는 시험 세션별 정확도의 평균, 재현율은 세 시험 세션을 합산한 값이다.
출력 : 표준 출력과 --out CSV

사용 예 (저장소 루트에서)
  python code/build_spatial.py --features data/b210_2ch_SP1-SP3/features_2ch.csv \
      --bursts data/b210_2ch_SP1-SP3/bursts_2ch.csv --cal data/b210_2ch_SP1-SP3/cal_S07.json \
      --enroll-sec 2 --out data/b210_2ch_SP1-SP3/features_2ch_sp_enroll2s.csv
  python code/eval_spatial.py --features data/b210_2ch_SP1-SP3/features_2ch_sp_enroll2s.csv --out out_spatial.csv
필요 패키지: numpy pandas scikit-learn
"""
import argparse, os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_sets import TIME, RF, TPLUS, SP, SPN, CL, fill, mlp, balance

S15 = TIME + RF + TPLUS
SETS = [("시간(5)", TIME), ("RF(5)", RF), ("시간+RF(10)", TIME + RF), ("시간+RF+개선 시간(15)", S15),
        ("15 + 공간(4)", S15 + SP), ("15 + 공간(4) + 정규화 공간(4)", S15 + SP + SPN), ("RF(5) + 공간(4)", RF + SP)]


def loso(d, cols, seeds):
    y = d.label.map({c: i for i, c in enumerate(CL)}).values
    X = d[cols].values.astype(float); idx = np.arange(len(d)); accs = []; cm = np.zeros((3, 3))
    for s in sorted(d.session.unique()):
        tr = idx[(d.session != s).values]; te = idx[(d.session == s).values]; acc = 0.0
        for seed in range(seeds):
            trb = balance(tr, y, np.random.default_rng(seed))
            p = mlp(seed).fit(X[trb], y[trb]).predict(X[te])
            acc += (p == y[te]).mean() / seeds
            for t, q in zip(y[te], p):
                cm[t, q] += 1 / seeds
        accs.append(acc)
    return float(np.mean(accs)) * 100, cm.diagonal() / cm.sum(axis=1) * 100


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", required=True); ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    d = pd.read_csv(a.features)
    d = d[d.label.isin(CL)]
    n_all = len(d)
    if "enroll" in d.columns:
        d = d[d.enroll.fillna(0) == 0]
    d = fill(d.copy(), "upper").reset_index(drop=True)
    print(f"윈도우 {n_all}개 중 기준 등록 구간 {n_all - len(d)}개 제외 → 평가 {len(d)}개 {d.label.value_counts().to_dict()}, 시드 {a.seeds}")
    rows = []
    for name, cols in SETS:
        acc, r = loso(d, cols, a.seeds)
        rows.append({"특징 집합": name, "특징 수": len(cols), "정확도": round(acc, 1), "매크로 재현율": round(float(r.mean()), 1),
                     "정상 재현율": round(r[0], 1), "재밍 재현율": round(r[1], 1), "하이재킹 재현율": round(r[2], 1)})
        print("  ".join(f"{k} {v}" for k, v in rows[-1].items()), flush=True)
    if a.out:
        pd.DataFrame(rows).to_csv(a.out, index=False, encoding="utf-8-sig")
        print(f"→ {a.out}")


if __name__ == "__main__":
    main()
