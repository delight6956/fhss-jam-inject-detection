#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_sets.py — 특징 집합별 세션 단위 교차검증 (시험 A·B·C 공통 평가)

입력 : extract_features_v3.py(및 build_spatial.py) 가 만든 특징 CSV. --glob 으로 여러 파일을 모은다.
       band 열(서브밴드)과 win_ms 열(윈도우 길이)이 여러 값이면 그 조합마다 따로 학습·평가한다.
절차 : 논문 5.5절과 같음 — 결측 대체 → 학습 집합 기준 z-점수 → 학습 집합 클래스 균형(복원 추출) →
       MLP(16, ReLU) → 세션 단위 교차검증(LOSO) × 시드. 세션이 1개면 윈도우 단위 5겹(예비 확인용)으로 대신한다.
       label 이 normal/jamming/hijack 이 아닌 행(background, cal)과 enroll=1 행(공간 기준 등록 구간)은 뺀다.

특징 집합(해당 열이 CSV에 있을 때만 평가)
  time        시간 5개(T1~T5)                    rf          RF 5개(R1~R5)             fused       시간+RF 10개
  time_plus   시간 5개 + 개선 시간 특징 5개      fused_plus  개선 시간 특징을 더한 15개
  fused14     10개 + 공간 4개(논문 정의)         fused14n    10개 + 공간 4개(주파수별 기준을 뺀 값)
  spatial / spatial_n  공간 특징만

출력 : <out>/t_sets.csv(집합별 지표), t_mcnemar.csv(짝 비교), fig_bandwidth.png / fig_window.png(해당 시), report.txt

사용 예
  python evaluate_sets.py --glob "features_S0*_bands.csv" --out results_A        # 시험 A: 대역폭별
  python evaluate_sets.py --glob "features_S0*_w*.csv" --out results_C           # 시험 C: 윈도우 길이별
  python evaluate_sets.py --glob "features_2ch_sp.csv" --out results_B           # 시험 B: 10차원 대 14차원
필요 패키지: numpy pandas scikit-learn matplotlib (scipy 있으면 정확 이항 검정)
"""
import argparse, glob, os, json, warnings
import numpy as np, pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings("ignore")

TIME = ["n_bursts", "dtoa_mean_ms", "dtoa_std_ms", "dur_median_ms", "n_fragments"]
RF = ["noise_floor_db", "occupancy", "env_cv_max", "env_cv_p90", "cw_dwell_max"]
TPLUS = ["ul_n_bursts", "ul_dtoa_mean_ms", "ul_dtoa_std_ms", "grid_frac", "grid_dev_ms"]
SP = ["sp_phase_mean", "sp_phase_std", "sp_rho_med", "sp_rho_std"]
SPN = ["sp_phase_dev_n", "sp_phase_std_n", "sp_rho_dev_n", "sp_rho_std_n"]
SETS = {"time": TIME, "rf": RF, "fused": TIME + RF, "time_plus": TIME + TPLUS, "fused_plus": TIME + TPLUS + RF,
        "spatial": SP, "spatial_n": SPN, "fused14": TIME + RF + SP, "fused14n": TIME + RF + SPN}
KO = {"time": "시간 단독", "rf": "RF 단독", "fused": "융합(10)", "time_plus": "개선 시간", "fused_plus": "융합+개선 시간(15)",
      "spatial": "공간 단독", "spatial_n": "공간 단독(기준 차감)", "fused14": "융합+공간(14)", "fused14n": "융합+공간(14, 기준 차감)"}
PAIRS = [("rf", "fused"), ("time", "time_plus"), ("fused", "fused_plus"), ("fused", "fused14"), ("fused", "fused14n")]
CL = ["normal", "jamming", "hijack"]
# 결측 대체: DTOA 계열은 윈도우 길이, 길이·CV·비율은 0, 격자 편차는 최댓값(10 ms), 공간 평균은 0·표준편차는 상한
FILL0 = ["dur_median_ms", "env_cv_max", "env_cv_p90", "grid_frac", "sp_phase_mean", "sp_rho_med", "sp_phase_dev_n", "sp_rho_dev_n"]
FILLW = ["dtoa_mean_ms", "dtoa_std_ms", "ul_dtoa_mean_ms", "ul_dtoa_std_ms"]
FILLV = {"grid_dev_ms": 10.0, "sp_phase_std": np.pi, "sp_phase_std_n": np.pi, "sp_rho_std": 20.0, "sp_rho_std_n": 20.0}


def fill(d, sp_std_fill="upper"):
    d = d.copy()
    w = d["win_ms"] if "win_ms" in d.columns else 100.0
    for c in FILL0:
        if c in d.columns:
            d[c] = d[c].fillna(0.0)
    for c in FILLW:
        if c in d.columns:
            d[c] = d[c].fillna(w)
    for c, v in FILLV.items():
        if c in d.columns:
            d[c] = d[c].fillna(0.0 if (sp_std_fill == "zero" and c.startswith("sp_")) else v)
    return d


def mlp(seed):
    return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(16,), activation="relu", solver="adam",
                         learning_rate_init=1e-3, alpha=1e-3, max_iter=2000, random_state=seed))


def balance(ix, y, rng):
    n = max((np.sum(y[ix] == k) for k in range(3)), default=0); out = []
    for k in range(3):
        ik = ix[y[ix] == k]
        if ik.size:
            out.append(np.concatenate([ik, rng.choice(ik, n - ik.size, replace=True)]))
    return np.concatenate(out) if out else ix


def evaluate(dfb, cols, cv, n_seeds):
    """반환: 지표 dict + 윈도우별 다수결 예측(짝 비교용)"""
    y = dfb.label.map({c: i for i, c in enumerate(CL)}).values
    X = dfb[cols].values.astype(float)
    idx = np.arange(len(dfb))
    sess = sorted(dfb.session.unique())
    use_loso = cv == "loso" and len(sess) >= 2
    votes = np.zeros((len(dfb), 3)); cm = np.zeros((3, 3)); fold_acc = {}
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        if use_loso:
            folds = [(idx[(dfb.session != s).values], idx[(dfb.session == s).values], s) for s in sess]
        else:
            skf = StratifiedKFold(5, shuffle=True, random_state=seed)
            folds = [(tr, te, f"k{i}") for i, (tr, te) in enumerate(skf.split(X, y))]
        for tr, te, name in folds:
            trb = balance(tr, y, rng)
            p = mlp(seed).fit(X[trb], y[trb]).predict(X[te])
            fold_acc.setdefault(name, []).append(np.mean(p == y[te]))
            for a_, b_ in zip(y[te], p):
                cm[a_, b_] += 1
            votes[te, p] += 1
    acc_f = np.array([np.mean(v) for v in fold_acc.values()]) * 100
    cm /= n_seeds
    rec = np.diag(cm) / np.maximum(cm.sum(axis=1), 1e-9) * 100
    prec = np.diag(cm) / np.maximum(cm.sum(axis=0), 1e-9) * 100
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    return dict(acc=acc_f.mean(), acc_sd=acc_f.std(ddof=0), macro_recall=rec.mean(), macro_f1=f1.mean(),
                rec_normal=rec[0], rec_jam=rec[1], rec_hijack=rec[2], prec_hijack=prec[2],
                fpr_normal_to_hijack=cm[0, 2] / max(cm[0].sum(), 1e-9) * 100,
                cv="LOSO" if use_loso else "window-5fold"), (votes.argmax(axis=1) == y)


def mcnemar(ca, cb):
    """ca, cb: 윈도우별 정답 여부(bool). b = A만 맞음, c = B만 맞음. 정확 이항 양측 p."""
    b = int(np.sum(ca & ~cb)); c = int(np.sum(~ca & cb))
    if b + c == 0:
        return b, c, 1.0
    try:
        from scipy.stats import binomtest
        p = binomtest(b, b + c, 0.5).pvalue
    except Exception:
        from math import erf, sqrt
        z = (abs(b - c) - 1) / sqrt(b + c); p = 2 * (1 - 0.5 * (1 + erf(z / sqrt(2))))
    return b, c, float(p)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glob", required=True); ap.add_argument("--out", default="results")
    ap.add_argument("--cv", choices=["loso", "window"], default="loso")
    ap.add_argument("--seeds", type=int, default=15)
    ap.add_argument("--sets", default="", help="쉼표 구분(비우면 열이 있는 집합 전부)")
    ap.add_argument("--sp-std-fill", choices=["upper", "zero"], default="upper",
                    help="버스트가 2개 미만이라 공간 표준편차(S2·S4)가 없는 윈도우의 대체값: upper = 상한(π rad, 20 dB; 논문 4.5.2항), zero = 0")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    files = sorted(glob.glob(a.glob))
    if not files:
        raise SystemExit(f"[오류] 일치하는 파일 없음: {a.glob}")
    raw = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    n0 = len(raw)
    raw = raw[raw.label.isin(CL)]
    if "enroll" in raw.columns:
        raw = raw[raw.enroll.fillna(0) == 0]
    for c, v in (("band", "full"), ("win_ms", 100.0)):
        if c not in raw.columns:
            raw[c] = v
    raw["win_ms"] = raw["win_ms"].round(0)
    d = fill(raw, a.sp_std_fill).reset_index(drop=True)
    rep = []
    def log(s=""):
        print(s, flush=True); rep.append(s)
    log(f"파일 {len(files)}개, 윈도우 {len(d)}개(제외 {n0 - len(d)}개), 세션 {sorted(d.session.unique())}, "
        f"클래스별 {d.label.value_counts().to_dict()}")
    want = [s.strip() for s in a.sets.split(",") if s.strip()] or list(SETS)
    sets = {k: v for k, v in SETS.items() if k in want and all(c in raw.columns and raw[c].notna().any() for c in v)}
    log(f"평가 집합: {list(sets)}")

    rows, mrows = [], []
    for (band, win), g in d.groupby(["band", "win_ms"], sort=False):
        g = g.reset_index(drop=True)
        bw = g.band_width_hz.iloc[0] / 1e6 if "band_width_hz" in g.columns else np.nan
        corr = {}
        for name, cols in sets.items():
            r, corr[name] = evaluate(g, cols, a.cv, a.seeds)
            rows.append(dict(band=band, 대역폭MHz=round(bw, 1), win_ms=win, 집합=name, 이름=KO[name], 특징수=len(cols),
                             **{k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items()}))
            log(f"[{band} {bw:.0f} MHz, {win:.0f} ms] {KO[name]:20s} 정확도 {r['acc']:.1f} ({r['acc_sd']:.1f})  매크로 재현율 {r['macro_recall']:.1f}  "
                f"재현율 정상 {r['rec_normal']:.1f} / 재밍 {r['rec_jam']:.1f} / 하이재킹 {r['rec_hijack']:.1f}  "
                f"하이재킹 정밀도 {r['prec_hijack']:.1f}  정상 오탐 {r['fpr_normal_to_hijack']:.1f}  [{r['cv']}]")
        for x, y_ in PAIRS:
            if x in corr and y_ in corr:
                b, c, p = mcnemar(corr[x], corr[y_])
                mrows.append(dict(band=band, win_ms=win, A=KO[x], B=KO[y_], A만_맞음=b, B만_맞음=c, p값=round(p, 4)))
                log(f"    McNemar {KO[x]} 대 {KO[y_]}: A만 맞음 {b}, B만 맞음 {c}, p = {p:.4f}")
    T = pd.DataFrame(rows); T.to_csv(os.path.join(a.out, "t_sets.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(mrows).to_csv(os.path.join(a.out, "t_mcnemar.csv"), index=False, encoding="utf-8-sig")
    log("※ McNemar p값은 윈도우를 독립으로 본 값이다. 연속 윈도우는 서로 상관되므로 실제보다 작게 나올 수 있다.")

    try:
        import matplotlib
        matplotlib.use("Agg"); import matplotlib.pyplot as plt
        from matplotlib import font_manager
        for cand in ["Malgun Gothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP", "AppleGothic"]:
            if any(cand == f.name for f in font_manager.fontManager.ttflist):
                plt.rcParams["font.family"] = cand; break
        plt.rcParams["axes.unicode_minus"] = False

        def curve(tt, xcol, xlabel, fname):
            fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
            for name in [s for s in ["time", "time_plus", "rf", "fused", "fused_plus"] if s in sets]:
                q = tt[tt.집합 == name].sort_values(xcol)
                ax[0].plot(q[xcol], q.macro_recall, "o-", label=KO[name])
                ax[1].plot(q[xcol], q.rec_hijack, "o-", label=KO[name])
            for i, yl in enumerate(["매크로 평균 재현율 (%)", "하이재킹 재현율 (%)"]):
                ax[i].set_xlabel(xlabel); ax[i].set_ylabel(yl); ax[i].grid(alpha=.3); ax[i].legend(fontsize=8)
            fig.tight_layout(); fig.savefig(os.path.join(a.out, fname), dpi=200); plt.close(fig); log(f"→ {fname}")

        main_b = T[~T.band.astype(str).str.startswith("kci")]
        if main_b.대역폭MHz.nunique() > 1:
            w0 = main_b.win_ms.mode().iloc[0]
            curve(main_b[main_b.win_ms == w0], "대역폭MHz", "관측 대역폭 (MHz)", "fig_bandwidth.png")
        if T.win_ms.nunique() > 1:
            b0 = T.sort_values("대역폭MHz").band.iloc[-1]
            curve(T[T.band == b0], "win_ms", "윈도우 길이 (ms)", "fig_window.png")
    except Exception as e:
        log(f"[그림 생략] {e}")

    # 대역별 시간 특징 품질(정상 윈도우): 윈도우당 버스트 수, 20 ms 정수배 비율
    if "grid_frac" in raw.columns:
        q = raw[raw.label == "normal"].groupby(["band", "win_ms"]).agg(
            윈도우당_버스트수=("n_bursts", "mean"), 업링크_버스트수=("ul_n_bursts", "mean"),
            정수배_비율=("grid_frac", "mean"), 정수배_편차ms=("grid_dev_ms", "mean")).round(3)
        q.to_csv(os.path.join(a.out, "t_time_quality.csv"), encoding="utf-8-sig")
        log("\n정상 윈도우의 시간 특징 품질:\n" + q.to_string())
    open(os.path.join(a.out, "report.txt"), "w", encoding="utf-8").write("\n".join(rep))
    log(f"\n→ {a.out}/ 저장 완료")


if __name__ == "__main__":
    main()
