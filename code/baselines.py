#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
baselines.py — 기존 기법과의 정량 비교 (시험 D, 중간심사 M6)

같은 특징 CSV(extract_features_v3.py 출력)와 같은 세션 단위 교차검증으로 다음을 비교한다.
  1) 에너지 검출        : 관측 대역 평균 전력(band_power_db)이 임계를 넘으면 공격
  2) 잡음바닥 임계      : 잡음바닥(noise_floor_db)이 학습 세션 정상 중앙값 + 3 dB 를 넘으면 공격(논문 6.11절과 같은 검출기)
  3) STFT+자기상관      : 슬라이스 활동 계열의 20 ms 배수 지연 자기상관(acf_frame)이 임계보다 낮으면 이상
  4) 제안(융합 MLP)     : 10개 특징 3클래스 분류기가 재밍 또는 하이재킹으로 판정하면 공격
1)과 3)의 임계는 학습 세션 정상 윈도우에서 목표 오탐률(--alpha, 기본 1 %)이 되도록 정한다(시험 세션은 쓰지 않음).
1)~3)은 공격 유무만 판정하는 이진 검출기이므로, 공격 종류별 탐지율과 정상 오탐률로 비교한다.

※ 3)은 Kaplan 등[20]의 STFT+자기상관을 그대로 구현한 것이 아니라, 같은 원리(도약 주기성의 자기상관)를
   윈도우 단위 이상 검출기로 옮긴 것이다. 논문에는 이 점을 밝혀 적는다.
※ CFO 기반 물리계층 인증은 복조·동기 획득이 필요하여 이 스크립트에는 포함하지 않았다.

출력 : <out>/t_baselines.csv, report.txt
사용 예
  python baselines.py --glob "features_S0*_bands.csv" --band kci20 --out results_D
필요 패키지: numpy pandas scikit-learn
"""
import argparse, glob, os, warnings
import numpy as np, pandas as pd
from evaluate_sets import TIME, RF, CL, fill, mlp, balance
warnings.filterwarnings("ignore")

ORDER = [("normal", "near"), ("normal", "far"), ("jamming", "barrage_lo"), ("jamming", "barrage_hi"),
         ("jamming", "cw"), ("hijack", "inject")]
KO = {"near": "정상(근거리) 오탐", "far": "정상(약전계) 오탐", "barrage_lo": "광대역(약) 탐지", "barrage_hi": "광대역(강) 탐지",
      "cw": "CW 탐지", "inject": "하이재킹 탐지"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glob", required=True); ap.add_argument("--out", default="results_D")
    ap.add_argument("--band", default="", help="평가할 band 이름(비우면 CSV의 첫 band)")
    ap.add_argument("--alpha", type=float, default=0.01, help="학습 정상에서의 목표 오탐률")
    ap.add_argument("--seeds", type=int, default=15)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    files = sorted(glob.glob(a.glob))
    if not files:
        raise SystemExit(f"[오류] 일치하는 파일 없음: {a.glob}")
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    d = d[d.label.isin(CL)]
    if "enroll" in d.columns:
        d = d[d.enroll.fillna(0) == 0]
    if "band" in d.columns:
        band = a.band or d.band.iloc[0]
        d = d[d.band == band]
    else:
        band = "full"
    if "win_ms" in d.columns:
        d = d[d.win_ms.round(0) == d.win_ms.round(0).mode().iloc[0]]
    d = fill(d).reset_index(drop=True)
    have = {c: c in d.columns and d[c].notna().any() for c in ("band_power_db", "acf_frame")}
    sess = sorted(d.session.unique())
    if len(sess) < 2:
        raise SystemExit("[오류] 세션이 2개 이상이어야 세션 단위 교차검증이 가능합니다.")
    y = d.label.map({c: i for i, c in enumerate(CL)}).values
    idx = np.arange(len(d))
    rep = []
    def log(s=""):
        print(s, flush=True); rep.append(s)
    log(f"band={band}, 윈도우 {len(d)}개, 세션 {sess}, 목표 오탐률 {a.alpha*100:g} %")

    alarms = {}                                   # 기법 → 윈도우별 공격 판정(평균, 0~1)
    def put(name, te, v):
        alarms.setdefault(name, np.full(len(d), np.nan))[te] = v

    prop = np.zeros(len(d)); prop3 = np.zeros(len(d))
    for s in sess:
        tr = idx[(d.session != s).values]; te = idx[(d.session == s).values]
        trn = tr[y[tr] == 0]
        if have["band_power_db"]:
            thr = np.quantile(d.band_power_db.values[trn], 1 - a.alpha)
            put("에너지 검출", te, (d.band_power_db.values[te] > thr).astype(float))
        thr = np.median(d.noise_floor_db.values[trn]) + 3.0
        put("잡음바닥 임계(+3 dB)", te, (d.noise_floor_db.values[te] > thr).astype(float))
        if have["acf_frame"]:
            acf = d.acf_frame.fillna(-1.0).values
            thr = np.quantile(acf[trn], a.alpha)
            put("STFT+자기상관", te, (acf[te] < thr).astype(float))
        X = d[TIME + RF].values.astype(float)
        for seed in range(a.seeds):
            trb = balance(tr, y, np.random.default_rng(seed))
            p = mlp(seed).fit(X[trb], y[trb]).predict(X[te])
            prop[te] += (p != 0) / a.seeds
            prop3[te] += (p == y[te]) / a.seeds
    alarms["제안(융합 MLP)"] = prop

    rows = []
    for name, al in alarms.items():
        r = {"기법": name}
        for lab, sub in ORDER:
            m = ((d.label == lab) & (d.subtype == sub)).values
            r[KO[sub]] = round(float(np.nanmean(al[m])) * 100, 1) if m.any() else np.nan
        m0 = (d.label == "normal").values
        r["정상 오탐(전체)"] = round(float(np.nanmean(al[m0])) * 100, 1)
        r["공격 종류 구분"] = "가능(3클래스)" if name.startswith("제안") else "불가(이진)"
        rows.append(r)
    T = pd.DataFrame(rows); T.to_csv(os.path.join(a.out, "t_baselines.csv"), index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    log(T.to_string(index=False))
    log(f"\n제안 기법의 3클래스 정확도(참고): {prop3.mean()*100:.1f} %")
    for c, ok in have.items():
        if not ok:
            log(f"[주의] {c} 열이 없어 해당 기법을 건너뜀 — extract_features_v3.py 로 다시 추출하세요.")
    open(os.path.join(a.out, "report.txt"), "w", encoding="utf-8").write("\n".join(rep))
    log(f"→ {a.out}/t_baselines.csv")


if __name__ == "__main__":
    main()
