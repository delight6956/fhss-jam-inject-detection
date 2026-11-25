#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_spatial.py — 버스트별 2채널 측정값 → 윈도우별 공간 특징(S1~S4) 산출 후 특징 CSV에 병합 (시험 B)

입력 : extract_features_v3.py 의 --out(윈도우 특징)과 --bursts-out(버스트 기록), calibrate_phase.py 의 cal.json
출력 : 공간 특징 열이 붙은 특징 CSV

두 가지 공간 특징을 함께 낸다.
  (1) 논문 4.5.2항 정의 그대로(보정만 적용)
        sp_phase_mean  S1 위상차 원형 평균(rad)      sp_phase_std  S2 위상차 원형 표준편차(rad, 상한 π)
        sp_rho_med     S3 전력비 중앙값(dB)          sp_rho_std    S4 전력비 표준편차(dB, 상한 20)
  (2) 도약 주파수별 기준을 뺀 값(…_n)
        실내 다중경로에서는 한 송신원이라도 도약 주파수마다 위상차·전력비가 달라져 (1)의 S2·S4가 정상에서도 커질 수 있다.
        세션마다 정상(근거리) 캡처의 앞 --enroll-sec 초를 기준 등록 구간으로 삼아 주파수별 기준 φ_ref(f), ρ_ref(f)를 만들고,
        각 버스트에서 그 기준을 뺀 잔차로 네 통계를 낸다:
          sp_phase_dev_n(잔차 원형 평균의 크기), sp_phase_std_n, sp_rho_dev_n(잔차 중앙값의 크기), sp_rho_std_n
        공격기의 위치는 세션마다 다르므로 평균·중앙값은 부호를 버리고 기준에서 벗어난 크기만 쓴다.
        등록 구간의 윈도우는 enroll=1 로 표시되어 평가에서 빠진다(등록과 평가를 같은 윈도우로 하지 않기 위함).

사용 예
  python build_spatial.py --features features_2ch.csv --bursts bursts_2ch.csv --cal cal_S07.json --out features_2ch_sp.csv
  python build_spatial.py --features f.csv --bursts b.csv --cal "S07=cal_S07.json,S08=cal_S08.json,S09=cal_S09.json" --out f_sp.csv
필요 패키지: numpy pandas
"""
import argparse, json
import numpy as np, pandas as pd

S2_MAX = float(np.pi); S4_MAX = 20.0
SP = ["sp_phase_mean", "sp_phase_std", "sp_rho_med", "sp_rho_std"]
SPN = ["sp_phase_dev_n", "sp_phase_std_n", "sp_rho_dev_n", "sp_rho_std_n"]


def wrap(x):
    return (np.asarray(x) + np.pi) % (2 * np.pi) - np.pi


def cstats(phi, rho):
    """윈도우 안 버스트들의 (원형 평균, 원형 표준편차, 전력비 중앙값, 전력비 표준편차)"""
    n = len(phi)
    if n == 0:
        return [np.nan] * 4
    z = np.mean(np.exp(1j * phi)); R = min(1.0, float(np.abs(z)))
    s2 = min(S2_MAX, float(np.sqrt(-2 * np.log(max(R, 1e-12))))) if n >= 2 else np.nan
    s4 = min(S4_MAX, float(np.std(rho, ddof=1))) if n >= 2 else np.nan
    return [float(np.angle(z)), s2, float(np.median(rho)), s4]


def load_cal(spec):
    if "=" not in spec:
        return {"*": json.load(open(spec, encoding="utf-8"))}
    out = {}
    for item in spec.split(","):
        k, v = item.split("=")
        out[k.strip()] = json.load(open(v.strip(), encoding="utf-8"))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", required=True); ap.add_argument("--bursts", required=True)
    ap.add_argument("--cal", required=True, help="cal.json 하나, 또는 '세션=파일,세션=파일'")
    ap.add_argument("--out", required=True)
    ap.add_argument("--enroll-sec", type=float, default=10.0, help="세션별 정상 캡처 앞부분을 기준 등록에 쓰는 길이(초)")
    ap.add_argument("--enroll-subtype", default="near")
    ap.add_argument("--min-coh", type=float, default=0.5, help="공간 특징에 쓸 버스트의 최소 채널 간 상관")
    ap.add_argument("--nb-hz", type=float, default=1.2e6, help="주파수별 기준을 만들 때 묶는 이웃 범위(±Hz)")
    a = ap.parse_args()

    F = pd.read_csv(a.features); B = pd.read_csv(a.bursts)
    cals = load_cal(a.cal)
    B = B[B.coh.notna()].copy()
    n_all = len(B)
    B = B[B.coh >= a.min_coh].copy()
    print(f"버스트 {n_all}개 중 상관 {a.min_coh} 이상 {len(B)}개 사용")

    # 보정 적용
    parts = []
    for ses, g in B.groupby("session"):
        c = cals.get(ses, cals.get("*"))
        if c is None:
            raise SystemExit(f"[오류] 세션 {ses} 의 보정 파일이 없습니다.")
        g = g.copy()
        g["phi_c"] = wrap(g.dphi_rad - (c["phi0_rad"] + 2 * np.pi * c["tau_s"] * g.f_off_hz))
        g["rho_c"] = g.rho_db - (c["rho0_db"] + c["rho_slope_db_per_hz"] * g.f_off_hz)
        parts.append(g)
    B = pd.concat(parts, ignore_index=True)

    # 주파수별 기준(세션 × band 단위)
    B["phi_n"] = np.nan; B["rho_n"] = np.nan
    B["enroll"] = ((B.label == "normal") & (B.subtype == a.enroll_subtype) & (B.t_s < a.enroll_sec)).astype(int)
    for (ses, band), g in B.groupby(["session", "band"]):
        e = g[g.enroll == 1]
        if len(e) < 5:
            print(f"[주의] {ses}/{band}: 기준 등록 버스트가 {len(e)}개뿐이라 주파수별 기준을 만들지 못함(…_n 열은 비움)")
            continue
        ef = e.f_off_hz.values; ez = np.exp(1j * e.phi_c.values); er = e.rho_c.values
        f = g.f_off_hz.values
        near = np.abs(f[:, None] - ef[None, :]) <= a.nb_hz
        cnt = near.sum(axis=1)
        zref = (near * ez[None, :]).sum(axis=1)
        rref = np.array([np.median(er[m]) if m.any() else np.nan for m in near])
        ok = cnt >= 1
        B.loc[g.index[ok], "phi_n"] = wrap(g.phi_c.values[ok] - np.angle(zref[ok]))
        B.loc[g.index[ok], "rho_n"] = g.rho_c.values[ok] - rref[ok]
        print(f"{ses}/{band}: 등록 버스트 {len(e)}개, 기준이 잡힌 버스트 {ok.mean()*100:.0f}%")

    # 윈도우 집계
    key = ["session", "file", "band", "win_idx"]
    recs = []
    for k, g in B.groupby(key):
        s = cstats(g.phi_c.values, g.rho_c.values)
        gn = g[g.phi_n.notna()]
        sn = cstats(gn.phi_n.values, gn.rho_n.values)
        sn[0], sn[2] = abs(sn[0]), abs(sn[2])      # 기준에서 벗어난 크기(부호는 세션의 배치에 따라 달라지므로 쓰지 않음)
        recs.append(list(k) + s + sn + [len(g)])
    W = pd.DataFrame(recs, columns=key + SP + SPN + ["sp_n"])
    out = F.merge(W, on=key, how="left")
    out["sp_n"] = out["sp_n"].fillna(0).astype(int)
    out["enroll"] = ((out.label == "normal") & (out.subtype == a.enroll_subtype) & (out.t_start_s < a.enroll_sec)).astype(int)
    out.to_csv(a.out, index=False, encoding="utf-8")

    pd.set_option("display.width", 200)
    ev = out[out.enroll == 0]
    print("\n상태별 공간 특징 평균(등록 구간 제외, 버스트 2개 이상인 윈도우):")
    print(ev[ev.sp_n >= 2].groupby(["label", "subtype"])[SP + SPN].mean().round(3).to_string())
    print(f"\n→ {a.out}  (윈도우 {len(out)}개, 등록 구간 {int(out.enroll.sum())}개)")


if __name__ == "__main__":
    main()
