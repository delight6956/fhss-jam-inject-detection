#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calibrate_phase.py — 2채널 수신기의 채널 간 고정 위상·이득 오프셋(φ_cal) 측정과 재현성 확인 (시험 B 1단계)

분배기로 하나의 신호를 두 채널에 동시에 넣은 캡처(파일 2개)에서 버스트별 위상차·전력비를 구하고,
  φ_cal(f) = φ0 + 2π·τ·f   (f: 캡처 중심 기준 주파수, τ: 두 경로의 지연 차)
  ρ_cal(f) = ρ0 + 기울기·f
를 맞춘다. 조종기 신호(FHSS)를 감쇠기·분배기로 넣으면 도약 주파수 전체에서 한 번에 측정된다.
단일 톤이면 τ를 구할 수 없으므로 τ = 0 으로 둔다(경고 출력).

합격 기준(기본): 잔차 원형 표준편차 ≤ 5°, 시간 구간별 드리프트 ≤ 5°, 재초기화 전후 차이 ≤ 5°

사용 예
  python calibrate_phase.py --file cal_a.00.sc16 --file2 cal_a.01.sc16 --fs 20e6 --out cal_S07.json
  python calibrate_phase.py --compare cal_S07.json cal_S07_b.json        # 재초기화 재현성
필요 패키지: numpy  (extract_features_v3.py 와 같은 폴더)
"""
import argparse, json, os, sys, time
import numpy as np
import extract_features_v3 as ex

LIM_DEG = 5.0


def wrap(x):
    return (np.asarray(x) + np.pi) % (2 * np.pi) - np.pi


def circ_mean(x):
    return float(np.angle(np.mean(np.exp(1j * np.asarray(x)))))


def circ_std(x):
    R = np.abs(np.mean(np.exp(1j * np.asarray(x))))
    return float(np.sqrt(-2 * np.log(max(R, 1e-12))))


def fit(f, phi):
    """φ ≈ φ0 + 2πτf 를 단위원 위에서 맞춤. 반환 (φ0, τ)."""
    if np.ptp(f) < 1e6:                       # 주파수 범위가 1 MHz 미만이면 지연은 구하지 않음
        return circ_mean(phi), 0.0, False
    taus = np.linspace(-60e-9, 60e-9, 2401)   # ±60 ns (케이블 길이 차 약 ±12 m 상당)
    z = np.exp(1j * (phi[None, :] - 2 * np.pi * taus[:, None] * f[None, :]))
    score = np.abs(z.mean(axis=1))
    i = int(np.argmax(score))
    lo, hi = taus[max(0, i - 1)], taus[min(len(taus) - 1, i + 1)]
    fine = np.linspace(lo, hi, 401)
    z = np.exp(1j * (phi[None, :] - 2 * np.pi * fine[:, None] * f[None, :]))
    j = int(np.argmax(np.abs(z.mean(axis=1))))
    return float(np.angle(z[j].mean())), float(fine[j]), True


def phi_cal(cal, f):
    return cal["phi0_rad"] + 2 * np.pi * cal["tau_s"] * np.asarray(f)


def rho_cal(cal, f):
    return cal["rho0_db"] + cal["rho_slope_db_per_hz"] * np.asarray(f)


def measure(a):
    fs_meta, _, _ = ex.sigmf_meta(a.file)
    fs = a.fs or fs_meta or 20e6
    _, br = ex.process_file(a.file, "cal", "splitter", "CAL", fs, 100.0, ex.parse_bands("", fs), False, a.fmt,
                            path2=a.file2, verbose=False)
    ok = [b for b in br if b["coh"] != "" and float(b["coh"]) >= a.min_coh]
    if len(ok) < 5:
        sys.exit(f"[오류] 쓸 수 있는 버스트가 {len(ok)}개뿐입니다(전체 {len(br)}개). 신호 세기·이득·분배기 연결을 확인하세요.")
    f = np.array([float(b["f_off_hz"]) for b in ok]); phi = np.array([float(b["dphi_rad"]) for b in ok])
    rho = np.array([float(b["rho_db"]) for b in ok]); t = np.array([float(b["t_s"]) for b in ok])
    phi0, tau, has_tau = fit(f, phi)
    res = wrap(phi - (phi0 + 2 * np.pi * tau * f))
    if has_tau:
        slope, rho0 = np.polyfit(f, rho, 1)
    else:
        slope, rho0 = 0.0, float(np.median(rho))
    blocks = []
    for b0 in np.arange(0, t.max() + 1e-9, a.block_sec):
        m = (t >= b0) & (t < b0 + a.block_sec)
        if m.sum() >= 3:
            blocks.append(np.degrees(circ_mean(res[m])))
    drift = float(np.max(np.abs(blocks))) if blocks else float("nan")
    cal = dict(phi0_rad=phi0, tau_s=tau, rho0_db=float(rho0), rho_slope_db_per_hz=float(slope),
               n_bursts=len(ok), n_bursts_all=len(br), resid_std_deg=float(np.degrees(circ_std(res))),
               rho_resid_std_db=float(np.std(rho - (rho0 + slope * f))), drift_max_deg=drift,
               f_min_hz=float(f.min()), f_max_hz=float(f.max()), fs=fs, has_delay_fit=bool(has_tau),
               file=os.path.basename(a.file), file2=os.path.basename(a.file2),
               created=time.strftime("%Y-%m-%d %H:%M:%S"))
    json.dump(cal, open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"버스트 {len(ok)}개 사용(전체 {len(br)}개), 주파수 범위 {f.min()/1e6:+.1f} ~ {f.max()/1e6:+.1f} MHz")
    print(f"φ0 = {np.degrees(phi0):+.1f}°, τ = {tau*1e9:+.2f} ns" + ("" if has_tau else "  (단일 주파수: τ = 0 으로 둠)"))
    print(f"ρ0 = {rho0:+.2f} dB, 기울기 = {slope*1e6:+.3f} dB/MHz, 전력비 잔차 표준편차 = {cal['rho_resid_std_db']:.2f} dB")
    verdict = lambda v: "합격" if v <= LIM_DEG else "불합격"
    print(f"위상 잔차 원형 표준편차 = {cal['resid_std_deg']:.2f}°  [{verdict(cal['resid_std_deg'])}, 기준 ≤ {LIM_DEG}°]")
    print(f"시간 구간({a.block_sec:g} s)별 드리프트 최대 = {drift:.2f}°  [{verdict(drift)}, 기준 ≤ {LIM_DEG}°]")
    if not has_tau:
        print("[주의] 주파수 범위가 좁아 지연을 맞추지 못했습니다. 조종기 FHSS 신호로 다시 측정하는 것을 권합니다.")
    print(f"→ {a.out}")


def compare(p1, p2):
    c1, c2 = json.load(open(p1, encoding="utf-8")), json.load(open(p2, encoding="utf-8"))
    lo = max(c1["f_min_hz"], c2["f_min_hz"]); hi = min(c1["f_max_hz"], c2["f_max_hz"])
    f = np.linspace(lo, hi, 51) if hi > lo else np.array([0.0])
    d = np.degrees(wrap(phi_cal(c1, f) - phi_cal(c2, f)))
    dr = rho_cal(c1, f) - rho_cal(c2, f)
    mx = float(np.max(np.abs(d)))
    print(f"φ_cal 차이: 중심 {d[len(d)//2]:+.2f}°, 대역 내 최대 {mx:.2f}°  [{'합격' if mx <= LIM_DEG else '불합격'}, 기준 ≤ {LIM_DEG}°]")
    print(f"τ: {c1['tau_s']*1e9:+.2f} ns 대 {c2['tau_s']*1e9:+.2f} ns,  ρ_cal 차이 최대 {np.max(np.abs(dr)):.2f} dB")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file"); ap.add_argument("--file2")
    ap.add_argument("--fs", type=float, default=None); ap.add_argument("--fmt", choices=["cs8", "ci16", "cf32"])
    ap.add_argument("--min-coh", type=float, default=0.9, help="쓸 버스트의 최소 채널 간 상관")
    ap.add_argument("--block-sec", type=float, default=1.0)
    ap.add_argument("--out", default="cal.json")
    ap.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"))
    a = ap.parse_args()
    if a.compare:
        compare(*a.compare)
    elif a.file and a.file2:
        measure(a)
    else:
        ap.error("--file 과 --file2, 또는 --compare A.json B.json 필요")


if __name__ == "__main__":
    main()
