#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sim_iq.py — 스크립트 점검용 합성 FHSS I/Q 생성기 (하드웨어 없이 추출·보정·평가 흐름을 끝까지 돌려 보기 위한 것)

실제 조종 링크나 주입기를 모사한 것이 아니라, 논문 3장의 신호 구조(2402 MHz 기준 400 kHz 간격 채널,
20 ms 프레임, 9.247 ms 2-FSK 버스트)만 흉내 낸 시험 신호이다. 여기서 나온 수치는 논문 결과로 쓰지 않는다.

시나리오: normal/near, normal/far, jamming/barrage_lo, jamming/barrage_hi, jamming/cw, hijack/inject, cal(분배기 보정)
2채널(--ch 2): 둘째 채널에 송신원별 주파수 의존 위상·이득과 고정 보정 위상(φ_cal + 지연)을 넣는다.

사용 예
  python sim_iq.py --out simdata --sessions S91,S92,S93 --fs 20e6 --fc 2450.2e6 --sec 3 --ch 2 --fmt cs8
"""
import argparse, csv, os
import numpy as np

F0 = 2402e6; CH_STEP = 400e3; N_CH = 200
FRAME_S = 20e-3; BURST_S = 9.247e-3; DEV = 38.1e3; BIT_S = 26.32e-6

SCEN = [("normal", "near"), ("normal", "far"), ("jamming", "barrage_lo"), ("jamming", "barrage_hi"),
        ("jamming", "cw"), ("hijack", "inject")]


def fsk_burst(n, fs, rng):
    nb = int(np.ceil(n / (BIT_S * fs))) + 1
    bits = rng.integers(0, 2, nb) * 2 - 1
    f = np.repeat(bits, int(np.ceil(BIT_S * fs)))[:n] * DEV
    return np.exp(1j * 2 * np.pi * np.cumsum(f) / fs)


def smooth_profile(rng, scale):
    """주파수(채널 번호)에 따라 완만히 변하는 무작위 함수 — 다중경로에 의한 채널 간 차이를 흉내."""
    a = rng.normal(0, scale, 3); ph = rng.uniform(0, 2 * np.pi, 3); k = np.array([1.0, 2.3, 4.1])
    return lambda ch: float(np.sum(a * np.sin(2 * np.pi * k * ch / N_CH + ph)))


def make(label, sub, fs, fc, sec, n_ch, rng, sess_rng, cal):
    n = int(sec * fs)
    t_len = n
    x = [np.zeros(n, np.complex64) for _ in range(n_ch)]
    amp = dict(near=0.25, far=0.06).get(sub, 0.25)
    if label == "cal":
        amp = 0.25
    # 송신원별 둘째 채널 응답(위상 rad, 이득 dB) — 세션마다 다름
    L_ph, L_g = smooth_profile(sess_rng, 0.9), smooth_profile(sess_rng, 1.5)
    I_ph, I_g = smooth_profile(sess_rng, 0.9), smooth_profile(sess_rng, 1.5)
    L_off, I_off = sess_rng.uniform(-1, 1), sess_rng.uniform(1.2, 2.4) * sess_rng.choice([-1, 1])

    def add(sig, start, f_rf, src):
        """신호를 중심주파수 기준으로 옮겨 두 채널에 더한다."""
        f_off = f_rf - fc
        if abs(f_off) > fs / 2 - 150e3:
            return
        a = max(0, start); b = min(t_len, start + sig.size)
        if b <= a:
            return
        seg = sig[a - start: b - start] * np.exp(1j * 2 * np.pi * f_off * (np.arange(a, b) / fs))
        x[0][a:b] += seg.astype(np.complex64)
        if n_ch == 2:
            ch = (f_rf - F0) / CH_STEP
            if src == "cal":
                ph, g = 0.0, 0.0
            elif src == "legit":
                ph, g = L_off + L_ph(ch), L_g(ch)
            else:
                ph, g = I_off + I_ph(ch), I_g(ch)
            ph_cal = cal["phi0"] + 2 * np.pi * cal["tau"] * f_off        # 수신기·케이블 고정 오프셋
            x[1][a:b] += (seg * (10 ** (g / 20)) * np.exp(1j * (ph + ph_cal))).astype(np.complex64)

    n_frames = int(sec / FRAME_S)
    nb = int(BURST_S * fs)
    t0 = rng.uniform(0, FRAME_S)
    for fr in range(n_frames):
        ch = int(rng.integers(0, N_CH)); f_rf = F0 + ch * CH_STEP
        start = int((t0 + fr * FRAME_S) * fs)
        add(amp * fsk_burst(nb, fs, rng), start, f_rf, "cal" if label == "cal" else "legit")
        if label == "hijack":
            # 덮기(정품 프레임 뒷부분과 겹침, 주파수 차 3 kHz) → 직후 주입 프레임
            cover = int(6e-3 * fs)
            add(0.22 * np.exp(1j * 2 * np.pi * 3e3 * np.arange(cover) / fs), start + nb - cover, f_rf, "inj")
            add(0.22 * fsk_burst(nb, fs, rng), start + nb + int(0.6e-3 * fs), f_rf + 2e3, "inj")
    sigma = 0.004
    if sub == "barrage_lo":
        sigma *= 10 ** (7 / 20)
    if sub == "barrage_hi":
        sigma *= 10 ** (19 / 20)
    for c in range(n_ch):
        x[c] += (rng.normal(0, sigma, n) + 1j * rng.normal(0, sigma, n)).astype(np.complex64)
        if sub == "cw":
            x[c] += (0.2 * np.exp(1j * 2 * np.pi * 1.7e6 * np.arange(n) / fs)).astype(np.complex64)
    return x


def write(path, x, fmt):
    full = 127 if fmt == "cs8" else 32767
    dt = np.int8 if fmt == "cs8" else "<i2"
    out = np.empty(2 * x.size, dtype=dt)
    out[0::2] = np.clip(np.round(x.real * full), -full, full)
    out[1::2] = np.clip(np.round(x.imag * full), -full, full)
    out.tofile(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="simdata"); ap.add_argument("--sessions", default="S91,S92,S93")
    ap.add_argument("--fs", type=float, default=20e6); ap.add_argument("--fc", type=float, default=2450.2e6)
    ap.add_argument("--sec", type=float, default=3.0); ap.add_argument("--ch", type=int, default=1, choices=[1, 2])
    ap.add_argument("--fmt", default="cs8", choices=["cs8", "ci16"])
    ap.add_argument("--scen", default="", help="쉼표 구분 subtype만 생성(예: near,inject). 비우면 전부")
    ap.add_argument("--cal", action="store_true", help="세션마다 분배기 보정 캡처(cal)도 생성")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ext = ".cs8" if a.fmt == "cs8" else ".sc16"
    cal = dict(phi0=0.7, tau=4e-9)        # 모든 세션 공통(같은 장비·케이블)
    rows = []
    want = set(a.scen.split(",")) if a.scen else None
    for si, ses in enumerate(a.sessions.split(",")):
        sess_seed = a.seed * 1000 + si
        scen = [("cal", "splitter")] * a.cal + [s for s in SCEN if want is None or s[1] in want]
        for li, (label, sub) in enumerate(scen):
            rng = np.random.default_rng(sess_seed * 100 + li)
            x = make(label, sub, a.fs, a.fc, a.sec, a.ch, rng, np.random.default_rng(sess_seed), cal)
            names = []
            for c in range(a.ch):
                nm = f"{ses}_{label}_{sub}_ch{c}{ext}" if a.ch == 2 else f"{ses}_{label}_{sub}{ext}"
                write(os.path.join(a.out, nm), x[c], a.fmt); names.append(nm)
            rows.append(dict(file=names[0], label=label, subtype=sub, session=ses, fs=a.fs, note="sim",
                             file_ch1=names[1] if a.ch == 2 else ""))
            print("생성:", names)
    with open(os.path.join(a.out, "manifest.csv"), "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=["file", "label", "subtype", "session", "fs", "note", "file_ch1"])
        wr.writeheader(); wr.writerows(rows)
    print("→", os.path.join(a.out, "manifest.csv"))


if __name__ == "__main__":
    main()
