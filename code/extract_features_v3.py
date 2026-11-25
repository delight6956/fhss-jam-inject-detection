#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_features_v3.py — I/Q 캡처 → 윈도우별 시간·RF 특징 CSV  (v3: 개선 시간 특징, 2채널 공간 측정, 기존 기법용 열)

v2(extract_features_v2.py)와의 차이
  * 기존 10개 특징(T1~T5, R1~R5)의 정의·임계·버스트 검출·포락선 CV 산출은 v2와 같다(논문 4.2~4.4절).
    1채널 입력이면 v2와 같은 값이 나와야 하며, 기존 HackRF 캡처 1개로 v2 출력과 대조해 확인한다.
  * 개선 시간 특징(시험 C): 규격 길이에 가까운 업링크 버스트(기본 7 ms 이상)만 골라
      ul_n_bursts, ul_dtoa_mean_ms, ul_dtoa_std_ms,
      grid_frac  (업링크 DTOA 중 20 ms 정수배 ±1 ms 이내의 비율),
      grid_dev_ms(업링크 DTOA와 가장 가까운 20 ms 정수배 사이 거리의 평균)
    를 추가로 낸다. --win-ms 로 윈도우 길이를 바꿀 수 있고 win_ms 열에 기록된다.
  * 기존 기법 비교용 열(시험 D): band_power_db(관측 대역 평균 전력, 에너지 검출기),
      acf_frame(슬라이스 활동 계열의 20 ms 배수 지연 자기상관 최대값, STFT+자기상관 방식).
  * 2채널(시험 B): --file2 또는 manifest의 file_ch1 열에 둘째 채널 파일을 주면
      두 채널 전력 합으로 버스트를 검출하고(논문 4.5.2항), 버스트마다
      rho_db = 10log10(P1/P2), dphi_rad = arg(Σ x1·x2*), coh(두 채널 상관 크기)를 구한다.
      보정(φ_cal)과 윈도우 집계는 calibrate_phase.py, build_spatial.py 가 한다.
  * --bursts-out 을 주면 버스트별 기록(시각, 길이, 주파수, 포락선 CV, 2채널이면 rho·dphi·coh)을 CSV로 남긴다.

사용 예
  # 시험 A: B210 56 MS/s 1채널 캡처에서 서브밴드 동시 추출
  python extract_features_v3.py --manifest manifest_S04.csv --out features_S04_bands.csv ^
      --bands "bw16:0:16;bw20:0:20;bw40:0:40;bw56:0:56;kci20:10:20" --emulate-dc
  # 시험 C: 윈도우 길이 200 ms
  python extract_features_v3.py --manifest manifest_S04.csv --out features_S04_w200.csv --bands "bw56:0:56" --win-ms 200
  # 시험 B: 2채널(파일 2개) — manifest에 file_ch1 열
  python extract_features_v3.py --manifest manifest_S07_2ch.csv --out features_S07.csv --bursts-out bursts_S07.csv

manifest.csv 열: file,label,subtype,session,fs,note[,file_ch1]   (fs 비우면 .sigmf-meta → --fs 순으로 결정)
필요 패키지: numpy
"""
import argparse, csv, json, os, sys, time
import numpy as np

THR_DB = 10.0
PERSIST_FRAC = 0.9
GAP_FILL = 2
FREQ_JUMP_BINS = 3
MIN_BURST_MS = 1.0
ENV_BW_HZ = 150e3
ENV_TRIM = 0.10
DC_EXCL = 2
EDGE_FRAC = 0.10
BIN_HZ = 78125.0           # 목표 bin 폭(20 Msps/256)
FRAME_MS = 20.0            # 업링크 프레임 주기
UL_MIN_MS = 7.0            # 업링크 버스트로 보는 최소 길이(규격 9.247 ms)
GRID_TOL_MS = 1.0          # 20 ms 정수배 판정 허용 오차
ACF_MAX_K = 4              # 자기상관 지연: 20 ms × 1..4

COLS = ["session", "label", "subtype", "file", "win_idx", "t_start_s",
        "n_bursts", "dtoa_mean_ms", "dtoa_std_ms", "dur_median_ms", "n_fragments",
        "noise_floor_db", "occupancy", "env_cv_max", "env_cv_p90", "cw_dwell_max",
        "proc_ms", "band", "band_center_hz", "band_width_hz", "fs", "nfft",
        # v3 추가
        "win_ms", "n_ch", "band_power_db", "acf_frame",
        "ul_n_bursts", "ul_dtoa_mean_ms", "ul_dtoa_std_ms", "grid_frac", "grid_dev_ms"]

BURST_COLS = ["session", "label", "subtype", "file", "band", "win_idx", "t_s", "dur_ms",
              "f_off_hz", "env_cv", "rho_db", "dphi_rad", "coh"]


class IQReader:
    """cs8 (int8), ci16 (int16 LE), cf32 (complex64). 정규화: int8/128, int16/32768."""
    def __init__(self, path, fmt=None):
        self.path = path
        ext = os.path.splitext(path)[1].lower()
        if fmt:
            self.fmt = fmt
        elif ext == ".sigmf-data" and sigmf_meta(path)[2]:
            self.fmt = sigmf_meta(path)[2]
        elif ext in (".cfile", ".cf32", ".fc32"):
            self.fmt = "cf32"
        elif ext in (".sc16", ".ci16", ".sigmf-data", ".int16", ".dat"):
            self.fmt = "ci16"
        else:
            self.fmt = "cs8"
        if self.fmt == "cs8":
            self.raw = np.memmap(path, dtype=np.int8, mode="r"); self.n = self.raw.size // 2; self.scale = 128.0
        elif self.fmt == "ci16":
            self.raw = np.memmap(path, dtype="<i2", mode="r"); self.n = self.raw.size // 2; self.scale = 32768.0
        else:
            self.raw = np.memmap(path, dtype=np.complex64, mode="r"); self.n = self.raw.size; self.scale = 1.0

    def read(self, start, count):
        start = max(0, int(start)); count = int(min(count, self.n - start))
        if count <= 0:
            return np.zeros(0, np.complex64)
        if self.fmt == "cf32":
            return np.asarray(self.raw[start:start + count], dtype=np.complex64)
        r = self.raw[2 * start: 2 * (start + count)].astype(np.float32) / self.scale
        return (r[0::2] + 1j * r[1::2]).astype(np.complex64)


def sigmf_meta(path):
    """옆에 있는 .sigmf-meta에서 (fs, fc, fmt) 읽기. 없으면 (None, None, None). fmt: cs8|ci16|cf32"""
    base = path[:-len(".sigmf-data")] if path.endswith(".sigmf-data") else os.path.splitext(path)[0]
    mp = base + ".sigmf-meta"
    if not os.path.exists(mp):
        return None, None, None
    try:
        m = json.load(open(mp, encoding="utf-8"))
        g = m.get("global", {})
        fs = g.get("core:sample_rate")
        fc = (m.get("captures") or [{}])[0].get("core:frequency")
        dt = (g.get("core:datatype") or "").lower()
        fmt = "cs8" if dt.startswith("ci8") else "ci16" if dt.startswith("ci16") else "cf32" if dt.startswith("cf32") else None
        return (float(fs) if fs else None), (float(fc) if fc else None), fmt
    except Exception:
        return None, None, None


def choose_nfft(fs):
    return max(64, int(round(fs / BIN_HZ)))


def band_mask(nfft, fs, center_off_hz, width_hz, emulate_dc):
    """fftshift 순서의 bin 인덱스 k(−N/2..N/2−1)와 관측 bin 마스크 (v2와 동일)."""
    k = np.arange(nfft) - nfft // 2
    df = fs / nfft
    width_hz = min(width_hz, fs)
    if width_hz >= fs and abs(center_off_hz) < df:
        lo, hi = 0, nfft                                  # 전체 대역: 모든 bin
    else:
        k_lo = int(np.ceil((center_off_hz - width_hz / 2) / df))
        k_hi = int(np.floor((center_off_hz + width_hz / 2) / df))
        lo = max(0, k_lo + nfft // 2); hi = min(nfft, k_hi + nfft // 2 + 1)
    ok = np.zeros(nfft, bool); ok[lo:hi] = True
    edge = int((hi - lo) * EDGE_FRAC)
    ok[lo:lo + edge] = False; ok[hi - edge:hi] = False
    ok[np.abs(k) <= DC_EXCL] = False
    if emulate_dc:
        kc = int(round(center_off_hz / df))
        ok[np.abs(k - kc) <= DC_EXCL] = False
    return k, ok


def _narrowband(rd, a, m, dec, f_off, fs):
    x = rd.read(a, m * dec)
    ph = (2 * np.pi * f_off / fs) * np.arange(x.size, dtype=np.float64)
    y = x * np.exp(-1j * ph).astype(np.complex64)
    return y.reshape(m, dec).mean(axis=1)


def envelope_cv(rd, s0, s1, f_off, fs):
    """버스트 포락선 변동계수 (v2와 동일)."""
    n = s1 - s0
    a = s0 + int(n * ENV_TRIM); b = s1 - int(n * ENV_TRIM)
    dec = max(1, int(fs // (2 * ENV_BW_HZ)) // 2 * 2)
    m = (b - a) // dec
    if m < 32:
        return np.nan
    e = np.abs(_narrowband(rd, a, m, dec, f_off, fs))
    mu = e.mean()
    return float(e.std() / mu) if mu > 0 else np.nan


def spatial_pair(rd, rd2, s0, s1, f_off, fs):
    """2채널 버스트의 채널 간 전력비(dB), 위상차(rad), 상관 크기. 보정 전 값."""
    n = s1 - s0
    a = s0 + int(n * ENV_TRIM); b = s1 - int(n * ENV_TRIM)
    dec = max(1, int(fs // (2 * ENV_BW_HZ)) // 2 * 2)
    m = (b - a) // dec
    if m < 32:
        return np.nan, np.nan, np.nan
    y1 = _narrowband(rd, a, m, dec, f_off, fs)
    y2 = _narrowband(rd2, a, m, dec, f_off, fs)
    p1 = float(np.sum(np.abs(y1) ** 2)); p2 = float(np.sum(np.abs(y2) ** 2))
    if p1 <= 0 or p2 <= 0:
        return np.nan, np.nan, np.nan
    c = np.sum(y1 * np.conj(y2))
    return 10 * np.log10(p1 / p2), float(np.angle(c)), float(np.abs(c) / np.sqrt(p1 * p2))


def frame_acf(act, slice_s):
    """슬라이스 활동 계열(bool)의 20 ms 배수 지연 정규화 자기상관 최대값."""
    a = act.astype(np.float64); a -= a.mean()
    den = float(np.dot(a, a))
    if den <= 0:
        return np.nan
    best = -1.0
    for k in range(1, ACF_MAX_K + 1):
        L = int(round(k * FRAME_MS * 1e-3 / slice_s))
        if L >= a.size - 1:
            break
        best = max(best, float(np.dot(a[:-L], a[L:]) / den))
    return best if best > -1.0 else np.nan


def process_file(path, label, subtype, session, fs, win_ms, bands, emulate_dc, fmt=None,
                 path2=None, verbose=True):
    """반환: (윈도우 행 리스트, 버스트 행 리스트)"""
    rd = IQReader(path, fmt)
    rd2 = IQReader(path2, fmt) if path2 else None
    n_samp = min(rd.n, rd2.n) if rd2 else rd.n
    nfft = choose_nfft(fs)
    slice_s = nfft / fs
    win_n = int(round(win_ms * 1e-3 * fs / nfft)) * nfft
    n_win = n_samp // win_n
    if n_win == 0:
        print(f"[경고] {path}: 윈도우 1개보다 짧음", file=sys.stderr); return [], []
    masks = {name: band_mask(nfft, fs, c, w, emulate_dc) for name, c, w in bands}
    hann = np.hanning(nfft).astype(np.float32)

    def spec(reader, w):
        x = reader.read(w * win_n, win_n)
        m = x.size // nfft
        X = x[: m * nfft].reshape(m, nfft) * hann
        return (np.abs(np.fft.fftshift(np.fft.fft(X, axis=1), axes=1)) ** 2).astype(np.float32)

    st = {name: dict(act=[], bin=[], wins=[]) for name, _, _ in bands}
    t_file = time.perf_counter()
    for w in range(n_win):
        t0 = time.perf_counter()
        P = spec(rd, w)
        if rd2 is not None:
            P = P + spec(rd2, w)                         # 두 채널 전력 합으로 검출
        Ps = P.copy(); Ps[:, 1:-1] = (P[:, :-2] + P[:, 1:-1] + P[:, 2:]) / 3     # 3-bin 평균(채널폭 ≈ 3 bin)
        t_spec = (time.perf_counter() - t0) * 1e3
        for name, _, _ in bands:
            t1 = time.perf_counter()
            k, ok = masks[name]
            nf = float(np.median(10 * np.log10(P[::4, ok] + 1e-12)))
            S = 10 * np.log10(Ps[:, ok] + 1e-12)
            nfs = float(np.median(S[::4]))
            A = S > nfs + THR_DB
            dwell = A.mean(axis=0)
            persist = dwell >= PERSIST_FRAC
            occ = float(A.any(axis=1).mean())
            S2 = np.where(persist[None, :], -np.inf, S)
            mx = S2.max(axis=1)
            act_w = mx > nfs + THR_DB
            st[name]["act"].append(act_w)
            st[name]["bin"].append(S2.argmax(axis=1))
            st[name]["wins"].append(dict(nf=nf, occ=occ, cw=float(dwell.max()),
                                         bp=float(10 * np.log10(P[:, ok].mean() + 1e-12)),
                                         acf=frame_acf(act_w, slice_s),
                                         t=t_spec + (time.perf_counter() - t1) * 1e3))
    rows, brows = [], []
    spw = win_n // nfft
    for name, c_hz, w_hz in bands:
        k, ok = masks[name]
        act = np.concatenate(st[name]["act"]); kb = k[ok][np.concatenate(st[name]["bin"])]
        wins = st[name]["wins"]
        idx = np.flatnonzero(act)
        if idx.size:
            gaps = np.diff(idx)
            for i in np.flatnonzero((gaps > 1) & (gaps <= GAP_FILL + 1)):
                act[idx[i] + 1: idx[i + 1]] = True
                kb[idx[i] + 1: idx[i + 1]] = kb[idx[i]]
        segs = []
        d = np.diff(np.concatenate([[0], act.astype(np.int8), [0]]))
        for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            cut = s
            jumps = np.flatnonzero(np.abs(np.diff(kb[s:e])) > FREQ_JUMP_BINS) + s + 1
            for j in list(jumps) + [e]:
                if j > cut:
                    segs.append((cut, j, int(np.median(kb[cut:j]))))
                cut = j
        bursts, frags = [], np.zeros(n_win, int)
        for s, e, b in segs:
            dur_ms = (e - s) * slice_s * 1e3
            w = s // spw
            if w >= n_win:
                continue
            if dur_ms >= MIN_BURST_MS:
                bursts.append((s, e, b, w, dur_ms))
            else:
                frags[w] += 1
        per_w = {w: [] for w in range(n_win)}
        prev_start = None; prev_ul = None
        for s, e, b, w, dur in bursts:
            dt = (s - prev_start) * slice_s * 1e3 if prev_start is not None else np.nan
            prev_start = s
            ul = dur >= UL_MIN_MS
            dt_ul = np.nan
            if ul:
                dt_ul = (s - prev_ul) * slice_s * 1e3 if prev_ul is not None else np.nan
                prev_ul = s
            t0 = time.perf_counter()
            f_off = b * fs / nfft
            cv = envelope_cv(rd, s * nfft, e * nfft, f_off, fs)
            rho = dphi = coh = np.nan
            if rd2 is not None:
                rho, dphi, coh = spatial_pair(rd, rd2, s * nfft, e * nfft, f_off, fs)
            per_w[w].append(dict(dt=dt, dur=dur, cv=cv, ul=ul, dt_ul=dt_ul, t=(time.perf_counter() - t0) * 1e3))
            brows.append(dict(session=session, label=label, subtype=subtype, file=os.path.basename(path), band=name,
                              win_idx=w, t_s=round(s * slice_s, 6), dur_ms=round(dur, 4), f_off_hz=round(f_off, 1),
                              env_cv=_f(cv), rho_db=_f(rho), dphi_rad=_f(dphi), coh=_f(coh)))
        for w in range(n_win):
            L = per_w[w]
            dts = np.array([x["dt"] for x in L if not np.isnan(x["dt"])])
            cvs = np.array([x["cv"] for x in L if not np.isnan(x["cv"])])
            uls = np.array([x["dt_ul"] for x in L if x["ul"] and not np.isnan(x["dt_ul"])])
            if uls.size:
                r = np.mod(uls, FRAME_MS); dev = np.minimum(r, FRAME_MS - r)
                gfrac, gdev = float(np.mean(dev <= GRID_TOL_MS)), float(np.mean(dev))
            else:
                gfrac = gdev = np.nan
            rows.append(dict(
                session=session, label=label, subtype=subtype, file=os.path.basename(path),
                win_idx=w, t_start_s=round(w * win_n / fs, 4),
                n_bursts=len(L),
                dtoa_mean_ms=_f(dts.mean() if dts.size >= 1 else np.nan),
                dtoa_std_ms=_f(dts.std(ddof=1) if dts.size >= 2 else np.nan),
                dur_median_ms=_f(np.median([x["dur"] for x in L]) if L else np.nan),
                n_fragments=int(frags[w]),
                noise_floor_db=_f(wins[w]["nf"]), occupancy=_f(wins[w]["occ"]),
                env_cv_max=_f(cvs.max() if cvs.size else np.nan),
                env_cv_p90=_f(np.percentile(cvs, 90) if cvs.size else np.nan),
                cw_dwell_max=_f(wins[w]["cw"]),
                proc_ms=_f(wins[w]["t"] + sum(x["t"] for x in L)),
                band=name, band_center_hz=round(c_hz, 1), band_width_hz=round(min(w_hz, fs), 1),
                fs=round(fs, 1), nfft=nfft,
                win_ms=round(win_n / fs * 1e3, 3), n_ch=2 if rd2 is not None else 1,
                band_power_db=_f(wins[w]["bp"]), acf_frame=_f(wins[w]["acf"]),
                ul_n_bursts=int(sum(1 for x in L if x["ul"])),
                ul_dtoa_mean_ms=_f(uls.mean() if uls.size >= 1 else np.nan),
                ul_dtoa_std_ms=_f(uls.std(ddof=1) if uls.size >= 2 else np.nan),
                grid_frac=_f(gfrac), grid_dev_ms=_f(gdev),
            ))
        if verbose:
            print(f"{os.path.basename(path)} [{name} {min(w_hz, fs)/1e6:.0f} MHz @ {c_hz/1e6:+.1f}]: "
                  f"{n_win} windows, {len(bursts)} bursts, label={label}/{subtype}, session={session}")
    if verbose:
        print(f"  ({time.perf_counter() - t_file:.1f} s, nfft={nfft}, fs={fs/1e6:.2f} MS/s, 채널 {2 if rd2 is not None else 1})")
    return rows, brows


def _f(v):
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), 5)


def parse_bands(spec, fs):
    """'name:center_offset_MHz:width_MHz;...' → [(name, center_hz, width_hz)]. 비우면 전체 대역 1개."""
    if not spec:
        return [("full", 0.0, fs)]
    out = []
    for item in spec.split(";"):
        item = item.strip()
        if not item:
            continue
        name, c, w = item.split(":")
        out.append((name.strip(), float(c) * 1e6, float(w) * 1e6))
    return out


def read_manifest(path, default_fs):
    """manifest → [(file, label, subtype, session, fs, file_ch1 or None)]"""
    jobs = []
    base = os.path.dirname(os.path.abspath(path))
    full = lambda p: p if os.path.isabs(p) else os.path.join(base, p)
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if not (r.get("file") or "").strip():
                continue
            p = full(r["file"].strip())
            fs_meta, _, _ = sigmf_meta(p)
            fs = float(r["fs"]) if (r.get("fs") or "").strip() else (fs_meta or default_fs or 20e6)
            p2 = full(r["file_ch1"].strip()) if (r.get("file_ch1") or "").strip() else None
            jobs.append((p, r["label"].strip(), (r.get("subtype") or "").strip(), r["session"].strip(), fs, p2))
    uniq = {}
    for j in jobs:                      # 같은 파일이 여러 줄이면 마지막 줄만 사용
        uniq[os.path.normcase(j[0])] = j
    if len(uniq) < len(jobs):
        print(f"[주의] manifest에 중복된 파일 {len(jobs) - len(uniq)}줄을 건너뜀", file=sys.stderr)
    return list(uniq.values())


def _append(path, cols, rows):
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols)
        if new:
            wr.writeheader()
        for r in rows:
            wr.writerow(r)


def main():
    global UL_MIN_MS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest"); ap.add_argument("--file"); ap.add_argument("--file2", help="2채널 캡처의 둘째 채널 파일")
    ap.add_argument("--label"); ap.add_argument("--subtype", default="")
    ap.add_argument("--session", default="S00")
    ap.add_argument("--fs", type=float, default=None, help="샘플레이트(Hz). manifest/sigmf-meta에 없을 때 사용(기본 20e6)")
    ap.add_argument("--win-ms", type=float, default=100.0)
    ap.add_argument("--fmt", choices=["cs8", "ci16", "cf32"])
    ap.add_argument("--bands", default="", help='예: "bw16:0:16;bw20:0:20;bw40:0:40;bw56:0:56;kci20:10:20" (이름:중심오프셋MHz:폭MHz)')
    ap.add_argument("--emulate-dc", action="store_true", help="서브밴드 중심 ±2 bin도 제외(실수신기 DC 제외 흉내)")
    ap.add_argument("--ul-min-ms", type=float, default=UL_MIN_MS, help="업링크 버스트 최소 길이(ms)")
    ap.add_argument("--out", default="features.csv")
    ap.add_argument("--bursts-out", default="", help="버스트별 기록 CSV(2채널 공간 특징 산출에 필요)")
    a = ap.parse_args()
    UL_MIN_MS = a.ul_min_ms

    if a.manifest:
        jobs = read_manifest(a.manifest, a.fs)
    elif a.file and a.label:
        fs_meta, _, _ = sigmf_meta(a.file)
        jobs = [(a.file, a.label, a.subtype, a.session, a.fs or fs_meta or 20e6, a.file2)]
    else:
        ap.error("--manifest 또는 (--file, --label) 필요")

    for p, lab, sub, ses, fs, p2 in jobs:
        bands = parse_bands(a.bands, fs)
        rows, brows = process_file(p, lab, sub, ses, fs, a.win_ms, bands, a.emulate_dc, a.fmt, path2=p2)
        _append(a.out, COLS, rows)
        if a.bursts_out:
            _append(a.bursts_out, BURST_COLS, brows)
    print(f"→ {a.out}" + (f", {a.bursts_out}" if a.bursts_out else ""))


if __name__ == "__main__":
    main()
