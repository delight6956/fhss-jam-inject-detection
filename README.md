# FHSS 드론 조종 링크 재밍·신호 주입 탐지 — 특징 데이터와 평가 코드

박사학위 논문 「드론 FHSS 조종 링크의 재밍 및 신호 주입 공격 탐지를 위한 시간·RF·공간 특징 융합 프레임워크」(황진하, 가천대학교, 2027)의 부록 자료입니다. 논문의 표를 재현할 수 있도록 윈도우 단위 특징 CSV와 평가 코드를 담았습니다.

- 원시 I/Q 캡처는 용량(캡처당 0.6~4.8 GB) 때문에 포함하지 않았습니다.
- 공격 장비(신호 주입기)의 구성과 파라미터도 포함하지 않았습니다. 하이재킹 데이터는 수신 측에서 관측한 특징값만 담고 있습니다.

## 구성

```
data/
  hackrf_S01-S03/   HackRF One, 중심 2450.2 MHz, 20 Msps, 세션 S01~S03 (논문 제6장·7.2~7.7절)
    features_v3_w100.csv / w200 / w500   윈도우 100·200·500 ms
    manifest.csv                         캡처 파일 목록(라벨·조건)
  b210_S04-S06/     USRP B210, 중심 2442 MHz, 40 Msps, 세션 S04~S06 (논문 7.8절)
    features_S0x_bands.csv               같은 캡처에서 추출한 서브밴드 bw16/bw20/bw40/kci20
    manifest_S0x.csv
code/
  extract_features_v3.py   I/Q → 윈도우 특징 추출 (제4장 정의)
  evaluate_sets.py         특징 집합별 세션 단위 교차검증(LOSO), McNemar
  baselines.py             기존 기법 비교: 에너지 검출, 잡음바닥 임계, STFT+자기상관 (7.10절)
  calibrate_phase.py, build_spatial.py   2채널 위상 보정과 공간 특징 (4.5절, 7.9절)
  sim_iq.py                검증용 합성 FHSS I/Q 생성
  eval_f1_mcc_ci.py        매크로 F1·MCC와 블록 부트스트랩 95% 신뢰구간
  eval_alarm.py            k-of-n 경보 규칙별 오경보·탐지 지연 (7.4절)
results/                   논문에 실은 결과 CSV
```

## 라벨

| label | subtype | 내용 |
|---|---|---|
| normal | near / far | 정상 링크. 근거리, 약전계(거리 증가·감쇠기) |
| jamming | barrage_lo / barrage_hi / cw | 광대역 재밍(약·강), CW 재밍 |
| hijack | inject | 신호 주입 하이재킹 |
| background, aux | — | 조종기 꺼짐, 보조 캡처(평가에서 제외) |

## 특징 열

| 영역 | 열 | 설명 |
|---|---|---|
| 시간(T1~T5) | n_bursts, dtoa_mean_ms, dtoa_std_ms, dur_median_ms, n_fragments | 윈도우 내 버스트 수, 도착 간격(DTOA) 평균·표준편차, 버스트 지속시간 중앙값, 파편 수 |
| 개선 시간(T6~T10) | ul_n_bursts, ul_dtoa_mean_ms, ul_dtoa_std_ms, grid_frac, grid_dev_ms | 업링크 버스트 수·DTOA, 20 ms 프레임 격자 정합 비율·편차 |
| RF(R1~R5) | noise_floor_db, occupancy, env_cv_max, env_cv_p90, cw_dwell_max | 잡음바닥, 채널 점유율, 포락선 변동계수 최대·90퍼센타일, 협대역 지속률 |
| 기타 | session, file, win_idx, t_start_s, band, band_width_hz, fs, nfft, win_ms, proc_ms 등 | 메타데이터 |

결측값(버스트가 없거나 1개인 윈도우)은 비어 있습니다. 평가 시 대체 규칙은 `evaluate_sets.py`의 `fill()`을 따릅니다.

## 재현

```bash
pip install numpy pandas scikit-learn scipy matplotlib
python code/evaluate_sets.py --glob "data/hackrf_S01-S03/features_v3_w*.csv" --out out_C          # <표 7-7>
python code/evaluate_sets.py --glob "data/b210_S04-S06/features_S0*_bands.csv" --out out_A --seeds 4  # <표 7-8>
python code/eval_f1_mcc_ci.py kci 15 && python code/eval_f1_mcc_ci.py b210 4                     # F1·MCC·CI
python code/eval_alarm.py                                                                         # 경보 성능
```

MLP(은닉 16, ReLU)와 시드 설정이 논문과 같으므로, scikit-learn 버전 차이에 따른 소수점 이하 차이 외에는 같은 값이 나옵니다(검증 환경: scikit-learn 1.9.1).

## 인용

황진하, "드론 FHSS 조종 링크의 재밍 및 신호 주입 공격 탐지를 위한 시간·RF·공간 특징 융합 프레임워크," 박사학위논문, 가천대학교 대학원, 2027.

## 라이선스

추후 명시 예정입니다. 그 전까지 재사용이 필요하면 저자에게 문의해 주세요.
