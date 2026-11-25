"""15차원 융합 판정에 k-of-n 경보 규칙을 적용한 오경보·지연 (eval_f1_mcc_ci.py 실행 후).
"""
import os,sys,numpy as np,pandas as pd
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from eval_f1_mcc_ci import alarm, OUT
rows=[]
for which in ['kci','b210']:
    d=pd.read_pickle(f'{OUT}/d_{which}.pkl'); y=np.load(f'{OUT}/y_{which}.npy'); P=np.load(f'{OUT}/P_{which}_fused_plus.npy')
    for k,n in [(1,1),(4,7),(7,9)]:
        r=alarm(d,y,P,k,n); no=r[r.label=='normal']; at=r[r.label!='normal']
        T=no.groupby('seed').dur.sum().iloc[0]
        rows.append(dict(data=which,rule=f'{k}-of-{n}',normal_s=T,
          fa_per_min=no.groupby('seed').rises.sum().mean()/T*60,
          fa_win=(no.frac_alarm*no.dur).groupby(no.seed).sum().mean()/T*100,
          jam_lat=at[at.label=='jamming'].latency.median(), hij_lat=at[at.label=='hijack'].latency.median(),
          hij_lat_max=at[at.label=='hijack'].latency.max(),
          miss=at.latency.isna().sum(), n_att=len(at), hij_cover=at[at.label=='hijack'].cover.mean()*100))
t=pd.DataFrame(rows).round(2); print(t.to_string()); t.to_csv(os.path.join(OUT,'alarm_fused15.csv'),index=False)
