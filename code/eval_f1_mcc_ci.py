"""윈도우 판정의 매크로 F1·MCC와 블록 부트스트랩 95% CI (LOSO).
사용: python eval_f1_mcc_ci.py kci 15   |   python eval_f1_mcc_ci.py b210 4
"""
import os, sys, numpy as np, pandas as pd, json
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
OUT=os.path.join(ROOT,'results','work'); os.makedirs(OUT,exist_ok=True)
sys.path.insert(0,HERE)
import evaluate_sets as E
from sklearn.metrics import f1_score, matthews_corrcoef, confusion_matrix

def run(d, cols, seeds):
    y = d.label.map({c:i for i,c in enumerate(E.CL)}).values
    X = d[cols].values.astype(float); idx=np.arange(len(d))
    P = np.zeros((seeds,len(d)),int)
    for s in range(seeds):
        rng=np.random.default_rng(s)
        for ss in sorted(d.session.unique()):
            tr=idx[(d.session!=ss).values]; te=idx[(d.session==ss).values]
            trb=E.balance(tr,y,rng)
            P[s,te]=E.mlp(s).fit(X[trb],y[trb]).predict(X[te])
    return y,P

def metrics(y,P):
    yy=np.tile(y,P.shape[0]); pp=P.ravel()
    return dict(acc=np.mean(yy==pp)*100, macro_f1=f1_score(yy,pp,average='macro')*100,
                mcc=matthews_corrcoef(yy,pp), macro_rec=np.mean(np.diag(confusion_matrix(yy,pp,normalize='true')))*100)

def boot(d,y,P,B=1000,blk=10):
    # 파일 내 연속 10윈도우 블록 단위 재표본(파일별 층화)
    blocks=[]
    for f,g in d.groupby('file',sort=False):
        ii=g.index.values
        bl=[ii[i:i+blk] for i in range(0,len(ii),blk)]
        blocks.append(bl)
    rng=np.random.default_rng(0); out=[]
    for b in range(B):
        sel=np.concatenate([np.concatenate([bl[j] for j in rng.integers(0,len(bl),len(bl))]) for bl in blocks])
        out.append(metrics(y[sel],P[:,sel]))
    o=pd.DataFrame(out)
    return {k:(np.percentile(o[k],2.5),np.percentile(o[k],97.5)) for k in o}

def alarm(d,y,P,k=4,n=7,win=0.1):
    res=[]
    for s in range(P.shape[0]):
        for f,g in d.groupby('file',sort=False):
            ii=g.sort_values('win_idx').index.values
            att=(P[s,ii]>0).astype(int)
            cs=np.convolve(att,np.ones(n,int))[:len(att)]
            al=cs>=k
            lab=g.label.iloc[0]; st=g.subtype.iloc[0]
            if lab=='normal':
                rises=np.sum(al[1:]&~al[:-1])+int(al[0])
                res.append(dict(seed=s,file=f,session=g.session.iloc[0],label=lab,sub=st,dur=len(ii)*win,rises=rises,frac_alarm=al.mean()))
            else:
                first=np.argmax(al) if al.any() else None
                lat=(first+1)*win if first is not None else np.nan
                # 종류 판정: 경보 시점까지 공격 판정 윈도우 다수결
                if first is not None:
                    w=P[s,ii[max(0,first-n+1):first+1]]; w=w[w>0]
                    typ=E.CL[np.bincount(w,minlength=3).argmax()]
                else: typ=None
                res.append(dict(seed=s,file=f,session=g.session.iloc[0],label=lab,sub=st,dur=len(ii)*win,latency=lat,type_ok=(typ==lab),cover=al.mean()))
    return pd.DataFrame(res)

def load_kci():
    d=pd.read_csv(os.path.join(ROOT,'data','hackrf_S01-S03','features_v3_w100.csv'))
    d=d[d.label.isin(E.CL)]; d['win_ms']=100.0
    return E.fill(d).reset_index(drop=True)
def load_b210(band='bw40'):
    fs=[os.path.join(ROOT,'data','b210_S04-S06',f'features_S0{i}_bands.csv') for i in (4,5,6)]
    d=pd.concat([pd.read_csv(f) for f in fs]); d=d[d.label.isin(E.CL)&(d.band==band)]
    d['win_ms']=d.win_ms.round(0)
    return E.fill(d).reset_index(drop=True)

if __name__=='__main__':
    which=sys.argv[1]; seeds=int(sys.argv[2])
    d=load_kci() if which=='kci' else load_b210()
    out={}
    for name in ['rf','fused','fused_plus']:
        y,P=run(d,E.SETS[name],seeds)
        np.save(f'{OUT}/P_{which}_{name}.npy',P)
        m=metrics(y,P); ci=boot(d,y,P)
        out[name]=dict(point=m,ci=ci)
        print(which,name,{k:round(v,3) for k,v in m.items()},{k:tuple(round(x,3) for x in v) for k,v in ci.items()},flush=True)
    d.to_pickle(f'{OUT}/d_{which}.pkl'); np.save(f'{OUT}/y_{which}.npy',y)
    json.dump(out,open(f'{OUT}/res_{which}.json','w'),default=float,ensure_ascii=False,indent=1)
