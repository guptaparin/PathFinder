import numpy as np, pandas as pd, os, sys, importlib.util, time
from multiprocessing import Pool
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','src'))
import pathfinder as PF
spec=importlib.util.spec_from_file_location('deg',os.environ.get('DEGRADE','degrade_imu.py')); deg=importlib.util.module_from_spec(spec); spec.loader.exec_module(deg)
E=os.environ.get('EUROC','data/euroc'); OUT=os.environ.get('TRAIN','data/train'); os.makedirs(OUT,exist_ok=True)
S=sorted(os.listdir(E))
def qR(q):
    return PF.q_to_R(q)
def feats(path,s):
    P=dict(PF.DEFAULT_PARAMS)
    ts,gyr,acc=PF.load_imu(path); t=(ts-ts[0])*1e-9
    good,_=PF.sanitize(ts,gyr,acc); tb,Gb,Ab,has,last,_vib=PF.bin_signals(ts,gyr,acc,good)
    valid=has.copy(); valid[last+1:]=False
    eu=PF.is_euroc_like(Ab,valid)
    Q,bias,ba,info=PF.estimate_attitude(tb[:last+1],Gb[:last+1],Ab[:last+1],valid[:last+1],P,eu)
    F=PF.features(Gb[:last+1],Ab[:last+1]-ba,bias,np.zeros(3),Q)
    gt=np.loadtxt(f'{E}/{s}/gt.csv',delimiter=',',comments='#'); tg=(gt[:,0]-ts[0])*1e-9
    Rg=qR(gt[:,4:8]); vb=np.einsum('nji,nj->ni',Rg,gt[:,8:11])
    tt=tb[:last+1]
    Y=np.stack([np.interp(tt,tg,vb[:,j],left=np.nan,right=np.nan) for j in range(3)],1)
    return dict(t=tt.astype(np.float32),X=F,Y=Y.astype(np.float32),Q=Q.astype(np.float32),valid=valid[:last+1],kp=info['kp'],eu=eu)
def make(args):
    s,k,mode,level,seed=args
    df=pd.read_csv(f'{E}/{s}/imu.csv'); c=list(df.columns)
    cfg=dict(deg.CONFIG); cfg['max_level']=level
    if mode=='B':
        d2=df[[c[0]]+c[4:7]+c[1:4]]; out,_=deg.degrade(d2,cfg,seed); oc=list(out.columns); out=out[[oc[0]]+oc[4:7]+oc[1:4]]
    elif mode=='A': out,_=deg.degrade(df,cfg,seed)
    else: out=df
    p=f'/tmp/tr2_{s}_{k}.csv'; out.to_csv(p,index=False)
    F=feats(p,s); os.remove(p); return s,k,mode,F
if __name__=='__main__':
    rng=np.random.default_rng(1); jobs=[]
    for s in S:
        jobs.append((s,0,'C',0,0))
        for k in range(1,11):
            mode='B' if k<=7 else 'A'
            jobs.append((s,k,mode,float(rng.uniform(0.3,1.2)),int(rng.integers(1,10**6))))
    t0=time.time()
    with Pool(2) as p:
        for s,k,mode,F in p.imap_unordered(make,jobs):
            np.savez(f'{OUT}/{s}_{k}.npz',**F); print(s,k,mode,len(F['t']),F['kp'],F['eu'],round(time.time()-t0),flush=True)
