"""TUM-VI handheld evaluation harness (degraded rooms in $PF_DATA/deg_tumvi/{A,B}/roomK/imu.csv, ground truth from
tools/tumvi_groundtruth.py in $PF_DATA/tumvi/roomK/gt.csv). Objective mirrors the official tier weights."""
import os as _os
_ROOT=_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),'..')
DATA=_os.environ.get('PF_DATA',_os.path.join(_ROOT,'data'))
import numpy as np, sys, json, os
sys.path.insert(0,_os.path.join(_ROOT,'src'))
import pathfinder as PF
from evaluate import umeyama
S=[f'room{k}' for k in range(1,7)]; FOLDS=[['room1','room2'],['room3','room4'],['room5','room6']]; fold_of={s:i for i,f in enumerate(FOLDS) for s in f}
GT={s:np.loadtxt(f'{DATA}/tumvi/{s}/gt.csv',delimiter=',',comments='#') for s in S}
D=os.environ.get('TVD',f'{DATA}/deg_tumvi')
WFMT=os.environ.get('TWFMT')
WT={i:dict(np.load(WFMT.format(i=i))) for i in range(3)} if WFMT else None
C={}
def get(m,s,P):
    key=(m,s)
    if key not in C: C[key]=PF.estimate(f'{D}/{m}/{s}/imu.csv',P,WT[fold_of[s]] if WT else None,return_debug=True)
    return C[key]
def score(res,s,pk):
    gt=GT[s]; tg=(gt[:,0]-res['ts'][0])*1e-9; g=gt[:,1:4]; tk=res['tk']; dur=res['t'][-1]; mk=(tg>=0)&(tg<=dur); tg=tg[mk]; g=g[mk]
    e=np.stack([np.interp(tg,tk,pk[:,j]) for j in range(3)],1); R,t=umeyama(e,g); err=np.linalg.norm(e@R.T+t-g,axis=1)
    bl=tg>=dur-5; wp=[err[np.argmin(np.abs(tg-x))] for x in np.arange(5,dur-5,5)]
    zero=np.sqrt(((g-g.mean(0))**2).sum(1).mean()); st=int(1/0.0083); dg=g[st:]-g[:-st]; de=(e[st:]-e[:-st])@R.T
    return dict(ate=np.sqrt((err**2).mean()),blind=np.sqrt((err[bl]**2).mean()),wp=np.sqrt(np.mean(np.square(wp))),zero=zero,
                blind0=np.sqrt((((g-g.mean(0))[bl])**2).sum(1).mean()),rpe=np.sqrt(((dg-de)**2).sum(1).mean()),rpe0=np.sqrt((dg**2).sum(1).mean()))
def evalP(P,modes='BA',verbose=False):
    P=dict(PF.DEFAULT_PARAMS,**P); out={}
    for m in modes:
        rows=[]
        for s in S:
            res=get(m,s,P); z,hz,vm,vv,hv=res['rts_in']
            if not P.get('use_net',True): hv=np.zeros_like(hv)
            pk,_=PF.smooth(z,hz,vm,vv,hv,P,None); rows.append(score(res,s,pk))
            if verbose: print(m,s,{k:round(float(v),2) for k,v in rows[-1].items()})
        o={k:float(np.mean([r[k] for r in rows])) for k in rows[0]}
        for k,b in (('rel','zero'),('relwp','zero'),('relbl','blind0'),('relrpe','rpe0')):
            o[k]=float(np.mean([r[{'rel':'ate','relwp':'wp','relbl':'blind','relrpe':'rpe'}[k]]/r[b] for r in rows]))
        out[m]=o
    return out
def obj(o): return np.mean([0.5*v['relwp']+0.33*v['relbl']+0.17*v['rel']+0.5*v['relrpe'] for v in o.values()])
if __name__=='__main__':
    P=json.loads(sys.argv[1])
    if len(sys.argv)>2:
        space=json.loads(sys.argv[2]); best=evalP(P); bo=obj(best); print('start',round(bo,4),flush=True)
        for rd in range(2):
            for k,vals in space.items():
                for v in vals:
                    Q=dict(P); Q[k]=v; o=evalP(Q); ob=obj(o)
                    if ob<bo-1e-4: bo,P,best=ob,Q,o; print(f'  {k}={v} -> {ob:.4f}',flush=True)
        print(json.dumps(P)); print({m:{k:round(v,3) for k,v in best[m].items()} for m in best})
    else:
        o=evalP(P,verbose=True); print({m:{k:round(v,3) for k,v in o[m].items()} for m in o})
