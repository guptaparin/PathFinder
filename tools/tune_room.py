"""Tunes the handheld smoother. `python tools/tune_room.py cv` = leave-two-rooms-out cross-validation (results/tumvi_cv.json);
`python tools/tune_room.py all` = tune on all six rooms."""
import numpy as np, sys, json
import e2e_room as ER, e2e_tv as E, room_smoother as RS
SPACE={"hp_tau":[0,1,2,4,8,16],"room":[1.0,1.5,2.0,3.0,4.0],"room_v":[0.2,0.3,0.5,1.0],"qa":[0.01,0.03,0.1,0.3],"qb":[0.01,0.03,0.1,0.3],"qb_sev":[0,0.07,0.15,0.3,0.6],
       "blk":[0.1,0.2,0.4],"tau_blind":[0.05,0.1,0.2,0.5],"qa_sev":[0,1,3],"v0":[0.03,0.1,0.5],"b0":[0.2,0.6,1.5],"lc":[0,0.5,1.0]}
def tune(rooms, R0, rounds=2):
    R=dict(R0); bo=E.obj(ER.evalR(R,rooms))
    for _ in range(rounds):
        for k,vals in SPACE.items():
            for v in vals:
                if R.get(k)==v: continue
                Q=dict(R); Q[k]=v; ob=E.obj(ER.evalR(Q,rooms))
                if ob<bo-1e-4: bo,R=ob,Q
    return R,bo
if __name__=='__main__':
    R0=dict(RS.DEF, **{"blk": 0.2, "qa": 0.03, "qb": 0.03, "qb_sev": 0.15, "tau_blind": 0.1, "room": 2.0, "v0": 0.1, "lc": 0.0, "room_v": 0.5, "hp_tau": 0})
    if sys.argv[1]=='cv':
        res=[]
        for f in E.FOLDS:
            tr=[s for s in E.S if s not in f]; R,_=tune(tr,R0)
            o=ER.evalR(R,f); res.append((f,o)); print(f, {m:{k:round(o[m][k],3) for k in('ate','wp','blind','zero','blind0','rpe')} for m in o}, json.dumps({k:R[k] for k in SPACE}),flush=True)
        for m in 'BA':
            print(m,'CV mean ATE %.3f wp %.3f blind %.3f rpe %.3f'%tuple(np.mean([o[m][k] for _,o in res]) for k in ('ate','wp','blind','rpe')))
    else:
        R,bo=tune(E.S,R0); o=ER.evalR(R); print(json.dumps(R)); print({m:{k:round(o[m][k],3) for k in('ate','wp','blind','zero','blind0','rpe')} for m in o}, bo)
