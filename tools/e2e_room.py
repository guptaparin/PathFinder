"""Scores the workspace-bounded handheld smoother (tools/room_smoother.py, same model as pathfinder.workspace_smooth)."""
import os as _os
_ROOT=_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),'..')
DATA=_os.environ.get('PF_DATA',_os.path.join(_ROOT,'data'))
import numpy as np, sys, json
sys.path.insert(0,_os.path.join(_ROOT,'src')); import pathfinder as PF
import e2e_tv as E, room_smoother as RS
PH=dict(PF.DEFAULT_PARAMS,**PF.HANDHELD_PARAMS)
def evalR(R, rooms=E.S, modes='BA', zroom=None):
    out={}
    for m in modes:
        rows=[]
        for s in rooms:
            res=E.get(m,s,PH); z,hz,*_=res['rts_in']
            pxy=RS.run(z[:,:2],hz,PF.KF_DT,R); Rz=dict(R,room=R.get('room_v',1.0)) 
            pz=RS.run(z[:,2:3],hz,PF.KF_DT,Rz)
            rows.append(E.score(res,s,np.c_[pxy,pz]))
        out[m]={k:float(np.mean([r[k] for r in rows])) for k in rows[0]}
        for k,(a,b) in {'rel':('ate','zero'),'relwp':('wp','zero'),'relbl':('blind','blind0'),'relrpe':('rpe','rpe0')}.items():
            out[m][k]=float(np.mean([r[a]/r[b] for r in rows]))
    return out
if __name__=='__main__':
    R=dict(RS.DEF,**json.loads(sys.argv[1]))
    o=evalR(R); print({m:{k:round(v,3) for k,v in o[m].items() if k in('ate','wp','blind','rpe','zero','blind0')} for m in o}, 'obj %.4f'%E.obj(o))
