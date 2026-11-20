#!/usr/bin/env python3
"""
Interactive trajectory viewer (single self-contained HTML, works offline).

usage: python tools/visualize.py estimated_trajectory.txt [--gt gt.csv] [--imu imu0/data.csv] [-o viewer.html]

  --gt   EuRoC state_groundtruth_estimate0/data.csv or TUM file: overlays ground truth (SE(3)-aligned like evo)
  --imu  the degraded input: marks dropouts and the dead (blind) window, plots accelerometer magnitude
"""
import argparse, json, os
import numpy as np


def load_tum(p):
    a = np.loadtxt(p, comments="#")
    return a[:, 0], a[:, 1:4], a[:, 4:8]


def load_gt(p):
    with open(p) as f:
        first = f.readline()
    if "," in first or first.startswith("#timestamp"):
        a = np.loadtxt(p, delimiter=",", comments="#")
        return a[:, 0] * 1e-9, a[:, 1:4]
    a = np.loadtxt(p, comments="#")
    return a[:, 0], a[:, 1:4]


def umeyama(src, dst):
    ms, md = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((dst - md).T @ (src - ms))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0: S[2, 2] = -1
    R = U @ S @ Vt
    return R, md - R @ ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("est"); ap.add_argument("--gt"); ap.add_argument("--imu"); ap.add_argument("-o", "--out", default="viewer.html")
    ap.add_argument("--title", default="PathFinder trajectory viewer")
    a = ap.parse_args()
    t, p, q = load_tum(a.est)
    keep = np.r_[True, np.diff(t) > 0]
    t, p = t[keep], p[keep]
    step = max(1, len(t) // 3000)
    data = {"title": a.title, "n": int(len(t))}
    if a.gt:
        tg, pg = load_gt(a.gt)
        i = np.clip(np.searchsorted(t, tg), 0, len(t) - 1)
        R, tr = umeyama(p[i], pg)
        p = p @ R.T + tr
        sel = slice(None, None, max(1, len(tg) // 3000))
        data["gt"] = np.c_[tg[sel] - t[0], pg[sel]].round(3).tolist()
        e = np.linalg.norm(p[i] - pg, axis=1)
        data["ate"] = float(np.sqrt((e ** 2).mean()))
    data["est"] = np.c_[t[::step] - t[0], p[::step]].round(3).tolist()
    if a.imu:
        d = np.loadtxt(a.imu, delimiter=",", comments="#")
        ti = (d[:, 0] - d[0, 0]) * 1e-9
        dt = np.diff(ti)
        med = np.median(dt)
        data["gaps"] = [[round(ti[k], 2), round(ti[k + 1], 2)] for k in np.flatnonzero(dt > 5 * med)]
        dead = np.flatnonzero((np.abs(d[:, 1:]).sum(1) == 0))
        if len(dead): data["blind"] = [round(ti[dead[0]], 2), round(ti[-1], 2)]
        an = np.linalg.norm(d[:, 4:7], axis=1)
        s2 = max(1, len(ti) // 1500)
        data["acc"] = np.c_[ti[::s2], an[::s2]].round(2).tolist()
    html = TEMPLATE.replace("__DATA__", json.dumps(data)).replace("__TITLE__", a.title)
    with open(a.out, "w") as f:
        f.write(html)
    print("wrote", os.path.abspath(a.out))


TEMPLATE = r"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>__TITLE__</title>
<style>
:root{--bg:#0A0E1A;--card:#121C33;--fg:#E6EDF7;--mut:#9AA8BF;--cyan:#22D3C5;--amber:#F5A524;--red:#F87171;--line:#26304A}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px system-ui,Segoe UI,Arial,sans-serif}
header{display:flex;align-items:baseline;gap:18px;padding:14px 20px}h1{font-size:20px;margin:0}
.stat{color:var(--mut)}.stat b{color:var(--cyan);font-size:18px}
main{display:grid;grid-template-columns:2fr 1fr;gap:14px;padding:0 20px}
.card{background:var(--card);border-radius:10px;padding:10px}canvas{width:100%;display:block}
.ctl{display:flex;gap:12px;align-items:center;padding:12px 20px}input[type=range]{flex:1}
button{background:var(--cyan);color:#05212A;border:0;border-radius:6px;padding:7px 16px;font-weight:700;cursor:pointer}
.legend span{margin-right:14px}.sw{display:inline-block;width:18px;height:3px;vertical-align:middle;margin-right:5px}
</style></head><body>
<header><h1>__TITLE__</h1><div class="stat" id="stats"></div></header>
<main><div class="card"><canvas id="top" height="520"></canvas></div>
<div><div class="card"><canvas id="side" height="240"></canvas></div><div class="card" style="margin-top:14px"><canvas id="sig" height="240"></canvas></div></div></main>
<div class="ctl"><button id="play">Pause</button><input id="slider" type="range" min="0" max="1000" value="0"><span id="clock" class="stat"></span></div>
<div class="ctl legend stat"><span><i class="sw" style="background:#9AA8BF"></i>ground truth (hidden from algorithm)</span>
<span><i class="sw" style="background:#22D3C5"></i>PathFinder, IMU only</span><span><i class="sw" style="background:#F5A524"></i>blind 5 s (IMU dead)</span><span><i class="sw" style="background:#F87171"></i>dropout</span></div>
<script>
const D=__DATA__;const est=D.est,gt=D.gt||null,T=est[est.length-1][0];let t=0,playing=true,last=null;
const blind=D.blind||null;
document.getElementById('stats').innerHTML=(D.ate!==undefined?`ATE <b>${D.ate.toFixed(2)} m</b> &nbsp; `:'')+`duration <b>${T.toFixed(1)} s</b> &nbsp; poses <b>${D.n}</b>`+(D.gaps?` &nbsp; dropouts <b>${D.gaps.length}</b>`:'');
function bounds(ix,iy){let a=[...est];if(gt)a=a.concat(gt);let xs=a.map(r=>r[ix]),ys=a.map(r=>r[iy]);return[Math.min(...xs),Math.max(...xs),Math.min(...ys),Math.max(...ys)]}
function setup(c){const r=window.devicePixelRatio||1;c.width=c.clientWidth*r;c.height=c.getAttribute('height')*r;const g=c.getContext('2d');g.setTransform(r,0,0,r,0,0);return g}
function view(c,b){const W=c.clientWidth,H=+c.getAttribute('height'),pad=30;const s=Math.min((W-2*pad)/(b[1]-b[0]||1),(H-2*pad)/(b[3]-b[2]||1));
 const ox=(W-s*(b[1]-b[0]))/2,oy=(H-s*(b[3]-b[2]))/2;return[(x)=>ox+(x-b[0])*s,(y)=>H-oy-(y-b[2])*s]}
function path(g,arr,ix,iy,X,Y,tmax,color,w,from){g.strokeStyle=color;g.lineWidth=w;g.beginPath();let st=false;
 for(const r of arr){if(r[0]>tmax)break;if(from!==undefined&&r[0]<from)continue;const x=X(r[ix]),y=Y(r[iy]);st?g.lineTo(x,y):g.moveTo(x,y);st=true}g.stroke()}
function at(arr,tt){let lo=0,hi=arr.length-1;while(hi-lo>1){const m=(lo+hi)>>1;arr[m][0]<tt?lo=m:hi=m}return arr[lo]}
function draw(c,ix,iy,label){const g=setup(c),b=bounds(ix,iy),[X,Y]=view(c,b);g.fillStyle='#9AA8BF';g.font='12px system-ui';g.fillText(label,8,16);
 if(gt)path(g,gt,ix,iy,X,Y,t,'#9AA8BF',1.4);
 const tb=blind?Math.min(t,blind[0]):t;path(g,est,ix,iy,X,Y,tb,'#22D3C5',2);if(blind&&t>blind[0])path(g,est,ix,iy,X,Y,t,'#F5A524',3,blind[0]);
 const e=at(est,t);g.fillStyle=blind&&t>blind[0]?'#F5A524':'#22D3C5';g.beginPath();g.arc(X(e[ix]),Y(e[iy]),6,0,7);g.fill();
 if(gt){const r=at(gt,t);g.fillStyle='#E6EDF7';g.beginPath();g.arc(X(r[ix]),Y(r[iy]),4,0,7);g.fill()}}
function sig(){const c=document.getElementById('sig'),g=setup(c),W=c.clientWidth,H=+c.getAttribute('height');g.fillStyle='#9AA8BF';g.font='12px system-ui';
 g.fillText('input |accel| (degraded) and sensor events',8,16);if(!D.acc)return;const X=x=>10+(W-20)*x/T,Y=y=>H-20-(H-50)*Math.min(y,25)/25;
 (D.gaps||[]).forEach(r=>{g.fillStyle='rgba(248,113,113,.55)';g.fillRect(X(r[0]),30,Math.max(2,X(r[1])-X(r[0])),H-50)});
 if(blind){g.fillStyle='rgba(245,165,36,.25)';g.fillRect(X(blind[0]),30,X(blind[1])-X(blind[0]),H-50)}
 g.strokeStyle='#22D3C5';g.lineWidth=1;g.beginPath();D.acc.forEach((r,i)=>i?g.lineTo(X(r[0]),Y(r[1])):g.moveTo(X(r[0]),Y(r[1])));g.stroke();
 g.strokeStyle='#E6EDF7';g.beginPath();g.moveTo(X(t),25);g.lineTo(X(t),H-15);g.stroke()}
function frame(ts){if(last!==null&&playing){t+=(ts-last)/1000*Math.max(6,T/25);if(t>T)t=0}last=ts;
 document.getElementById('slider').value=Math.round(1000*t/T);
 document.getElementById('clock').textContent=`t = ${t.toFixed(1)} s`+(blind&&t>blind[0]?'  ·  IMU DEAD: dead reckoning':'');
 draw(document.getElementById('top'),1,2,'top view x–y [m]');draw(document.getElementById('side'),1,3,'side view x–z [m]');sig();requestAnimationFrame(frame)}
document.getElementById('play').onclick=e=>{playing=!playing;e.target.textContent=playing?'Pause':'Play'};
document.getElementById('slider').oninput=e=>{t=T*e.target.value/1000;playing=false;document.getElementById('play').textContent='Play'};
const h=location.hash.match(/t=([\d.]+)/);if(h){t=+h[1];playing=false;document.getElementById('play').textContent='Play'}
requestAnimationFrame(frame);
</script></body></html>"""

if __name__ == "__main__":
    main()
