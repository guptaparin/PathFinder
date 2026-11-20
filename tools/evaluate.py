"""Evaluation harness mimicking the organiser pipeline (evo ATE with SE(3) Umeyama alignment).
Reports: full ATE RMSE, waypoint error (every 5 s), blind-5s error, post-dropout error, RPE(1s), rot err."""
import numpy as np, os, sys, json

def load_gt(path):
    a = np.genfromtxt(path, delimiter=',', comments='#')
    ts = np.loadtxt(path, delimiter=',', comments='#', usecols=0, dtype=np.int64)
    return ts, a[:, 1:4], a[:, 4:8]  # q wxyz

def load_est(path):
    rows = [l.split() for l in open(path) if l.strip() and not l.startswith('#')]
    t = np.array([r[0] for r in rows])
    ns = np.array([int(x.split('.')[0]) * 10**9 + int((x.split('.')[1] + '000000000')[:9]) for x in t], dtype=np.int64)
    a = np.array([[float(v) for v in r[1:]] for r in rows])
    return ns, a[:, :3], a[:, 3:7]  # q xyzw

def load_input_ts(path):
    return np.loadtxt(path, delimiter=',', comments='#', usecols=0, dtype=np.int64)

def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S = (dst - mu_d).T @ (src - mu_s) / len(src)
    U, D, Vt = np.linalg.svd(S)
    W = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0: W[2, 2] = -1
    R = U @ W @ Vt
    return R, mu_d - R @ mu_s

def qmat_wxyz(q):
    w, x, y, z = q.T
    return np.stack([np.stack([1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], -1),
                     np.stack([2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], -1),
                     np.stack([2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)], -1)], 1)

def evaluate(est_path, gt_path, inp_path):
    gts, gp, gq = load_gt(gt_path)
    ens, ep, eq = load_est(est_path)
    its = load_input_ts(inp_path)
    out = {'rows_ok': bool(len(ens) == len(its) and np.all(ens == its))}
    # association: for each GT stamp, nearest est stamp within 5ms
    idx = np.searchsorted(ens, gts)
    idx = np.clip(idx, 1, len(ens) - 1)
    left = ens[idx - 1]; right = ens[idx]
    idx = np.where(np.abs(gts - left) < np.abs(gts - right), idx - 1, idx)
    ok = np.abs(ens[idx] - gts) < 5_000_000
    g_t, g_p, g_q = gts[ok], gp[ok], gq[ok]
    e_p, e_q = ep[idx[ok]], eq[idx[ok]]
    R, t = umeyama(e_p, g_p)
    a_p = e_p @ R.T + t
    err = np.linalg.norm(a_p - g_p, axis=1)
    out['ate'] = float(np.sqrt(np.mean(err**2)))
    T0, T1 = its[0], its[-1]
    tsec = (g_t - T0) * 1e-9; dur = (T1 - T0) * 1e-9
    blind = tsec >= dur - 5.0
    out['blind'] = float(np.sqrt(np.mean(err[blind]**2))) if blind.any() else float('nan')
    # waypoints every 5 s before blind
    wp = []
    for s in np.arange(5.0, dur - 5.0, 5.0):
        j = np.argmin(np.abs(tsec - s));
        if abs(tsec[j] - s) < 0.05: wp.append(err[j])
    out['wp'] = float(np.sqrt(np.mean(np.square(wp))))
    # post-dropout: 2 s after gaps > 50 ms
    d = np.diff(its); gaps = np.flatnonzero(d > 50_000_000)
    m = np.zeros(len(g_t), bool)
    for gi in gaps:
        m |= (g_t >= its[gi + 1]) & (g_t < its[gi + 1] + 2_000_000_000)
    m &= ~blind
    out['drop'] = float(np.sqrt(np.mean(err[m]**2))) if m.any() else float('nan')
    # RPE translation over 1s (no alignment needed, compare relative displacements in world after rotation align)
    step = 200
    dg = g_p[step:] - g_p[:-step]; de = (e_p[step:] - e_p[:-step]) @ R.T
    out['rpe1s'] = float(np.sqrt(np.mean(np.sum((dg - de)**2, 1))))
    # rotation error (deg) after alignment
    Rg = qmat_wxyz(g_q)
    x, y, z, w = e_q.T
    Re = qmat_wxyz(np.stack([w, x, y, z], 1))
    Ra = R[None] @ Re
    # relative rotation angle; body frames match (both IMU frame)
    tr = np.einsum('nij,nij->n', Ra, Rg)
    ang = np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))
    out['rot'] = float(np.sqrt(np.mean(ang**2)))
    # baseline: constant position
    out['zero'] = float(np.sqrt(np.mean(np.sum((g_p - g_p.mean(0))**2, 1))))
    return out

if __name__ == '__main__':
    print(json.dumps(evaluate(*sys.argv[1:4]), indent=1))
