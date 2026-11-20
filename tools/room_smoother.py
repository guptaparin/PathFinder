"""Experimental handheld translation smoother (our implementation of the idea):
state per axis [p, v, b, c]; world acceleration enters as an input; b = slowly varying world-acceleration offset
(tilt/bias residual) whose random walk grows with the documented degradation ramp; a workspace pseudo-measurement
(p - c = 0, sigma_room) is applied at every block so position bounds also calibrate v and b; IMU-dead window:
velocity decays with tau_blind. Runs on blocks of `blk` seconds."""
import numpy as np, math
def run(z, has_z, dt0, P):
    nb0 = len(z); m = max(1, int(round(P["blk"] / dt0))); dt = m * dt0
    nb = (nb0 + m - 1) // m; pad = nb * m - nb0
    hz = np.r_[has_z, np.zeros(pad, bool)].reshape(nb, m); zz = np.r_[z, np.zeros((pad, z.shape[1]))].reshape(nb, m, -1)
    cnt = hz.sum(1); u = (zz * hz[:, :, None]).sum(1) / np.maximum(cnt, 1)[:, None]
    if P.get("hp_tau", 0) > 0:     # track slowly varying gravity/offset (first-order low-pass) and remove it
        al = 1.0 - math.exp(-dt / P["hp_tau"]); g = u[cnt > 0][:max(1, int(2 / dt))].mean(0) if (cnt > 0).any() else np.zeros(u.shape[1])
        for k in range(len(u)):
            if cnt[k] > 0:
                g = g + al * (u[k] - g); u[k] = u[k] - g
    meas = cnt >= max(1, m // 2)
    last = np.flatnonzero(cnt > 0)[-1] if (cnt > 0).any() else nb - 1
    blind = np.arange(nb) > last
    D = z.shape[1]; T = nb
    av = 1.0 - dt / P["tau_v"] if P.get("tau_v", 0) > 0 else 1.0
    F = np.array([[1, dt, -0.5 * dt * dt, 0], [0, av, -dt, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])
    G = np.array([0.5 * dt * dt, dt, 0, 0])
    tn = np.arange(T) / max(last, 1)
    sev = P["sev_floor"] + (1 - P["sev_floor"]) * np.clip(tn, 0, 1) ** P["sev_gamma"]
    qa = P["qa"] * (1 + P["qa_sev"] * sev)
    qb = np.sqrt(P["qb"] ** 2 + (P["qb_sev"] * sev) ** 2)
    ed = math.exp(-dt / P["tau_blind"])
    Fb = np.array([[1, P["tau_blind"] * (1 - ed), 0, 0], [0, ed, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])
    GG = np.outer(G, G) / dt
    QV = np.zeros((4, 4))
    if P.get("tau_v", 0) > 0: QV[1, 1] = 2 * P["sig_v"] ** 2 * dt / P["tau_v"]
    Hr = np.array([1.0, 0, 0, -1.0]); rr = P["room"] ** 2
    x = np.zeros((4, D)); C = np.diag([1e-4, P["v0"] ** 2, P["b0"] ** 2, P["c0"] ** 2])
    xf = np.empty((T, 4, D)); Pf = np.empty((T, 4, 4)); Fs = np.empty((T, 4, 4)); Qs = np.empty((T, 4, 4))
    for k in range(T):
        if k:
            j = k - 1
            if blind[j]:
                Fk = Fb; Qk = GG * P["blind_qa"] ** 2; Qk = Qk.copy(); Qk[2, 2] += qb[j] ** 2 * dt
                x = Fk @ x
            elif meas[j]:
                Fk = F; Qk = GG * qa[j] ** 2 + QV; Qk[2, 2] += qb[j] ** 2 * dt
                x = F @ x + np.outer(G, u[j])
            else:
                Fk = F; Qk = GG * (qa[j] ** 2 + P["gap_qa"] ** 2) + QV; Qk[2, 2] += qb[j] ** 2 * dt
                x = F @ x
            C = Fk @ C @ Fk.T + Qk; Fs[j] = Fk; Qs[j] = Qk
        Ch = C @ Hr; S = Hr @ Ch + rr; K = Ch / S
        x = x - np.outer(K, Hr @ x); C = C - np.outer(K, Ch)
        if k == T - 1 and P.get("lc", 0) > 0:
            S = C[0, 0] + P["lc"] ** 2; K = C[:, 0] / S; x = x - np.outer(K, x[0]); C = C - np.outer(K, C[0])
        xf[k] = x; Pf[k] = C
    xs = xf.copy(); Ps = Pf[-1]
    for k in range(T - 2, -1, -1):
        Pp = Fs[k] @ Pf[k] @ Fs[k].T + Qs[k]
        Cg = Pf[k] @ Fs[k].T @ np.linalg.inv(Pp)
        xp = Fs[k] @ xf[k] + (np.outer(G, u[k]) if (meas[k] and not blind[k]) else 0)
        xs[k] = xf[k] + Cg @ (xs[k + 1] - xp)
    # back to the fine grid (block centres -> linear interpolation)
    tb = (np.arange(T) + 0.5) * dt; tf = (np.arange(nb0) + 0.5) * dt0
    p = np.stack([np.interp(tf, tb, xs[:, 0, j]) for j in range(D)], 1)
    return p
DEF = dict(blk=0.4, qa=0.1, qa_sev=0.0, qb=0.003, qb_sev=0.15, sev_floor=0.02, sev_gamma=1.7, tau_blind=0.2,
           blind_qa=0.3, gap_qa=0.3, room=3.0, v0=0.5, b0=0.6, c0=10.0, lc=0.0)
