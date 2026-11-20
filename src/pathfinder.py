"""
PathFinder core: 6-DoF trajectory estimation from a degraded IMU stream.

Pipeline
  1. ingest      streaming CSV parse, unit/column-order detection
  2. sanitize    duplicates, dead (all-zero) frames, stuck runs, Hampel spike rejection
  3. bin         time-binned 50 Hz representation (gap aware, anti-aliased)
  4. attitude    gyro strapdown via parallel quaternion prefix-scan + batch self-calibration
                 (gyro bias & drift, accel bias & scale, initial tilt) + noise-adaptive
                 gravity correction
  5. velocity    tiny 1-D CNN (numpy inference) -> body-frame velocity + uncertainty
  6. smoother    4-state Rauch-Tung-Striebel smoother (p, v, a, accel-bias) fusing world
                 acceleration and network velocity with mean-reverting physical priors;
                 the same model propagates through the 5 s blind window
  7. output      one pose per input row (TUM format)

Only numpy is required.
"""
import math
import numpy as np

G = 9.80665
BIN = 0.02            # 50 Hz working rate
KF_DT = 0.05          # 20 Hz smoother rate
SC_DS = 5             # self-calibration at 10 Hz (every 5th bin)

# Calibration priors for the EuRoC MAV ADIS16448 IMU (measured from public sequences)
EUROC_BG = np.array([-0.0021, 0.0222, 0.0778])
EUROC_BA = np.array([-0.020, 0.125, 0.070])
EUROC_THRUST_DIR = np.array([0.938, -0.018, -0.346])

DEFAULT_PARAMS = dict(
    # self-calibration
    sc_iter=3, sc_sigv=0.6, sc_corr=50.0, sc_sbg=0.003, sc_sbs=0.003, sc_sba=0.1,
    # adaptive gravity correction
    kp_lo=0.0, kp_mid=0.1, kp_hi=3.0, gnoise_ref=0.03, gnoise_switch=0.06,
    # smoother (horizontal axes); z_* override for the vertical axis
    tau_p=2000.0, sig_p=100.0, tau_v=8.0, sig_v=0.6, tau_a=0.3, sig_a=0.5, q_b=1e-2, r_z=0.5,
    v_scale=200.0, sig_c=0.0, sig_lc=1.0, vstride=5, ens_spread_w=1.0,
    z_tau_p=200.0, z_tau_v=1.0, z_r_z=2.0, z_sig_v=0.1,
    # rotor drag (EuRoC multirotor)
    use_drag=True, zupt=True, zu_win=1.0, zu_sd=0.05, zu_vib=0.1, zu_sig=0.02, drag_k1=-3.2689, drag_k2=-2.5118, drag_sig=1.0180, drag_block=0.4, drag_vib=0.25, drag_irls=2, drag_cauchy=0.7,)


# Non-EuRoC platforms (handheld TUM-VI and anything unknown): physics-only profile tuned on TUM-VI rooms
# with leave-rooms-out validation. Strong workspace prior + degradation-aware noise; never far from standing still.
HANDHELD_PARAMS = dict(use_net=False, use_drag=False, smoother="workspace",
                       rm_blk=0.2, rm_qa=0.03, rm_qb=0.03, rm_qb_sev=0.15, rm_room=2.0, rm_room_v=0.5, rm_v0=0.1, rm_b0=0.6,
                       rm_c0=10.0, rm_tau_blind=0.1, rm_blind_qa=0.3, rm_gap_qa=0.3, rm_sev_floor=0.02, rm_sev_gamma=1.7)


# ----------------------------------------------------------------------------- ingest
def resolve_input(path):
    """Accept a data.csv, a folder (…/imu0/data.csv or …/mav0/imu0/data.csv) or a .zip."""
    import os, zipfile, tempfile
    if os.path.isdir(path):
        for cand in ("imu0/data.csv", "mav0/imu0/data.csv", "data.csv"):
            p = os.path.join(path, cand)
            if os.path.isfile(p):
                return p
        for root, _, files in os.walk(path):
            if "data.csv" in files and os.path.basename(root) == "imu0":
                return os.path.join(root, "data.csv")
        raise FileNotFoundError("no imu0/data.csv under " + path)
    if path.lower().endswith(".zip"):
        z = zipfile.ZipFile(path)
        names = [n for n in z.namelist() if n.endswith("imu0/data.csv")] or \
                [n for n in z.namelist() if n.endswith(".csv")]
        d = tempfile.mkdtemp(prefix="pathfinder_")
        z.extract(names[0], d)
        return os.path.join(d, names[0])
    return path


def load_imu(path):
    """Two-pass streaming parser -> ts (int64 ns), gyr, acc (float64, rad/s and m/s^2)."""
    n = 0
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            s = line.strip()
            if s and s[0] not in "#tT" and s[0] != ",":
                n += 1
    ts = np.empty(n, np.int64)
    d = np.empty((n, 6), np.float32)
    frac = None                                  # fractional part (1e-9 units) when timestamps carry decimals
    i = 0
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            s = line.strip()
            if not s or s[0] in "#tT" or s[0] == ",":
                continue
            p = s.split(",")
            if len(p) < 7:
                continue
            t0 = p[0].strip()
            try:
                ts[i] = int(t0)
            except ValueError:                      # decimal timestamp: keep all 9 fractional digits exactly
                a_, _, b_ = t0.partition(".")
                try:
                    ts[i] = int(a_ or 0)
                    if frac is None:
                        frac = np.zeros(n, np.int64)
                    frac[i] = int((b_ + "000000000")[:9]) if b_.isdigit() else 0
                except ValueError:
                    v_ = float(t0); ts[i] = int(v_)
                    if frac is None:
                        frac = np.zeros(n, np.int64)
                    frac[i] = int(round((v_ - int(v_)) * 1e9))
            for j in range(6):
                try:
                    d[i, j] = float(p[j + 1])
                except ValueError:
                    d[i, j] = np.nan
            i += 1
    ts = ts[:i]; d = d[:i]
    d[~np.isfinite(d)] = 0.0
    # timestamp unit -> ns (decided from the median sample interval)
    if i > 1:
        tv = ts[: min(i, 2000)].astype(np.float64) + (frac[: min(i, 2000)] * 1e-9 if frac is not None else 0.0)
        dt = float(np.median(np.diff(tv)))
        f9 = frac[:i] if frac is not None else np.zeros(i, np.int64)
        if dt < 1:          # seconds
            ts = ts * 1_000_000_000 + f9
        elif dt < 1e3:      # ms
            ts = ts * 1_000_000 + f9 // 1000
        elif dt < 1e6:      # us
            ts = ts * 1000 + f9 // 1_000_000
        elif frac is not None:
            ts = ts + (f9 >= 500_000_000)
    del frac
    # column order (EuRoC = gyro first). The accelerometer is the triple dominated by gravity: its mean vector
    # is nearly as long as its mean norm, whereas rotation rates average out. Units cannot fool this test.
    h = d[: min(i, 4000)].astype(np.float64)
    live = np.abs(h).sum(1) > 0
    h = h[live] if live.sum() >= 10 else h
    def _grav(x):
        nm = np.linalg.norm(x, axis=1).mean()
        return np.linalg.norm(x.mean(0)) / nm if nm > 0 else 0.0
    s1, s2 = _grav(h[:, :3]), _grav(h[:, 3:])
    if abs(s1 - s2) < 0.05:          # ambiguous (e.g. perfectly still): fall back to magnitude (gyro is the smaller)
        swap = np.median(np.abs(h[:, :3]).sum(1)) > np.median(np.abs(h[:, 3:]).sum(1))
    else:
        swap = s1 > s2
    if swap:
        gyr, acc = d[:, 3:], d[:, :3]
    else:
        gyr, acc = d[:, :3], d[:, 3:]
    del h
    # accel in g?
    an = np.median(np.linalg.norm(acc[: min(i, 4000)], axis=1))
    if 0.5 < an < 2.0:
        acc *= G
    # gyro in deg/s? (a typical rotation rate above ~6 rad/s is implausible for a device in rad/s)
    if np.percentile(np.linalg.norm(gyr[: min(i, 4000)], axis=1), 90) > 6.0:
        gyr *= math.pi / 180
    return ts, gyr, acc


# ----------------------------------------------------------------------------- sanitize
def sanitize(ts, gyr, acc, hampel_w=7, hampel_k=5.0, chunk=384):
    """Return boolean mask of trustworthy rows and stats."""
    n = len(ts)
    dup = np.zeros(n, bool); dup[1:] = ts[1:] <= ts[:-1]
    dead = ~np.any(gyr != 0, 1) & ~np.any(acc != 0, 1)
    a2 = np.einsum("ij,ij->i", acc, acc)
    dead |= (a2 < 0.25) | (a2 > 160.0 ** 2)                   # physically impossible specific force (<0.5 or >160 m/s^2)
    dead |= np.einsum("ij,ij->i", gyr, gyr) > 35.0 ** 2       # > 2000 deg/s: saturated / garbage gyro
    same = np.zeros(n, bool)
    same[1:] = np.all(gyr[1:] == gyr[:-1], 1) & np.all(acc[1:] == acc[:-1], 1)
    stuck = same & ~dup & ~dead
    good = ~dup & ~dead & ~stuck
    # chunked Hampel filter
    floor = np.array([0.05, 0.05, 0.05, 0.5, 0.5, 0.5], np.float32)
    spk = np.zeros(n, bool)
    w = hampel_w
    for s in range(0, n, chunk):
        a = max(0, s - w); b = min(n, s + chunk + w)
        X = np.empty((b - a, 6), np.float32); X[:, :3] = gyr[a:b]; X[:, 3:] = acc[a:b]
        pad = np.pad(X, ((w, w), (0, 0)), mode="edge")
        win = np.lib.stride_tricks.sliding_window_view(pad, 2 * w + 1, axis=0)
        med = np.median(win, axis=2)
        mad = np.median(np.abs(win - med[:, :, None]), axis=2) * 1.4826
        del win
        o = s - a; m = min(chunk, n - s)
        thr = hampel_k * np.maximum(mad[o:o + m], floor)
        spk[s:s + m] = np.any(np.abs(X[o:o + m] - med[o:o + m]) > thr, 1)
        del pad, med, mad, X
    spk &= good
    good &= ~spk
    stats = dict(rows=int(n), duplicates=int(dup.sum()), dead=int(dead.sum()), stuck=int(stuck.sum()),
                 spikes=int(spk.sum()))
    return good, stats


# ----------------------------------------------------------------------------- binning
def bin_signals(ts, gyr, acc, good, dtb=BIN, chunk=8192):
    """Average good samples into fixed time bins (chunked, works from int64 ns timestamps).
    Returns bin centre times, gyr, acc, has-data mask, last valid bin."""
    t0 = ts[0]
    tend = (ts[-1] - t0) * 1e-9
    nb = int(tend / dtb) + 1
    cnt = np.zeros(nb); G_ = np.zeros((nb, 3)); A_ = np.zeros((nb, 3)); VIB = np.zeros(nb)
    prev = None
    for s in range(0, len(ts), chunk):
        g = good[s:s + chunk]
        if not g.any():
            continue
        b = np.minimum(((ts[s:s + chunk][g] - t0) * (1e-9 / dtb)).astype(np.int64), nb - 1)
        cnt += np.bincount(b, minlength=nb)
        gg = gyr[s:s + chunk][g]; aa = acc[s:s + chunk][g]
        for j in range(3):
            G_[:, j] += np.bincount(b, gg[:, j], minlength=nb)
            A_[:, j] += np.bincount(b, aa[:, j], minlength=nb)
        an = np.sqrt((aa.astype(np.float64) ** 2).sum(1))      # rotor vibration: |d|a|| between samples
        dv = np.abs(np.diff(an, prepend=an[0] if prev is None else prev))
        prev = an[-1]
        VIB += np.bincount(b, dv, minlength=nb)
    has = cnt > 0
    c = np.maximum(cnt, 1)[:, None]
    G_ /= c; A_ /= c; VIB /= c[:, 0]
    tb = (np.arange(nb) + 0.5) * dtb
    last = int(np.flatnonzero(has)[-1]) if has.any() else 0
    hv = np.flatnonzero(has)
    if len(hv) < nb:
        for j in range(3):
            G_[:, j] = np.interp(tb, tb[hv], G_[hv, j])
            A_[:, j] = np.interp(tb, tb[hv], A_[hv, j])
    return tb, G_, A_, has, last, VIB


# ----------------------------------------------------------------------------- quaternions
def qmul_v(a, b):
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw], -1)


def rotvec_to_q(v):
    a = np.sqrt((v * v).sum(-1, keepdims=True))
    s = np.where(a > 1e-12, np.sin(a / 2) / np.maximum(a, 1e-12), 0.5)
    return np.concatenate([np.cos(a / 2), v * s], -1)


def q_to_R(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3))
    R[..., 0, 0] = 1 - 2 * (y * y + z * z); R[..., 0, 1] = 2 * (x * y - z * w); R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w); R[..., 1, 1] = 1 - 2 * (x * x + z * z); R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w); R[..., 2, 1] = 2 * (y * z + x * w); R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def prefix_scan(dq, inplace=False):
    """Inclusive prefix product dq0*dq1*...*dqk via Hillis-Steele doubling (vectorised)."""
    q = dq if inplace else dq.copy()
    d = 1
    n = len(q)
    while d < n:
        q[d:] = qmul_v(q[:-d], q[d:])
        d *= 2
    q /= np.sqrt((q * q).sum(1, keepdims=True))
    return q


def tilt_quat(f):
    f = f / np.linalg.norm(f)
    z = np.array([0.0, 0.0, 1.0]); v = np.cross(f, z); s = np.linalg.norm(v)
    if s < 1e-9:
        return np.array([1.0, 0, 0, 0])
    ang = math.atan2(s, float(f @ z))
    return np.concatenate([[math.cos(ang / 2)], math.sin(ang / 2) * v / s])


def integrate(tb, gyr, bias, q0, dtb=BIN):
    Q = prefix_scan(rotvec_to_q((gyr - bias) * dtb), inplace=True)
    # orientation at bin centre i is after integrating bins 0..i-1 (shift written in place, no extra copy)
    Q[1:] = qmul_v(np.broadcast_to(q0, Q[:-1].shape), Q[:-1]); Q[0] = q0
    return Q


# ----------------------------------------------------------------------------- attitude
def self_calibrate(tb, gyr, acc, valid, bg0, ba0, P):
    """Batch linear self-calibration minimising integrated-velocity energy.
    Unknowns: initial tilt (3), gyro bias (3) + linear drift (3), accel bias (3), initial velocity (3)."""
    T = tb[-1] + 1e-9
    nf = min(len(acc), 50)
    q0 = tilt_quat(acc[:nf].mean(0))
    bg = np.array(bg0, float); bgs = np.zeros(3); ba = np.array(ba0, float)
    sel = np.flatnonzero(valid)
    sel = sel[::max(SC_DS, -(-len(sel) // 4000))]   # <= 4000 points: bounded memory on long files (unchanged below ~400 s)
    t = tb[sel]; dt = np.diff(t, prepend=t[0])
    for _ in range(P["sc_iter"]):
        Q = integrate(tb, gyr, bg + np.outer(tb / T, bgs), q0)
        R = q_to_R(Q[sel]); del Q; f = acc[sel] - ba
        fw = np.einsum("nij,nj->ni", R, f)
        a0 = fw.copy(); a0[:, 2] -= fw[:, 2].mean()
        Fx = np.zeros((len(t), 3, 3))
        Fx[:, 0, 1] = -fw[:, 2]; Fx[:, 0, 2] = fw[:, 1]; Fx[:, 1, 0] = fw[:, 2]
        Fx[:, 1, 2] = -fw[:, 0]; Fx[:, 2, 0] = -fw[:, 1]; Fx[:, 2, 1] = fw[:, 0]
        J = np.empty((len(t), 3, 15))
        J[:, :, 0:3] = -Fx
        A = np.cumsum(R * dt[:, None, None], 0)
        J[:, :, 3:6] = np.einsum("nij,njk->nik", Fx, A)
        A = np.cumsum(R * (dt * t / T)[:, None, None], 0)
        J[:, :, 6:9] = np.einsum("nij,njk->nik", Fx, A)
        del A
        J[:, :, 9:12] = -R
        J[:, :, :12] *= dt[:, None, None]; np.cumsum(J[:, :, :12], 0, out=J[:, :, :12])   # in place: no 2 MB temporary
        J[:, :, 12:15] = np.eye(3)
        y = -np.cumsum(a0 * dt[:, None], 0)
        M = J.reshape(-1, 15); yv = y.reshape(-1)
        wgt = (1.0 / P["sc_corr"]) / P["sc_sigv"] ** 2
        prior = np.array([0.1, 0.1, 1e-3] + [P["sc_sbg"]] * 3 + [P["sc_sbs"]] * 3 + [P["sc_sba"]] * 3 + [5.0] * 3)
        H = M.T @ M * wgt + np.diag(1 / prior ** 2)
        th = np.linalg.solve(H, M.T @ yv * wgt)
        del J, M, R, f, fw, a0, Fx, y, yv             # free this pass before the next one allocates
        bg = bg + th[3:6]; bgs = bgs + th[6:9]; ba = ba + th[9:12]
        q0 = qmul_v(rotvec_to_q(th[0:3]), q0); q0 /= np.linalg.norm(q0)
    bias = bg + np.outer(tb / T, bgs)
    return q0, bias, ba


def gyro_noise_level(gyr, valid):
    """Robust high-frequency gyro noise estimate on the last third of the valid data."""
    v = np.flatnonzero(valid)
    seg = v[2 * len(v) // 3:]
    if len(seg) < 20:
        return 0.0
    d = np.diff(gyr[seg], axis=0)
    return float(np.median(np.abs(d - np.median(d, 0)), 0).mean() * 1.4826 / math.sqrt(2))


def adaptive_attitude(tb, gyr, acc, bias, q0, kp, dtb=BIN, gref=None):
    """Complementary (Mahony-type) gravity correction at 50 Hz. kp=0 -> pure strapdown."""
    if kp <= 0:
        return integrate(tb, gyr, bias, q0)
    n = len(tb); Q = np.empty((n, 4))
    if gref is None:
        gref = float(np.median(np.linalg.norm(acc, axis=1)))
    q = [float(x) for x in q0]
    CH = 1024
    for s in range(0, n, CH):
        W = (gyr[s:s + CH] - bias[s:s + CH]).tolist(); A = acc[s:s + CH].tolist()
        for i in range(len(W)):
            Q[s + i] = q
            gx, gy, gz = W[i]; ax, ay, az = A[i]
            qw, qx, qy, qz = q
            an = math.sqrt(ax * ax + ay * ay + az * az)
            if an > 1e-6:
                g = kp * math.exp(-((an - gref) / 1.5) ** 2) / an
                vx = 2 * (qx * qz - qw * qy); vy = 2 * (qy * qz + qw * qx); vz = qw * qw - qx * qx - qy * qy + qz * qz
                gx += g * (ay * vz - az * vy); gy += g * (az * vx - ax * vz); gz += g * (ax * vy - ay * vx)
            h = 0.5 * dtb
            gx *= h; gy *= h; gz *= h
            q = [qw - qx * gx - qy * gy - qz * gz, qx + qw * gx + qy * gz - qz * gy,
                 qy + qw * gy - qx * gz + qz * gx, qz + qw * gz + qx * gy - qy * gx]
            nq = 1.0 / math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
            q = [q[0] * nq, q[1] * nq, q[2] * nq, q[3] * nq]
    return Q


def is_euroc_like(acc, valid):
    f = acc[valid].mean(0); f = f / (np.linalg.norm(f) + 1e-12)
    return float(f @ EUROC_THRUST_DIR) > math.cos(math.radians(15))


def estimate_attitude(tb, gyr, acc, valid, P, euroc):
    bg0 = EUROC_BG if euroc else np.zeros(3)
    ba0 = EUROC_BA if euroc else np.zeros(3)
    PP = dict(P)
    if not euroc:
        PP["sc_sbg"] = 0.02; PP["sc_sba"] = 0.3
    gn = gyro_noise_level(gyr, valid)
    noisy = gn > P["gnoise_switch"]
    if noisy:
        # heavily corrupted gyro: strapdown linearisation is invalid -> trust gravity, prior biases
        q0 = tilt_quat(acc[:min(len(acc), 50)].mean(0))
        bias = np.broadcast_to(bg0, (len(tb), 3)).copy(); ba = np.array(ba0, float)
    else:
        q0, bias, ba = self_calibrate(tb, gyr, acc, valid, bg0, ba0, PP)
    # noise-adaptive gravity-correction gain: 0 for a healthy gyro, rises as the gyro degrades
    x = max(0.0, gn / P["gnoise_ref"] - 1.0)
    kp = P["kp_lo"] + (P["kp_mid"] - P["kp_lo"]) * min(1.0, x / 3.0)
    if noisy:
        kp = P["kp_hi"]
    Q = adaptive_attitude(tb, gyr, acc - ba, bias, q0, kp)
    return Q, bias, ba, dict(gyro_noise=gn, kp=kp, bg=bias[0].tolist(), ba=ba.tolist())


# ----------------------------------------------------------------------------- network
def features(gyr, acc, bias, ba_prior, Q):
    """[acc - ba, gyro - bias, gravity direction in body] as float32 (built in place, no temporaries)."""
    n = len(Q)
    F = np.empty((n, 9), np.float32)
    F[:, 0:3] = acc; F[:, 0:3] -= ba_prior
    F[:, 3:6] = gyr; F[:, 3:6] -= bias
    w, x, y, z = Q[:, 0], Q[:, 1], Q[:, 2], Q[:, 3]
    F[:, 6] = 2 * (x * z - y * w); F[:, 7] = 2 * (y * z + x * w); F[:, 8] = 1 - 2 * (x * x + y * y)  # R^T e_z
    return F


def conv1d(x, w, b, stride=2, pad=2):
    """x: (B, C, L) ; w: (O, C, K). im2col + one BLAS matmul (fast on any numpy version)."""
    B, C, L = x.shape; O, _, K = w.shape
    xp = np.zeros((B, C, L + 2 * pad), np.float32); xp[:, :, pad:pad + L] = x
    Lo = (L + 2 * pad - K) // stride + 1
    cols = np.empty((C, K, B, Lo), np.float32)
    for k in range(K):
        cols[:, k] = xp[:, :, k:k + stride * (Lo - 1) + 1:stride].transpose(1, 0, 2)
    y = w.reshape(O, C * K) @ cols.reshape(C * K, B * Lo)
    return (y.reshape(O, B, Lo) + b[:, None, None]).transpose(1, 0, 2)


def net_forward(Wt, x):
    f = Wt.__getitem__
    h = np.maximum(conv1d(x, f("c1.weight"), f("c1.bias")), 0)
    h = np.maximum(conv1d(h, f("c2.weight"), f("c2.bias")), 0)
    h = np.maximum(conv1d(h, f("c3.weight"), f("c3.bias")), 0)
    h = h.reshape(len(h), -1)
    h = np.maximum(h @ f("f1.weight").T + f("f1.bias"), 0)
    return h @ f("f2.weight").T + f("f2.bias")


def split_ensemble(Wt):
    """A weights file may hold several members with keys 'm<k>.<name>'; shared keys: win, mu, sd."""
    if not any(k.startswith("m0.") for k in Wt):
        return [Wt]
    members = sorted({k.split(".", 1)[0] for k in Wt if k[0] == "m" and k[1].isdigit()})
    out = []
    for m in members:
        d = {k.split(".", 1)[1]: v for k, v in Wt.items() if k.startswith(m + ".")}
        for k in ("win", "mu", "sd"):
            if k not in d: d[k] = Wt[k]
        out.append(d)
    return out


def predict_velocity(F, Q, last, Wt, P):
    """Ensemble velocity prior: mean of members; variance = mean predicted variance + member spread."""
    members = split_ensemble(Wt)
    W0 = members[0]
    win = int(W0["win"]); h = win // 2; st = int(P["vstride"])
    centers = np.arange(h, min(last + 1, len(F)) - h, st)
    if len(centers) == 0:
        return None, None, None
    mu = W0["mu"].astype(np.float32); sd = W0["sd"].astype(np.float32)
    Fn = F; Fn -= mu; Fn /= sd                      # in place
    K = len(members)
    out_v = np.zeros((len(centers), 3)); m2 = np.zeros((len(centers), 3)); out_var = np.zeros(len(centers))
    B = 8
    for Wm in members:                              # member-outer: cast one member to fp32 at a time
        W32 = {k: (v.astype(np.float32) if v.dtype == np.float16 else v) for k, v in Wm.items()}
        for s in range(0, len(centers), B):
            c = centers[s:s + B]
            x = np.stack([Fn[i - h:i + h] for i in c]).transpose(0, 2, 1)
            o = net_forward(W32, x)
            out_v[s:s + B] += o[:, :3] / K
            m2[s:s + B] += o[:, :3] ** 2 / K
            out_var[s:s + B] += np.exp(np.clip(o[:, 3:], -6, 3)).mean(1) / K
        del W32
    spread = np.maximum(m2 - out_v ** 2, 0).mean(1)
    out_var += P.get("ens_spread_w", 1.0) * spread
    R = q_to_R(Q[centers])
    out_v = np.einsum("nij,nj->ni", R, out_v)
    return centers, out_v, out_var


# ----------------------------------------------------------------------------- rotor drag
DRAG_E1 = np.array([-0.0191862, -0.9998159, 0.0])
DRAG_E2 = np.array([-0.3459564, 0.0066388, -0.9382271])


def drag_measure(tb, Ac, Q, valid, vib, nk, P):
    """Rotor-drag body velocity (EuRoC multirotor): v_perp = K * (f_perp - mean), 0.4 s blocks,
    used only while rotors spin (vibration gate). Returned as world-frame xy velocity measurements."""
    n = len(Ac); NB = int(round(P["drag_block"] / BIN))
    a = np.stack([Ac @ DRAG_E1, Ac @ DRAG_E2], 1)
    on = valid & (vib[:n] > P["drag_vib"])
    if on.sum() < 5 * NB:
        return None
    a -= a[on].mean(0)
    m = n // NB * NB
    ab = a[:m].reshape(-1, NB, 2).mean(1); onb = on[:m].reshape(-1, NB).mean(1) > 0.8
    tbb = tb[:m].reshape(-1, NB).mean(1)
    qb = Q[:m].reshape(-1, NB, 4).mean(1); qb /= np.sqrt((qb * qb).sum(1, keepdims=True))
    vbody = ab[:, 0:1] * P["drag_k1"] * DRAG_E1 + ab[:, 1:2] * P["drag_k2"] * DRAG_E2
    vw = np.einsum("nij,nj->ni", q_to_R(qb), vbody)
    ki = np.minimum((tbb / KF_DT).astype(np.int64), nk - 1)
    vm = np.zeros((nk, 2)); has = np.zeros(nk, bool)
    vm[ki[onb]] = vw[onb, :2]; has[ki[onb]] = True
    return vm, has


def rest_steps(tb, Ac, valid, vib, nk, P):
    """Rest periods with the rotors off (multirotor on the ground): |a| steady over zu_win seconds and rotor
    vibration below zu_vib. Returns a boolean mask on the smoother grid (a step counts only if all its bins rest)."""
    n = len(Ac); W = max(1, int(round(P["zu_win"] / BIN)))
    an = np.sqrt((Ac.astype(np.float64) ** 2).sum(1))
    c1 = np.r_[0.0, np.cumsum(an)]; c2 = np.r_[0.0, np.cumsum(an * an)]; del an
    cv = np.r_[0.0, np.cumsum(vib[:n].astype(np.float64))]
    rest = np.zeros(n, bool)
    for s0 in range(0, n, 4096):                 # chunked: no full-length index/temporary arrays
        k = np.arange(s0, min(n, s0 + 4096)); lo = np.maximum(k - W // 2, 0); hi = np.minimum(k + W // 2 + 1, n)
        m = (c1[hi] - c1[lo]) / (hi - lo); sd = np.sqrt(np.maximum((c2[hi] - c2[lo]) / (hi - lo) - m * m, 0.0))
        rest[k] = valid[k] & (sd < P["zu_sd"]) & ((cv[hi] - cv[lo]) / (hi - lo) < P["zu_vib"])
    del c1, c2, cv
    ki = np.minimum((tb[:n] / KF_DT).astype(np.int64), nk - 1)
    cnt = np.bincount(ki, minlength=nk); rc = np.bincount(ki, rest, minlength=nk)
    return (cnt > 0) & (rc >= cnt)


# ----------------------------------------------------------------------------- smoother
def rts(z, has_z, vm, vvar, has_v, dt, P, vm2=None, var2=None, has2=None):
    """Shared-covariance RTS smoother for 3 independent axes.
    state [p, v, a, b, c]: position, velocity, acceleration, accel-bias (tilt residual), hover centre.
    Position mean-reverts to the latent centre c; optional soft return-to-launch at the final step."""
    tp, tv, ta = P["tau_p"], P["tau_v"], P["tau_a"]
    sp, sv, sa = P["sig_p"], P["sig_v"], P["sig_a"]
    F = np.array([[1 - dt / tp, dt, 0.5 * dt * dt, 0, dt / tp],
                  [0, 1 - dt / tv, dt, 0, 0],
                  [0, 0, 1 - dt / ta, 0, 0],
                  [0, 0, 0, 1.0, 0],
                  [0, 0, 0, 0, 1.0]])
    Qn = np.diag([2 * sp ** 2 * dt / tp, 2 * sv ** 2 * dt / tv, 2 * sa ** 2 * dt / ta, P["q_b"] * dt, 1e-10])
    T = len(z)
    D = z.shape[1]
    xs = np.empty((T, 5, D)); Ps = np.empty((T, 5, 5), np.float32)
    x = np.zeros((5, D)); C = np.diag([1e-4, sv ** 2, sa ** 2, 0.01, max(P["sig_c"], 1e-3) ** 2])
    rz = P["r_z"]; vs = P["v_scale"]; lc = P["sig_lc"]
    Hz = np.array([0, 0, 1.0, 1.0, 0])
    # degradation-aware noise: the jammer's severity ramps up over the file (organiser model ~ (t/T)^1.7)
    rr = P.get("r_ramp", 0.0)
    rz_t = rz * (1.0 + rr * (np.arange(T) / max(T - 1, 1)) ** P.get("r_gamma", 1.7)) if rr > 0 else None
    for k in range(T):
        if k:
            x = F @ x; C = F @ C @ F.T + Qn
        if has_z[k]:
            Ch = C @ Hz; S = Hz @ Ch + (rz if rz_t is None else rz_t[k]); K = Ch / S
            x += K[:, None] * (z[k] - Hz @ x)[None, :]; C = C - K[:, None] * Ch[None, :]
        if has_v[k]:
            S = C[1, 1] + vvar[k] * vs; K = C[:, 1] / S
            x += K[:, None] * (vm[k] - x[1])[None, :]; C = C - K[:, None] * C[1][None, :]
        if has2 is not None and has2[k]:
            S = C[1, 1] + var2[k]; K = C[:, 1] / S
            x += K[:, None] * (vm2[k] - x[1])[None, :]; C = C - K[:, None] * C[1][None, :]
        if k == T - 1 and lc > 0:
            S = C[0, 0] + lc ** 2; K = C[:, 0] / S
            x += np.outer(K, -x[0]); C = C - np.outer(K, C[0])
        xs[k] = x; Ps[k] = C
    for k in range(T - 2, -1, -1):
        Pk = Ps[k].astype(np.float64)
        Pp1 = F @ Pk @ F.T + Qn                 # recomputed prediction (saves O(T) memory)
        Cg = Pk @ F.T @ np.linalg.inv(Pp1)
        xs[k] = xs[k] + Cg @ (xs[k + 1] - F @ xs[k])
        Ps[k] = Pk + Cg @ (Ps[k + 1].astype(np.float64) - Pp1) @ Cg.T
    return xs[:, 0, :].copy(), xs[:, 1, :].copy()


def smooth(z, has_z, vm, vvar, has_v, P, drag=None):
    """Horizontal axes (+ robust rotor-drag velocity) and vertical axis use separate priors (z_* parameters)."""
    if drag is None:
        pxy, vxy = rts(z[:, :2], has_z, vm[:, :2], vvar, has_v, KF_DT, P)
    else:
        dvm, dhas = drag
        var2 = np.full(len(z), P["drag_sig"] ** 2)
        for _ in range(P["drag_irls"] + 1):
            pxy, vxy = rts(z[:, :2], has_z, vm[:, :2], vvar, has_v, KF_DT, P, dvm, var2, dhas)
            r2 = ((dvm - vxy) ** 2).sum(1) / (2 * P["drag_sig"] ** 2)
            var2 = P["drag_sig"] ** 2 * (1 + r2 / P["drag_cauchy"] ** 2)        # Cauchy re-weighting
    zk = {k[2:]: v for k, v in P.items() if k.startswith("z_")}
    Pz = dict(P, **zk)
    pz, vz = rts(z[:, 2:3], has_z, vm[:, 2:3], vvar, has_v, KF_DT, Pz)
    return np.c_[pxy, pz], np.c_[vxy, vz]


def workspace_smooth(z, has_z, dt0, P, room):
    """Handheld / unknown-platform smoother. Per axis state [p, v, b, c]: position, velocity, slowly varying
    world-acceleration offset (tilt and bias residual) and workspace centre. World acceleration is the input.
    The offset's random walk grows with the degradation ramp; a workspace pseudo-measurement p - c = 0 (sigma =
    room) at every block bounds position and, through the covariance, also calibrates v and b. While the IMU
    is dead the velocity decays (tau_blind). Kalman forward pass + RTS backward pass on rm_blk-second blocks."""
    nb0, D = z.shape; m = max(1, int(round(P["rm_blk"] / dt0))); dt = m * dt0
    nb = (nb0 + m - 1) // m; pad = nb * m - nb0
    h = np.r_[has_z, np.zeros(pad, bool)].reshape(nb, m)
    zz = np.r_[z, np.zeros((pad, D))].reshape(nb, m, D)
    cnt = h.sum(1); u = (zz * h[:, :, None]).sum(1) / np.maximum(cnt, 1)[:, None]
    meas = cnt >= max(1, m // 2)
    last = int(np.flatnonzero(cnt > 0)[-1]) if (cnt > 0).any() else nb - 1
    blind = np.arange(nb) > last
    sev = P["rm_sev_floor"] + (1 - P["rm_sev_floor"]) * np.clip(np.arange(nb) / max(last, 1), 0, 1) ** P["rm_sev_gamma"]
    qb = np.sqrt(P["rm_qb"] ** 2 + (P["rm_qb_sev"] * sev) ** 2)
    F = np.array([[1, dt, -0.5 * dt * dt, 0], [0, 1, -dt, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])
    G = np.array([0.5 * dt * dt, dt, 0, 0]); GG = np.outer(G, G) / dt
    tb_ = P["rm_tau_blind"]; ed = math.exp(-dt / tb_)
    Fb = np.array([[1, tb_ * (1 - ed), 0, 0], [0, ed, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])
    Hr = np.array([1.0, 0, 0, -1.0]); rr = room ** 2
    x = np.zeros((4, D)); C = np.diag([1e-4, P["rm_v0"] ** 2, P["rm_b0"] ** 2, P["rm_c0"] ** 2])
    xf = np.empty((nb, 4, D)); Pf = np.empty((nb, 4, 4)); kind = np.zeros(nb, np.int8)
    def trans(j):
        if blind[j]:
            return Fb, GG * P["rm_blind_qa"] ** 2, False
        q = GG * (P["rm_qa"] ** 2 + (0.0 if meas[j] else P["rm_gap_qa"] ** 2))
        return F, q, bool(meas[j])
    for k in range(nb):
        if k:
            Fk, Qk, use = trans(k - 1); Qk = Qk.copy(); Qk[2, 2] += qb[k - 1] ** 2 * dt
            x = Fk @ x + (np.outer(G, u[k - 1]) if use else 0.0); C = Fk @ C @ Fk.T + Qk
        Ch = C @ Hr; K = Ch / (Hr @ Ch + rr)
        x = x - np.outer(K, Hr @ x); C = C - np.outer(K, Ch)
        xf[k] = x; Pf[k] = C
    xs = xf
    for k in range(nb - 2, -1, -1):
        Fk, Qk, use = trans(k); Qk = Qk.copy(); Qk[2, 2] += qb[k] ** 2 * dt
        Pp = Fk @ Pf[k] @ Fk.T + Qk
        Cg = np.linalg.solve(Pp, Fk @ Pf[k]).T
        xs[k] = xf[k] + Cg @ (xs[k + 1] - (Fk @ xf[k] + (np.outer(G, u[k]) if use else 0.0)))
    tbk = (np.arange(nb) + 0.5) * dt; tf = (np.arange(nb0) + 0.5) * dt0
    return np.stack([np.interp(tf, tbk, xs[:, 0, j]) for j in range(D)], 1)


# ----------------------------------------------------------------------------- full pipeline
def estimate(path, P=None, weights=None, return_debug=False):
    user = dict(P or {})
    P = dict(DEFAULT_PARAMS, **user)
    ts, gyr, acc = load_imu(path)
    good, stats = sanitize(ts, gyr, acc)
    tb, Gb, Ab, has, last, VIB = bin_signals(ts, gyr, acc, good)
    del gyr, acc, good
    n = last + 1
    valid = has[:n].copy(); del has
    euroc = is_euroc_like(Ab[:n], valid)
    if not euroc:
        P = {**DEFAULT_PARAMS, **HANDHELD_PARAMS, **user}
    Q, bias, ba, info = estimate_attitude(tb[:n], Gb[:n], Ab[:n], valid, P, euroc)
    Ac = Ab[:n] - ba; del Ab
    use_net = weights is not None and euroc and P.get("use_net", True)
    F = features(Gb[:n], Ac, bias, np.zeros(3), Q) if use_net else None
    tend = (ts[-1] - ts[0]) * 1e-9
    nk = int(tend / KF_DT) + 1
    drag = drag_measure(tb[:n], Ac, Q, valid, VIB, nk, P) if (euroc and P.get("use_drag", True)) else None
    rest = rest_steps(tb[:n], Ac, valid, VIB, nk, P) if (euroc and P.get("zupt", True)) else None
    dbg_drag_in = (tb[:n], Ac, Q, valid, VIB, nk) if return_debug else None
    dbg_gb = (Gb[:n] - bias) if return_debug else None
    del VIB
    del Gb, bias
    # world acceleration, binned straight onto the smoother grid
    kb = np.minimum((tb[:n] / KF_DT).astype(np.int64), nk - 1)[valid]
    dbg_ac = (Ac[valid].copy(), tb[:n][valid].copy(), kb.copy(), nk) if return_debug else None
    aw = np.einsum("nij,nj->ni", q_to_R(Q[valid]), Ac[valid]); del Ac
    aw[:, 2] -= aw[:, 2].mean()
    cnt = np.bincount(kb, minlength=nk)
    z = np.stack([np.bincount(kb, aw[:, j], minlength=nk) for j in range(3)], 1) / np.maximum(cnt, 1)[:, None]
    has_z = cnt > 0
    dbg_aw = aw if return_debug else None
    del aw, kb, cnt
    # learned velocity prior
    vm = np.zeros((nk, 3)); vvar = np.ones(nk); has_v = np.zeros(nk, bool)
    vt = vw = None
    if use_net:
        c, vw, vv = predict_velocity(F, Q, last, weights, P)
        del F
        if c is not None:
            vt = tb[c]
            ki = np.minimum((vt / KF_DT).astype(np.int64), nk - 1)
            vm[ki] = vw; vvar[ki] = vv; has_v[ki] = True
    if rest is not None and rest.any():         # zero-velocity updates while parked with the rotors off
        vm[rest] = 0.0; vvar[rest] = P["zu_sig"] ** 2 / P["v_scale"]; has_v[rest] = True
        if drag is not None:
            drag = (drag[0], drag[1] & ~rest)
    if P.get("smoother") == "workspace":
        pk = np.c_[workspace_smooth(z[:, :2], has_z, KF_DT, P, P["rm_room"]), workspace_smooth(z[:, 2:3], has_z, KF_DT, P, P["rm_room_v"])]
        vk = None
    else:
        pk, vk = smooth(z, has_z, vm, vvar, has_v, P, drag)
    tk = (np.arange(nk) + 0.5) * KF_DT
    # orientation on the full bin grid: hold last valid attitude through the blind window
    nb = len(tb)
    Qfull = np.empty((nb, 4)); Qfull[:n] = Q; Qfull[n:] = Q[last]
    info.update(stats); info.update(euroc_profile=bool(euroc), profile="euroc-multirotor" if euroc else "handheld/generic", net=vt is not None, drag=drag is not None, rest_s=float(rest.sum() * KF_DT) if rest is not None else 0.0,
                                     blind_s=float(max(0.0, tend - tb[last])), duration_s=float(tend))
    res = dict(ts=ts, tk=tk, pk=pk, tb=tb, Q=Qfull, info=info)
    if return_debug:
        res.update(t=(ts - ts[0]) * 1e-9, aw=dbg_aw, vk=vk, vt=vt, vw=vw, rts_in=(z, has_z, vm, vvar, has_v), ac_in=dbg_ac, drag=drag, drag_in=dbg_drag_in, gyr_in=dbg_gb)
    return res


def write_tum(res, out_path, chunk=1024):
    ts = res["ts"]; t0 = ts[0]
    tk, pk, tb, Q = res["tk"], res["pk"], res["tb"], res["Q"]
    with open(out_path, "w") as fo:
        pass                                  # no header line: exactly one line per input row
        for s in range(0, len(ts), chunk):
            tt = (ts[s:s + chunk] - t0) * 1e-9
            p = np.stack([np.interp(tt, tk, pk[:, j]) for j in range(3)], 1)
            # nlerp orientation between neighbouring bins (sign-consistent)
            i1 = np.clip(np.searchsorted(tb, tt), 1, len(tb) - 1); i0 = i1 - 1
            w = np.clip((tt - tb[i0]) / (tb[i1] - tb[i0]), 0, 1)[:, None]
            q0 = Q[i0]; q1 = Q[i1]
            q1 = np.where(((q0 * q1).sum(1) < 0)[:, None], -q1, q1)
            q = q0 * (1 - w) + q1 * w
            q /= np.sqrt((q * q).sum(1, keepdims=True))
            lines = []
            for i in range(len(tt)):
                sns = int(ts[s + i])
                lines.append("%d.%09d %.6f %.6f %.6f %.8f %.8f %.8f %.8f\n" % (
                    sns // 1_000_000_000, sns % 1_000_000_000, p[i, 0], p[i, 1], p[i, 2],
                    q[i, 1], q[i, 2], q[i, 3], q[i, 0]))
            fo.write("".join(lines))
