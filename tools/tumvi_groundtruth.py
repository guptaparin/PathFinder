#!/usr/bin/env python3
"""
Build IMU-frame ground truth for TUM-VI room sequences (used for the handheld profile).

Sources: IMU  = ORB-SLAM3 Examples/*/TUM_IMU/dataset-roomK_512.txt (EuRoC IMU format)
         pose = dm-vio-python-tools groundtruth_files/tumvi/gtFiles/tumvi_dataset-roomK_512_16.txt (TUM format)

Conventions found and verified (see README):
  * the stored quaternion is the INVERSE of the body->world rotation (use R^T);
  * the body is a camera frame, rotated ~122 deg from the IMU. The rotation is fitted from gyro vs ground-truth
    angular velocity (hand-eye, Wahba); it is identical across all 6 rooms to 0.23 deg;
  * the mocap world is not z-up: it is rotated so that gravity is +z;
  * quaternions are made sign-continuous (otherwise interpolation produces garbage);
  * check: gyro vs ground-truth relative rotation over 2 s agrees to ~0.45 deg (EuRoC V1_01: 0.64 deg).

usage: python tools/tumvi_groundtruth.py <imu.csv> <dmvio_gt.txt> <out_gt.csv> [--extrinsic C.npy]
"""
import sys, numpy as np
sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__file__), '..', 'src'))
import pathfinder as PF

def wahba(A, B):
    U, S, Vt = np.linalg.svd(A.T @ B); D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]); return Vt.T @ D @ U.T

def logR(M):
    th = np.arccos(np.clip((np.trace(M, axis1=1, axis2=2) - 1) / 2, -1, 1))
    v = np.stack([M[:, 2, 1] - M[:, 1, 2], M[:, 0, 2] - M[:, 2, 0], M[:, 1, 0] - M[:, 0, 1]], 1)
    return v * np.where(th > 1e-8, th / (2 * np.sin(np.maximum(th, 1e-8))), 0.5)[:, None]

def main(imu_p, gt_p, out_p):
    a = np.loadtxt(imu_p, delimiter=',', comments='#'); g = np.loadtxt(gt_p)
    t = g[:, 0]; Rcam = np.transpose(PF.q_to_R(np.c_[g[:, 7], g[:, 4:7]]), (0, 2, 1))     # inverse convention
    rel = np.einsum('nji,njk->nik', Rcam[:-1], Rcam[1:]); dt = np.diff(t); ok = dt < 0.012
    w_gt = (logR(rel) / dt[:, None])[ok]; tm = (0.5 * (t[1:] + t[:-1]))[ok]
    w_imu = np.stack([np.interp(tm * 1e9, a[:, 0], a[:, 1 + j]) for j in range(3)], 1)
    m = np.linalg.norm(w_gt, axis=1) > 0.3; C = wahba(w_gt[m], w_imu[m])                  # w_imu = C w_cam
    Rimu = np.einsum('nij,jk->nik', Rcam, C.T)
    f = np.stack([np.interp(t * 1e9, a[:, 0], np.convolve(a[:, 4 + j], np.ones(400) / 400, 'same')) for j in range(3)], 1)
    u = np.einsum('nij,nj->ni', Rimu, f / np.linalg.norm(f, axis=1, keepdims=True)).mean(0); u /= np.linalg.norm(u)
    z = np.array([0, 0, 1.]); ax = np.cross(u, z); sn = np.linalg.norm(ax); cs = u @ z
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]]) / max(sn, 1e-12); Rw = np.eye(3) + sn * K + (1 - cs) * K @ K
    Rimu = np.einsum('ij,njk->nik', Rw, Rimu); p = g[:, 1:4] @ Rw.T
    v = np.gradient(p, t, axis=0); v = np.stack([np.convolve(v[:, j], np.ones(9) / 9, 'same') for j in range(3)], 1)
    from scipy.spatial.transform import Rotation as Rot          # tooling only (not used by the submission)
    q = Rot.from_matrix(Rimu).as_quat()[:, [3, 0, 1, 2]]
    for i in range(1, len(q)):
        if q[i] @ q[i - 1] < 0: q[i] = -q[i]
    ts = np.round(t * 1e9).astype(np.int64)
    with open(out_p, 'w') as fo:
        fo.write('#timestamp, p_x, p_y, p_z, q_w, q_x, q_y, q_z, v_x, v_y, v_z, bw_x, bw_y, bw_z, ba_x, ba_y, ba_z\n')
        for i in range(len(ts)): fo.write('%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,0,0,0,0,0,0\n' % (ts[i], *p[i], *q[i], *v[i]))
    print('extrinsic angle %.1f deg ->' % np.degrees(np.arccos((np.trace(C) - 1) / 2)), out_p)

if __name__ == '__main__':
    main(*sys.argv[1:4])
