"""Fine-tune velocity CNN with an integrated-displacement loss.
Segments of K consecutive windows (stride S bins) are predicted; velocity errors are rotated to the
(estimated) world frame and integrated; displacement error over several horizons is penalised.
This directly targets the time-correlated velocity errors that become position drift."""
import numpy as np, torch, os, sys, glob, time
from train_velnet import Net, W, FOLDS, ALL, export, TR
SEED = int(os.environ.get('SEED', '0')); torch.manual_seed(SEED); np.random.seed(SEED)
torch.set_num_threads(int(os.environ.get('NT', '2')))
LAM = float(os.environ.get('LAM', '1.0')); STEPS = int(os.environ.get('STEPS', '1500'))
S = 5; K = int(os.environ.get('K', '150')); BINDT = 0.02; NSEG = int(os.environ.get('NSEG', '4'))
LR = float(os.environ.get('LR', '3e-4'))
HZ = [int(h / (S * BINDT)) for h in (1, 2, 5, 10)] + [K - 1]


def qR(q):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([np.stack([1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], -1),
                     np.stack([2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], -1),
                     np.stack([2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)], -1)], 1).astype(np.float32)


def load_files(seqs):
    D = []
    for f in sorted(glob.glob(f'{TR}/*.npz')):
        s = os.path.basename(f).rsplit('_', 1)[0]
        if s not in seqs: continue
        d = np.load(f)
        X, Y, Q = d['X'], d['Y'], d['Q'].astype(np.float64)
        ok = ~np.isnan(Y).any(1)
        D.append((X, Y, qR(Q), ok))
    return D


def sample(D, mu, sd):
    xs, ys, rs = [], [], []
    for _ in range(NSEG):
        while True:
            X, Y, R, ok = D[np.random.randint(len(D))]
            lo = W // 2; hi = len(X) - W // 2 - K * S
            if hi <= lo: continue
            st = np.random.randint(lo, hi); c = st + np.arange(K) * S
            if ok[c].all(): break
        xs.append(np.stack([X[i - W // 2:i + W // 2] for i in c])); ys.append(Y[c]); rs.append(R[c])
    x = torch.tensor(((np.concatenate(xs) - mu) / sd).transpose(0, 2, 1), dtype=torch.float32)
    return x, torch.tensor(np.stack(ys)), torch.tensor(np.stack(rs))


def finetune(init_npz, seqs, out_npz):
    d = dict(np.load(init_npz)); mu, sd = d['mu'], d['sd']
    net = Net(); sdict = net.state_dict()
    net.load_state_dict({k: torch.tensor(d[k].astype(np.float32)) for k in sdict})
    D = load_files(seqs)
    opt = torch.optim.Adam(net.parameters(), LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    t0 = time.time(); run = []
    for it in range(STEPS):
        x, y, R = sample(D, mu, sd)
        o = net(x); m, lv = o[:, :3], o[:, 3:].clamp(-6, 3)
        yf = y.reshape(-1, 3)
        lvel = (((m - yf) ** 2) * torch.exp(-lv) + lv).mean() * 0.5 + ((m - yf) ** 2).mean()
        e = (m - yf).reshape(NSEG, K, 3)
        ew = torch.einsum('bkij,bkj->bki', R, e)
        dp = torch.cumsum(ew * (S * BINDT), 1)
        dp = torch.cat([torch.zeros(NSEG, 1, 3), dp], 1)
        lpos = 0
        for h in HZ:
            dd = dp[:, h:] - dp[:, :-h]
            lpos = lpos + (dd ** 2).sum(-1).mean() / (h * S * BINDT)
        lpos = lpos / len(HZ)
        loss = lvel + LAM * lpos
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step(); sched.step()
        run.append((lvel.item(), lpos.item()))
        if (it + 1) % 250 == 0:
            a = np.mean(run[-250:], 0); print(f'  it {it+1} lvel {a[0]:.3f} lpos {a[1]:.3f} {time.time()-t0:.0f}s', flush=True)
    export(net, mu, sd, out_npz)


if __name__ == '__main__':
    tag = os.environ.get('TAG', 'disp'); src = os.environ.get('SRC', 'v2')
    if os.environ.get('FINAL'):
        finetune(f'{src}_final.npz', ALL, f'{tag}_final.npz'); print('final saved'); sys.exit()
    folds = [int(f) for f in os.environ.get('FOLDS', '0,1,2').split(',')]
    for fi in folds:
        test = FOLDS[fi]
        print('fold', fi, flush=True)
        finetune(f'{src}_fold{fi}.npz', [s for s in ALL if s not in test], f'{tag}_fold{fi}.npz')
