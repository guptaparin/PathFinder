import numpy as np, torch, torch.nn as nn, os, sys, json, glob, time
torch.set_num_threads(2); torch.manual_seed(0); np.random.seed(0)
TR=os.environ.get('TRAIN','data/train'); W=int(os.environ.get('WIN','200')); STRIDE=int(os.environ.get('STRIDE','8')); EPOCHS=int(os.environ.get('EPOCHS','25'))
H=int(os.environ.get('HID','32'))
FOLDS=[['MH_01_easy','MH_04_difficult','V1_02_medium','V2_03_difficult'],['MH_02_easy','MH_05_difficult','V1_03_difficult'],['MH_03_medium','V1_01_easy','V2_01_easy','V2_02_medium']]
ALL=sum(FOLDS,[])
def windows(X,idx): return np.stack([X[i-W//2:i+W//2] for i in idx])
def load(seqs, stride):
    Xs=[];Ys=[]
    for f in sorted(glob.glob(f'{TR}/*.npz')):
        s=os.path.basename(f).rsplit('_',1)[0]
        if s not in seqs: continue
        d=np.load(f); X,Y=d['X'],d['Y']
        idx=np.arange(W//2, len(X)-W//2, stride); idx=idx[~np.isnan(Y[idx]).any(1)]
        Xs.append(windows(X,idx)); Ys.append(Y[idx])
    return np.concatenate(Xs),np.concatenate(Ys)
class Net(nn.Module):
    def __init__(s,c=9,h=H):
        super().__init__()
        s.c1=nn.Conv1d(c,h,5,stride=2,padding=2); s.c2=nn.Conv1d(h,h,5,stride=2,padding=2); s.c3=nn.Conv1d(h,2*h,5,stride=2,padding=2)
        L=W
        for _ in range(3): L=(L+4-5)//2+1
        s.f1=nn.Linear(2*h*L,64); s.f2=nn.Linear(64,6)
    def forward(s,x):
        x=torch.relu(s.c1(x)); x=torch.relu(s.c2(x)); x=torch.relu(s.c3(x)); x=torch.relu(s.f1(x.flatten(1))); return s.f2(x)
def train(seqs, epochs=EPOCHS):
    X,Y=load(seqs,STRIDE)
    mu=X.reshape(-1,9).mean(0); sd=X.reshape(-1,9).std(0)+1e-6
    Xt=torch.tensor(((X-mu)/sd).transpose(0,2,1),dtype=torch.float32); Yt=torch.tensor(Y); del X
    net=Net(); opt=torch.optim.Adam(net.parameters(),1e-3,weight_decay=1e-4); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs)
    n=len(Xt)
    for ep in range(epochs):
        perm=torch.randperm(n)
        for i in range(0,n,256):
            b=perm[i:i+256]; o=net(Xt[b]); m,lv=o[:,:3],o[:,3:].clamp(-6,3)
            loss=(((m-Yt[b])**2)*torch.exp(-lv)+lv).mean()*0.5+((m-Yt[b])**2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    return net,mu,sd
def export(net,mu,sd,path):
    d={k:v.detach().numpy().astype(np.float32) for k,v in net.state_dict().items()}
    d['mu']=mu.astype(np.float32); d['sd']=sd.astype(np.float32); d['win']=np.array(W)
    np.savez(path,**d)
if __name__=='__main__':
    tag=os.environ.get('TAG','v2')
    if os.environ.get('FINAL'):
        net,mu,sd=train(ALL); export(net,mu,sd,f'{tag}_final.npz'); print('final saved'); sys.exit()
    res={}
    for fi,test in enumerate(FOLDS):
        t0=time.time(); net,mu,sd=train([s for s in ALL if s not in test]); export(net,mu,sd,f'{tag}_fold{fi}.npz')
        net.eval()
        for s in test:
            for f in sorted(glob.glob(f'{TR}/{s}_*.npz')):
                k=int(f.rsplit('_',1)[1][:-4]); d=np.load(f)
                idx=np.arange(W//2,len(d['X'])-W//2,2); idx=idx[~np.isnan(d['Y'][idx]).any(1)]
                with torch.no_grad(): o=net(torch.tensor(((windows(d['X'],idx)-mu)/sd).transpose(0,2,1),dtype=torch.float32)).numpy()
                res.setdefault(s,{})[k]=(np.sqrt(((o[:,:3]-d['Y'][idx])**2).mean(0))/d['Y'][idx].std(0)).tolist()
            print(fi,s,'C',np.round(res[s][0],2),'B',np.round(np.mean([res[s][k] for k in range(1,8)],0),2),'A',np.round(np.mean([res[s][k] for k in (8,9,10)],0),2),round(time.time()-t0),flush=True)
    print('MEAN B',np.mean([np.mean([res[s][k] for k in range(1,8)],0) for s in ALL],0).round(3),'A',np.mean([np.mean([res[s][k] for k in (8,9,10)],0) for s in ALL],0).round(3))
