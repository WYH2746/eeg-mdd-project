# eeg_code/train.py  ——  subject-level 5-fold training
import argparse, time, pathlib, numpy as np, torch, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from eeg_code.model  import EEGNetAttn
from eeg_code.utils  import set_seed, focal_loss

# ----------------------------------------------------------- #
def train_fold(model, dl_train, dl_val, lr=1e-3, epochs=50, device="cuda", fid=0):
    opt = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    best_auc, best_state = 0.0, None
    for ep in range(epochs):
        model.train(); cum_loss = 0.0
        for xb, yb in dl_train:
            xb, yb = xb.to(device), yb.to(device)
            loss = focal_loss(model(xb), yb)
            loss.backward(); opt.step(); opt.zero_grad()
            cum_loss += loss.item() * len(xb)

        # ---------- validation ----------
        model.eval(); preds, gts = [], []
        with torch.no_grad():
            for xb, yb in dl_val:
                p = torch.softmax(model(xb.to(device)), 1)[:, 1].cpu()
                preds.append(p); gts.append(yb)
        auc = roc_auc_score(torch.cat(gts), torch.cat(preds))

        if ep % 5 == 0 or ep == epochs - 1:
            print(f"[Fold {fid}] Ep {ep:02d}/50 | loss {(cum_loss/len(dl_train.dataset)):.4f} | AUC {auc:.4f}")

        if auc > best_auc:
            best_auc, best_state = auc, model.state_dict()
    return best_auc, best_state
# ----------------------------------------------------------- #

def main(args):
    t0 = time.time(); set_seed(42)
    pathlib.Path("results").mkdir(exist_ok=True, parents=True)

    npz = np.load(args.data, allow_pickle=True)
    X   = torch.tensor(npz["X"]).float()     # (N, 128, 500)
    y   = torch.tensor(npz["y"]).long()      # (N,)
    sid = npz["sid"]                         # (N,) str array
    print("Loaded:", X.shape, " pos/neg =", y.sum().item(), "/", len(y)-y.sum().item())

    gkf = GroupKFold(5)
    metrics = []

    for k, (tr, va) in enumerate(gkf.split(X, y, groups=sid)):
        print(f"\n======== Fold {k}  subjects(train)={len(np.unique(sid[tr]))}"
              f"  subjects(val)={len(np.unique(sid[va]))} ========")

        model = EEGNetAttn(chans=X.shape[1], samples=X.shape[2]).cuda()
        tr_loader = DataLoader(TensorDataset(X[tr], y[tr]), 64, shuffle=True)
        va_loader = DataLoader(TensorDataset(X[va], y[va]), 256)

        auc, state = train_fold(model, tr_loader, va_loader, fid=k)
        model.load_state_dict(state); model.cuda().eval()

        # ----------- batched inference for ACC -----------
        with torch.no_grad():
            pr_cls = []
            for xb, _ in va_loader:                  # ← 分批推断
                pr_cls.append(torch.argmax(model(xb.cuda()), 1).cpu())
        acc = accuracy_score(y[va], torch.cat(pr_cls))

        metrics.append({"fold": k, "auc": auc, "acc": acc})
        torch.save(state, f"results/best_fold{k}.ckpt")
        print(f"[Fold {k}] best  AUC {auc:.4f} | ACC {acc:.4f}  ✅ ckpt saved")

    pd.DataFrame(metrics).to_csv("results/metrics.csv", index=False)
    print("\nAll folds finished — results/metrics.csv  |  total {:.1f}s".format(time.time()-t0))

# ----------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",
        default="data/processed/modma_erp_withsid.npz",
        help="npz file that contains X, y, sid")
    main(parser.parse_args())
