# eeg_code/explain_avg.py  —— 生成两张平均时序热条

import numpy as np, torch, matplotlib.pyplot as plt, pathlib, warnings
from captum.attr import LayerGradCam, IntegratedGradients
from eeg_code.model import EEGNetAttn

DATA  = "data/processed/modma_erp_withsid.npz"
CKPT  = "results/best_fold0.ckpt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_first_n_per_class(X, y, n=50):
    idx0 = np.where(y == 0)[0][:n]
    idx1 = np.where(y == 1)[0][:n]
    if len(idx0) < n or len(idx1) < n:
        raise RuntimeError(f"样本不足：HC={len(idx0)}, MDD={len(idx1)}")
    return X[idx0], X[idx1]

def gradcam_batch(model, xs, layer):
    cam = LayerGradCam(model, layer)
    outs = []
    for x in xs:
        a = cam.attribute(torch.tensor(x).unsqueeze(0).float().to(DEVICE), target=1)
        outs.append(a.squeeze().cpu().detach().numpy())
    return np.array(outs)  # (N,C,T') or (N,T')

def ig_batch(model, xs):
    ig = IntegratedGradients(model)
    outs = []
    for x in xs:
        a = ig.attribute(torch.tensor(x).unsqueeze(0).float().to(DEVICE), target=1)
        outs.append(a.squeeze().cpu().detach().numpy())
    return np.array(outs)

def save_heat(name, arr):
    arr = np.squeeze(arr)            # 保证 ≤2 维
    if arr.ndim == 1:
        arr = arr[None, :]
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    plt.imsave(name, arr, cmap="hot")

def main():
    # ---------- 数据 ----------
    npz = np.load(DATA, allow_pickle=True)
    X, y = npz["X"], npz["y"]
    X0, X1 = load_first_n_per_class(X, y, n=50)
    print(f"Loaded 50 HC + 50 MDD")

    # ---------- 模型 ----------
    model = EEGNetAttn(chans=128, samples=500).to(DEVICE).eval()
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE))

    # ---------- attribution ----------
    cam0 = gradcam_batch(model, X0, model.block1[-1]).mean(axis=0)
    cam1 = gradcam_batch(model, X1, model.block1[-1]).mean(axis=0)
    ig0  = ig_batch(model, X0).mean(axis=0)
    ig1  = ig_batch(model, X1).mean(axis=0)

    out = pathlib.Path("results/avg_explain"); out.mkdir(parents=True, exist_ok=True)
    save_heat(out / "gradcam_HC.png",  cam0)
    save_heat(out / "gradcam_MDD.png", cam1)
    save_heat(out / "ig_HC.png",       ig0)
    save_heat(out / "ig_MDD.png",      ig1)

    print("✅ 已生成 4 张平均热条 → results/avg_explain/")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
