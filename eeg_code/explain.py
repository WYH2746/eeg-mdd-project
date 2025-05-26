#!/usr/bin/env python
"""
Explain EEGNet-Attention predictions with Grad-CAM & Integrated Gradients
------------------------------------------------------------------------
* 对每段输入 (128×T) 生成 128 × T′ attribution
* 结果目录结构
    results/
      cam_heatmaps/   cam_0.png …
      ig_matrices/    ig_0.npy …
      topo/           topo_0.png  (可选)
"""

import argparse, pathlib, numpy as np, torch, matplotlib.pyplot as plt
from tqdm import tqdm
from captum.attr import LayerGradCam, IntegratedGradients
from eeg_code.model import EEGNetAttn

# ---------- CLI ----------
p = argparse.ArgumentParser()
p.add_argument("--data",     default="data/processed/modma_erp_withsid.npz")
p.add_argument("--ckpt",     default="results/best_fold0.ckpt")
p.add_argument("--max_n",    type=int, default=100, help="samples to explain")
p.add_argument("--topomap",  action="store_true",   help="save 300 ms scalp map")
args = p.parse_args() if __name__ == "__main__" else None

# ---------- 环境 ----------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.switch_backend("Agg")           # 防止无 X11 报错
torch.set_grad_enabled(False)

# ---------- 输出文件夹 ----------
root_out = pathlib.Path("results")
d_cam = root_out / "cam_heatmaps"; d_cam.mkdir(parents=True, exist_ok=True)
d_ig  = root_out / "ig_matrices";  d_ig.mkdir(parents=True, exist_ok=True)
if args and args.topomap:
    d_topo = root_out / "topo";    d_topo.mkdir(exist_ok=True)

# ---------- 载模型 ----------
model = EEGNetAttn(chans=128, samples=500).to(DEVICE).eval()
state = torch.load(args.ckpt, map_location="cpu")
model.load_state_dict(state)

# Grad-CAM 工具：选更靠前卷积保持空间分辨率 (32×T′)
cam_explainer = LayerGradCam(model, model.block1[-1])
ig_explainer  = IntegratedGradients(model)

# ---------- 读数据 ----------
X_all = np.load(args.data, mmap_mode="r")["X"][: args.max_n]
print(f"Explaining {len(X_all)} samples on {DEVICE} …")

# ---------- 用于统一配色的全局最大值 ----------
global_max = 0.0
cams_cache = []

for x in tqdm(X_all, desc="Forward attribution"):
    x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    # Grad-CAM  (32×T′) → repeat 到 128×T′
    cam = cam_explainer.attribute(x_t, target=1).squeeze(0).cpu().numpy()
    cam = np.repeat(cam, 4, axis=0)[:128]
    cams_cache.append(cam)
    global_max = max(global_max, cam.max())

# ---------- 再次遍历保存 PNG & IG ----------
for idx, (x, cam) in enumerate(zip(X_all, cams_cache)):
    # ----- Grad-CAM PNG -----
    cam_norm = cam / (global_max + 1e-9)           # 0-1 统一配色
    arr = cam_norm                   # (≥2-D 皆可)
    arr = np.squeeze(arr)            # 去掉 size=1 维；变 (128, T′)
    if arr.ndim != 2:
        # 若仍是 3-D（说明第三维 k>1，非法），可选做均值或取最大
        arr = arr.mean(axis=0)       # -> (128, T′)

    plt.imsave(d_cam/f"cam_{idx}.png",
            arr, cmap="hot", vmin=0, vmax=1, format="png")

    plt.imsave(d_cam / f"cam_{idx}.png",
            arr, cmap="hot", vmin=0, vmax=1, format="png")
    # ----- IG npy -----
    x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    ig = ig_explainer.attribute(x_t, target=1).squeeze(0).cpu().numpy()
    np.save(d_ig / f"ig_{idx}.npy", ig.astype(np.float32))

    # ----- 可选头皮拓扑图 (300 ms) -----
    if args and args.topomap:
        try:
            import mne
            info = mne.create_info(
                ch_names=[f"E{i+1}" for i in range(128)],
                sfreq=250, ch_types="eeg")
            evk = mne.EvokedArray(cam[:, 75:325].mean(axis=1, keepdims=True),
                                  info, tmin=0)
            evk.set_montage("GSN-HydroCel-128")
            evk.plot_topomap(times=[0], ch_type="eeg", cmap="hot",
                             time_format="", show=False)
            plt.savefig(d_topo / f"topo_{idx}.png", dpi=150,
                        bbox_inches="tight", pad_inches=0)
            plt.close()
        except Exception as e:
            print(f"[WARN] topo {idx}: {e}")

print("✅  Grad-CAM & IG 生成完毕   ->  results/")
