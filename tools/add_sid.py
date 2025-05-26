"""
tools/add_sid.py  –  append subject_id array to an existing npz

Usage
-----
python tools/add_sid.py \
       --npz data/processed/modma_erp.npz \
       --raw_dir data/EEG_128channels_ERP_lanzhou_2015 \
       --keep            # 可选：保留原文件，写 *.withsid.npz
"""
import argparse, pathlib, re, numpy as np, mne, sys

# ---------- 尝试读取 .raw（CNT → EGI） ----------
def read_raw_auto(fp):
    try:
        return mne.io.read_raw_cnt(fp, preload=False, verbose=False)
    except Exception:
        return mne.io.read_raw_egi(fp, preload=False, verbose=False)

def epoch_count(raw_fp):
    raw = read_raw_auto(raw_fp)           # ← 改这里
    sfreq = raw.info["sfreq"]
    return int(raw.n_times // (sfreq * 2.0))  # 2-s epoch 数

def main(a):
    npz_path = pathlib.Path(a.npz)
    raw_dir  = pathlib.Path(a.raw_dir)
    data     = np.load(npz_path, allow_pickle=True)
    N        = data["X"].shape[0]

    sid_all = []
    for raw_fp in sorted(raw_dir.glob("**/*.raw")):
        m = re.match(r"(\d{8})", raw_fp.name)
        if not m:
            continue
        sub_id = m.group(1)
        n_ep   = epoch_count(raw_fp)
        sid_all.extend([sub_id] * n_ep)
        if len(sid_all) >= N:
            sid_all = sid_all[:N]
            break

    if len(sid_all) != N:
        print(f"[ERR] epoch mismatch  sid={len(sid_all)}  X={N}")
        sys.exit(1)

    out_path = (npz_path.with_stem(npz_path.stem + "_withsid")  # pathlib>=3.10
                if a.keep else npz_path)
    np.savez_compressed(
        out_path,
        X=data["X"], y=data["y"],
        sid=np.array(sid_all, dtype="U8")
    )
    print("✅  sid added →", out_path)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--npz",     required=True)
    p.add_argument("--raw_dir", required=True)
    p.add_argument("--keep",    action="store_true")
    main(p.parse_args())
