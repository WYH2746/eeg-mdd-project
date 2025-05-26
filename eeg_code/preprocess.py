# eeg_code/preprocess.py  ★最终稳定版
import argparse, pathlib, re, numpy as np, pandas as pd, mne

# ---------- 常量：EGI 128 通道（大写） ----------
EGI_128_ORDER = [
    "E1","E2","E3","E4","E5","E6","E7","E8","E9","E10","E11","E12","E13","E14","E15","E16",
    "E17","E18","E19","E20","E21","E22","E23","E24","E25","E26","E27","E28","E29","E30","E31","E32",
    "E33","E34","E35","E36","E37","E38","E39","E40","E41","E42","E43","E44","E45","E46","E47","E48",
    "E49","E50","E51","E52","E53","E54","E55","E56","E57","E58","E59","E60","E61","E62","E63","E64",
    "E65","E66","E67","E68","E69","E70","E71","E72","E73","E74","E75","E76","E77","E78","E79","E80",
    "E81","E82","E83","E84","E85","E86","E87","E88","E89","E90","E91","E92","E93","E94","E95","E96",
    "E97","E98","E99","E100","E101","E102","E103","E104","E105","E106","E107","E108","E109","E110",
    "E111","E112","E113","E114","E115","E116","E117","E118","E119","E120","E121","E122","E123","E124",
    "E125","E126","E127","E128"
]

# ------------------ 信号工具 ------------------
def bandpass(raw, l_freq=1.0, h_freq=45.0):
    raw.load_data().filter(l_freq, h_freq, fir_design="firwin")
    return raw

def ica_clean(raw, n_components=20):
    ica = mne.preprocessing.ICA(n_components=n_components, random_state=0, max_iter="auto", verbose=False)
    ica.fit(raw)
    try:
        eog_idx, _ = ica.find_bads_eog(raw, threshold=3.0)
        ecg_idx, _ = ica.find_bads_ecg(raw, threshold=0.6)
        ica.exclude = list(set(eog_idx + ecg_idx))
    except Exception:
        ica.exclude = []
    ica.apply(raw, verbose=False)
    return raw

def epoch_and_label(raw, label, t_len=2.0):
    epochs = mne.make_fixed_length_epochs(raw, duration=t_len, preload=True)
    X = epochs.get_data().astype(np.float32)               # ★ 强制 float32
    y = np.full(len(X), label, dtype=np.int64)
    return X, y

def read_raw_auto(fp: pathlib.Path):
    try:
        return mne.io.read_raw_cnt(fp, preload=False, verbose=False)
    except Exception:
        return mne.io.read_raw_egi(fp, preload=False, verbose=False)

# ------------------ 主流程 ------------------
def main(args):
    root = pathlib.Path(args.data_dir)

    # 1) 解析标签
    info_file = next(root.glob("*subject*info*.xls*"), None)
    if info_file is None:
        raise FileNotFoundError("subject_info.xlsx / .xls / .csv 不存在")
    df = (pd.read_excel if info_file.suffix in (".xlsx", ".xls") else pd.read_csv)(info_file)
    df.columns = [c.lower().strip() for c in df.columns]
    subj_col = next(c for c in df.columns if any(k in c for k in ["subject", "sub", "id"]))
    diag_col = next(c for c in df.columns if any(k in c for k in ["group", "type", "mdd", "label", "diag"]))
    to_label = lambda v: 1 if str(v).upper() in ("1","MDD","PATIENT","DEPRESSION") else 0
    id2label = {str(r[subj_col]).zfill(8): to_label(r[diag_col]) for _, r in df.iterrows()}

    Xs, ys = [], []
    for raw_fp in root.glob("**/*.raw"):
        m = re.match(r"(\d{8})", raw_fp.name)
        if not m or m.group(1) not in id2label:
            continue
        label = id2label[m.group(1)]

        # 2) 读取 + 仅 EEG 信道
        raw = read_raw_auto(raw_fp)
        raw.pick_types(eeg=True, stim=False)                       # ★ 去 Stim
        raw.rename_channels({ch: ch.upper() for ch in raw.ch_names})

        # 3) 对齐到固定 128 通道
        extra = [ch for ch in raw.ch_names if ch not in EGI_128_ORDER]
        if extra:
            raw.drop_channels(extra)
        missing = [ch for ch in EGI_128_ORDER if ch not in raw.ch_names]
        if missing:
            zeros = np.zeros((len(missing), raw.n_times), dtype=np.float32)
            raw.add_channels([mne.io.RawArray(zeros, mne.create_info(missing, raw.info["sfreq"], "eeg"))],
                             force_update_info=True)
        raw.reorder_channels(EGI_128_ORDER)

        # 4) Filter+ICA
        raw = bandpass(raw)
        raw = ica_clean(raw)

        # 5) 切片
        Xi, yi = epoch_and_label(raw, label)
        Xs.append(Xi); ys.append(yi)
        print(f"✔ {raw_fp.name:30s}  epochs={len(Xi):4d} label={label}")

    if not Xs:
        raise RuntimeError("无有效样本")

    # 6) 拼接保存
    X = np.concatenate(Xs, 0)
    y = np.concatenate(ys, 0)
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, X=X, y=y)
    print(f"\n✅ 已保存 {args.out} — shape={X.shape},  pos/neg={y.sum()}/{len(y)-y.sum()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="包含 *.raw 的目录")
    parser.add_argument("--out", default="../data/processed/modma_erp.npz")
    main(parser.parse_args())
