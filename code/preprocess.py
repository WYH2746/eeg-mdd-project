# code/preprocess.py
"""
Pre-process MODMA (ERP .raw) for MDD/HC classification
-----------------------------------------------------
Example
    python -m code.preprocess \
        --data_dir ../data/EEG_128channels_ERP_lanzhou_2015 \
        --out      ../data/processed/modma_erp.npz
"""

import argparse, pathlib, re
import numpy as np
import pandas as pd
import mne

# --------------------------------------------------------------------
# ----------  预处理步骤：滤波 → ICA 去伪迹 → 切 2 s epoch  ----------
# --------------------------------------------------------------------
def bandpass(raw, l_freq=1.0, h_freq=45.0):
    raw.load_data().filter(l_freq, h_freq, fir_design="firwin")
    return raw


def ica_clean(raw, n_components=20):
    ica = mne.preprocessing.ICA(n_components=n_components, random_state=0)
    ica.fit(raw)
    ica.detect_artifacts(raw)
    ica.apply(raw)
    return raw


def epoch_and_label(raw, label, t_len=2.0):
    epochs = mne.make_fixed_length_epochs(raw, duration=t_len, preload=True)
    X = epochs.get_data()            # (N, C, T)
    y = np.full(len(X), label, dtype=np.int64)
    return X, y


# --------------------------------------------------------------------
def read_raw_auto(fp: pathlib.Path):
    """
    Try multiple MNE readers to open a *.raw file.  Extend if needed.
    """
    try:
        return mne.io.read_raw_cnt(fp, preload=False)   # Neuroscan
    except Exception:
        try:
            return mne.io.read_raw_egi(fp, preload=False)  # EGI
        except Exception as e:
            raise RuntimeError(
                f"Cannot read {fp.name}. "
                "Please replace 'read_raw_auto' with the correct MNE reader. "
                f"Underlying error: {e}"
            )


# --------------------------------------------------------------------
def main(args):
    root = pathlib.Path(args.data_dir)

    # 1) 读取 subject_info.xlsx → dict: {subject_id: label}
    info_file = next(root.glob("subject_info.*"), None)
    if info_file is None:
        raise FileNotFoundError("subject_info.xlsx / .xls / .csv not found in data_dir")
    if info_file.suffix in (".xlsx", ".xls"):
        df = pd.read_excel(info_file)
    else:
        df = pd.read_csv(info_file)
    # 约定列名：subject / group；自定义请自行修改
    id2label = {
        str(row["subject"]).zfill(8): 1 if str(row["group"]).upper() == "MDD" else 0
        for _, row in df.iterrows()
    }

    # 2) 遍历所有 .raw 文件
    Xs, ys = [], []
    for raw_fp in root.glob("**/*.raw"):
        # 文件名前 8 位数字为被试 ID，例如 02010002erp****.raw
        m = re.match(r"(\d{8})", raw_fp.name)
        if not m:
            print(f"[WARN] skip {raw_fp.name} (cannot parse subject id)")
            continue
        sub_id = m.group(1)
        if sub_id not in id2label:
            print(f"[WARN] {sub_id} not in subject_info → skip")
            continue
        label = id2label[sub_id]

        # 读取、预处理
        raw = read_raw_auto(raw_fp)
        raw = bandpass(raw)
        raw = ica_clean(raw)
        Xi, yi = epoch_and_label(raw, label)
        Xs.append(Xi)
        ys.append(yi)
        print(f"✔  {raw_fp.name:40s}  epochs={len(Xi)},  label={label}")

    if not Xs:
        raise RuntimeError("No valid .raw files processed — check paths & formats")

    # 3) 保存
    X = np.concatenate(Xs)          # (N_total, C, T)
    y = np.concatenate(ys)
    np.savez_compressed(args.out, X=X, y=y)
    print(f"\nSaved {args.out} …  shape={X.shape},  pos/neg={y.sum()}/{len(y)-y.sum()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True, help="Folder with *.raw + subject_info.*")
    p.add_argument("--out", default="../data/processed/modma_erp.npz")
    main(p.parse_args())
