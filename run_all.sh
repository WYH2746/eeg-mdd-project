#!/usr/bin/env bash
set -e
echo "[1/4] Pre-processing"
python -m eeg_code.preprocess --data_dir data\EEG_128channels_ERP_lanzhou_2015 --out data\processed\modma_erp.npz

echo "[2/4] Training 5-fold"
python -m eeg_code.train --data data/processed/modma_erp_withsid.npz

echo "[3/4] Generating explanations (fold0)"
python -m eeg_code.explain

echo "[4/4] Done. Results under ./results"
