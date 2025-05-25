#!/usr/bin/env bash
set -e
echo "[1/5] Pre-processing"
python -m code.preprocess --data_dir data/MODMA --out data/processed/modma.npz

echo "[2/5] Training 5-fold"
python -m code.train --data data/processed/modma.npz

echo "[3/5] Generating explanations (fold0)"
python -m code.explain

echo "[4/5] Statistics"
python -m code.stats

echo "[5/5] Done. Results under ./results"
