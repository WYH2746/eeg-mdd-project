import numpy as np, sys
npz = np.load("data/processed/modma_erp.npz")
print("X shape:", npz["X"].shape, "   y shape:", npz["y"].shape)
