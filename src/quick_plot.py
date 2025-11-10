# src/quick_plot.py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

X = np.load("data/X_moons.npy")
labels = np.load("labels/labels_spectral_moons.npy")

plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=labels, s=12)
plt.title("Spectral clustering on two moons")
plt.tight_layout()
plt.savefig("figs/moons_spectral.png", dpi=150)
print("Saved figs/moons_spectral.png")
