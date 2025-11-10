from sklearn.cluster import KMeans
import numpy as np

# --- Load data produced in Step 1 ---
X = np.load("data/X_moons.npy")   # shape (n, d)