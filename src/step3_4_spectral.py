# src/step3_4_spectral.py
# Build the normalized Laplacian, compute the k smallest eigenvectors,
# row-normalize them, run k-means in that space, and save outputs.

import numpy as np
from scipy.sparse import load_npz, diags
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans

# --- Load similarity matrix from Step 2 ---
W = load_npz("graphs/W_moons.npz")  # sparse (n x n)

# --- Normalized Laplacian: L_sym = D^{-1/2} (D - W) D^{-1/2} ---
deg = np.array(W.sum(axis=1)).ravel()
deg_safe = np.where(deg > 0, deg, 1.0)  # avoid divide-by-zero
D_inv_sqrt = diags(1.0 / np.sqrt(deg_safe))
L = diags(deg) - W
Lsym = D_inv_sqrt @ L @ D_inv_sqrt

# Sanity: symmetry (required by eigsh with "SM")
sym_ok = (Lsym - Lsym.T).nnz == 0
print("L_sym symmetric:", sym_ok)

# --- Eigenvectors: take the k smallest (k = number of clusters) ---
k = 2  # two moons => 2 clusters
eigvals, eigvecs = eigsh(Lsym, k=k, which="SM")  # smallest magnitude eigenpairs
print("Eigenvalues (smallest k):", eigvals)

# --- Row-normalize the embedding U ---
U = eigvecs / (np.linalg.norm(eigvecs, axis=1, keepdims=True) + 1e-12)

# --- k-means on rows of U ---
km = KMeans(n_clusters=k, n_init=10, random_state=0)
labels = km.fit_predict(U)

# --- Save artifacts for teammates/plots ---
np.save("spectral/U_moons.npy", U)
np.save("labels/labels_spectral_moons.npy", labels)

# --- Console checks ---
print("U shape:", U.shape)
print("labels shape:", labels.shape, "| unique labels:", np.unique(labels))
row_norms = np.linalg.norm(U, axis=1)
print("U row-norm stats -> min:", float(row_norms.min()), "max:", float(row_norms.max()))
