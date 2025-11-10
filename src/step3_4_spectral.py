
import numpy as np
from scipy.sparse import load_npz, diags
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans


W = load_npz("graphs/W_moons.npz")  


deg = np.array(W.sum(axis=1)).ravel()
deg_safe = np.where(deg > 0, deg, 1.0)  
D_inv_sqrt = diags(1.0 / np.sqrt(deg_safe))
L = diags(deg) - W
Lsym = D_inv_sqrt @ L @ D_inv_sqrt


sym_ok = (Lsym - Lsym.T).nnz == 0
print("L_sym symmetric:", sym_ok)


k = 2  
eigvals, eigvecs = eigsh(Lsym, k=k, which="SM")  
print("Eigenvalues (smallest k):", eigvals)


U = eigvecs / (np.linalg.norm(eigvecs, axis=1, keepdims=True) + 1e-12)


km = KMeans(n_clusters=k, n_init=10, random_state=0)
labels = km.fit_predict(U)


np.save("spectral/U_moons.npy", U)
np.save("labels/labels_spectral_moons.npy", labels)


print("U shape:", U.shape)
print("labels shape:", labels.shape, "| unique labels:", np.unique(labels))
row_norms = np.linalg.norm(U, axis=1)
print("U row-norm stats -> min:", float(row_norms.min()), "max:", float(row_norms.max()))
