

import json
import numpy as np
from scipy.sparse import csr_matrix, save_npz
from sklearn.neighbors import kneighbors_graph


X = np.load("data/X_moons.npy") 


n_neighbors = 10   


A = kneighbors_graph(
    X,
    n_neighbors=n_neighbors,
    mode="distance",
    include_self=False
)
A = 0.5 * (A + A.T) 


d = A.data
sigma = float(np.median(d)) if d.size else 1.0 
W_data = np.exp(-(d**2) / (2 * sigma**2))

W = csr_matrix((W_data, A.indices, A.indptr), shape=A.shape)
W = 0.5 * (W + W.T)  


save_npz("graphs/W_moons.npz", W)
with open("graphs/params.json", "w") as f:
    json.dump(
        {"moons": {"n_neighbors": n_neighbors, "sigma": "median_neighbor_distance"}},
        f,
        indent=2
    )


deg = np.array(W.sum(axis=1)).ravel()
print("W shape:", W.shape)
print("Nonzeros:", W.nnz)
print("Symmetric:", (W - W.T).nnz == 0)
print("Zero-degree nodes:", int(np.sum(deg == 0)))
print("Sigma (median neighbor distance):", sigma)
