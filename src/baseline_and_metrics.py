
import numpy as np, csv
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

X = np.load("data/X_moons.npy")
y_true = np.load("data/y_moons.npy")
y_spec = np.load("labels/labels_spectral_moons.npy")


y_kmeans = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(X)

def scores(y_true, y_pred):
    return (
        adjusted_rand_score(y_true, y_pred),
        normalized_mutual_info_score(y_true, y_pred),
        silhouette_score(X, y_pred),
    )

ari_s, nmi_s, sil_s = scores(y_true, y_spec)
ari_k, nmi_k, sil_k = scores(y_true, y_kmeans)

print("Spectral:", {"ARI": ari_s, "NMI": nmi_s, "Silhouette": sil_s})
print("KMeans  :", {"ARI": ari_k, "NMI": nmi_k, "Silhouette": sil_k})


with open("results/metrics.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Method", "ARI", "NMI", "Silhouette"])
    w.writerow(["Spectral", ari_s, nmi_s, sil_s])
    w.writerow(["KMeans_raw", ari_k, nmi_k, sil_k])


pct_diff = 100.0 * np.mean(y_spec != y_kmeans)
with open("results/differences.txt", "w") as f:
    f.write(f"% of points clustered differently (Spectral vs KMeans): {pct_diff:.2f}%\n")

print(f"Saved results/metrics.csv and differences.txt (diff={pct_diff:.2f}%)")
