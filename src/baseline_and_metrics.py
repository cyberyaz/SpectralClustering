import numpy as np
import csv
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import numpy as np, csv
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

# SPECTRAL
X = np.load("data/X_moons.npy")
y_true = np.load("data/y_moons.npy")
y_spec = np.load("labels/labels_spectral_moons.npy")

# K-MEANS
y_kmeans = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(X)

# GAUSSIAN MIXTURE
gm = GaussianMixture(n_components=2, covariance_type='full', random_state=0)
y_gm = gm.fit_predict(X)

# AGGLOMERATIVE
agg = AgglomerativeClustering(n_clusters=2, linkage='single')
y_agg = agg.fit_predict(X)

# Adjusted random score: compares predicted clustering to ground truth clustering
#                        and adjusts for randomness (some points will recieve the
#                        correct label even if labeled randomly)
# Normalized mutual info score: just finds the percentage of common points between
#                               the two clusters (similar to adjusted random, but
#                                does not adjust for randomness)
# Silhouette score: compares each datapoint to its cluster
def scores(y_true, y_pred):
    return (
        adjusted_rand_score(y_true, y_pred),
        normalized_mutual_info_score(y_true, y_pred),
        silhouette_score(X, y_pred),
    )

spec_rand_score, spec_normalized_score, spec_silhouette_score = scores(y_true, y_spec)
kmeans_rand_score, kmeans_normalized_score, kmeans_silhouette_score = scores(y_true, y_kmeans)
gm_rand_score, gm_normalized_score, gm_silhouette_score = scores(y_true, y_gm)
agg_rand_score, aggs_normalized_score, agg_silhouette_score = scores(y_true, y_agg)

print("Spectral:", {"AR": spec_rand_score, "Norm": spec_normalized_score, "Silhouette": spec_silhouette_score})
print("Spectral:", {"AR": kmeans_rand_score, "Norm": kmeans_normalized_score, "Silhouette": kmeans_silhouette_score})
print("Spectral:", {"AR": gm_rand_score, "Norm": gm_normalized_score, "Silhouette": gm_silhouette_score})
print("Spectral:", {"AR": agg_rand_score, "Norm": aggs_normalized_score, "Silhouette": agg_silhouette_score})

# Write results
with open("results/metrics.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Method", "AR", "Norm", "Silhouette"])
    w.writerow(["Spectral", spec_rand_score, spec_normalized_score, spec_silhouette_score])
    w.writerow(["KMeans_raw", kmeans_rand_score, kmeans_normalized_score, kmeans_silhouette_score])
    w.writerow(["Spectral", gm_rand_score, gm_normalized_score, gm_silhouette_score])
    w.writerow(["Spectral", agg_rand_score, aggs_normalized_score, agg_silhouette_score])

# Compare spectral to other methods
kmean_diff = 100.0 * np.mean(y_spec != y_kmeans)
gm_diff = 100.0 * np.mean(y_spec != y_gm)
agg_diff = 100.0 * np.mean(y_spec != y_agg)

with open("results/metrics.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Method", "ARI", "NMI", "Silhouette"])
    w.writerow(["Spectral", ari_s, nmi_s, sil_s])
    w.writerow(["KMeans_raw", ari_k, nmi_k, sil_k])


pct_diff = 100.0 * np.mean(y_spec != y_kmeans)
with open("results/differences.txt", "w") as f:
    f.write(f"% of points clustered differently (Spectral vs KMeans): {kmean_diff:.2f}%\n")
    f.write(f"% of points clustered differently (Spectral vs Gaussian Mix): {gm_diff:.2f}%\n")
    f.write(f"% of points clustered differently (Spectral vs Agglomerative): {agg_diff:.2f}%\n")
