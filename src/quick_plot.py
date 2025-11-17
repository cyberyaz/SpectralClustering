
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture


X = np.load("data/X_moons.npy")
y_true = np.load("data/y_moons.npy")
y_spectral = np.load("labels/labels_spectral_moons.npy")

# Recreates results of other models
print("Regenerating clustering results for visualization...")
y_kmeans = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(X)
y_gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=0).fit_predict(X)
y_agg = AgglomerativeClustering(n_clusters=2, linkage='single').fit_predict(X)


fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle("Clustering Methods Comparison on Two Moons Dataset", fontsize=16, fontweight='bold')
methods = [
    ("Ground Truth", y_true, axes[0, 0]),
    ("Spectral Clustering", y_spectral, axes[0, 1]),
    ("K-Means", y_kmeans, axes[0, 2]),
    ("Gaussian Mixture Model", y_gmm, axes[1, 0]),
    ("Agglomerative (Single Link)", y_agg, axes[1, 1]),
]
colors = ['#2E86AB', '#A23B72']  # Blue and Purple
cmap = matplotlib.colors.ListedColormap(colors)


for title, labels, ax in methods:
    # which cluster gets which label is arbitrary so take whichever performs
    # better
    accuracy1 = np.mean(labels == y_true)
    accuracy2 = np.mean(labels == (1 - y_true))
    if accuracy2 > accuracy1:
        labels_plot = 1 - labels
    else:
        labels_plot = labels
    
    scatter = ax.scatter(X[:, 0], X[:, 1], c=labels_plot, s=20, 
                        cmap=cmap, alpha=0.8, edgecolors='black', linewidth=0.5)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

ax_table = axes[1, 2]
ax_table.axis('tight')
ax_table.axis('off')

import csv
metrics_data = []
with open("results/metrics.csv", 'r') as f:
    reader = csv.reader(f)
    headers = next(reader)
    for row in reader:
        metrics_data.append(row)


table_data = [headers]
for row in metrics_data:
    formatted_row = [row[0]] + [f"{float(val):.3f}" for val in row[1:]]
    table_data.append(formatted_row)

cell_colors = [['lightgray'] * len(headers)]
for row in metrics_data:
    row_colors = ['white']
    for val in row[1:3]:
        if float(val) > 0.9:
            row_colors.append('lightgreen')
        elif float(val) > 0.5:
            row_colors.append('lightyellow')
        else:
            row_colors.append('lightcoral')
    row_colors.append('white')
    cell_colors.append(row_colors)

table = ax_table.table(cellText=table_data, cellColours=cell_colors,
                      cellLoc='center', loc='center',
                      colWidths=[0.3, 0.2, 0.2, 0.3])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 2)

ax_table.set_title('Performance Metrics\n(Green=Good, Yellow=Moderate, Red=Poor)', 
                   fontsize=12, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig("figs/all_methods_comparison.png", dpi=200, bbox_inches='tight')
print("Saved: figs/all_methods_comparison.png")

fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
fig2.suptitle("Disagreement Analysis: Points Where Methods Differ from Spectral", 
              fontsize=14, fontweight='bold')

comparisons = [
    ("K-Means vs Spectral", y_kmeans, axes2[0]),
    ("GMM vs Spectral", y_gmm, axes2[1]),
    ("Agglomerative vs Spectral", y_agg, axes2[2])
]

for title, labels, ax in comparisons:
    disagree1 = (y_spectral != labels)
    disagree2 = (y_spectral != (1 - labels))
    if disagree2.sum() < disagree1.sum():
        disagreement = disagree2
        labels_aligned = 1 - labels
    else:
        disagreement = disagree1
        labels_aligned = labels
    
    colors_disagree = ['gray' if not d else 'red' for d in disagreement]
    sizes = [15 if not d else 40 for d in disagreement]
    
    ax.scatter(X[:, 0], X[:, 1], c=colors_disagree, s=sizes, alpha=0.7)
    ax.set_title(f"{title}\n({disagreement.sum()} points differ = {100*disagreement.mean():.1f}%)", 
                fontsize=11)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.savefig("figs/disagreement_analysis.png", dpi=200, bbox_inches='tight')
print("Saved: figs/disagreement_analysis.png")

print("\nGenerating individual method plots...")

fig_spectral = plt.figure(figsize=(6, 5))

if np.mean(y_spectral == (1 - y_true)) > np.mean(y_spectral == y_true):
    y_spectral_plot = 1 - y_spectral
else:
    y_spectral_plot = y_spectral

plt.scatter(X[:, 0], X[:, 1], c=y_spectral_plot, s=30, cmap=cmap,
           alpha=0.8, edgecolors='black', linewidth=0.5)
plt.title("Spectral Clustering", fontsize=14, fontweight='bold')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True, alpha=0.3)
plt.gca().set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.savefig("figs/moons_spectral.png", dpi=150)
plt.close()
print("Saved: figs/moons_spectral.png")

for method_name, labels, full_name in [
    ("kmeans", y_kmeans, "K-Means"),
    ("gmm", y_gmm, "Gaussian Mixture Model"),
    ("agg", y_agg, "Agglomerative (Single Link)")
]:
    fig_single = plt.figure(figsize=(6, 5))
    
    if np.mean(labels == (1 - y_true)) > np.mean(labels == y_true):
        labels = 1 - labels
    
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=30, cmap=cmap, 
               alpha=0.8, edgecolors='black', linewidth=0.5)
    plt.title(f"{full_name} Clustering Result", fontsize=14, fontweight='bold')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True, alpha=0.3)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig(f"figs/{method_name}_clustering.png", dpi=150)
    plt.close()
    print(f"Saved: figs/{method_name}_clustering.png")

print("\nAll visualizations saved successfully!")
print("\nFiles created:")
print("  - figs/all_methods_comparison.png : Main comparison grid with metrics")
print("  - figs/disagreement_analysis.png : Shows where methods differ from spectral")  
print("  - figs/moons_spectral.png : Spectral clustering result")
print("  - figs/kmeans_clustering.png : K-means result")
print("  - figs/gmm_clustering.png : GMM result")
print("  - figs/agg_clustering.png : Agglomerative result")