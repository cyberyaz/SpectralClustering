
import numpy as np
from sklearn.datasets import make_moons


X, y = make_moons(n_samples=600, noise=0.08, random_state=0)


np.save("data/X_moons.npy", X)


np.save("data/y_moons.npy", y)


print("Saved:", "data/X_moons.npy", "data/y_moons.npy", "| shapes:", X.shape, y.shape)
