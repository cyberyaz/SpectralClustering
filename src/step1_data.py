# src/step1_data.py
import numpy as np
from sklearn.datasets import make_moons

# Generate a synthetic two-dimensional dataset using the "moons" shape, 
# with 600 samples and some noise
X, y = make_moons(n_samples=600, noise=0.08, random_state=0)

# Save the feature matrix X to a NumPy binary file for later use
np.save("data/X_moons.npy", X)

# Save the target vector y to a separate NumPy binary file
np.save("data/y_moons.npy", y)

# Print confirmation that the files were saved, including their file paths and the shapes of X and y
print("Saved:", "data/X_moons.npy", "data/y_moons.npy", "| shapes:", X.shape, y.shape)
