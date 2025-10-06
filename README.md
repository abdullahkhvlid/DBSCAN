## DBSCAN Clustering Project

### Overview

This project demonstrates the implementation of the **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** algorithm in Python.
The goal is to understand how DBSCAN groups data points into clusters based on density, and how it detects noise and border points.

---

### Key Concepts

**DBSCAN Parameters**

* **eps:** The maximum distance between two samples for them to be considered neighbors.
* **min_samples:** The minimum number of points required to form a dense region (cluster).
* **metric:** Defines how the distance between points is calculated (default is Euclidean).
* **algorithm:** Determines the method used to find nearest neighbors (auto, ball_tree, kd_tree, brute).
* **leaf_size:** Controls the size of leaf nodes in BallTree or KDTree algorithms.
* **metric_params:** Dictionary of additional parameters for the chosen metric.
* **p:** Power parameter for Minkowski metric (p=2 means Euclidean).

---

### Point Categories

* **Core Points:** Points having at least `min_samples` neighbors within `eps`.
* **Border Points:** Points that are within the neighborhood of a core point but are not themselves core points.
* **Noise Points:** Points that do not belong to any cluster.

---

### Implementation Steps

1. Import the required libraries:

   ```python
   import numpy as np
   import matplotlib.pyplot as plt
   from sklearn.datasets import make_moons
   from sklearn.preprocessing import StandardScaler
   from sklearn.cluster import DBSCAN
   ```

2. Create and scale the dataset:

   ```python
   X, y = make_moons(n_samples=200, noise=0.1, random_state=42)
   scaled_X = StandardScaler().fit_transform(X)
   ```

3. Apply DBSCAN:

   ```python
   model = DBSCAN(eps=0.3, min_samples=5)
   model.fit(scaled_X)
   labels = model.labels_
   ```

4. Identify core samples:

   ```python
   core_mask = np.zeros_like(labels, dtype=bool)
   core_mask[model.core_sample_indices_] = True
   ```

5. Visualize clusters:

   ```python
   plt.scatter(scaled_X[:, 0], scaled_X[:, 1], c=labels, cmap='rainbow')
   plt.title('DBSCAN Clustering Results')
   plt.show()
   ```

---

### Outputs

* **labels:** Cluster label assigned to each point. `-1` indicates a noise point.
* **core_sample_indices_:** Indices of all core points in the dataset.
* **components_:** Coordinates of all core points.

---

### Insights

* DBSCAN is ideal for datasets with irregular cluster shapes (e.g., moons).
* Unlike K-Means, DBSCAN automatically detects the number of clusters.
* Scaling the data is essential to ensure distance-based accuracy.

---

### Dependencies

* Python 3.x
* numpy
* matplotlib
* scikit-learn

---

