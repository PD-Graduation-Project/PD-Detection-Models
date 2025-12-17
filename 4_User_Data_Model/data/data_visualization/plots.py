import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def plot_age_histogram_and_find_best_split(csv_path, age_column="age"):
    """
    Plots a histogram of the age column and finds the best split
    of ages into 3 or 4 classes using K-Means clustering.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.
    age_column : str
        Name of the age column (default: 'age').

    Returns
    -------
    dict
        Contains:
        - best_k: chosen number of classes (3 or 4)
        - boundaries: age boundaries between classes
        - silhouette_scores: scores for k=3 and k=4
    """

    # -----------------------------
    # Load and clean data
    # -----------------------------
    df = pd.read_csv(csv_path)

    if age_column not in df.columns:
        raise ValueError(f"Column '{age_column}' not found in CSV file.")

    ages = df[age_column].dropna().values.reshape(-1, 1)

    # -----------------------------
    # Try K = 3 and K = 4
    # -----------------------------
    results = {}
    silhouette_scores = {}
    models = {}

    for k in [3, 4]:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = kmeans.fit_predict(ages)

        score = silhouette_score(ages, labels)
        silhouette_scores[k] = score
        models[k] = (kmeans, labels)

        centers = np.sort(kmeans.cluster_centers_.flatten())
        boundaries = [(centers[i] + centers[i + 1]) / 2 for i in range(len(centers) - 1)]
        results[k] = boundaries

    # -----------------------------
    # Select best K
    # -----------------------------
    best_k = max(silhouette_scores, key=silhouette_scores.get)
    kmeans, labels = models[best_k]

    centers = kmeans.cluster_centers_.flatten()
    order = np.argsort(centers)

    sorted_labels = np.zeros_like(labels)
    for new_id, old_id in enumerate(order):
        sorted_labels[labels == old_id] = new_id

    boundaries = results[best_k]
    min_age = ages.min()
    max_age = ages.max()

    # Build readable class ranges
    class_ranges = []
    for i in range(best_k):
        if i == 0:
            class_ranges.append(f"Age < {boundaries[0]:.1f}")
        elif i == best_k - 1:
            class_ranges.append(f"Age ≥ {boundaries[-1]:.1f}")
        else:
            class_ranges.append(
                f"{boundaries[i-1]:.1f} ≤ Age < {boundaries[i]:.1f}"
            )

    # -----------------------------
    # Colored histogram by class
    # -----------------------------
    plt.figure()

    for i in range(best_k):
        cluster_ages = ages[sorted_labels == i].flatten()
        plt.hist(
            cluster_ages,
            bins=15,
            alpha=0.7,
            label=f"Class {i + 1}: {class_ranges[i]}"
        )

    for b in boundaries:
        plt.axvline(b, linestyle="--")

    plt.xlabel("Age")
    plt.ylabel("Frequency")
    plt.title(f"Age Distribution Split into {best_k} Classes")
    plt.legend()
    plt.show()

    return {
        "best_k": best_k,
        "boundaries": boundaries,
        "silhouette_scores": silhouette_scores
    }


def plot_height_weight_and_find_best_classes(
    csv_path,
    height_column="height",
    weight_column="weight",
    height_range=(100, 300)  # tuple: (min_height, max_height)
):
    """
    Plots height vs weight and finds the best clustering
    into 3 or 4 classes using K-Means.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.
    height_column : str
        Name of the height column.
    weight_column : str
        Name of the weight column.
    height_range : tuple or None
        (min_height, max_height). Use None to disable filtering.

    Returns
    -------
    dict
        Contains:
        - best_k: chosen number of clusters (3 or 4)
        - centers: cluster centers (height, weight)
        - silhouette_scores: scores for k=3 and k=4
    """

    # -----------------------------
    # Load data
    # -----------------------------
    df = pd.read_csv(csv_path)

    for col in [height_column, weight_column]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in CSV file.")

    # -----------------------------
    # Apply height limit if given
    # -----------------------------
    if height_range is not None:
        min_h, max_h = height_range
        df = df[
            (df[height_column] >= min_h) &
            (df[height_column] <= max_h)
        ]

    # Drop missing values after filtering
    data = df[[height_column, weight_column]].dropna().values

    if len(data) < 4:
        raise ValueError("Not enough data points after height filtering.")

    # -----------------------------
    # Scatter plot (raw data)
    # -----------------------------
    plt.figure()
    plt.scatter(data[:, 0], data[:, 1])
    plt.xlabel("Height")
    plt.ylabel("Weight")
    plt.title("Height vs Weight")
    plt.show()

    # -----------------------------
    # Clustering (k = 3, 4)
    # -----------------------------
    silhouette_scores = {}
    models = {}

    for k in [3, 4]:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = kmeans.fit_predict(data)
        silhouette_scores[k] = silhouette_score(data, labels)
        models[k] = (kmeans, labels)

    best_k = max(silhouette_scores, key=silhouette_scores.get)
    kmeans, labels = models[best_k]

    # -----------------------------
    # Order clusters by height
    # -----------------------------
    centers = kmeans.cluster_centers_
    order = np.argsort(centers[:, 0])  # sort by height

    sorted_labels = np.zeros_like(labels)
    for new_id, old_id in enumerate(order):
        sorted_labels[labels == old_id] = new_id

    sorted_centers = centers[order]

    # -----------------------------
    # Build height & weight boundaries
    # -----------------------------
    height_bounds = [
        (sorted_centers[i, 0] + sorted_centers[i + 1, 0]) / 2
        for i in range(best_k - 1)
    ]
    weight_bounds = [
        (sorted_centers[i, 1] + sorted_centers[i + 1, 1]) / 2
        for i in range(best_k - 1)
    ]

    class_descriptions = []
    for i in range(best_k):
        if i == 0:
            desc = (
                f"Height < {height_bounds[0]:.1f}\n"
                f"Weight < {weight_bounds[0]:.1f}"
            )
        elif i == best_k - 1:
            desc = (
                f"Height ≥ {height_bounds[-1]:.1f}\n"
                f"Weight ≥ {weight_bounds[-1]:.1f}"
            )
        else:
            desc = (
                f"{height_bounds[i-1]:.1f} ≤ Height < {height_bounds[i]:.1f}\n"
                f"{weight_bounds[i-1]:.1f} ≤ Weight < {weight_bounds[i]:.1f}"
            )
        class_descriptions.append(desc)

    # -----------------------------
    # Plot clustering with legend
    # -----------------------------
    plt.figure()

    for i in range(best_k):
        class_data = data[sorted_labels == i]
        plt.scatter(
            class_data[:, 0],
            class_data[:, 1],
            label=f"Class {i + 1}:\n{class_descriptions[i]}"
        )

    plt.scatter(
        sorted_centers[:, 0],
        sorted_centers[:, 1],
        marker="X"
    )

    plt.xlabel("Height")
    plt.ylabel("Weight")
    plt.title(f"Height vs Weight — {best_k} Classes")
    plt.legend()
    plt.show()

    return {
        "best_k": best_k,
        "centers": sorted_centers,
        "silhouette_scores": silhouette_scores
    }