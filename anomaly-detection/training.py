from pathlib import Path
import numpy as np
from scipy import stats as scipy_stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    roc_curve,
)

# Configuration
DATASET_PATH = Path("datasets/ac")
NORMAL_OPS = ["silent_0_baseline"]
ANOMALY_OPS = [
    "medium_0",
    "high_0",
    "silent_1",
    "medium_1",
    "high_1",
]
SAMPLE_RATE = 200  # Hz
SAMPLE_TIME = 0.5  # seconds

# Training parameters
VAL_RATIO = 0.2
TEST_RATIO = 0.2
MAX_ANOMALY_SAMPLES = 100
MAX_NORMAL_SAMPLES = 100
MODEL_PATH = Path("models/mahalanobis_model.npz")


def get_data_files(operations):
    files = []
    for op in operations:
        files.extend(list((DATASET_PATH / op).glob("*.csv")))
    return files


def load_and_extract_features(file_path):
    # Load and preprocess
    data = np.genfromtxt(file_path, delimiter=",")
    data = data - np.mean(data, axis=0)  # Remove DC

    # Add noise for robustness
    noise = np.random.normal(0, 0.3, data.shape)
    data = data + noise

    # Extract features per axis
    features = []
    for axis_idx in range(data.shape[1]):
        axis_data = data[:, axis_idx]
        features.extend(
            [
                np.std(axis_data),
                scipy_stats.kurtosis(axis_data),
                np.max(np.abs(axis_data)),
                np.sqrt(np.mean(np.square(axis_data))),
                np.max(axis_data) - np.min(axis_data),
            ]
        )
    return np.array(features)


def create_dataset(files, max_samples=50):
    # Randomly sample files if we have more than max_samples
    if len(files) > max_samples:
        files = np.random.choice(files, max_samples, replace=False)

    features = [load_and_extract_features(f) for f in files]
    return np.array(features)


def mahalanobis_distance(x, mu, cov):
    x_mu = x - mu
    inv_covmat = np.linalg.inv(cov + 1e-6 * np.eye(cov.shape[0]))
    return np.sqrt(np.sum(np.dot(x_mu, inv_covmat) * x_mu, axis=1))


def find_optimal_threshold(normal_dist, anomaly_dist, n_splits=5):
    """Find threshold with more conservative constraints"""
    # Calculate percentile-based thresholds
    normal_range = np.percentile(normal_dist, [75, 99])  # More conservative
    anomaly_range = np.percentile(anomaly_dist, [1, 25])  # More conservative

    thresholds = np.linspace(normal_range[0], anomaly_range[1], 100)

    best_score = -np.inf
    best_threshold = None
    best_metrics = None

    for threshold in thresholds:
        fold_metrics = []
        for _ in range(n_splits):
            # Increase randomization in validation
            normal_mask = np.random.choice(
                [True, False], len(normal_dist), p=[0.7, 0.3]
            )
            anomaly_mask = np.random.choice(
                [True, False], len(anomaly_dist), p=[0.7, 0.3]
            )

            normal_pred = normal_dist[normal_mask] > threshold
            anomaly_pred = anomaly_dist[anomaly_mask] > threshold

            fp_rate = np.mean(normal_pred)
            tp_rate = np.mean(anomaly_pred)

            # More conservative scoring with higher penalties
            score = tp_rate - (5 * fp_rate)  # Increased penalty for false positives

            # Stronger penalties for perfect performance
            if fp_rate == 0 or tp_rate == 1:
                score *= (
                    0.5  # Penalize perfect scores as they likely indicate overfitting
                )

            fold_metrics.append(
                {"score": score, "fp_rate": fp_rate, "tp_rate": tp_rate}
            )

        avg_score = np.mean([m["score"] for m in fold_metrics])
        score_std = np.std([m["score"] for m in fold_metrics])

        # Prefer stable solutions with reasonable performance
        final_score = avg_score - (2 * score_std)  # Increased stability penalty

        if final_score > best_score:
            best_score = final_score
            best_threshold = threshold
            best_metrics = {
                "fp_rate": np.mean([m["fp_rate"] for m in fold_metrics]),
                "tp_rate": np.mean([m["tp_rate"] for m in fold_metrics]),
            }

    return best_threshold, best_metrics


def validate_model(normal_distances, anomaly_distances, threshold):
    """Validate model with multiple metrics"""
    y_true = np.concatenate(
        [np.zeros(len(normal_distances)), np.ones(len(anomaly_distances))]
    )
    distances = np.concatenate([normal_distances, anomaly_distances])
    y_pred = (distances > threshold).astype(int)

    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, distances),
    }

    # Calculate false positive rate
    fp = np.sum((y_true == 0) & (y_pred == 1))
    results["false_positive_rate"] = fp / len(normal_distances)

    return results


def plot_distance_distributions(normal_dist, anomaly_dist, threshold=None):
    plt.figure(figsize=(12, 6))
    n_bins = int(np.sqrt(len(normal_dist) + len(anomaly_dist)))

    plt.hist(
        normal_dist, bins=n_bins, alpha=0.7, label="Normal", color="blue", density=True
    )
    plt.hist(
        anomaly_dist, bins=n_bins, alpha=0.7, label="Anomaly", color="red", density=True
    )

    if threshold is not None:
        plt.axvline(
            x=threshold, color="k", linestyle="--", label=f"Threshold: {threshold:.2f}"
        )

    plt.xlabel("Mahalanobis Distance")
    plt.ylabel("Density")
    plt.title("Distribution of Mahalanobis Distances")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_roc_curve(normal_distances, anomaly_distances):
    y_true = np.concatenate(
        [np.zeros(len(normal_distances)), np.ones(len(anomaly_distances))]
    )
    distances = np.concatenate([normal_distances, anomaly_distances])

    fpr, tpr, _ = roc_curve(y_true, distances)
    auc = roc_auc_score(y_true, distances)

    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        pd.DataFrame(cm, index=["Normal", "Anomaly"], columns=["Normal", "Anomaly"]),
        annot=True,
        fmt="d",
        cmap="Blues",
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()


def train_model():
    # Load and prepare data
    normal_files = get_data_files(NORMAL_OPS)
    anomaly_files = get_data_files(ANOMALY_OPS)
    print(
        f"Found {len(normal_files)} normal files and {len(anomaly_files)} anomaly files"
    )

    # Split normal data
    train_files, test_files = train_test_split(
        normal_files, test_size=0.4, random_state=42
    )

    # Create datasets
    X_train = create_dataset(train_files)
    X_test = create_dataset(test_files)
    X_anomaly = create_dataset(anomaly_files)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_anomaly_scaled = scaler.transform(X_anomaly)

    # Train model
    mu = np.mean(X_train_scaled, axis=0)
    cov = np.cov(X_train_scaled.T)

    # Find threshold
    normal_dist = mahalanobis_distance(X_test_scaled, mu, cov)
    anomaly_dist = mahalanobis_distance(X_anomaly_scaled, mu, cov)
    threshold = np.percentile(normal_dist, 95)  # 5% false positive rate

    # Evaluate
    y_true = np.concatenate([np.zeros(len(X_test)), np.ones(len(X_anomaly))])
    y_pred = np.concatenate([normal_dist > threshold, anomaly_dist > threshold]).astype(
        int
    )

    # Print results
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Anomaly"]))
    print(
        f"AUC Score: {roc_auc_score(y_true, np.concatenate([normal_dist, anomaly_dist])):.3f}"
    )

    # Plot results
    plot_distance_distributions(normal_dist, anomaly_dist, threshold)
    plot_confusion_matrix(y_true, y_pred)

    # Save model
    np.savez(MODEL_PATH, mu=mu, cov=cov, threshold=threshold, scaler=scaler)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    train_model()
