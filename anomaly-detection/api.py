import numpy as np
from scipy import stats as scipy_stats  # Renamed to avoid shadowing
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class AnomalyDetector:
    def __init__(self, model_path: str):
        model = np.load(model_path)
        self.mu = model["mu"]
        self.cov = model["cov"]
        self.threshold = model["threshold"]
        self.last_predictions = [False, False, False]
        self.recent_distances = []  # Store recent distances for stability calculation
        logger.info(
            "Model loaded with threshold: %.2f", self.threshold
        )  # Fixed f-string

    def preprocess(self, data, remove_dc=True):
        if remove_dc:
            data = data - np.mean(data, axis=0)
        return data

    def extract_features(self, sample):
        """Extract reduced set of statistical features from sample"""
        features = []
        for axis_idx in range(sample.shape[1]):
            axis_data = sample[:, axis_idx]

            # Reduced set of time domain features
            features.extend(
                [
                    np.std(axis_data),  # Standard deviation
                    scipy_stats.kurtosis(axis_data),  # Kurtosis
                    np.max(np.abs(axis_data)),  # Peak amplitude
                    np.sqrt(np.mean(np.square(axis_data))),  # RMS
                    np.max(axis_data) - np.min(axis_data),  # Peak-to-peak
                ]
            )

        return np.array(features)

    def mahalanobis_distance(self, x):
        x_mu = x - self.mu
        epsilon = 1e-6
        cov_reg = self.cov + epsilon * np.eye(self.cov.shape[0])

        try:
            scale = np.median(np.diag(cov_reg))
            cov_scaled = cov_reg / scale
            inv_covmat = np.linalg.inv(cov_scaled) / scale

            if x_mu.ndim == 1:
                mahal = np.sqrt(np.dot(np.dot(x_mu, inv_covmat), x_mu))
            else:
                mahal = np.sqrt(np.sum(np.dot(x_mu, inv_covmat) * x_mu, axis=1))
            return mahal
        except np.linalg.LinAlgError:
            return np.inf

    def calculate_confidence(self, distance):
        """Calculate confidence using adaptive threshold bands and improved stability"""
        # Keep track of recent distances
        self.recent_distances.append(distance)
        self.recent_distances = self.recent_distances[
            -20:
        ]  # Increased history size to 20

        # Adaptive threshold bands based on the magnitude of the threshold
        threshold_magnitude = np.log10(self.threshold)

        # Adjust bands based on threshold magnitude
        lower_band_factor = np.exp(
            -threshold_magnitude / 2
        )  # Smaller for larger thresholds
        upper_band_factor = np.exp(
            threshold_magnitude / 2
        )  # Larger for larger thresholds

        lower_bound = self.threshold * lower_band_factor
        upper_bound = self.threshold * upper_band_factor

        # Base confidence calculation with sigmoid function for smoother transition
        if distance < lower_bound:
            # Clearly normal
            base_confidence = 0.95
        elif distance > upper_bound:
            # Clearly anomalous
            base_confidence = 0.90
        else:
            # Smooth transition in uncertainty zone using sigmoid
            x = (distance - lower_bound) / (upper_bound - lower_bound)
            base_confidence = 0.9 / (1 + np.exp((x - 0.5) * 10))

        # Enhanced stability calculation
        if len(self.recent_distances) > 5:  # Need at least 5 points for stability
            # Calculate trends and variations
            recent_mean = np.mean(self.recent_distances[-5:])
            recent_std = np.std(self.recent_distances[-5:])

            # Trend stability (are we consistently high/low?)
            trend_stability = np.exp(-abs(distance - recent_mean) / (recent_std + 1e-6))

            # Variation stability (how much are we fluctuating?)
            variation_coefficient = recent_std / (recent_mean + 1e-6)
            variation_stability = np.exp(-variation_coefficient)

            # Combined stability factor
            stability_factor = (trend_stability + variation_stability) / 2

            # Weight stability more when we have more history
            history_weight = min(len(self.recent_distances) / 20, 1.0)
            final_confidence = (
                base_confidence * (1 - history_weight)
                + (base_confidence + stability_factor) / 2 * history_weight
            )
        else:
            final_confidence = base_confidence

        # Ensure confidence is within bounds and handle numerical issues
        return float(np.clip(final_confidence, 0.0, 1.0))

    def predict(self, data):
        processed_data = self.preprocess(data)
        features = self.extract_features(processed_data)
        distance = float(self.mahalanobis_distance(features))

        # Current prediction
        is_anomaly = distance > self.threshold

        # Update prediction history
        self.last_predictions.pop(0)
        self.last_predictions.append(is_anomaly)

        # Only mark as anomaly if 2 out of 3 last predictions are anomalies
        stable_anomaly = sum(self.last_predictions) >= 2

        # Calculate confidence with improved method
        confidence = self.calculate_confidence(distance)

        # Calculate feature statistics for debugging
        feature_names = [
            "std",
            "kurtosis",
            "peak_amplitude",
            "rms",
            "peak_to_peak",
        ]
        feature_stats = {}

        # Organize features by axis
        n_features_per_axis = len(feature_names)
        n_axes = len(features) // n_features_per_axis

        for axis_idx in range(n_axes):
            start_idx = axis_idx * n_features_per_axis
            axis_features = features[start_idx : start_idx + n_features_per_axis]
            feature_stats[f"axis_{axis_idx}"] = {
                name: float(value) for name, value in zip(feature_names, axis_features)
            }

        result = {
            "is_anomaly": bool(stable_anomaly),
            "confidence": float(confidence),
            "distance": float(distance),
            "threshold": float(self.threshold),
            "feature_values": feature_stats,
            "timestamp": datetime.now().isoformat(),
        }

        # Log prediction details
        logger.info("=" * 50)
        logger.info("Prediction Details:")
        logger.info("Timestamp: %s", result["timestamp"])
        logger.info("Is Anomaly: %s", result["is_anomaly"])
        logger.info("Confidence: %.3f", result["confidence"])
        logger.info(
            "Distance: %.3f (threshold: %.3f)", result["distance"], result["threshold"]
        )
        logger.info("Feature Values:")
        for axis_name, stats in feature_stats.items():
            logger.info("  %s:", axis_name)
            for feat, val in stats.items():
                logger.info("    %s: %.3f", feat, val)
        logger.info("=" * 50)

        return result


class AccelerometerData(BaseModel):
    data: List[List[float]]
    sensor_id: str = "default"


app = FastAPI()

# Add CORS middleware to allow requests from your Next.js app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = AnomalyDetector("models/mahalanobis_model.npz")


@app.post("/predict")
async def predict_anomaly(data: AccelerometerData):
    try:
        array_data = np.array(data.data)
        logger.info(
            "Received data shape: %s from sensor %s", array_data.shape, data.sensor_id
        )

        result = detector.predict(array_data)
        return result
    except Exception as e:
        logger.error("Error during prediction: %s", str(e))
        return {"error": str(e), "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
