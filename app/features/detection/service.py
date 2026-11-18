import os
import joblib
import numpy as np
from typing import List, Tuple
from app.core.config import settings
from app.features.detection.schemas import URLFeatureValues, PredictionResponse
from app.features.detection.extractor import URLExtractor


class DetectionService:
    """Service layer for URL feature extraction and ML model inference."""

    @classmethod
    def extract_from_url(cls, url: str) -> URLFeatureValues:
        """Extract 30 features from the provided URL."""
        return URLExtractor.extract_features(url)

    @classmethod
    def predict(cls, features: URLFeatureValues, model_name: str = "random_forest") -> PredictionResponse:
        """Run prediction against the chosen ML model."""
        feature_dict = features.model_dump()
        feature_vector = np.array(list(feature_dict.values())).reshape(1, -1)

        # Identify suspicious features (risk factors)
        risk_factors: List[str] = [
            k for k, v in feature_dict.items() if v == 1
        ]

        model_path = os.path.join(settings.MODEL_DIR, f"{model_name}.joblib")

        if os.path.exists(model_path):
            try:
                model = joblib.load(model_path)
                pred_label = int(model.predict(feature_vector)[0])
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(feature_vector)[0]
                    # assuming class 1 is phishing, -1 is legitimate
                    confidence = float(np.max(probs))
                else:
                    confidence = 0.90
                
                prediction_text = "Phishing" if pred_label == 1 else "Legitimate"
                return PredictionResponse(
                    prediction=prediction_text,
                    label=pred_label,
                    confidence=confidence,
                    model_used=model_name,
                    risk_factors=risk_factors,
                )
            except Exception as e:
                # Log error and fallback
                pass

        # Heuristic fallback if model file is not yet trained/serialized
        suspicious_count = len(risk_factors)
        is_phish = suspicious_count >= 3 or features.SSLfinal_State == 1 or features.having_IP_Address == 1
        label = 1 if is_phish else -1
        prediction_text = "Phishing" if is_phish else "Legitimate"
        confidence = min(0.50 + (suspicious_count * 0.1), 0.98) if is_phish else max(0.95 - (suspicious_count * 0.1), 0.55)

        return PredictionResponse(
            prediction=prediction_text,
            label=label,
            confidence=round(confidence, 2),
            model_used=f"{model_name} (baseline)",
            risk_factors=risk_factors,
        )
