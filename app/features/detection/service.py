import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from app.core.config import settings
from app.features.detection.schemas import URLFeatureValues, PredictionResponse
from app.features.detection.extractor import URLExtractor


class DetectionService:
    """Service layer for URL feature extraction and ML model inference."""

    # In-memory cache untuk model ML yang telah dimuat agar inferensi < 300ms
    _model_cache: Dict[str, Any] = {}

    @classmethod
    def _get_normalized_model_name(cls, raw_name: str) -> str:
        """Normalisasi nama model dari request user/frontend ke nama file model."""
        mapping = {
            "random_forest": "random_forest",
            "random forest": "random_forest",
            "decision_tree": "decision_tree",
            "decision tree": "decision_tree",
            "knn": "knn",
            "k-nn": "knn",
            "k_nearest_neighbors": "knn",
            "k-nearest neighbors": "knn",
            "k-nearest neighbors (k-nn)": "knn",
        }
        clean = raw_name.lower().strip()
        return mapping.get(clean, clean.replace(" ", "_").replace("-", "_"))

    @classmethod
    def _load_model(cls, normalized_name: str):
        """Memuat model dari disk atau memori cache."""
        if normalized_name in cls._model_cache:
            return cls._model_cache[normalized_name]

        model_path = os.path.join(settings.MODEL_DIR, f"{normalized_name}.joblib")
        if os.path.exists(model_path):
            try:
                model = joblib.load(model_path)
                cls._model_cache[normalized_name] = model
                return model
            except Exception:
                return None
        return None

    @classmethod
    def extract_from_url(cls, url: str) -> URLFeatureValues:
        """Extract 30 features from the provided URL."""
        return URLExtractor.extract_features(url)

    @classmethod
    def predict(cls, features: URLFeatureValues, model_name: str = "random_forest") -> PredictionResponse:
        """Run prediction against the chosen ML model."""
        feature_dict = features.model_dump()
        feature_df = pd.DataFrame([feature_dict])

        # Identify suspicious features (risk factors where value is 1)
        risk_factors: List[str] = [
            k for k, v in feature_dict.items() if v == 1
        ]

        normalized_model_name = cls._get_normalized_model_name(model_name)
        model = cls._load_model(normalized_model_name)

        if model is not None:
            try:
                pred_label = int(model.predict(feature_df)[0])
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(feature_df)[0]
                    confidence = float(np.max(probs))
                else:
                    confidence = 0.90
                
                prediction_text = "Phishing" if pred_label == 1 else "Legitimate"
                return PredictionResponse(
                    prediction=prediction_text,
                    label=pred_label,
                    confidence=round(confidence, 4),
                    model_used=model_name,
                    risk_factors=risk_factors,
                )
            except Exception:
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
