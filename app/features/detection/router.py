from fastapi import APIRouter, HTTPException, status
from app.features.detection.schemas import (
    URLInputRequest,
    ExtractedFeaturesResponse,
    PredictionRequest,
    PredictionResponse,
)
from app.features.detection.service import DetectionService

router = APIRouter(tags=["Detection & Inference"])


@router.post(
    "/extract-url",
    response_model=ExtractedFeaturesResponse,
    status_code=status.HTTP_200_OK,
    summary="Extract 30 features from target URL (PRD 5.2)",
)
@router.post(
    "/detection/extract-url",
    response_model=ExtractedFeaturesResponse,
    status_code=status.HTTP_200_OK,
    include_in_schema=False,
)
def extract_url_features(payload: URLInputRequest) -> ExtractedFeaturesResponse:
    """Extract structural URL features to populate detection form."""
    if not payload.url or not payload.url.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="URL parameter cannot be empty.",
        )
    features = DetectionService.extract_from_url(payload.url.strip())
    return ExtractedFeaturesResponse(url=payload.url, features=features)


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict phishing probability using ML model (PRD 5.2)",
)
@router.post(
    "/detection/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    include_in_schema=False,
)
def predict_phishing(payload: PredictionRequest) -> PredictionResponse:
    """Evaluate 30 feature parameters against ML classifier."""
    selected_model = payload.model_name or "random_forest"
    result = DetectionService.predict(payload.features, model_name=selected_model)
    return result
