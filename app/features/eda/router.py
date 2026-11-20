from fastapi import APIRouter, status
from app.features.eda.schemas import EDASummaryResponse
from app.features.eda.service import EDAService

router = APIRouter(tags=["Exploratory Data Analysis"])


@router.get(
    "/dataset/eda-summary",
    response_model=EDASummaryResponse,
    status_code=status.HTTP_200_OK,
    summary="Get exploratory dataset summary and distributions (PRD 5.2)",
)
@router.get(
    "/eda/summary",
    response_model=EDASummaryResponse,
    status_code=status.HTTP_200_OK,
    include_in_schema=False,
)
@router.get(
    "/summary",
    response_model=EDASummaryResponse,
    status_code=status.HTTP_200_OK,
    include_in_schema=False,
)
def get_eda_summary() -> EDASummaryResponse:
    """Retrieve statistical distribution, class balance, and correlation pairs from dataset."""
    return EDAService.get_eda_summary()
