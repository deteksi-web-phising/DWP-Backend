from fastapi import APIRouter, status
from app.features.eda.schemas import EDASummaryResponse
from app.features.eda.service import EDAService

router = APIRouter(prefix="/eda", tags=["Exploratory Data Analysis"])


@router.get(
    "/summary",
    response_model=EDASummaryResponse,
    status_code=status.HTTP_200_OK,
    summary="Get exploratory dataset summary and distributions",
)
def get_eda_summary() -> EDASummaryResponse:
    """Retrieve statistical distribution, class balance, and correlation pairs from dataset."""
    return EDAService.get_eda_summary()
