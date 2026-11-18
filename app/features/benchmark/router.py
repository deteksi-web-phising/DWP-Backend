from fastapi import APIRouter, status
from app.features.benchmark.schemas import BenchmarkResponse
from app.features.benchmark.service import BenchmarkService

router = APIRouter(prefix="/benchmark", tags=["Model Benchmark"])


@router.get(
    "",
    response_model=BenchmarkResponse,
    status_code=status.HTTP_200_OK,
    summary="Get multi-model performance comparison metrics",
)
def get_model_benchmark() -> BenchmarkResponse:
    """Retrieve performance benchmarks across Decision Tree, Random Forest, k-NN, and K-Means."""
    return BenchmarkService.get_benchmark_results()
