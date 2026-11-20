from fastapi import APIRouter, status
from app.features.benchmark.schemas import BenchmarkResponse
from app.features.benchmark.service import BenchmarkService

router = APIRouter(tags=["Model Benchmark"])


@router.get(
    "/models/benchmark",
    response_model=BenchmarkResponse,
    status_code=status.HTTP_200_OK,
    summary="Get multi-model performance comparison metrics (PRD 5.2)",
)
@router.get(
    "/benchmark",
    response_model=BenchmarkResponse,
    status_code=status.HTTP_200_OK,
    include_in_schema=False,
)
def get_model_benchmark() -> BenchmarkResponse:
    """Retrieve performance benchmarks across Decision Tree, Random Forest, k-NN, and K-Means."""
    return BenchmarkService.get_benchmark_results()
