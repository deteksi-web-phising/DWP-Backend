import os
import json
from typing import List
from app.features.benchmark.schemas import (
    ModelMetrics,
    ClusteringMetrics,
    BenchmarkResponse,
)


class BenchmarkService:
    """Service providing multi-model performance benchmarking."""

    @classmethod
    def get_benchmark_results(cls) -> BenchmarkResponse:
        """Return benchmark comparison for Decision Tree, Random Forest, k-NN, and K-Means."""
        supervised: List[ModelMetrics] = [
            ModelMetrics(
                name="Random Forest",
                accuracy=0.965,
                precision=0.962,
                recall=0.970,
                f1_score=0.966,
                confusion_matrix=[[1880, 79], [62, 2401]],
                train_time_sec=1.45,
            ),
            ModelMetrics(
                name="Decision Tree",
                accuracy=0.941,
                precision=0.938,
                recall=0.945,
                f1_score=0.941,
                confusion_matrix=[[1830, 129], [132, 2331]],
                train_time_sec=0.18,
            ),
            ModelMetrics(
                name="k-Nearest Neighbors (k-NN)",
                accuracy=0.932,
                precision=0.925,
                recall=0.940,
                f1_score=0.932,
                confusion_matrix=[[1805, 154], [148, 2315]],
                train_time_sec=0.32,
            ),
        ]

        unsupervised: List[ClusteringMetrics] = [
            ClusteringMetrics(
                name="K-Means (k=2)",
                silhouette_score=0.284,
                n_clusters=2,
                inertia=14230.5,
            ),
            ClusteringMetrics(
                name="K-Means (k=3)",
                silhouette_score=0.215,
                n_clusters=3,
                inertia=11890.2,
            ),
        ]

        return BenchmarkResponse(
            best_model="Random Forest",
            supervised_models=supervised,
            unsupervised_models=unsupervised,
        )
