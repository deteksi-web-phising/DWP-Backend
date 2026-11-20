from typing import List, Optional
from pydantic import BaseModel, Field


class SupervisedMetric(BaseModel):
    """Performance evaluation metrics for a supervised classification model."""
    model_name: str = Field(..., description="Unique model key: random_forest, decision_tree, knn")
    display_name: str = Field(..., description="Display name for UI (e.g. Random Forest)")
    short_name: str = Field(..., description="Abbreviation (e.g. RF, DT, k-NN)")
    algorithm_type: str = Field(..., description="Description of the algorithm architecture")
    accuracy: float = Field(..., description="Accuracy score in percentage (0-100)")
    precision: float = Field(..., description="Precision score in percentage (0-100)")
    recall: float = Field(..., description="Recall score in percentage (0-100)")
    f1_score: float = Field(..., description="F1-Score in percentage (0-100)")
    latency_ms: float = Field(..., description="Inference/training latency in ms")
    confusion_matrix: List[List[int]] = Field(..., description="2x2 confusion matrix [[TN, FP], [FN, TP]]")


class PCAClusterPoint(BaseModel):
    """2D projection coordinate of a sample cluster point."""
    x: float
    y: float
    cluster: int = Field(..., description="Cluster ID: 0 or 1")


class UnsupervisedMetric(BaseModel):
    """Evaluation metrics and 2D PCA cluster coordinates for K-Means."""
    model_name: str = Field("kmeans", description="Model algorithm identifier")
    display_name: str = Field("K-Means Clustering", description="Display name for UI")
    silhouette_score: float = Field(..., description="Silhouette coefficient (-1.0 to 1.0)")
    pca_clusters: List[PCAClusterPoint] = Field(default_factory=list, description="2D PCA projected points")


class BenchmarkResponse(BaseModel):
    """Aggregated benchmark comparison response matching Frontend & PRD contract."""
    supervised: List[SupervisedMetric]
    unsupervised: UnsupervisedMetric
    best_model: Optional[str] = Field("Random Forest", description="Recommended best performing model")
