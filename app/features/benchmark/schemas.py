from typing import List
from pydantic import BaseModel, Field


class ModelMetrics(BaseModel):
    """Performance evaluation metrics for a supervised classification model."""
    name: str = Field(..., description="Algorithm name (e.g. Random Forest, Decision Tree, k-NN)")
    accuracy: float = Field(..., description="Accuracy score (0.0 to 1.0)")
    precision: float = Field(..., description="Precision score")
    recall: float = Field(..., description="Recall score")
    f1_score: float = Field(..., description="F1-Score")
    confusion_matrix: List[List[int]] = Field(..., description="2x2 confusion matrix [[TN, FP], [FN, TP]]")
    train_time_sec: float = Field(..., description="Training time in seconds")


class ClusteringMetrics(BaseModel):
    """Evaluation metrics for unsupervised clustering (K-Means)."""
    name: str = Field(..., description="Clustering algorithm name")
    silhouette_score: float = Field(..., description="Silhouette coefficient (-1.0 to 1.0)")
    n_clusters: int = Field(..., description="Number of clusters")
    inertia: float = Field(..., description="Sum of squared distances to closest cluster center")


class BenchmarkResponse(BaseModel):
    """Aggregated benchmark comparison response."""
    best_model: str = Field(..., description="Recommended best performing model")
    supervised_models: List[ModelMetrics]
    unsupervised_models: List[ClusteringMetrics]
