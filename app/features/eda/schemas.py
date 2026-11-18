from typing import Dict, List
from pydantic import BaseModel, Field


class ClassDistribution(BaseModel):
    """Target class distribution summary."""
    name: str = Field(..., description="Class name (e.g. Legitimate, Phishing)")
    code: int = Field(..., description="Numerical code (-1, 1)")
    count: int = Field(..., description="Sample count")
    percentage: float = Field(..., description="Percentage of dataset")


class CorrelationPair(BaseModel):
    """Correlation pair between two features."""
    feature_a: str
    feature_b: str
    correlation: float


class EDASummaryResponse(BaseModel):
    """Exploratory Data Analysis summary."""
    total_records: int
    total_features: int
    class_distribution: List[ClassDistribution]
    top_correlations: List[CorrelationPair]
    feature_names: List[str]
