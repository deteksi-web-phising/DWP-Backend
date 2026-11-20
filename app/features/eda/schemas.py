from typing import List
from pydantic import BaseModel, Field


class KeyFeatureStat(BaseModel):
    """Percentage distribution of values (-1, 0, 1) for a key feature."""
    name: str = Field(..., description="Feature attribute name")
    neg: float = Field(..., description="Percentage of value -1 (Legitimate)")
    zero: float = Field(..., description="Percentage of value 0 (Moderate / Neutral)")
    pos: float = Field(..., description="Percentage of value 1 (Phishing / Suspicious)")


class ClassStats(BaseModel):
    """Count and percentage for a class."""
    count: int
    percentage: float


class ClassDistribution(BaseModel):
    """Dataset distribution for legitimate and phishing samples."""
    legitimate: ClassStats
    phishing: ClassStats


class CorrelationMatrix(BaseModel):
    """Correlation matrix of predictive features."""
    features: List[str]
    matrix: List[List[float]]


class EDASummaryResponse(BaseModel):
    """Exploratory Data Analysis summary matching Frontend & PRD contract."""
    total_rows: int
    total_features: int
    class_distribution: ClassDistribution
    key_features_stats: List[KeyFeatureStat]
    correlation_matrix: CorrelationMatrix
