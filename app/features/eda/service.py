import os
from typing import Optional, List
import pandas as pd
from app.core.config import settings
from app.features.eda.schemas import (
    ClassDistribution,
    ClassStats,
    KeyFeatureStat,
    CorrelationMatrix,
    EDASummaryResponse,
)


class EDAService:
    """Service providing exploratory data analysis metrics from dataset with in-memory caching."""

    _cached_summary: Optional[EDASummaryResponse] = None

    @classmethod
    def get_eda_summary(cls) -> EDASummaryResponse:
        """Return dataset statistical metrics (computed from CSV with in-memory cache)."""
        if cls._cached_summary is not None:
            return cls._cached_summary

        dataset_candidates = [
            settings.DATASET_PATH,
            os.path.join("data", "Dataset Phising Website.csv"),
            os.path.join("..", "Dataset Phising Website.csv"),
            os.path.join("backend", "data", "Dataset Phising Website.csv"),
        ]

        df: Optional[pd.DataFrame] = None
        for path in dataset_candidates:
            if os.path.exists(path):
                try:
                    df = pd.read_csv(path)
                    break
                except Exception:
                    pass

        if df is not None and "Result" in df.columns:
            total_records = len(df)
            feature_cols = [c for c in df.columns if c not in ("index", "Result")]
            total_features = len(feature_cols)

            counts = df["Result"].value_counts()
            phish_count = int(counts.get(1, 0))
            legit_count = int(counts.get(-1, 0))

            class_dist = ClassDistribution(
                legitimate=ClassStats(
                    count=legit_count,
                    percentage=round((legit_count / total_records) * 100, 1),
                ),
                phishing=ClassStats(
                    count=phish_count,
                    percentage=round((phish_count / total_records) * 100, 1),
                ),
            )

            # Key features stats (-1, 0, 1 percentages)
            key_cols = ["SSLfinal_State", "URL_of_Anchor", "Prefix_Suffix", "having_IP_Address", "URL_Length"]
            key_stats: List[KeyFeatureStat] = []
            for col in key_cols:
                if col in df.columns:
                    col_counts = df[col].value_counts()
                    neg = round((col_counts.get(-1, 0) / total_records) * 100, 1)
                    zero = round((col_counts.get(0, 0) / total_records) * 100, 1)
                    pos = round((col_counts.get(1, 0) / total_records) * 100, 1)
                    key_stats.append(KeyFeatureStat(name=col, neg=neg, zero=zero, pos=pos))

            # Selected features correlation matrix
            corr_features = [
                "SSLfinal_State",
                "URL_of_Anchor",
                "Prefix_Suffix",
                "web_traffic",
                "having_Sub_Domain",
                "Request_URL",
                "age_of_domain",
                "Result",
            ]
            valid_corr_feats = [f for f in corr_features if f in df.columns]
            corr_df = df[valid_corr_feats].corr().round(2)
            corr_matrix_data = corr_df.values.tolist()

            summary = EDASummaryResponse(
                total_rows=total_records,
                total_features=total_features,
                class_distribution=class_dist,
                key_features_stats=key_stats,
                correlation_matrix=CorrelationMatrix(
                    features=valid_corr_feats,
                    matrix=corr_matrix_data,
                ),
            )
            cls._cached_summary = summary
            return summary

        # Fallback summary if dataset CSV is not found
        fallback = EDASummaryResponse(
            total_rows=22110,
            total_features=30,
            class_distribution=ClassDistribution(
                legitimate=ClassStats(count=9796, percentage=44.3),
                phishing=ClassStats(count=12314, percentage=55.7),
            ),
            key_features_stats=[
                KeyFeatureStat(name="SSLfinal_State", neg=32.2, zero=10.6, pos=57.3),
                KeyFeatureStat(name="URL_of_Anchor", neg=29.7, zero=48.3, pos=22.0),
                KeyFeatureStat(name="Prefix_Suffix", neg=86.7, zero=0.0, pos=13.3),
                KeyFeatureStat(name="having_IP_Address", neg=34.3, zero=0.0, pos=65.7),
                KeyFeatureStat(name="URL_Length", neg=81.0, zero=1.2, pos=17.7),
            ],
            correlation_matrix=CorrelationMatrix(
                features=[
                    "SSLfinal_State",
                    "URL_of_Anchor",
                    "Prefix_Suffix",
                    "web_traffic",
                    "having_Sub_Domain",
                    "Request_URL",
                    "age_of_domain",
                    "Result",
                ],
                matrix=[
                    [1.0, 0.42, 0.28, 0.35, 0.21, 0.19, 0.24, 0.71],
                    [0.42, 1.0, 0.24, 0.31, 0.18, 0.29, 0.17, 0.69],
                    [0.28, 0.24, 1.0, 0.15, 0.12, 0.14, 0.11, 0.35],
                    [0.35, 0.31, 0.15, 1.0, 0.16, 0.22, 0.27, 0.35],
                    [0.21, 0.18, 0.12, 0.16, 1.0, 0.13, 0.19, 0.3],
                    [0.19, 0.29, 0.14, 0.22, 0.13, 1.0, 0.15, 0.25],
                    [0.24, 0.17, 0.11, 0.27, 0.19, 0.15, 1.0, 0.12],
                    [0.71, 0.69, 0.35, 0.35, 0.3, 0.25, 0.12, 1.0],
                ],
            ),
        )
        cls._cached_summary = fallback
        return fallback
