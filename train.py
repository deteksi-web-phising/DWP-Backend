"""
Script Pelatihan & Benchmark Model Machine Learning
Sesuai Panduan PRD: Web Analisis & Deteksi Website Phishing

Model yang dilatih & dievaluasi:
1. Random Forest (Supervised Ensemble)
2. Decision Tree (Supervised Tree-based)
3. k-Nearest Neighbors / k-NN (Supervised Instance-based)
4. K-Means Clustering (Unsupervised k=2 dan k=3)

Output yang dihasilkan:
- models/random_forest.joblib
- models/decision_tree.joblib
- models/knn.joblib
- models/kmeans.joblib
- models/benchmark_metrics.json
- models/feature_importance.json
"""

import os
import time
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    silhouette_score,
)

# Konfigurasi path
DATASET_PATH = os.path.join("data", "Dataset Phising Website.csv")
MODEL_DIR = "models"


def load_and_preprocess_dataset(filepath: str):
    """Load dataset dan validasi kolom fitur serta label target."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset tidak ditemukan di path: {filepath}")

    print(f"[*] Membaca dataset dari '{filepath}'...")
    df = pd.read_csv(filepath)
    print(f"    Dimensi data awal: {df.shape[0]} baris, {df.shape[1]} kolom")

    # Drop kolom indeks jika ada
    if "index" in df.columns:
        X = df.drop(columns=["index", "Result"])
    else:
        X = df.drop(columns=["Result"])

    y = df["Result"]

    print(f"    Jumlah fitur: {X.shape[1]}")
    print(f"    Distribusi kelas: Legitimate (-1) = {(y == -1).sum()}, Phishing (1) = {(y == 1).sum()}")

    return X, y


def train_and_evaluate_supervised(X_train, X_test, y_train, y_test) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Latih dan evaluasi 3 model klasifikasi (Random Forest, Decision Tree, k-NN)."""
    models = {
        "Random Forest": {
            "key": "random_forest",
            "estimator": RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42, n_jobs=-1),
        },
        "Decision Tree": {
            "key": "decision_tree",
            "estimator": DecisionTreeClassifier(random_state=42),
        },
        "k-Nearest Neighbors (k-NN)": {
            "key": "knn",
            "estimator": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        },
    }

    metrics_list: List[Dict[str, Any]] = []
    trained_estimators: Dict[str, Any] = {}
    best_f1 = -1.0
    best_model_name = ""

    print("\n" + "=" * 65)
    print(" MEMULAI PELATIHAN SUPERVISED CLASSIFICATION ")
    print("=" * 65)

    for name, config in models.items():
        estimator = config["estimator"]
        key = config["key"]

        print(f"\n[*] Melatih model: {name} ...")
        start_time = time.time()
        estimator.fit(X_train, y_train)
        train_time = round(time.time() - start_time, 4)

        # Evaluasi pada data uji (Test Set)
        y_pred = estimator.predict(X_test)

        acc = round(float(accuracy_score(y_test, y_pred)), 4)
        prec = round(float(precision_score(y_test, y_pred, pos_label=1, zero_division=0)), 4)
        rec = round(float(recall_score(y_test, y_pred, pos_label=1, zero_division=0)), 4)
        f1 = round(float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)), 4)

        # Confusion Matrix: [[TN, FP], [FN, TP]] dengan labels=[-1, 1]
        cm = confusion_matrix(y_test, y_pred, labels=[-1, 1]).tolist()

        print(f"    -> Waktu Training: {train_time:.2f} detik")
        print(f"    -> Akurasi       : {acc * 100:.2f}%")
        print(f"    -> Precision     : {prec * 100:.2f}%")
        print(f"    -> Recall        : {rec * 100:.2f}%")
        print(f"    -> F1-Score      : {f1 * 100:.2f}%")
        print(f"    -> Confusion Matrix: TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}")

        model_metric = {
            "name": name,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "confusion_matrix": cm,
            "train_time_sec": train_time,
        }
        metrics_list.append(model_metric)
        trained_estimators[key] = estimator

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name

    return metrics_list, trained_estimators, best_model_name


def train_and_evaluate_clustering(X) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Latih dan evaluasi K-Means Clustering (k=2 dan k=3)."""
    print("\n" + "=" * 65)
    print(" MEMULAI PELATIHAN UNSUPERVISED CLUSTERING (K-MEANS) ")
    print("=" * 65)

    clustering_configs = [
        {"name": "K-Means (k=2)", "k": 2},
        {"name": "K-Means (k=3)", "k": 3},
    ]

    cluster_metrics: List[Dict[str, Any]] = []
    cluster_estimators: Dict[str, Any] = {}

    for cfg in clustering_configs:
        k = cfg["k"]
        name = cfg["name"]
        print(f"\n[*] Menjalankan K-Means dengan k={k} ...")

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        inertia = round(float(kmeans.inertia_), 2)
        # Menghitung silhouette score
        sil_score = round(float(silhouette_score(X, labels, sample_size=5000, random_state=42)), 4)

        print(f"    -> Inertia          : {inertia}")
        print(f"    -> Silhouette Score : {sil_score}")

        cluster_metrics.append({
            "name": name,
            "silhouette_score": sil_score,
            "n_clusters": k,
            "inertia": inertia,
        })
        cluster_estimators[f"kmeans_k{k}"] = kmeans

    return cluster_metrics, cluster_estimators


def extract_feature_importances(trained_models: Dict[str, Any], feature_names: List[str]) -> Dict[str, Any]:
    """Ekstraksi bobot pengaruh fitur (Feature Importance) untuk analisis siber."""
    importance_data: Dict[str, Any] = {}

    # Random Forest feature importance
    if "random_forest" in trained_models:
        rf = trained_models["random_forest"]
        rf_importances = dict(
            sorted(
                zip(feature_names, [round(float(v), 5) for v in rf.feature_importances_]),
                key=lambda x: x[1],
                reverse=True,
            )
        )
        importance_data["random_forest"] = rf_importances

    # Decision Tree feature importance
    if "decision_tree" in trained_models:
        dt = trained_models["decision_tree"]
        dt_importances = dict(
            sorted(
                zip(feature_names, [round(float(v), 5) for v in dt.feature_importances_]),
                key=lambda x: x[1],
                reverse=True,
            )
        )
        importance_data["decision_tree"] = dt_importances

    return importance_data


def save_artifacts(
    supervised_metrics: List[Dict[str, Any]],
    clustering_metrics: List[Dict[str, Any]],
    best_model_name: str,
    trained_supervised: Dict[str, Any],
    trained_clustering: Dict[str, Any],
    feature_importances: Dict[str, Any],
    output_dir: str = MODEL_DIR,
):
    """Serialisasi model .joblib dan simpan metrik evaluasi ke JSON."""
    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "=" * 65)
    print(" MENYIMPAN ARTIFAK MODEL & METRIK BENCHMARK ")
    print("=" * 65)

    # 1. Simpan model klasifikasi terawasi (.joblib)
    for model_key, estimator in trained_supervised.items():
        save_path = os.path.join(output_dir, f"{model_key}.joblib")
        joblib.dump(estimator, save_path)
        print(f"[+] Model tersimpan: {save_path}")

    # 2. Simpan model clustering (.joblib)
    for model_key, estimator in trained_clustering.items():
        save_path = os.path.join(output_dir, f"{model_key}.joblib")
        joblib.dump(estimator, save_path)
        print(f"[+] Model clustering tersimpan: {save_path}")

    # Simpan juga default kmeans.joblib (k=2)
    default_kmeans = trained_clustering.get("kmeans_k2")
    if default_kmeans:
        save_path = os.path.join(output_dir, "kmeans.joblib")
        joblib.dump(default_kmeans, save_path)
        print(f"[+] Model clustering default tersimpan: {save_path}")

    # 3. Simpan Benchmark Metrics JSON
    benchmark_payload = {
        "best_model": best_model_name,
        "supervised_models": supervised_metrics,
        "unsupervised_models": clustering_metrics,
    }
    benchmark_file = os.path.join(output_dir, "benchmark_metrics.json")
    with open(benchmark_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_payload, f, indent=2)
    print(f"[+] Metrik benchmark tersimpan: {benchmark_file}")

    # 4. Simpan Feature Importance JSON
    fi_file = os.path.join(output_dir, "feature_importance.json")
    with open(fi_file, "w", encoding="utf-8") as f:
        json.dump(feature_importances, f, indent=2)
    print(f"[+] Feature importances tersimpan: {fi_file}")


def main():
    print("=" * 65)
    print(" PIPELINE PELATIHAN MODEL PHISHING DETECTION (PRD WORKSHOP ML) ")
    print("=" * 65)

    # 1. Muat dataset
    X, y = load_and_preprocess_dataset(DATASET_PATH)
    feature_names = list(X.columns)

    # 2. Split train-test (80% train, 20% test, stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[*] Pembagian data: Train = {len(X_train)} sampel, Test = {len(X_test)} sampel")

    # 3. Pelatihan Model Supervised
    supervised_metrics, trained_supervised, best_model_name = train_and_evaluate_supervised(
        X_train, X_test, y_train, y_test
    )

    # 4. Pelatihan Model Unsupervised (Clustering K-Means)
    clustering_metrics, trained_clustering = train_and_evaluate_clustering(X)

    # 5. Ekstraksi Feature Importance
    feature_importances = extract_feature_importances(trained_supervised, feature_names)

    # 6. Simpan semua artifak ke direktori models/
    save_artifacts(
        supervised_metrics=supervised_metrics,
        clustering_metrics=clustering_metrics,
        best_model_name=best_model_name,
        trained_supervised=trained_supervised,
        trained_clustering=trained_clustering,
        feature_importances=feature_importances,
    )

    # 7. Validasi Acceptance Criteria PRD
    rf_metrics = next(m for m in supervised_metrics if m["name"] == "Random Forest")
    rf_acc = rf_metrics["accuracy"]
    print("\n" + "=" * 65)
    print(" VALIDASI KRITERIA PENERIMAAN PRD (SECTION 7) ")
    print("=" * 65)
    print(f"-> Target Minimal Akurasi Random Forest : >= 88.00%")
    print(f"-> Capaian Akurasi Random Forest        : {rf_acc * 100:.2f}%")
    if rf_acc >= 0.88:
        print("[SUCCESS] Memenuhi kriteria penerimaan PRD! Kinerja model sangat unggul.")
    else:
        print("[WARNING] Akurasi di bawah target minimal PRD!")

    print("\n[+] Pelatihan dan benchmark selesai dengan sukses!\n")


if __name__ == "__main__":
    from typing import Tuple
    main()
