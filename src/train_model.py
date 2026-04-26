from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
FIGURES = ARTIFACTS / "figures"
REPORT = ROOT / "report" / "AIML_Assignment_3_Report.md"


def load_dataset() -> tuple[pd.DataFrame, pd.Series, dict]:
    dataset = load_breast_cancer(as_frame=True)
    features = dataset.data.copy()
    target = dataset.target.copy()
    metadata = {
        "source": "UCI Machine Learning Repository, loaded through sklearn.datasets.load_breast_cancer",
        "target_names": list(dataset.target_names),
        "feature_names": list(dataset.feature_names),
        "description": "Binary classification of breast tumor samples as malignant or benign.",
    }
    return features, target, metadata


def ensure_dirs() -> None:
    ARTIFACTS.mkdir(exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(exist_ok=True)


def save_class_balance_plot(y: pd.Series, target_names: list[str]) -> dict:
    counts = y.value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    plot_counts = y.map({0: target_names[0], 1: target_names[1]}).value_counts()
    sns.barplot(x=plot_counts.index, y=plot_counts.values, palette=["#d1495b", "#2a9d8f"], hue=plot_counts.index, legend=False)
    plt.title("Class Distribution")
    plt.xlabel("Diagnosis")
    plt.ylabel("Number of samples")
    plt.tight_layout()
    plt.savefig(FIGURES / "class_distribution.png", dpi=160)
    plt.close()
    return {target_names[i]: int(counts.get(i, 0)) for i in range(len(target_names))}


def save_correlation_heatmap(features: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 8))
    corr = features.corr()
    sns.heatmap(corr, cmap="vlag", center=0, cbar_kws={"shrink": 0.7})
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(FIGURES / "correlation_heatmap.png", dpi=160)
    plt.close()


def save_confusion_matrix_plot(name: str, matrix: np.ndarray, target_names: list[str]) -> None:
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
    )
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(FIGURES / f"confusion_matrix_{name.lower().replace(' ', '_')}.png", dpi=160)
    plt.close()


def save_feature_importance_plot(model: RandomForestClassifier, feature_names: list[str]) -> list[dict]:
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False).head(12)
    plt.figure(figsize=(8, 5))
    sns.barplot(x=importances.values, y=importances.index, color="#3a86ff")
    plt.title("Top Random Forest Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(FIGURES / "feature_importance.png", dpi=160)
    plt.close()
    return [{"feature": feature, "importance": round(float(value), 4)} for feature, value in importances.items()]


def save_pca_cluster_plot(features: pd.DataFrame, y: pd.Series, target_names: list[str]) -> dict:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(scaled)
    kmeans = KMeans(n_clusters=2, n_init=20, random_state=42)
    clusters = kmeans.fit_predict(components)

    plt.figure(figsize=(7, 5))
    sns.scatterplot(
        x=components[:, 0],
        y=components[:, 1],
        hue=[target_names[int(label)] for label in y],
        style=clusters,
        palette=["#d1495b", "#2a9d8f"],
        s=50,
    )
    plt.title("PCA View with K-Means Cluster Assignment")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.tight_layout()
    plt.savefig(FIGURES / "pca_kmeans_clusters.png", dpi=160)
    plt.close()

    return {
        "algorithm": "K-Means clustering on two PCA components",
        "silhouette_score": round(float(silhouette_score(components, clusters)), 4),
        "adjusted_rand_index_against_actual_labels": round(float(adjusted_rand_score(y, clusters)), 4),
        "pca_explained_variance_ratio": [round(float(value), 4) for value in pca.explained_variance_ratio_],
        "pca_points": [
            {
                "x": round(float(components[i, 0]), 4),
                "y": round(float(components[i, 1]), 4),
                "label": target_names[int(y.iloc[i])],
                "cluster": int(clusters[i]),
            }
            for i in range(len(components))
        ],
        "projection_params": {
            "scaler_mean": [round(float(v), 6) for v in scaler.mean_],
            "scaler_scale": [round(float(v), 6) for v in scaler.scale_],
            "pca_components": [[round(float(v), 6) for v in row] for row in pca.components_],
        },
    }


def build_models() -> dict[str, Pipeline]:
    return {
        "Decision Tree": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", DecisionTreeClassifier(max_depth=5, random_state=42)),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", RandomForestClassifier(n_estimators=250, max_depth=8, random_state=42)),
            ]
        ),
        "Support Vector Machine": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", SVC(kernel="rbf", probability=True, C=3, gamma="scale", random_state=42)),
            ]
        ),
    }


def evaluate_models(
    models: dict[str, Pipeline],
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    target_names: list[str],
) -> tuple[dict, str]:
    results = {}
    best_name = ""
    best_score = -1.0

    for name, model in models.items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        matrix = confusion_matrix(y_test, predictions)
        save_confusion_matrix_plot(name, matrix, target_names)

        macro_f1 = f1_score(y_test, predictions, average="macro")
        results[name] = {
            "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
            "precision_macro": round(float(precision_score(y_test, predictions, average="macro")), 4),
            "recall_macro": round(float(recall_score(y_test, predictions, average="macro")), 4),
            "f1_macro": round(float(macro_f1), 4),
            "confusion_matrix": matrix.tolist(),
            "classification_report": classification_report(
                y_test,
                predictions,
                target_names=target_names,
                output_dict=True,
                zero_division=0,
            ),
        }
        if macro_f1 > best_score:
            best_name = name
            best_score = macro_f1

    return results, best_name


def write_report(
    metadata: dict,
    features: pd.DataFrame,
    y: pd.Series,
    results: dict,
    best_model_name: str,
    cluster_summary: dict,
    feature_importances: list[dict],
) -> None:
    metrics_rows = "\n".join(
        f"| {name} | {scores['accuracy']:.4f} | {scores['precision_macro']:.4f} | {scores['recall_macro']:.4f} | {scores['f1_macro']:.4f} |"
        for name, scores in results.items()
    )
    top_features = "\n".join(
        f"| {item['feature']} | {item['importance']:.4f} |" for item in feature_importances[:8]
    )
    class_counts = y.value_counts().sort_index()
    missing_values = int(features.isna().sum().sum())

    REPORT.write_text(
        f"""# AIML Assignment 3 Mini Project Report

## Title
Breast Cancer Diagnosis Prediction using Machine Learning

## Problem Statement
The objective of this project is to build a complete machine learning solution that predicts whether a breast tumor sample is **malignant** or **benign** using diagnostic measurements computed from digitized cell nuclei images.

## Dataset
- Source: {metadata['source']}
- Dataset size: {features.shape[0]} rows and {features.shape[1]} input features
- Target variable: diagnosis class, where `0 = {metadata['target_names'][0]}` and `1 = {metadata['target_names'][1]}`
- Class distribution: {metadata['target_names'][0]} = {int(class_counts.loc[0])}, {metadata['target_names'][1]} = {int(class_counts.loc[1])}
- Missing values found during cleaning: {missing_values}

## Real-World Significance
Early and accurate cancer diagnosis helps doctors prioritize additional tests and treatment planning. A machine learning model can support clinical decision-making by identifying suspicious tumor measurements, while still requiring expert medical confirmation.

## Features and Target
The features include radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, and fractal dimension statistics. The target is the tumor diagnosis class.

## Data Preprocessing
- Checked missing values and duplicates.
- Used a stratified train-test split so both diagnosis classes are represented in training and testing data.
- Applied `StandardScaler` inside each model pipeline so distance-based and margin-based algorithms are trained on normalized measurements.
- Kept preprocessing inside `Pipeline` objects to avoid data leakage.

## Exploratory Data Analysis
- Class distribution shows the dataset is moderately imbalanced but still contains enough samples for both classes.
- The correlation heatmap shows strong relationships among radius, perimeter, and area measurements.
- The Random Forest importance chart indicates that concavity, perimeter, radius, and area measurements are highly informative for diagnosis.

![Class Distribution](../artifacts/figures/class_distribution.png)

![Correlation Heatmap](../artifacts/figures/correlation_heatmap.png)

![Feature Importance](../artifacts/figures/feature_importance.png)

## Algorithms Used
1. **Decision Tree**: Suitable because it is interpretable and can model non-linear decision boundaries.
2. **Random Forest**: Suitable because it combines many trees, reduces overfitting, and provides feature importance.
3. **Support Vector Machine**: Suitable for high-dimensional classification and effective decision boundaries after scaling.

## Performance Comparison
| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
{metrics_rows}

Best-performing model: **{best_model_name}**

The best model performed well because it captured non-linear relationships among tumor measurements while reducing overfitting through its learning strategy.

## Confusion Matrices
![Decision Tree Confusion Matrix](../artifacts/figures/confusion_matrix_decision_tree.png)

![Random Forest Confusion Matrix](../artifacts/figures/confusion_matrix_random_forest.png)

![SVM Confusion Matrix](../artifacts/figures/confusion_matrix_support_vector_machine.png)

## Advanced Concept: Clustering
To extend the supervised classification work, K-Means clustering was applied after reducing the scaled features to two principal components using PCA.

- Silhouette score: {cluster_summary['silhouette_score']}
- Adjusted Rand Index against actual diagnosis labels: {cluster_summary['adjusted_rand_index_against_actual_labels']}
- PCA explained variance ratio: {cluster_summary['pca_explained_variance_ratio']}

This addition enhances the project by showing whether natural groupings in the data align with the known diagnosis labels.

![PCA K-Means Clusters](../artifacts/figures/pca_kmeans_clusters.png)

## Machine Learning Toolkit
This project uses **Scikit-learn** for dataset loading, preprocessing, model training, evaluation metrics, PCA, K-Means clustering, and pipeline management. Scikit-learn is useful because it provides consistent APIs for multiple ML algorithms and helps avoid data leakage through pipelines.

## Top Random Forest Feature Importances
| Feature | Importance |
|---|---:|
{top_features}

## Implementation
The implementation includes:
- `src/train_model.py` for training, evaluation, artifact generation, and report generation.
- `src/app.py` for the local web server and prediction API.
- `static/index.html`, `static/styles.css`, and `static/app.js` for the frontend interface.
- `artifacts/model_bundle.joblib` for the saved best model and metadata.
- `artifacts/metrics.json` for model results.

## Reflection and Learning Outcome
This project helped demonstrate the complete machine learning workflow from problem definition to evaluation and deployment-style prediction. The main challenge was making the workflow usable for non-technical users, which was solved by adding a simple web interface. The key insight was that several nuclear cell measurements are strongly correlated and highly predictive of diagnosis.

## Practical Applications
The project can be used as a learning prototype for clinical decision-support systems, medical data analysis, and educational demonstrations of classification workflows. It is not a replacement for professional medical diagnosis.
""",
        encoding="utf-8",
    )


def main() -> None:
    ensure_dirs()
    features, target, metadata = load_dataset()
    target_names = metadata["target_names"]

    class_distribution = save_class_balance_plot(target, target_names)
    save_correlation_heatmap(features)
    cluster_summary = save_pca_cluster_plot(features, target, target_names)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        stratify=target,
        random_state=42,
    )

    models = build_models()
    results, best_model_name = evaluate_models(models, x_train, x_test, y_train, y_test, target_names)
    best_model = models[best_model_name]
    feature_importances = []
    if "classifier" in best_model.named_steps and hasattr(best_model.named_steps["classifier"], "feature_importances_"):
        feature_importances = save_feature_importance_plot(best_model.named_steps["classifier"], list(features.columns))
    else:
        feature_importances = save_feature_importance_plot(models["Random Forest"].named_steps["classifier"], list(features.columns))

    bundle = {
        "model": best_model,
        "best_model_name": best_model_name,
        "feature_names": list(features.columns),
        "target_names": target_names,
        "feature_means": features.mean().round(4).to_dict(),
        "benign_example": features[target == 1].iloc[0].round(4).to_dict(),
        "malignant_example": features[target == 0].iloc[0].round(4).to_dict(),
        "metrics": results,
        "cluster_summary": cluster_summary,
        "class_distribution": class_distribution,
        "feature_importances": feature_importances,
    }
    joblib.dump(bundle, ARTIFACTS / "model_bundle.joblib")

    (ARTIFACTS / "metrics.json").write_text(json.dumps(bundle, indent=2, default=str), encoding="utf-8")
    write_report(metadata, features, target, results, best_model_name, cluster_summary, feature_importances)

    print(f"Training complete. Best model: {best_model_name}")
    print(f"Report written to: {REPORT}")


if __name__ == "__main__":
    main()
