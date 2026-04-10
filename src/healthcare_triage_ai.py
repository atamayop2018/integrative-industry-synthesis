"""Integrated Industry Synthesis artifact.

Industry focus: Healthcare
Use case: AI-assisted chronic care triage for outpatient diabetes follow-up.

This executable demo intentionally combines four capstone-style components:
1. Data analysis and statistical profiling of patient risk indicators.
2. A supervised machine learning model for intervention prioritization.
3. A generative-style natural language explanation layer.
4. An agentic routing workflow that assigns the next operational action.

The dataset is synthetic and privacy-safe by design.
"""

from __future__ import annotations

from pathlib import Path
import json
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEED = 42
RNG = np.random.default_rng(SEED)
ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def build_synthetic_dataset(n_rows: int = 240) -> pd.DataFrame:
    """Create a privacy-safe synthetic dataset for outpatient follow-up triage."""
    age = RNG.integers(28, 86, size=n_rows)
    a1c = np.round(RNG.normal(7.8, 1.3, size=n_rows).clip(5.2, 13.5), 1)
    systolic_bp = np.round(RNG.normal(132, 17, size=n_rows).clip(96, 210)).astype(int)
    missed_appointments = RNG.poisson(1.3, size=n_rows).clip(0, 6)
    medication_adherence = np.round(RNG.normal(0.77, 0.14, size=n_rows).clip(0.35, 0.99), 2)
    prior_er_visits = RNG.poisson(0.8, size=n_rows).clip(0, 5)

    logit = (
        -2.8
        + 0.04 * (age - 50)
        + 0.70 * (a1c - 7.0)
        + 0.020 * (systolic_bp - 120)
        + 0.50 * missed_appointments
        + 0.60 * prior_er_visits
        - 3.0 * (medication_adherence - 0.70)
    )
    probability = 1 / (1 + np.exp(-logit))
    needs_intervention = RNG.binomial(1, probability)

    return pd.DataFrame(
        {
            "patient_id": [f"PT-{idx:03d}" for idx in range(1, n_rows + 1)],
            "age": age,
            "a1c": a1c,
            "systolic_bp": systolic_bp,
            "missed_appointments": missed_appointments,
            "medication_adherence": medication_adherence,
            "prior_er_visits": prior_er_visits,
            "needs_intervention": needs_intervention,
        }
    )


def label_risk_band(probability: float) -> str:
    if probability >= 0.75:
        return "Critical"
    if probability >= 0.55:
        return "High"
    if probability >= 0.35:
        return "Moderate"
    return "Low"


def assign_agent(row: pd.Series) -> str:
    """Simple agentic routing policy with human escalation."""
    if row["risk_probability"] >= 0.75 or row["prior_er_visits"] >= 2:
        return "Physician review agent"
    if row["risk_probability"] >= 0.55 or row["missed_appointments"] >= 2:
        return "Nurse outreach agent"
    return "Education and reminder agent"


def generate_case_summary(row: pd.Series) -> str:
    drivers: list[str] = []
    if row["a1c"] >= 8.5:
        drivers.append(f"elevated A1C ({row['a1c']})")
    if row["systolic_bp"] >= 145:
        drivers.append(f"high systolic blood pressure ({row['systolic_bp']})")
    if row["missed_appointments"] >= 2:
        drivers.append(f"{int(row['missed_appointments'])} missed appointments")
    if row["medication_adherence"] < 0.70:
        drivers.append(f"low medication adherence ({row['medication_adherence']:.0%})")
    if row["prior_er_visits"] >= 1:
        drivers.append(f"{int(row['prior_er_visits'])} prior ER visit(s)")

    risk_drivers = ", ".join(drivers) if drivers else "stable recent indicators"
    summary = (
        f"Patient {row['patient_id']} is in the {row['risk_band'].lower()} risk tier "
        f"with a predicted intervention probability of {row['risk_probability']:.1%}. "
        f"Key drivers include {risk_drivers}. Recommended next step: {row['assigned_agent']}."
    )
    return textwrap.fill(summary, width=96)


def generate_patient_message(row: pd.Series) -> str:
    if row["risk_band"] in {"Critical", "High"}:
        return (
            "Please schedule a follow-up within 72 hours and confirm medication access. "
            "A care team member should review barriers and recent symptoms."
        )
    if row["risk_band"] == "Moderate":
        return (
            "Send a check-in reminder, reinforce glucose monitoring, and encourage attendance "
            "at the next appointment."
        )
    return "Send an automated wellness reminder and keep the current routine follow-up cadence."


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(values, -50, 50)))


def fit_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.08,
    epochs: int = 2500,
) -> tuple[np.ndarray, float]:
    """Fit a simple logistic regression model with gradient descent using NumPy only."""
    weights = np.zeros(X.shape[1], dtype=float)
    bias = 0.0

    for _ in range(epochs):
        linear_output = X @ weights + bias
        predictions = sigmoid(linear_output)
        errors = predictions - y

        weights -= learning_rate * (X.T @ errors) / len(X)
        bias -= learning_rate * float(errors.mean())

    return weights, bias


def roc_auc_from_scores(y_true: np.ndarray, scores: np.ndarray) -> float:
    positive_scores = scores[y_true == 1]
    negative_scores = scores[y_true == 0]

    if len(positive_scores) == 0 or len(negative_scores) == 0:
        return 0.5

    wins = (positive_scores[:, None] > negative_scores).mean()
    ties = (positive_scores[:, None] == negative_scores).mean()
    return float(wins + 0.5 * ties)


def train_and_score(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float], pd.Series]:
    features = [
        "age",
        "a1c",
        "systolic_bp",
        "missed_appointments",
        "medication_adherence",
        "prior_er_visits",
    ]

    X = df[features].astype(float)
    y = df["needs_intervention"].to_numpy(dtype=float)

    positive_idx = np.where(y == 1)[0]
    negative_idx = np.where(y == 0)[0]
    RNG.shuffle(positive_idx)
    RNG.shuffle(negative_idx)

    pos_test_size = max(1, int(0.25 * len(positive_idx)))
    neg_test_size = max(1, int(0.25 * len(negative_idx)))
    test_idx = np.concatenate([positive_idx[:pos_test_size], negative_idx[:neg_test_size]])
    train_mask = np.ones(len(df), dtype=bool)
    train_mask[test_idx] = False

    X_train = X.loc[train_mask].to_numpy()
    X_test = X.loc[~train_mask].to_numpy()
    y_train = y[train_mask]
    y_test = y[~train_mask]

    train_mean = X_train.mean(axis=0)
    train_std = X_train.std(axis=0)
    train_std[train_std == 0] = 1.0

    X_train_scaled = (X_train - train_mean) / train_std
    X_test_scaled = (X_test - train_mean) / train_std
    X_all_scaled = (X.to_numpy() - train_mean) / train_std

    weights, bias = fit_logistic_regression(X_train_scaled, y_train)

    test_probabilities = sigmoid(X_test_scaled @ weights + bias)
    test_predictions = (test_probabilities >= 0.50).astype(int)

    scored_df = df.copy()
    scored_df["risk_probability"] = sigmoid(X_all_scaled @ weights + bias)
    scored_df["risk_band"] = scored_df["risk_probability"].apply(label_risk_band)

    coefficients = pd.Series(weights, index=features, name="coefficient").sort_values(
        key=np.abs,
        ascending=False,
    )

    metrics = {
        "records": int(len(df)),
        "positive_rate": round(float(y.mean()), 3),
        "test_accuracy": round(float((test_predictions == y_test).mean()), 3),
        "test_roc_auc": round(float(roc_auc_from_scores(y_test, test_probabilities)), 3),
        "high_or_critical_share": round(
            float(scored_df["risk_band"].isin(["High", "Critical"]).mean()), 3
        ),
    }
    return scored_df, metrics, coefficients


def save_visuals(scored_df: pd.DataFrame, coefficients: pd.Series) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    order = ["Low", "Moderate", "High", "Critical"]
    (
        scored_df["risk_band"]
        .value_counts()
        .reindex(order)
        .plot(kind="bar", color=["#8ecae6", "#ffb703", "#fb8500", "#d62828"], ax=axes[0])
    )
    axes[0].set_title("Predicted Patient Risk Tiers")
    axes[0].set_xlabel("Risk Band")
    axes[0].set_ylabel("Patients")
    axes[0].tick_params(axis="x", rotation=0)

    coefficients.sort_values().plot(kind="barh", color="#457b9d", ax=axes[1])
    axes[1].set_title("Model Coefficient Importance")
    axes[1].set_xlabel("Standardized Weight")
    axes[1].set_ylabel("Feature")

    fig.suptitle("AI-Assisted Chronic Care Triage Dashboard", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "risk_dashboard.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def export_outputs(scored_df: pd.DataFrame, metrics: dict[str, float]) -> None:
    scored_df = scored_df.copy()
    scored_df["assigned_agent"] = scored_df.apply(assign_agent, axis=1)
    scored_df["case_summary"] = scored_df.apply(generate_case_summary, axis=1)
    scored_df["patient_message"] = scored_df.apply(generate_patient_message, axis=1)

    scored_df.sort_values("risk_probability", ascending=False).to_csv(
        OUTPUT_DIR / "patient_recommendations.csv",
        index=False,
    )

    with (OUTPUT_DIR / "evaluation_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    preview_columns = [
        "patient_id",
        "risk_probability",
        "risk_band",
        "assigned_agent",
        "case_summary",
    ]
    print("\n=== Integrated Healthcare AI Triage Demo ===")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\nTop 5 routed cases:")
    print(
        scored_df.sort_values("risk_probability", ascending=False)[preview_columns]
        .head(5)
        .to_string(index=False)
    )


def main() -> None:
    df = build_synthetic_dataset()
    scored_df, metrics, coefficients = train_and_score(df)
    save_visuals(scored_df, coefficients)
    export_outputs(scored_df, metrics)


if __name__ == "__main__":
    main()
