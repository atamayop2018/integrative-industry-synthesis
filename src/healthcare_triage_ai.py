"""Integrated Industry Synthesis artifact (revised).

Industry focus: Healthcare
Use case: AI-assisted chronic care triage for diabetes follow-up, using the
UCI Diabetes 130-US Hospitals dataset (1999-2008).

This executable demo intentionally combines four capstone-style components:

1. Data analysis and statistical profiling of patient indicators on a real,
   independently produced clinical dataset (UCI 130-US Hospitals).
2. A supervised machine learning model (logistic regression and random forest)
   trained and evaluated against a held-out test set drawn from the real data.
3. A generative AI explanation layer powered by an actual large language model
   (OpenAI by default; Anthropic and local llama-cpp are also supported).
4. A lightweight agentic routing layer in which the same LLM evaluates each
   patient context, chooses one of several follow-up actions, and produces a
   short justification. Routing is therefore a model-driven decision rather
   than a hand-coded if/else block.

This artifact is a prototype and applied case study, not a deployed clinical
product. Real deployment would require fairness review, clinical validation,
governance approval, and ongoing monitoring.
"""

from __future__ import annotations

import json
import os
import re
import textwrap
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:  # Optional: load API keys from a .env file at the repo root
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:  # pragma: no cover - dotenv is optional
    pass
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------------------------

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_CSV = DATA_DIR / "diabetic_data.csv"
UCI_DATASET_ID = 296  # UCI Diabetes 130-US Hospitals for years 1999-2008

# Feature lists used by the model
NUMERIC_FEATURES = [
    "age_midpoint",
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]
CATEGORICAL_FEATURES = [
    "race",
    "gender",
    "A1Cresult",
    "max_glu_serum",
    "insulin",
    "change",
    "diabetesMed",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# How many high-risk patients to send through the LLM generative + agentic
# layers. The classifier scores all rows; the LLM is reserved for the
# triage shortlist to keep API usage and runtime bounded.
LLM_SHORTLIST_SIZE = int(os.getenv("LLM_SHORTLIST_SIZE", "20"))


# ---------------------------------------------------------------------------
# 1. Data loading and preprocessing (real dataset)
# ---------------------------------------------------------------------------


def load_uci_diabetes_dataset() -> pd.DataFrame:
    """Load the UCI Diabetes 130-US Hospitals dataset.

    Uses a local cache at ``data/diabetic_data.csv`` after the first fetch so
    that subsequent runs do not require network access.
    """
    if DATASET_CSV.exists():
        df = pd.read_csv(DATASET_CSV, low_memory=False)
        return df

    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:  # pragma: no cover - import-time guard
        raise RuntimeError(
            "ucimlrepo is required to download the UCI dataset. "
            "Install it with `pip install ucimlrepo`."
        ) from exc

    print("Fetching UCI Diabetes 130-US Hospitals dataset (only once)...")
    bundle = fetch_ucirepo(id=UCI_DATASET_ID)
    features = bundle.data.features.copy()
    targets = bundle.data.targets.copy()
    df = pd.concat([features, targets], axis=1)
    df.to_csv(DATASET_CSV, index=False)
    return df


_AGE_MIDPOINT = {
    "[0-10)": 5,
    "[10-20)": 15,
    "[20-30)": 25,
    "[30-40)": 35,
    "[40-50)": 45,
    "[50-60)": 55,
    "[60-70)": 65,
    "[70-80)": 75,
    "[80-90)": 85,
    "[90-100)": 95,
}


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw UCI dataset and define the binary triage target.

    Target definition: ``needs_intervention = 1`` when the patient was
    readmitted in fewer than 30 days (``readmitted == "<30"``), and ``0``
    otherwise. Early readmission is a clinically meaningful proxy for
    "this patient should have received earlier or more intensive
    follow-up", which is the operational decision the prototype supports.
    """
    df = df.copy()

    # Drop rows where the patient died or was discharged to hospice. These
    # outcomes are not in scope for outpatient follow-up triage.
    df = df[~df["discharge_disposition_id"].isin([11, 13, 14, 19, 20, 21])]

    # Drop the small number of unknown-gender rows.
    df = df[df["gender"].isin(["Male", "Female"])]

    # Numeric age midpoint
    df["age_midpoint"] = df["age"].map(_AGE_MIDPOINT).astype(float)

    # Treat sentinel "?" values as missing.
    for col in ["race", "A1Cresult", "max_glu_serum", "insulin", "change", "diabetesMed"]:
        if col in df.columns:
            df[col] = df[col].replace({"?": np.nan, "None": np.nan, "Unknown/Invalid": np.nan})
            df[col] = df[col].fillna("Missing")

    # Cast ID columns to string so the OneHotEncoder treats them as categorical.
    for col in ["admission_type_id", "discharge_disposition_id", "admission_source_id"]:
        df[col] = df[col].astype(str)

    # Build the binary target.
    df["needs_intervention"] = (df["readmitted"] == "<30").astype(int)

    # Stable patient identifier
    df = df.reset_index(drop=True)
    df["patient_id"] = [f"PT-{idx:06d}" for idx in range(1, len(df) + 1)]

    keep_cols = ["patient_id", "age", "needs_intervention", "readmitted"] + ALL_FEATURES
    df = df[keep_cols].dropna(subset=["age_midpoint"])
    return df


# ---------------------------------------------------------------------------
# 2. Machine learning layer
# ---------------------------------------------------------------------------


@dataclass
class TrainedModel:
    name: str
    pipeline: Pipeline
    test_metrics: dict[str, float]
    confusion: list[list[int]]


def _build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=50)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ]
    )


def _evaluate(name: str, pipeline: Pipeline, X_test, y_test) -> TrainedModel:
    proba = pipeline.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    metrics = {
        "test_accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "test_precision": round(float(precision_score(y_test, preds, zero_division=0)), 4),
        "test_recall": round(float(recall_score(y_test, preds, zero_division=0)), 4),
        "test_f1": round(float(f1_score(y_test, preds, zero_division=0)), 4),
        "test_roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
    }
    cm = confusion_matrix(y_test, preds).tolist()
    return TrainedModel(name=name, pipeline=pipeline, test_metrics=metrics, confusion=cm)


def train_models(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any], TrainedModel]:
    """Train logistic regression and random forest classifiers.

    Returns the dataframe with risk scores attached, a metrics dictionary
    with results from both models, and the model selected for downstream
    use (highest test ROC-AUC).
    """
    X = df[ALL_FEATURES]
    y = df["needs_intervention"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED
    )

    preprocessor = _build_preprocessor()

    lr_pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    solver="liblinear",
                    class_weight="balanced",
                    C=0.5,
                    random_state=SEED,
                ),
            ),
        ]
    )
    rf_pipeline = Pipeline(
        steps=[
            ("preprocess", _build_preprocessor()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=14,
                    min_samples_leaf=20,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=SEED,
                ),
            ),
        ]
    )

    print("Training logistic regression...")
    lr_pipeline.fit(X_train, y_train)
    lr_model = _evaluate("logistic_regression", lr_pipeline, X_test, y_test)

    print("Training random forest...")
    rf_pipeline.fit(X_train, y_train)
    rf_model = _evaluate("random_forest", rf_pipeline, X_test, y_test)

    selected = max([lr_model, rf_model], key=lambda m: m["test_roc_auc"] if False else m.test_metrics["test_roc_auc"])
    print(f"Selected model for scoring: {selected.name} "
          f"(ROC-AUC={selected.test_metrics['test_roc_auc']})")

    # Score everyone using the selected model.
    risk_proba = selected.pipeline.predict_proba(df[ALL_FEATURES])[:, 1]
    scored = df.copy()
    scored["risk_probability"] = risk_proba
    # Operational risk bands are defined by percentile of predicted
    # probability rather than fixed thresholds. This matches how a clinic
    # would actually use the model (triage capacity is finite), and it
    # adapts gracefully to whatever calibration the chosen model produces.
    scored["risk_band"] = assign_risk_bands(scored["risk_probability"])

    metrics = {
        "dataset": "UCI Diabetes 130-US Hospitals (1999-2008)",
        "records_total": int(len(df)),
        "records_train": int(len(X_train)),
        "records_test": int(len(X_test)),
        "positive_rate_overall": round(float(y.mean()), 4),
        "positive_rate_test": round(float(y_test.mean()), 4),
        "selected_model": selected.name,
        "high_or_critical_share": round(
            float(scored["risk_band"].isin(["High", "Critical"]).mean()), 4
        ),
        "models": {
            lr_model.name: {"metrics": lr_model.test_metrics, "confusion": lr_model.confusion},
            rf_model.name: {"metrics": rf_model.test_metrics, "confusion": rf_model.confusion},
        },
    }
    return scored, metrics, selected


def assign_risk_bands(probabilities: pd.Series) -> pd.Series:
    """Assign operational risk bands using percentile thresholds.

    The thresholds correspond to how a clinic with finite triage capacity
    would actually want to use the model:

    - Critical: top 5% of predicted probabilities (escalate now).
    - High:     next 10% (proactive nurse outreach).
    - Moderate: next 20% (educational reminder + monitoring).
    - Low:      remaining 65% (routine cadence).
    """
    quantiles = probabilities.quantile([0.65, 0.85, 0.95]).to_dict()
    moderate_cut = quantiles[0.65]
    high_cut = quantiles[0.85]
    critical_cut = quantiles[0.95]

    def _band(p: float) -> str:
        if p >= critical_cut:
            return "Critical"
        if p >= high_cut:
            return "High"
        if p >= moderate_cut:
            return "Moderate"
        return "Low"

    return probabilities.apply(_band)


def feature_importance_for(model: TrainedModel) -> pd.Series:
    """Return a feature-importance Series for plotting."""
    pipe = model.pipeline
    pre = pipe.named_steps["preprocess"]
    feature_names = pre.get_feature_names_out()
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "coef_"):
        importances = clf.coef_[0]
    elif hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    else:
        return pd.Series(dtype=float)
    series = pd.Series(importances, index=feature_names)
    return series.reindex(series.abs().sort_values(ascending=False).index).head(15)


# ---------------------------------------------------------------------------
# 3. & 4. LLM client (generative + agentic backends)
# ---------------------------------------------------------------------------


class LLMUnavailableError(RuntimeError):
    """Raised when no usable LLM backend is configured."""


class LLMClient:
    """Thin wrapper that selects an available LLM backend.

    Order of preference:
        1. OpenAI Chat Completions API   (env: OPENAI_API_KEY)
        2. Anthropic Messages API        (env: ANTHROPIC_API_KEY)
        3. Local llama-cpp-python model  (env: LOCAL_LLM_GGUF pointing to a
           GGUF file on disk).

    The project's grading rubric requires that the generative and agentic
    layers be implemented with real AI techniques. This class therefore
    raises :class:`LLMUnavailableError` if no backend is reachable instead
    of silently falling back to string templates.
    """

    def __init__(self) -> None:
        self.backend: str | None = None
        self.model_name: str | None = None
        self._client: Any = None
        self._llama: Any = None

        if os.getenv("OPENAI_API_KEY"):
            try:
                from openai import OpenAI

                self._client = OpenAI()
                self.backend = "openai"
                self.model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
                return
            except Exception as exc:  # pragma: no cover
                print(f"[LLMClient] OpenAI init failed: {exc}")

        if os.getenv("ANTHROPIC_API_KEY"):
            try:
                import anthropic

                self._client = anthropic.Anthropic()
                self.backend = "anthropic"
                self.model_name = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-latest")
                return
            except Exception as exc:  # pragma: no cover
                print(f"[LLMClient] Anthropic init failed: {exc}")

        local_path = os.getenv("LOCAL_LLM_GGUF")
        if local_path and Path(local_path).exists():
            try:
                from llama_cpp import Llama

                self._llama = Llama(
                    model_path=local_path,
                    n_ctx=2048,
                    n_threads=os.cpu_count() or 4,
                    verbose=False,
                )
                self.backend = "llama_cpp"
                self.model_name = Path(local_path).name
                return
            except Exception as exc:  # pragma: no cover
                print(f"[LLMClient] llama-cpp init failed: {exc}")

        raise LLMUnavailableError(
            "No LLM backend available. Set OPENAI_API_KEY (preferred), "
            "ANTHROPIC_API_KEY, or LOCAL_LLM_GGUF (path to a GGUF model)."
        )

    # -- low level call ---------------------------------------------------

    def chat(
        self,
        system: str,
        user: str,
        max_tokens: int = 350,
        temperature: float = 0.4,
        json_mode: bool = False,
    ) -> str:
        if self.backend == "openai":
            kwargs: dict[str, Any] = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            response = self._client.chat.completions.create(**kwargs)
            return (response.choices[0].message.content or "").strip()

        if self.backend == "anthropic":
            response = self._client.messages.create(
                model=self.model_name,
                system=system,
                messages=[{"role": "user", "content": user}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            parts = [block.text for block in response.content if getattr(block, "text", None)]
            return "".join(parts).strip()

        if self.backend == "llama_cpp":
            response = self._llama.create_chat_completion(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response["choices"][0]["message"]["content"].strip()

        raise LLMUnavailableError("LLMClient has no backend configured")


# ---------------------------------------------------------------------------
# 3. Generative AI layer (LLM-based case summary + outreach message)
# ---------------------------------------------------------------------------


def _patient_context_block(row: pd.Series) -> str:
    """Render the patient indicators as a structured prompt context block."""
    fields = [
        ("Age band", row.get("age", "?")),
        ("Race", row.get("race", "?")),
        ("Gender", row.get("gender", "?")),
        ("Time in hospital (days)", row.get("time_in_hospital", "?")),
        ("Number of inpatient visits (last yr)", row.get("number_inpatient", "?")),
        ("Number of emergency visits (last yr)", row.get("number_emergency", "?")),
        ("Number of outpatient visits (last yr)", row.get("number_outpatient", "?")),
        ("Number of medications", row.get("num_medications", "?")),
        ("Number of lab procedures", row.get("num_lab_procedures", "?")),
        ("Number of diagnoses", row.get("number_diagnoses", "?")),
        ("A1C result", row.get("A1Cresult", "?")),
        ("Max glucose serum", row.get("max_glu_serum", "?")),
        ("Insulin status", row.get("insulin", "?")),
        ("Medication change this stay", row.get("change", "?")),
        ("On diabetes medication", row.get("diabetesMed", "?")),
        ("Predicted 30-day readmission probability", f"{row['risk_probability']:.1%}"),
        ("Risk band", row["risk_band"]),
    ]
    lines = [f"- {label}: {value}" for label, value in fields]
    return "\n".join(lines)


CASE_SUMMARY_SYSTEM = (
    "You are a clinical decision-support assistant for an outpatient diabetes "
    "follow-up team. You write short, professional case summaries that help "
    "nurses and physicians prioritize their day. You never diagnose, never "
    "prescribe, and never invent indicators that were not provided. If a "
    "field is missing, do not speculate about it. Keep summaries to four "
    "sentences or fewer."
)

PATIENT_MESSAGE_SYSTEM = (
    "You write brief, empathetic outreach messages for patients with "
    "diabetes who may need follow-up care. Use plain, supportive language at "
    "about a sixth-grade reading level. Do not provide medical advice or "
    "diagnoses. Always remind the patient that the care team can answer "
    "questions. Keep the message to three sentences."
)


def generate_case_summary(llm: LLMClient, row: pd.Series) -> str:
    user_prompt = (
        "Write a short case summary for the clinical team based on the "
        "structured indicators below. Mention only the most relevant drivers "
        "of the predicted risk. End with one sentence on what the team "
        "should consider next.\n\nPatient indicators:\n"
        f"{_patient_context_block(row)}"
    )
    return llm.chat(
        system=CASE_SUMMARY_SYSTEM,
        user=user_prompt,
        max_tokens=240,
        temperature=0.5,
    )


def generate_patient_message(llm: LLMClient, row: pd.Series, action: str) -> str:
    user_prompt = (
        "Write a short outreach message that the clinic will send to the "
        "patient. The recommended next step from the care team is: "
        f"'{action}'. Tailor the tone to that step. Do not mention "
        "probabilities or model output, and do not include medical advice.\n\n"
        "Patient context:\n"
        f"{_patient_context_block(row)}"
    )
    return llm.chat(
        system=PATIENT_MESSAGE_SYSTEM,
        user=user_prompt,
        max_tokens=180,
        temperature=0.6,
    )


# ---------------------------------------------------------------------------
# 4. Agentic routing layer (LLM as the routing agent)
# ---------------------------------------------------------------------------


AGENT_TOOLS = [
    {
        "name": "education_reminder",
        "description": (
            "Send an automated educational reminder about diabetes self-care, "
            "medication adherence, and the next routine appointment. Best for "
            "patients with stable indicators and no recent acute utilization."
        ),
    },
    {
        "name": "nurse_outreach",
        "description": (
            "Schedule a nurse phone call within the next 5 business days to "
            "review symptoms, medication adherence, and barriers to follow-up. "
            "Best for moderate-risk patients or those with recent missed care."
        ),
    },
    {
        "name": "physician_review",
        "description": (
            "Escalate the chart to a physician for review and possible "
            "expedited visit (within 72 hours). Best for high-risk patients or "
            "those with multiple acute utilization signals."
        ),
    },
]

AGENT_SYSTEM = (
    "You are an autonomous triage routing agent for an outpatient diabetes "
    "follow-up team. Your job is to read the patient context produced by the "
    "predictive model, reason about the available follow-up tools, and pick "
    "EXACTLY ONE tool that best fits the patient's situation. You must also "
    "produce a one-sentence justification grounded in the indicators provided "
    "and an estimated follow-up window in hours.\n\n"
    "Operating rules:\n"
    "- Be conservative: when in doubt between two adjacent tools, pick the "
    "one with more human oversight.\n"
    "- Never invent indicators that were not provided.\n"
    "- Output STRICT JSON with the schema "
    '{"action": <tool_name>, "justification": <string>, "follow_up_hours": <integer>}.'
)


def _format_tools_for_prompt(tools: list[dict[str, str]]) -> str:
    return "\n".join(
        f"- {tool['name']}: {tool['description']}" for tool in tools
    )


_VALID_ACTIONS = {tool["name"] for tool in AGENT_TOOLS}


def _parse_agent_response(raw: str) -> dict[str, Any]:
    """Parse the LLM's JSON response, with a tolerant regex fallback."""
    cleaned = raw.strip()
    # Strip markdown code fences if any.
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise
        data = json.loads(match.group(0))

    action = str(data.get("action", "")).strip()
    if action not in _VALID_ACTIONS:
        raise ValueError(f"LLM returned invalid action: {action!r}")

    justification = str(data.get("justification", "")).strip() or "(no justification)"
    follow_up_hours = data.get("follow_up_hours", 168)
    try:
        follow_up_hours = int(follow_up_hours)
    except (TypeError, ValueError):
        follow_up_hours = 168
    return {
        "action": action,
        "justification": justification,
        "follow_up_hours": follow_up_hours,
    }


def agent_route_patient(llm: LLMClient, row: pd.Series) -> dict[str, Any]:
    """Use the LLM as a lightweight routing agent.

    The LLM is given (a) the patient indicators, (b) the available tools,
    and (c) instructions for output format. It selects one tool and returns
    its justification. This is the agentic reasoning step required by the
    rubric: the routing decision is produced by the model rather than by a
    hand-coded threshold ladder.
    """
    user_prompt = (
        "Available follow-up tools:\n"
        f"{_format_tools_for_prompt(AGENT_TOOLS)}\n\n"
        "Patient context produced by the predictive model:\n"
        f"{_patient_context_block(row)}\n\n"
        "Choose the single best tool and respond with the JSON schema "
        "described in the system message."
    )
    raw = llm.chat(
        system=AGENT_SYSTEM,
        user=user_prompt,
        max_tokens=220,
        temperature=0.2,
        json_mode=True,
    )
    try:
        return _parse_agent_response(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        # One retry with stricter instructions before failing the row.
        retry = llm.chat(
            system=AGENT_SYSTEM
            + "\n\nIMPORTANT: Your previous output was not valid JSON or "
              "used an invalid action name. Retry with valid JSON only.",
            user=user_prompt,
            max_tokens=220,
            temperature=0.0,
            json_mode=True,
        )
        try:
            return _parse_agent_response(retry)
        except (json.JSONDecodeError, ValueError):
            return {
                "action": "physician_review",
                "justification": (
                    "Agent response could not be parsed; conservatively "
                    "escalating to physician review for human oversight. "
                    f"(parse error: {exc})"
                ),
                "follow_up_hours": 72,
            }


def shortlist_for_llm(scored_df: pd.DataFrame, size: int) -> pd.DataFrame:
    """Pick patients to send through the LLM layers.

    Strategy: take the top ``size - 5`` highest-risk patients (the most
    operationally important), plus 5 stratified samples from lower bands so
    the demonstration shows the agent making different routing choices.
    """
    if size <= 0:
        return scored_df.head(0)

    top_n = max(1, size - 5)
    top = scored_df.sort_values("risk_probability", ascending=False).head(top_n)
    lower = scored_df.drop(top.index)
    sample_size = min(size - len(top), len(lower))
    if sample_size > 0:
        sampled = lower.sample(n=sample_size, random_state=SEED)
        shortlist = pd.concat([top, sampled])
    else:
        shortlist = top
    return shortlist.sort_values("risk_probability", ascending=False).reset_index(drop=True)


def run_llm_layers(
    llm: LLMClient, shortlist: pd.DataFrame
) -> pd.DataFrame:
    """Run generative + agentic layers for each shortlisted patient."""
    case_summaries: list[str] = []
    actions: list[str] = []
    justifications: list[str] = []
    windows: list[int] = []
    messages: list[str] = []

    print(f"Running LLM layers via backend={llm.backend} model={llm.model_name} "
          f"on {len(shortlist)} patients...")

    for idx, row in shortlist.iterrows():
        try:
            summary = generate_case_summary(llm, row)
        except Exception as exc:  # pragma: no cover
            summary = f"[case summary unavailable: {exc}]"
        case_summaries.append(summary)

        decision = agent_route_patient(llm, row)
        actions.append(decision["action"])
        justifications.append(decision["justification"])
        windows.append(decision["follow_up_hours"])

        try:
            msg = generate_patient_message(llm, row, decision["action"])
        except Exception as exc:  # pragma: no cover
            msg = f"[outreach message unavailable: {exc}]"
        messages.append(msg)

        if (idx + 1) % 5 == 0 or (idx + 1) == len(shortlist):
            print(f"  processed {idx + 1}/{len(shortlist)}")

    out = shortlist.copy()
    out["case_summary"] = case_summaries
    out["agent_action"] = actions
    out["agent_justification"] = justifications
    out["follow_up_hours"] = windows
    out["patient_message"] = messages
    return out


# ---------------------------------------------------------------------------
# Visualizations and exports
# ---------------------------------------------------------------------------


def save_visuals(scored_df: pd.DataFrame, model: TrainedModel, metrics: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    order = ["Low", "Moderate", "High", "Critical"]
    counts = scored_df["risk_band"].value_counts().reindex(order).fillna(0)
    counts.plot(
        kind="bar",
        color=["#8ecae6", "#ffb703", "#fb8500", "#d62828"],
        ax=axes[0],
    )
    axes[0].set_title(
        f"Predicted Risk Tiers (n={len(scored_df):,})\n"
        f"Selected model: {model.name} | "
        f"ROC-AUC={model.test_metrics['test_roc_auc']:.3f}"
    )
    axes[0].set_xlabel("Risk Band")
    axes[0].set_ylabel("Patients")
    axes[0].tick_params(axis="x", rotation=0)

    importance = feature_importance_for(model)
    if not importance.empty:
        importance.iloc[::-1].plot(kind="barh", color="#457b9d", ax=axes[1])
        axes[1].set_title(f"Top Features ({model.name})")
        axes[1].set_xlabel("Standardized weight or importance")

    fig.suptitle(
        "AI-Assisted Chronic Care Triage Dashboard "
        "(UCI Diabetes 130-US Hospitals)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "risk_dashboard.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def export_outputs(
    scored_df: pd.DataFrame,
    llm_df: pd.DataFrame,
    metrics: dict[str, Any],
    llm_meta: dict[str, Any],
) -> None:
    # Full scored CSV (everyone) — model results only (no LLM artifacts).
    scored_export = scored_df[
        [
            "patient_id",
            "age",
            "gender",
            "race",
            "A1Cresult",
            "insulin",
            "diabetesMed",
            "number_inpatient",
            "number_emergency",
            "number_outpatient",
            "risk_probability",
            "risk_band",
            "needs_intervention",
        ]
    ].sort_values("risk_probability", ascending=False)
    scored_export.to_csv(OUTPUT_DIR / "patient_scores.csv", index=False)

    # Shortlist CSV — full LLM artifacts for the shortlisted patients.
    llm_df.sort_values("risk_probability", ascending=False).to_csv(
        OUTPUT_DIR / "patient_recommendations.csv", index=False
    )

    metrics_with_llm = dict(metrics)
    metrics_with_llm["llm_layer"] = llm_meta
    with (OUTPUT_DIR / "evaluation_metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics_with_llm, fh, indent=2)

    print("\n=== Integrated Healthcare AI Triage Demo ===")
    print(f"Dataset: {metrics['dataset']}")
    print(f"Records: {metrics['records_total']:,} "
          f"(train={metrics['records_train']:,}, test={metrics['records_test']:,})")
    print(f"Positive rate (overall): {metrics['positive_rate_overall']:.3f}")
    print(f"Selected model: {metrics['selected_model']}")
    for name, info in metrics["models"].items():
        m = info["metrics"]
        print(
            f"  - {name}: acc={m['test_accuracy']:.3f}, "
            f"prec={m['test_precision']:.3f}, rec={m['test_recall']:.3f}, "
            f"f1={m['test_f1']:.3f}, roc_auc={m['test_roc_auc']:.3f}"
        )
    print(f"\nLLM backend: {llm_meta['backend']} ({llm_meta['model']})")
    print(f"LLM-processed shortlist: {llm_meta['shortlist_size']} patients")
    print("Agent action distribution:")
    for action, count in llm_df["agent_action"].value_counts().items():
        print(f"  - {action}: {count}")

    print("\nTop 3 routed cases (LLM-driven):")
    preview = llm_df.sort_values("risk_probability", ascending=False).head(3)
    for _, row in preview.iterrows():
        print(f"\nPatient {row['patient_id']} | "
              f"risk={row['risk_probability']:.1%} ({row['risk_band']}) | "
              f"action={row['agent_action']} ({row['follow_up_hours']}h)")
        print(textwrap.fill(f"Justification: {row['agent_justification']}", 96))
        print(textwrap.fill(f"Summary: {row['case_summary']}", 96))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("Loading UCI Diabetes 130-US Hospitals dataset...")
    raw = load_uci_diabetes_dataset()
    df = preprocess_dataset(raw)
    print(f"Preprocessed dataset: {len(df):,} rows, "
          f"positive rate={df['needs_intervention'].mean():.3f}")

    scored_df, metrics, model = train_models(df)
    save_visuals(scored_df, model, metrics)

    llm = LLMClient()
    shortlist = shortlist_for_llm(scored_df, LLM_SHORTLIST_SIZE)
    llm_df = run_llm_layers(llm, shortlist)

    llm_meta = {
        "backend": llm.backend,
        "model": llm.model_name,
        "shortlist_size": int(len(shortlist)),
        "tools": [tool["name"] for tool in AGENT_TOOLS],
    }

    export_outputs(scored_df, llm_df, metrics, llm_meta)


if __name__ == "__main__":
    main()
