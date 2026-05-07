# Integrative Industry Synthesis

## Overview
This repository contains a **capstone-level AI solution** for the healthcare industry focused on **AI-assisted chronic care triage for diabetes follow-up**. The artifact runs against the **UCI Diabetes 130-US Hospitals (1999-2008)** dataset (~99,000 inpatient encounters across 130 U.S. hospitals) and combines four genuinely different AI techniques into one workflow:

- **data analysis and statistical reasoning** on a real, independently produced clinical dataset
- **machine learning risk prediction** (logistic regression + random forest, selected on held-out ROC-AUC)
- **LLM-driven generative AI** for clinical case summaries and patient outreach messages
- **LLM-driven agentic routing** that selects a follow-up tool from a fixed action space and produces a justification

This is the **revised** version of an earlier submission. Two methodological problems flagged in review have been addressed:

1. The synthetic logistic-on-logistic dataset has been replaced with a real public dataset (UCI Diabetes 130-US Hospitals), removing the circular evaluation.
2. The previously rule-based "generative" and "agentic" layers have been replaced with real LLM calls. The agentic layer in particular is now a structured tool-selection step that returns JSON with action + justification + follow-up window.

---

## Industry Problem
Outpatient and post-discharge clinics need to identify which patients require faster follow-up to avoid unnecessary readmission. Early 30-day readmission is the operational target used here because it is well documented in the dataset and clinically meaningful. The system supports prioritization while keeping a **human in the loop** for high-risk cases.

---

## Integrated Solution
Four connected layers:

1. **Data analysis layer** – loads the UCI dataset, defines `needs_intervention = (readmitted < 30 days)`, drops out-of-scope discharges, encodes age bands as midpoints, and prepares numeric and categorical features.
2. **Predictive model layer** – trains a logistic regression and a random forest on a stratified split with class weighting, evaluates accuracy / precision / recall / F1 / ROC-AUC, and selects the better model for scoring.
3. **Generative AI layer** – calls an LLM to produce a short clinical case summary and a separate plain-language outreach message for each shortlisted patient, with prompts that forbid diagnosis, prescription, or invented indicators.
4. **Agentic routing layer** – the same LLM acts as a tool-using agent: given the patient context and a fixed list of follow-up tools, it returns strict JSON `{action, justification, follow_up_hours}`. Output is validated against an allow-list; invalid output is conservatively escalated to physician review.

> This artifact is a **prototype and applied case study**, not a deployed clinical product.

---

## Repository Structure

| Path | Purpose |
| --- | --- |
| `src/healthcare_triage_ai.py` | Main executable integrated artifact |
| `notebooks/integrated_industry_synthesis.ipynb` | Notebook walkthrough of the same workflow |
| `docs/system_architecture.md` | Architecture explanation and Mermaid diagram |
| `Reflective_Synthesis_Paper.md` | Reflective paper (markdown source) |
| `Reflective_Synthesis_Paper.pdf` | PDF version of the paper |
| `presentation_outline.md` | 15-minute mentor presentation and defense outline |
| `outputs/` | Generated metrics, recommendations, and dashboard image |
| `scripts/export_paper_pdf.py` | Helper that renders the markdown paper to PDF |
| `requirements.txt` | Environment snapshot generated with `pip freeze` |
| `.env.example` | Example environment file showing required LLM keys |

---

## Setup Instructions
Use Python **3.11+** (3.13 is what the prototype was tested against).

### 1. Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

The key runtime dependencies are: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `ucimlrepo` (dataset download), `openai` (default LLM backend), `anthropic` (alternative LLM backend), `llama-cpp-python` (local LLM fallback), and `python-dotenv` (loads `.env`).

### 3. Configure an LLM backend
Copy `.env.example` to `.env` and set **at least one** of:

```
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini

# or:
# ANTHROPIC_API_KEY=...
# ANTHROPIC_MODEL=claude-3-5-haiku-latest

# or, fully offline:
# LOCAL_LLM_GGUF=./models/qwen2.5-0.5b-instruct-q4_k_m.gguf
```

The script intentionally raises if no backend is configured. This avoids silently falling back to string-template "fake" generative output, which was one of the issues in the prior submission.

### 4. Run the integrated artifact
```bash
python src/healthcare_triage_ai.py
```

On the first run the UCI dataset (~20 MB) is fetched from the UCI ML Repository and cached at `data/diabetic_data.csv` for subsequent runs. The classifier scores all ~99k patients; the LLM layers run on a configurable shortlist (env `LLM_SHORTLIST_SIZE`, default 20).

### 5. Optional: open the notebook
Open `notebooks/integrated_industry_synthesis.ipynb` in VS Code or Jupyter and run the cells in order.

---

## Generated Outputs
After running the script, the following files are created in `outputs/`:

- `risk_dashboard.png` — risk-band counts and feature importance for the selected model.
- `patient_scores.csv` — risk probability and risk band for **every** patient in the dataset.
- `patient_recommendations.csv` — full LLM artifacts (case summary, agent action, justification, follow-up window, patient message) for the triage shortlist.
- `evaluation_metrics.json` — metrics for both models, the selected model, and metadata about the LLM backend that was used.

### Verified evaluation snapshot (real data)
The current prototype run on the UCI dataset, with class-weighted training and a stratified 80/20 split, produced:

- **Dataset:** UCI Diabetes 130-US Hospitals (1999-2008)
- **Records:** 99,340 (train 79,472 / test 19,868)
- **Positive rate (30-day readmission):** 0.1139
- **Logistic regression:** acc=0.677, precision=0.182, recall=0.528, F1=0.271, **ROC-AUC=0.665**
- **Random forest:** acc=0.690, precision=0.191, recall=0.533, F1=0.282, **ROC-AUC=0.671** (selected)
- **High or critical share by percentile bands:** 0.15 (top 5% Critical + next 10% High)
- **LLM backend:** `openai` / `gpt-4o-mini`
- **Agent action distribution on a 10-patient shortlist:** 5 physician_review, 4 nurse_outreach, 1 education_reminder

These numbers are intentionally lower than the original synthetic submission's headline 0.888 ROC-AUC. Real, noisy, independently produced clinical data is harder to predict than data generated from the same logistic equation the model is fitting, and the realistic score is what should be reported. The agentic routing distribution (not all rows pushed to the same action) is a useful signal that the LLM agent is actually reasoning over the patient context rather than collapsing to a constant decision.

---

## Submission-Ready Artifacts
- ✅ integrated industry artifact (real dataset + real ML + real LLM-driven generative + real LLM-driven agent)
- ✅ revised reflective synthesis paper (markdown + PDF)
- ✅ notebook walkthrough
- ✅ architecture description with Mermaid diagram
- ✅ presentation outline
- ✅ `requirements.txt` for reproducibility
- ✅ `.env.example` showing required LLM configuration

---

## Responsible AI Notes
- The dataset is **real but historical** (1999-2008 U.S. inpatient encounters); generalization to current outpatient populations should not be assumed.
- The workflow is designed for **decision support**, not diagnosis or prescription.
- High-risk and ambiguous cases are routed to **human review**.
- LLM prompts forbid diagnosis, prescription, and invented indicators.
- The agent's action space is closed and validated against an allow-list; invalid output triggers conservative escalation.
- Any real deployment would require subgroup fairness analysis, governance review, secure data handling, an explicit policy on what is sent to a third-party LLM provider (or use of the local `llama-cpp` backend), and ongoing monitoring.

---

## Notes on the Revision
The original submission was flagged for two specific issues:

1. **Circular methodology in the data**: synthetic data was generated from a logistic equation and a logistic regression was then fit to it, producing inflated metrics that confirmed the data-generating process rather than measuring generalization.
2. **Rule-based components mislabeled as AI**: `generate_case_summary`, `generate_patient_message`, and `assign_agent` were string templates and `if/else` thresholds, not generative AI or agentic reasoning.

Both have been addressed:

- The data is now the **UCI Diabetes 130-US Hospitals** dataset, fetched at runtime via `ucimlrepo` and cached locally.
- `generate_case_summary` and `generate_patient_message` are now real LLM calls with constrained system prompts.
- `assign_agent` has been replaced with `agent_route_patient`, a structured tool-selection step in which the LLM reads the patient context, picks one tool from a fixed list, and returns JSON with action + justification + follow-up window. Output is validated and conservatively escalated on parse failure.
- The model layer now compares logistic regression and random forest on a stratified split and selects on held-out ROC-AUC, replacing the from-scratch gradient-descent demo on circular data.
