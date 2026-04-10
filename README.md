# Integrative Industry Synthesis

## Overview
This repository contains a **capstone-level AI solution** for the healthcare industry focused on **AI-assisted chronic care triage for outpatient diabetes follow-up**.

The project was designed to satisfy the Integrative Industry Synthesis requirements by combining methods from multiple prior capstone domains into one cohesive workflow:
- **data analysis and statistical reasoning**
- **machine learning risk prediction**
- **generative AI style explanation**
- **agentic routing for operational follow-up**

---

## Industry Problem
Outpatient clinics often need to identify which patients require faster follow-up based on risk indicators such as:
- elevated A1C
- high blood pressure
- missed appointments
- medication adherence issues
- recent emergency room visits

This project demonstrates how an integrated AI system can help prioritize outreach while keeping **human oversight** in the loop for high-risk cases.

---

## Integrated Solution
The system includes four connected layers:

1. **Data profiling layer**  
   Reviews structured patient indicators and summarizes risk patterns.

2. **Predictive model layer**  
   Estimates the probability that a patient needs intervention.

3. **Generative explanation layer**  
   Produces a short natural-language case summary and outreach guidance.

4. **Agentic routing layer**  
   Routes each case to one of the following actions:
   - education and reminder agent
   - nurse outreach agent
   - physician review agent

> This artifact is a **prototype and applied case study**, not a deployed clinical product.

---

## Repository Structure

| Path | Purpose |
| --- | --- |
| `src/healthcare_triage_ai.py` | Main executable integrated artifact |
| `notebooks/integrated_industry_synthesis.ipynb` | Notebook version of the workflow |
| `docs/system_architecture.md` | Architecture explanation and system diagram |
| `Reflective_Synthesis_Paper.md` | Editable paper draft |
| `Reflective_Synthesis_Paper.pdf` | PDF submission version |
| `presentation_outline.md` | 15-minute mentor presentation and defense outline |
| `outputs/` | Generated metrics, recommendations, and dashboard image |
| `requirements.txt` | Environment snapshot generated with `pip freeze` |

---

## Setup Instructions
Use Python **3.9+**.

### 1. Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the integrated artifact
```bash
python src/healthcare_triage_ai.py
```

### 4. Optional: open the notebook
Open `notebooks/integrated_industry_synthesis.ipynb` in VS Code or Jupyter and run the cells in order.

---

## Generated Outputs
After running the script or notebook, the following files are created in `outputs/`:
- `risk_dashboard.png`
- `patient_recommendations.csv`
- `evaluation_metrics.json`

### Verified evaluation snapshot
The current prototype run produced:
- **Records:** 240
- **Positive rate:** 0.367
- **Test accuracy:** 0.85
- **Test ROC AUC:** 0.888
- **High or critical share:** 0.246

---

## Submission-Ready Artifacts
This workspace includes the required capstone deliverables:
- ✅ integrated industry artifact
- ✅ reflective synthesis paper draft and PDF
- ✅ notebook and supporting documentation
- ✅ architecture description
- ✅ presentation outline
- ✅ `requirements.txt` for reproducibility

---

## Responsible AI Notes
- The dataset is **synthetic** and privacy-safe.
- The workflow is designed for **decision support**, not diagnosis.
- High-risk cases are routed to **human review**.
- Any real deployment would require fairness testing, governance review, secure data handling, and ongoing monitoring.

---

## Final Personalization Step
Before submission, update the synthesis paper so it references your **actual prior project titles** and explains exactly how each one informed this integrated design.