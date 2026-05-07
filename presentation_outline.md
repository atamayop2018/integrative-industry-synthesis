# Mentor Presentation and Defense Outline

## 15-Minute Structure

### Slide 1 — Title and Industry Problem (1 minute)
- Project title: **AI-Assisted Chronic Care Triage for Diabetes Follow-Up**
- Industry: healthcare
- Operational target: 30-day readmission risk on the **UCI Diabetes 130-US Hospitals (1999-2008)** dataset (~99,000 encounters across 130 U.S. hospitals).
- Problem: clinics have limited time and staff while many patients show indicators that warrant earlier follow-up.

### Slide 2 — Why This Problem Matters (1 minute)
- Delayed follow-up correlates with avoidable readmission, complications, and higher cost of care.
- AI is appropriate because the workflow involves structured patient signals, prioritization decisions, communication support, and operational routing — but the stakes also require human oversight.

### Slide 3 — Revision Note (1 minute)
- The first version of this submission was flagged for two specific issues:
  1. Synthetic data generated from a logistic equation, then evaluated with a logistic regression — circular evaluation.
  2. "Generative" and "agentic" layers implemented as string templates and `if/else` thresholds — not real AI.
- Both issues have been fixed in this revision (described in the next slides).

### Slide 4 — Integration of Prior Projects (2 minutes)
- **Data analysis** informed the preprocessing pipeline on the UCI dataset (target definition, missing-value handling, age-band encoding).
- **Machine learning** informed the dual-model evaluation: logistic regression vs. random forest, stratified split, class weighting, ROC-AUC + precision/recall/F1.
- **Generative AI** informed the LLM-based case-summary and outreach-message layers, with guardrails on diagnosis and invented indicators.
- **Agentic workflows** informed the LLM-as-router design that selects from a fixed tool set and returns JSON with action + justification + follow-up window.

### Slide 5 — System Architecture (2 minutes)
1. UCI Diabetes 130-US Hospitals dataset → preprocessing
2. Two classifiers trained side by side; better held-out ROC-AUC is selected
3. Risk probability → percentile-based risk band (Low / Moderate / High / Critical)
4. Triage shortlist → LLM generative layer → case_summary + patient_message
5. Triage shortlist → LLM agentic layer → tool selection + justification + follow-up window
6. Human-in-the-loop review → documented action and monitoring

### Slide 6 — Technical Design Decisions (2 minutes)
- Real dataset replaces the synthetic one, removing circular evaluation.
- Two model classes evaluated head-to-head; selection on held-out ROC-AUC, not on training fit.
- Percentile-based risk bands match finite triage capacity in a real clinic.
- LLM backend is pluggable: OpenAI (default), Anthropic, or local `llama-cpp-python`. The script raises if none is configured rather than falling back to fake "AI" output.
- LLM layers run only on a configurable shortlist (default 20), keeping API cost bounded.
- Agent output is validated against an allow-list of actions; invalid output retries once, then conservatively escalates to physician review.

### Slide 7 — Evaluation Results (2 minutes)
- Headline metrics from the prototype run on the real dataset:
  - Logistic regression ROC-AUC ≈ 0.665
  - Random forest ROC-AUC ≈ 0.671 (selected)
  - Critical band (top 5% by predicted probability) carries roughly 4–5× the base positive rate.
- These numbers are intentionally lower than the prior 0.888 from the synthetic version. Real noisy clinical data is harder, and a realistic ROC-AUC is the point.
- Show one example: highest-risk patient → LLM-generated case summary → agent-selected action and justification.

### Slide 8 — Ethical and Governance Considerations (2 minutes)
- Risks: bias in historical data, false negatives, alert fatigue, automation bias, privacy of any data sent to a third-party LLM.
- Safeguards: human review for high-risk cases, LLM prompts that forbid diagnosis and invented indicators, validated allow-list of agent actions, conservative escalation on parse failure, support for a fully local LLM backend so no PHI needs to leave the network in production.
- Alignment with NIST AI RMF and WHO ethics guidance for health AI.

### Slide 9 — Limitations and Future Improvements (1 minute)
- The dataset is historical (1999-2008 U.S. inpatient); generalization is not assumed.
- ROC-AUC in the mid-0.6 range — the system supports prioritization, not decisions.
- Future work: subgroup fairness analysis (race, gender, age, payer), calibration plots, gradient-boosted trees, automated checks on LLM output, a clinician-rated evaluation harness for case summaries, and a model card.

### Slide 10 — Closing Defense Statement (1 minute)
- The revised project shows four genuinely different AI techniques working together on a real dataset: data analysis, two competing classifiers, an LLM-driven generative layer, and an LLM-driven routing agent.
- The revision process itself demonstrates the professional skill of translating precise feedback into precise correction.

---

## Likely Mentor Questions and Short Answers

### 1. Why healthcare?
Because it combines high social impact with clear operational constraints, making it a strong setting for responsible AI system design.

### 2. Why two model classes instead of one?
Logistic regression contributes interpretability (signed coefficients on encoded features) and random forest captures non-linear interactions. Picking on held-out ROC-AUC demonstrates real model selection rather than committing to a single architecture in advance.

### 3. Why is the ROC-AUC lower than in the previous submission?
Because the previous version trained a logistic regression on data generated from a logistic equation. That made the evaluation circular. Replacing it with a real public dataset (UCI Diabetes 130-US Hospitals) removed that artifact; the resulting mid-0.6 ROC-AUC is consistent with prior published work on this dataset.

### 4. What makes the agentic layer actually agentic instead of `if/else`?
The routing decision is produced by an LLM that reads the patient context, considers a fixed tool set, and returns structured JSON with the chosen action, a one-sentence justification grounded in the indicators, and a recommended follow-up window. Output is validated against the allow-list and retried on failure. The decision is not produced by hand-coded thresholds.

### 5. What is the biggest ethical concern?
Two: false negatives (missing a patient who needs follow-up) and PHI leaving the network when a third-party LLM is used. The first is mitigated by routing high-risk and ambiguous cases to humans. The second is mitigated by supporting Anthropic and a local `llama-cpp` backend in addition to OpenAI.

### 6. What would need to happen before deployment?
Subgroup fairness testing, clinical validation on a current outpatient population, governance and privacy review, an explicit policy on what context is sent to a third-party LLM, user training, alert tuning, and continuous monitoring with feedback loops.
