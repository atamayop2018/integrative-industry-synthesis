# Industry-specific AI solution.
## AI Capstone
### Alexis Tamayo
### Udacity - Woolf University.
### Master's degree in AI.

## Project Overview
This project presents an integrated AI system for healthcare triage, designed to support outpatient diabetes follow-up prioritization. The system addresses a real-world problem: clinics often lack the time and resources to proactively identify patients whose condition may be worsening between visits.

The solution combines multiple AI domains into a cohesive workflow, including:
- statistical data analysis on a real public clinical dataset
- machine learning prediction (logistic regression and random forest)
- generative AI explanations driven by a large language model
- agentic decision routing in which an LLM selects a follow-up tool and produces a justification

Rather than focusing on a single model, the project emphasizes system-level design, ethical reasoning, and professional applicability, aligning with industry expectations for AI deployment.

This document is the **revised** version of the original module summary. The first submission was flagged for two issues: a circular synthetic-on-synthetic data evaluation, and "generative" and "agentic" layers that were actually deterministic string templates and `if/else` thresholds. Both issues are addressed in the revised system described below.

## Dataset Description
The dataset used in this project is the **UCI Diabetes 130-US Hospitals (1999-2008) dataset**, a publicly available collection of approximately 99,340 inpatient encounters across 130 U.S. hospitals (Strack et al., 2014). It is fetched at runtime from the UCI ML Repository via the `ucimlrepo` package and cached locally at `data/diabetic_data.csv`.

Key features used by the pipeline include:
- demographics: age band (encoded as a numeric midpoint), gender, race
- utilization history: time in hospital, prior outpatient/emergency/inpatient visits
- treatment intensity: number of medications, number of lab procedures, number of diagnoses
- diabetes-specific signals: A1C result, max glucose serum, insulin status, whether diabetes medication was changed during the encounter
- encounter context: admission type, discharge disposition, admission source

The binary target `needs_intervention` is defined as `1` when the patient was readmitted in fewer than 30 days (`readmitted == "<30"`), and `0` otherwise. Early 30-day readmission is a clinically meaningful proxy for "this patient should have received earlier or more intensive follow-up", which is the operational decision the prototype supports.

The dataset replaces the original synthetic data, which had been generated from a hand-picked logistic equation. That replacement removes the circular evaluation problem identified in review.

## Data Preparation & Exploration
Data preparation involved:
- dropping out-of-scope discharges (deceased, hospice, transferred to another facility) so the cohort matches the outpatient follow-up problem
- dropping rows with unknown gender
- mapping age bands such as `[60-70)` to numeric midpoints
- treating sentinel `"?"`, `"None"`, and `"Unknown/Invalid"` values as missing and explicitly imputing them
- casting admission and discharge IDs to strings so the model treats them as categorical rather than ordinal numeric features

Exploratory analysis included:
- the overall positive rate (~11.4% of encounters end in early readmission), confirming class imbalance and motivating class-weighted training
- distributional checks on numeric features such as `time_in_hospital`, `num_medications`, and `number_inpatient`
- frequency review of categorical features such as A1C result and medication change

This step ensured that the model inputs were interpretable, clinically meaningful, and aligned with real-world signals rather than relying on arbitrary or self-generated features.

## Model Design
The system trains and compares two supervised classifiers and selects the better-performing one for downstream scoring:
- **Logistic regression** with L1/L2 regularization, a `liblinear` solver, and `class_weight="balanced"`. Contributes interpretability through signed coefficients on the encoded features.
- **Random forest** (300 trees, max depth 14, min samples per leaf 20) with `class_weight="balanced_subsample"`. Contributes the ability to model non-linear interactions among utilization, treatment-intensity, and diabetes-specific features.

Both models share the same scikit-learn `Pipeline` and `ColumnTransformer` preprocessor: `SimpleImputer` plus `StandardScaler` for numeric features and `SimpleImputer` plus `OneHotEncoder(handle_unknown="ignore", min_frequency=50)` for categorical features. Selection is automatic: the model with the higher held-out ROC-AUC is used to score every patient.

The model is embedded within a four-layer architecture:
1. Data analysis and preprocessing on the UCI dataset
2. Risk prediction (LR vs. RF, selected by held-out ROC-AUC)
3. Generative explanation via an LLM
4. Agentic routing via the same LLM acting as a tool-using agent

This design transforms predictions into actionable decisions, not just scores.

## Training Process
The training process followed standard machine learning practices on the real dataset:
- stratified 80/20 train/test split keyed on the `needs_intervention` target
- class weighting to compensate for the ~11% positive rate
- side-by-side training of logistic regression and random forest on the same split
- evaluation on the held-out test set, with both models reporting accuracy, precision, recall, F1, ROC-AUC, and a confusion matrix
- automatic selection of the model with the higher test ROC-AUC

Because the dataset is now real and was not produced by the same functional form as the model, the evaluation measures genuine generalization rather than recovering a known answer key.

## Model Evaluation
The current run on the UCI dataset produced the following held-out test metrics:

- **Records:** 99,340 total (79,472 train / 19,868 test)
- **Positive rate (overall):** 0.114
- **Logistic regression:** accuracy 0.677, precision 0.182, recall 0.528, F1 0.271, ROC-AUC **0.665**
- **Random forest:** accuracy 0.690, precision 0.191, recall 0.533, F1 0.282, ROC-AUC **0.671** (selected)

These values are intentionally lower than the previous synthetic submission's headline 0.888 ROC-AUC. Real, noisy, independently produced clinical data is harder to predict than data generated from the same logistic equation a logistic regression is fitting. The mid-0.6 ROC-AUC on this dataset is consistent with prior published work on it and represents an honest measurement of generalization on a hard real-world problem.

Evaluation also included:
- per-band positive rates (the top 5% Critical band carries roughly 4-5x the base positive rate, demonstrating meaningful discrimination)
- agent action distribution on the LLM shortlist (e.g., 5 physician_review, 4 nurse_outreach, 1 education_reminder for a 10-patient run), showing that the routing agent makes patient-specific decisions rather than collapsing to one constant action.

## Results & Interpretation
The system successfully:
- scored all ~99,000 patients with a real classifier evaluated on held-out data
- categorized patients into actionable percentile-based tiers (Low, Moderate, High, Critical)
- generated readable LLM-driven case summaries grounded in the available indicators
- produced LLM-driven routing decisions that select one tool from a fixed set, with a one-sentence justification and an estimated follow-up window
- routed cases to appropriate follow-up actions through human review for the high-risk and ambiguous cases

Key insight: the value of the system is not just prediction accuracy, but its ability to integrate prediction with explanation and action, all built from real techniques rather than templates. End-to-end workflow design with genuine multi-domain AI is what differentiates this from a standalone classifier.

## Non-Technical Explanation
This system helps clinics decide which patients need attention first. Instead of reviewing every patient manually, the system:
1. Reads important health signals from the patient's record
2. Calculates a risk level using a predictive model
3. Asks an AI assistant to write a short summary of why the patient is at risk
4. Asks the same AI assistant to choose the next step from a small fixed list (reminder, nurse phone call, or doctor review) and explain its choice in one sentence
5. Sends those outputs to a human clinician for the final call

For example:
- Lower-risk, stable patient → automated educational reminder
- Moderate-risk patient with recent missed care → nurse phone call within five business days
- High-risk patient with multiple acute utilization signals → physician review within 72 hours

The system does not replace doctors. It helps them focus on the patients who need help the most, and it always defers to human judgment for the high-stakes decisions.

## Experimental Design Justification
The experimental design was guided by:
- real-world healthcare constraints (finite triage capacity, the cost of false negatives)
- the need for interpretability and defensibility
- ethical considerations around bias, privacy, and automation risk
- the rubric requirement that each claimed AI domain be implemented with a real technique appropriate to that domain

Key decisions:
- **Real public dataset (UCI Diabetes 130-US Hospitals)** instead of synthetic data → eliminates the circular evaluation problem and produces honest generalization metrics
- **Two model classes evaluated head-to-head** → demonstrates real model selection rather than committing to a single architecture in advance
- **Percentile-based risk bands** → adapts to whatever calibration the chosen model produces and matches finite triage capacity
- **LLM-driven generative layer with explicit guardrails** → real generative AI, with prompts that forbid diagnosis, prescription, or invented indicators
- **LLM-driven agentic layer with structured output** → real agentic reasoning, with the model selecting one tool from a fixed allow-list and returning JSON validated against that allow-list
- **Conservative escalation on parse failure** → safer default behavior (`physician_review`) when the agent's output cannot be validated
- **Bounded LLM cost** → the classifier scores everyone; the LLM layers run only on a configurable triage shortlist (default 20 patients)
- **Pluggable LLM backend** → OpenAI (default), Anthropic, or local `llama-cpp-python`, supporting deployments where data cannot be sent to a third-party API

This design prioritizes:
- reliability over complexity
- transparency over optimization
- safety over autonomy

## Workflow Completeness
The system demonstrates a complete AI workflow, including:
- data ingestion from a public source with local caching
- preprocessing and feature engineering on real clinical data
- exploratory analysis and target definition
- supervised training of two models with stratified evaluation
- automated model selection based on held-out ROC-AUC
- LLM-driven generative case summaries and outreach messages
- LLM-driven agentic routing with validated tool selection
- output generation: dashboard image, full patient scores, shortlist with LLM artifacts, evaluation metrics JSON
- documentation: README, architecture document, reflective paper (markdown + PDF), presentation outline, notebook walkthrough
- reproducibility: `requirements.txt`, `.env.example`, git-ignored `data/` cache and `.env` for secrets

It also includes:
- architecture design with a Mermaid diagram
- both a notebook and an executable script
- structured outputs and metrics
- a presentation outline and a synthesis paper

This ensures the project is end-to-end and production-aware, even as a prototype.

## Bias and Risk Awareness
Several risks were identified:
- **Bias risk:** the UCI dataset reflects U.S. inpatient practice between 1999 and 2008. Patterns in that data may not generalize, and may encode historical disparities in access and treatment (Obermeyer et al., 2019).
- **False negatives:** missing a high-risk patient who needs follow-up is more clinically damaging than over-flagging a stable patient. ROC-AUC in the mid-0.6 range means this risk is real and should be monitored.
- **Automation bias:** staff may trust an AI output simply because it appears technical. The case-summary layer is intended to make the drivers of each decision legible and challengeable.
- **Privacy and PHI exposure:** sending patient context to a third-party LLM provider raises governance questions. The system supports a fully local `llama-cpp-python` backend so that, in a real deployment, no PHI needs to leave the network.
- **LLM hallucination:** even with constrained prompts, generative output is not guaranteed to be free of error. Prompts forbid diagnosis and invented indicators, the agent's action space is closed, and JSON output is validated.

Mitigation strategies:
- transparent feature set with reported feature importance for the linear model
- human review for high-risk cases and any case the agent cannot route confidently
- LLM prompts with explicit clinical guardrails
- validated allow-list for the agent's action space, with conservative escalation on parse failure
- explicit framing as decision-support, not decision-making

These align with responsible AI frameworks such as:
- NIST AI RMF (NIST, 2023)
- WHO AI ethics guidelines (WHO, 2021)

## Future Improvements & Integration
Future enhancements include:
- formal subgroup performance analysis (race, gender, age band, payer code) and calibration plots in addition to ROC-AUC
- gradient-boosted tree models and probability calibration
- a clinician-rated evaluation harness for the LLM-generated case summaries and routing justifications
- automated checks on LLM output (for example, that the case summary never names a diagnosis the patient does not have in the record)
- integration with real electronic health records under proper governance
- longitudinal patient tracking and trend features
- monitoring dashboards, feedback loops, and alert tuning based on clinic capacity
- a formal model card documenting intended use, training data lineage, performance, and known risks

Additionally, the system could evolve into a production-ready clinical decision support tool with proper governance and validation, including a mandatory local-LLM mode for any environment where data cannot be sent to a third-party API.

## Conclusion
This project demonstrates how an AI system can:
- integrate multiple AI domains into a unified workflow, each implemented with a real technique appropriate to that domain
- support real-world decision-making on a real public clinical dataset
- balance performance with ethical responsibility through guardrails, validated tool spaces, and human oversight

More importantly, the revised version shows that modern AI work is about system design grounded in honest evaluation, not just models trained on data they generated themselves. The project reflects professional readiness by combining:
- technical skills (real ML, real LLM-driven generation, real LLM-driven routing)
- ethical awareness (bias, privacy, automation risk, PHI handling)
- practical application (real dataset, real metrics, conservative human-in-the-loop design)
- the ability to translate precise feedback into a precise correction

## Reproducibility
To reproduce the project:

Clone the repository: https://github.com/atamayop2018/integrative-industry-synthesis

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env and set OPENAI_API_KEY (or ANTHROPIC_API_KEY, or LOCAL_LLM_GGUF)
python src/healthcare_triage_ai.py
```

Additional options:
- Run the Jupyter notebook version at `notebooks/integrated_industry_synthesis.ipynb`
- Review outputs in the `outputs/` folder (`risk_dashboard.png`, `patient_scores.csv`, `patient_recommendations.csv`, `evaluation_metrics.json`)
- Adjust the LLM shortlist size with `LLM_SHORTLIST_SIZE` in `.env`

The `requirements.txt` file (regenerated via `pip freeze`) ensures consistent environment setup. The UCI dataset is downloaded automatically on first run and then cached locally.

## References
- National Institute of Standards and Technology (2023). *AI Risk Management Framework (AI RMF 1.0).*
- Obermeyer, Z., Powers, B., Vogeli, C., & Mullainathan, S. (2019). Dissecting racial bias in an algorithm used to manage the health of populations. *Science*, 366(6464), 447-453.
- Strack, B., DeShazo, J. P., Gennings, C., Olmo, J. L., Ventura, S., Cios, K. J., & Clore, J. N. (2014). Impact of HbA1c measurement on hospital readmission rates: Analysis of 70,000 clinical database patient records. *BioMed Research International*, 2014, Article 781670.
- Topol, E. (2019). High-performance medicine: The convergence of human and artificial intelligence. *Nature Medicine*, 25, 44-56.
- UCI Machine Learning Repository. *Diabetes 130-US Hospitals for Years 1999-2008 Data Set.* https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
- World Health Organization (2021). *Ethics and Governance of AI for Health.*
- Udacity AI Nanodegree Course Materials.
