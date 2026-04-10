# Mentor Presentation and Defense Outline

## 15-Minute Structure

### Slide 1 — Title and Industry Problem (1 minute)
- Project title: **AI-Assisted Chronic Care Triage for Outpatient Diabetes Follow-Up**
- Industry: healthcare
- Problem: clinics have limited time and staff, while many patients need timely follow-up after risk signals begin to rise.

### Slide 2 — Why This Problem Matters (1 minute)
- Delayed follow-up can increase complications, avoidable ER use, and missed preventive care.
- AI is appropriate because the workflow involves structured patient signals, prioritization decisions, and communication support.

### Slide 3 — Integration of Prior Projects (2 minutes)
- Prior work in **data analysis** informed feature selection and risk pattern exploration.
- Prior work in **machine learning** informed the supervised prediction layer.
- Prior work in **generative AI** informed the explanation and outreach message layer.
- Prior work in **agentic workflows** informed the routing logic and human escalation path.

### Slide 4 — System Architecture (2 minutes)
- Walk through the full pipeline:
  1. patient data input
  2. feature engineering
  3. risk scoring
  4. summary generation
  5. routing to education, nurse, or physician review
- Emphasize that the system is integrated rather than a set of disconnected tools.

### Slide 5 — Technical Design Decisions (2 minutes)
- Chose an interpretable logistic model instead of a black-box architecture.
- Used synthetic data to demonstrate flow without privacy risk.
- Added human review for high-risk outputs to support safe deployment.

### Slide 6 — Evaluation Results (2 minutes)
- Share model metrics from the prototype run.
- Show example outputs: risk tiers, routed cases, and one generated summary.
- Discuss strengths: workflow clarity, explainability, and operational usefulness.

### Slide 7 — Ethical and Governance Considerations (2 minutes)
- Key risks: bias, false negatives, alert fatigue, over-reliance on automation, privacy.
- Safeguards: synthetic development data, governance review, human oversight, monitoring, transparent documentation.

### Slide 8 — Limitations and Future Improvements (1 minute)
- Synthetic data means the artifact is technically plausible, not clinically validated.
- Future work: real-world validation, subgroup fairness review, secure EHR integration, stronger monitoring.

### Slide 9 — Professional Relevance (1 minute)
- Demonstrates cross-domain integration, responsible AI reasoning, and stakeholder communication.
- Connects technical skills to a realistic healthcare operations problem.

### Slide 10 — Closing Defense Statement (1 minute)
- The project shows how to connect analytics, predictive modeling, generative outputs, and agentic orchestration into one defensible system.

---

## Likely Mentor Questions and Short Answers

### 1. Why healthcare?
Because it combines high social impact with clear operational constraints, making it a strong setting for responsible AI system design.

### 2. Why not use a more complex model?
Interpretability and defendability were more important than squeezing out marginal performance gains in a safety-sensitive context.

### 3. What makes this integrative instead of just multi-part?
Each component depends on the previous one: the data informs the model, the model informs the generated explanation, and both drive the routing decision.

### 4. What is the biggest ethical concern?
A false sense of objectivity: if the model encodes historical bias or misses underserved patients, it can reinforce inequitable care.

### 5. What would need to happen before deployment?
Clinical validation, subgroup fairness testing, privacy review, governance approval, user training, and continuous monitoring.
