# AIpolicy

# Stakeholder Power and Governance Asymmetries in AI Policy

From recognition to enforceable accountability: a reproducible analysis of AI policy records from the AI Policy Tracker.

**Team MK** (Interdisciplinary Elective: *AI, Power & Responsibility: Governing Intelligence*)
Maryam Moghaddas Bayat, Kelvin Obour Coleman
Target journal: *Technology in Society* (Elsevier)

## Overview

AI policies increasingly name the people affected by AI systems (consumers, workers, children, vulnerable groups, civil society). Naming a group is not the same as giving it an enforceable right or making someone answerable when harm occurs. This repository contains the analysis notebook that measures that gap across policy instruments and jurisdictions.

The notebook follows a four-stage framework: **Recognition** (who is named) → **Protection** (what right or safeguard is stated) → **Accountability** (who has a duty and who checks it) → **Implementation** (what measure supports delivery). Not every instrument is expected to contain all four stages.

## Research questions

| RQ | Question |
|---|---|
| RQ1 | How are different stakeholder groups represented across AI governance policies and jurisdictions? |
| RQ2 | To what extent do AI policies translate stakeholder recognition into clearly defined rights, protections and obligations? |
| RQ3 | What accountability mechanisms (oversight, enforcement and remedies) are established to support compliance? |
| RQ4 | How do stakeholder protection and enforceable accountability differ across jurisdictions and implementation stages? |

## Data

- **Source:** [AI Policy Tracker Open Data Portal](https://aipolicytracker.org/open-data), licensed **CC BY 4.0** (attribution required).
- **Snapshot accessed:** 2 November 2026.
- **Files used (four CSV tables):** `policies`, `obligations`, `jurisdictions`, `implementation`.
  The notebook finds them by file name under `/kaggle/input`, preferring the folder named in `DATA_FOLDER` (cell 2, currently `AIpolicy_data`).
- The raw CSV files are not modified; the notebook keeps unchanged copies for audit.

> Record counts of the working snapshot must be confirmed from the notebook output after each run (cell 4 prints them).

## How to run

1. Create a Kaggle notebook and upload `aipolicy-8nov_corrected_fixed.ipynb` (or open it in any Python 3 environment).
2. Attach the dataset containing the four CSV files. Internet access is not needed.
3. Check `DATA_FOLDER` in cell 2 matches your dataset folder name.
4. Run all cells from top to bottom. The random seed is fixed (42).
5. Download `aipolicy_results.zip` from the notebook output.

**Important:** after any change to the notebook or the data, re-run it from top to bottom. Results from earlier versions are superseded and should not be reused.

## Outputs

Every table is displayed and saved as **CSV**; every figure is displayed and saved as vector **PDF**. The last cell zips everything into one archive (`aipolicy_results.zip`) with a run-metadata file and a manifest (35 tables, `t01`–`t35`, and 18 figures, `fig01`–`fig18`; some conditional tables may be absent).

| Notebook section | Content | RQ |
|---|---|---|
| 1–2 | Load, quality checks, merge, derived variables, corpus description | all |
| 3–4 | Codebook, two rule-based coders, inter-method agreement, reconciled coding | all |
| 5 | Representation of stakeholder groups | RQ1 |
| 6 | Recognition-to-protection funnel, binding vs non-binding, obligation profile | RQ2 |
| 7 | Remedy levels, accountability features and index, exploratory regression | RQ3 |
| 8 | Actors in obligations, jurisdiction indices, concentration, implementation records | RQ4 |
| 9 | Sensitivity to coder choice | all |
| 10–11 | Evidence-at-a-glance summary, results archive | all |

## Method in brief

- **Unit of analysis:** the policy instrument; obligations are linked records.
- **Coding:** two rule-based coders (keyword / regular-expression programs; no machine-learning or generative-AI classifier).
  - *Coder A*: conservative, whole-word patterns.
  - *Coder B*: broader stem patterns plus modal-verb logic (aspirational wording caps the remedy at symbolic level).
- **Scores:** a stakeholder group scores 0 (not mentioned), 1 (passive recognition), 2 (procedural protection) or 3 (explicit right or remedy) only when the group and the safeguard or right appear in the **same sentence**. Public enforcement (fines, sanctions) is a separate indicator.
- **Working coding:** the lower of Coder A and Coder B for each score (conservative, provisional).
- **Accountability index (0–5):** right/protection, duty-holder, oversight and enforcement, complaint and remedy, implementation support.
- **Statistics:** Wilson intervals, Fisher exact tests, odds ratios, Mann–Whitney and Kruskal–Wallis tests, Cohen's kappa with bootstrap intervals, logistic regression with HC1 robust standard errors, ordered logit, Herfindahl index.

## Human validation and adjudication

Human validation is **not complete**. The notebook only prepares it:

- `human_validation_template.csv`: reproducible stratified sample (up to 60 policies, seed 42). Reviewers inspect the **official policy sources** and code each case independently.
- `human_validation_completed.csv` (optional input): the completed worksheet; if present, human–human weighted kappa is computed.
- `human_adjudication.csv` (optional input): columns `slug`, `adjudicated_remedy` (0–3), `adjudicated_inclusion` (0/1). If present, it overrides document-level remedy and inclusion codes and rebuilds dependent variables. Group-level scores stay rule-based.

Do not describe human validation as completed until the worksheet has been completed and analysed.

## Limitations and interpretation

- Coding uses **tracker-derived text** (title, summary, applicability statement, linked obligation text), not necessarily the full official legal instruments. Findings describe what is identifiable in that text; they do not show that a right or remedy is absent from the law itself.
- Cohen's kappa between Coder A and Coder B measures **inter-method agreement**, not human inter-coder reliability or construct validity.
- **Document-level remedies** (a remedy anywhere in a document) are reported separately from **stakeholder-linked protections** (a named group linked to a safeguard or right).
- Regression models are **exploratory associations**. Binding status and statute format overlap strongly and some format-by-legal-status cells are sparse, so independent or causal effects are not claimed. Tests are not corrected for multiple comparisons.
- The implementation table is small and weakly linked to policies; it is analysed **descriptively only**.
- Actor matches show who is *named* in an obligation, not who is legally responsible.

## Citation and attribution

Please cite the data source: AI Policy Tracker. (2026). *AI Policy Tracker Open Data Portal*. https://aipolicytracker.org/open-data (CC BY 4.0).

## AI assistance

It were used to assist with code development, debugging, and refinement of the computational workflow. The researchers reviewed, executed, tested, and validated the code and take responsibility for its functionality, analytical procedures, and outputs. The research design, research questions, dataset selection, methodological decisions, interpretation of results, and conclusions remain the responsibility of the human researchers.
