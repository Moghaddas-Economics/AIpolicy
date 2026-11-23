# Reproducibility and execution guide

## 1. Scope

This repository contains a Python/Jupyter workflow for the four AIPolicyTracker CSV tables. The notebook is designed for execution in Kaggle or a local Python environment. It uses deterministic regular-expression coding procedures and conventional statistical summaries/models; it does not use a machine-learning or generative-AI classifier.

## 2. Environment

Install dependencies from the repository root:

```bash
python -m venv .venv
# Windows PowerShell / Command Prompt:
.venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Alternatively, use the supplied Conda `environment.yml`.

The principal libraries used by the notebook are Python, NumPy, pandas, Matplotlib, SciPy, scikit-learn (Cohen's kappa), statsmodels (regression), IPython/Jupyter and the Python standard library. Exact versions from a run are written to `run_metadata.csv` when the final packaging cell is executed.

## 3. Data acquisition and placement

Download the four tables (`policies`, `obligations`, `jurisdictions`, `implementation`) from the [AIPolicyTracker Open Data Portal](https://aipolicytracker.org/open-data). The notebook searches recursively for CSV filenames containing each table name.

- **Local:** place the four CSVs under `data/raw/` (subfolders are also searched).
- **Kaggle:** attach the dataset containing the four CSVs as an input dataset. The setup cell has `DATA_FOLDER = "AIpolicy_data"` as a preferred-folder hint. If the attached Kaggle folder has another name, update that constant. The locator chooses the best matching path; ensure there is only one intended candidate per table name.

The repository does not include the raw dataset. Verify the current licence, attribution and redistribution conditions before sharing source CSVs. The notebook currently records the source URL, `CC BY 4.0`, and access date `2026-11-02`; confirm that these values accurately describe the actual downloaded snapshot before publication.

## 4. Execute the analysis

Open `notebooks/aipolicy_analysis.ipynb` in JupyterLab or Kaggle and run **all cells from top to bottom**. Do not execute only selected downstream cells because later sections depend on objects created earlier.

The workflow is:

1. Import libraries, set parameters and initialise output folders.
2. Locate and load the four CSV tables; retain raw copies and produce data-quality checks.
3. Join policy, obligation and jurisdiction records; derive legal status, instrument format, jurisdictional tier and analysis text.
4. Apply Coder A and Coder B to policy text.
5. Calculate inter-method agreement, bootstrap confidence intervals and raw agreement; export a disagreement audit list.
6. Construct the conservative minimum-of-A-and-B working coding and export a stratified human-validation template.
7. Analyse stakeholder representation (RQ1), recognition/protection (RQ2), accountability and exploratory models (RQ3), actors/jurisdictions/implementation (RQ4).
8. Conduct coder-sensitivity analysis under A, B and reconciled coding.
9. Generate the evidence-at-a-glance summary, metadata, manifest and ZIP archive.


## Reusable source-code module (`src/`)

The reusable implementation is in `src/aipolicy_analysis.py`. It exposes the following functions:

- `load_tracker_tables(data_dir)`: recursively finds and loads the four CSV tables by filename token.
- `build_policy_corpus(policies, obligations, jurisdictions)`: joins obligation and jurisdiction information and assembles the tracker-derived text field used by the coding rules.
- `code_policy(text, method)` and `apply_coder(frame, method)`: apply Coder A or Coder B.
- `derive_coding_indicators(frame, prefix)`: derive inclusion, protection and accountability variables from one coding set.
- `reconcile_coders(frame)`: applies the pre-specified element-wise minimum of A and B and derives provisional indicators. This is not human adjudication.
- `bootstrap_kappa(...)` and `inter_method_agreement(...)`: estimate Cohen's kappa, bootstrap confidence intervals and raw agreement.
- `stratified_validation_sample(...)` and `create_human_validation_template(...)`: prepare the reproducible human-review sample and blank worksheet.
- `validate_adjudication_file(...)`: checks the structure and allowed values of the notebook-compatible override file.
- `src/single_reviewer_validation.py`: creates a blinded single-reviewer worksheet, validates scores, compares human scores with A/B/reconciled outputs, exports confusion matrices and creates reviewer-assessed override files.

Example from the repository root:

```python
from src.aipolicy_analysis import (
    load_tracker_tables, build_policy_corpus, apply_coder,
    reconcile_coders, create_human_validation_template,
)

tables = load_tracker_tables("data/raw")
pol = build_policy_corpus(tables["policies"], tables["obligations"], tables["jurisdictions"])
pol = pol.join(apply_coder(pol, "A"))
pol = pol.join(apply_coder(pol, "B"))
pol = reconcile_coders(pol)
create_human_validation_template(pol, n=60).to_csv(
    "human_validation_template.csv", index=False
)
```

The module supports reuse of core preparation and coding steps. The notebook remains the complete analysis workflow for descriptive results, figures, regression models, jurisdiction analysis, robustness checks and result packaging.

## 5. Parameters recorded in the notebook

| Parameter | Current configured value | Function |
|---|---:|---|
| `SEED` | 42 | Reproducible random sampling and plotting jitter. |
| `N_BOOT` | 1,000 | Bootstrap resamples for kappa confidence intervals. |
| `MIN_EVENTS` | 10 | Minimum events and non-events required before logistic regression is estimated. |
| `MIN_DOCS` | 3 | Minimum policy documents for inclusion in jurisdiction-level comparison tables. |
| `VALIDATION_N` | `min(60, number of policies)` | Maximum size of stratified human-validation sample. |

The validation sample is stratified by the reconciled document-level remedy category where possible, with a reproducible random sample used to fill the target. This does not mean the sample has been human-reviewed.

## 6. Statistical and interpretive safeguards

- Cohen's kappa between A and B is **inter-method agreement**. It is not human inter-rater reliability or proof of construct validity.
- Quadratic weights are used for ordinal scores; binary indicators use unweighted kappa. Raw agreement is also reported because kappa can be affected by prevalence.
- The reconciled coding is the minimum of A and B, a conservative working rule rather than human ground truth.
- Logistic regression outcomes are exploratory. Models are estimated only when event and non-event counts meet the configured minimum and model diagnostics permit estimation. Robust HC1 standard errors and checks for convergence/separation are used; categories may be collapsed under the implemented rules. Do not interpret associations causally.
- The implementation table is analysed descriptively because its number of records is small.
- Actor categories overlap; do not sum them as mutually exclusive groups.
- Tracker-derived text is not necessarily the full legal instrument. Absence of a match is not proof of legal absence.

## 7. Single-reviewer human validation workflow

The Kaggle notebook exports a legacy two-reviewer worksheet. If only one human reviewer is available, use the dedicated single-reviewer workflow instead; do not fill the second-reviewer columns with duplicated or inferred values. The full protocol is in `docs/single_reviewer_validation_protocol.md`.

After the notebook has created the policy-level `pol` data frame and reconciled `remedy` scores, create a blinded sample in Kaggle or locally from the repository root:

```python
from src.single_reviewer_validation import create_single_reviewer_template

review = create_single_reviewer_template(pol, n=60, seed=42)
review.to_csv("human_validation_single_reviewer_template.csv", index=False)
```

The worksheet omits A/B scores while the reviewer codes. Inspect the official policy document, record source details and date checked, code remedy (0–3), any-stakeholder inclusion (0/1), and (where feasible) the five group-level scores (0–3). Record legal status, instrument type and current status as checked against the official source. Never code an inaccessible or missing source as zero. Record the article/section/page and supporting quotation or precise paraphrase.

Once completed, compare human scores with A, B and the provisional minimum(A,B):

```python
import pandas as pd
from src.single_reviewer_validation import compare_human_with_automated

review = pd.read_csv("human_validation_single_reviewer_completed.csv")
comparison, confusion_matrices = compare_human_with_automated(review, pol)
comparison.to_csv("single_reviewer_comparison.csv", index=False)
for name, matrix in confusion_matrices.items():
    matrix.to_csv(f"confusion_{name}.csv")
```

The comparison reports human–algorithm concordance, not human-human inter-rater reliability. Do not calculate human-human kappa or claim double coding. For ordinal remedy scores, report quadratic weighted kappa, exact agreement, mean absolute category difference and within-one-category agreement; for binary inclusion, report unweighted kappa and exact agreement. Include confusion matrices and paired sample sizes. Kappa may be unstable or undefined in a small/prevalence-skewed sample, so always show raw agreement and the confusion matrix.

The sample is stratified by the provisional remedy category. Unweighted sample agreement describes only the reviewed sample. If estimating corpus-wide error rates, calculate and disclose design weights. The optional `create_review_override_file()` creates the notebook-compatible `slug`, `adjudicated_remedy`, `adjudicated_inclusion` schema. With one reviewer, these are reviewer-assessed overrides, not multi-reviewer adjudications. The notebook hook changes document-level remedy and inclusion only; it does not update group-level scores or all legal classifications. Preserve original values and report before/after sensitivity where appropriate.

## 8. Official-source verification

For the validation sample and consequential classifications, check the original official legal/policy document rather than relying solely on the tracker summary. Prioritise:

- binding/non-binding status and whether a proposal/bill was enacted;
- instrument type and current status;
- stakeholder-linked protections and explicit individual rights;
- the distinction between public sanctions and individual remedies;
- official source URL, publication date and relevant article/section.

Record the official document title, issuing body, URL, relevant article/paragraph, date checked and a short evidence quotation or paraphrase. Do not infer that a right is legally enforceable solely from a keyword match.

## 9. Outputs

The notebook writes output files to `aipolicy_results/` locally or `/kaggle/working/aipolicy_results/` on Kaggle. Tables are saved as CSV and figures as vector PDF. The final packaging cell writes:

- `run_metadata.csv` — source and run parameters, record counts, Python/library versions and timestamp;
- `manifest.csv` — saved table/figure names, file type and file size;
- `aipolicy_results.zip` — archive of the results directory.

The notebook documents up to 35 CSV tables and 18 PDF figures, but some outputs are conditional on available columns and estimable models. The evidence-at-a-glance cell and final packaging cell must both be executed for the final summary figure/table and archive to be generated. In the uploaded notebook version, these two final code cells were unexecuted in the saved file; run them after the preceding cells complete successfully.

## 10. Tests and what they do not establish

Run repository checks from the project root:

```bash
pytest
```

The tests check required repository files, notebook validity and Python syntax. They do **not** execute the complete empirical pipeline or validate substantive classifications. Full reproducibility requires access to the four source CSV files and a successful top-to-bottom notebook run.

## 11. Recommended archival practice

For a publication-ready replication package, preserve the final notebook, source snapshot or lawful source retrieval instructions, run metadata, manifest, all CSV/PDF outputs, human-validation records (if completed), and a Git commit identifier. Record the exact data access date and verify the portal's current licence/citation terms. Do not include source data if redistribution is not permitted.
