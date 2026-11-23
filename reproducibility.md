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

## 7. Human validation workflow

The notebook exports `human_validation_template.csv` in the results folder. It contains up to 60 sampled policies and columns for official-source checking, two reviewer remedy scores, adjudicated remedy/inclusion scores, and notes/evidence.

Recommended workflow:

1. Two reviewers independently inspect the official source for each sampled policy.
2. Reviewers code remedy strength (0–3) and any-stakeholder inclusion (0/1) without consulting each other's decisions.
3. Preserve a source reference and brief evidence note for each decision.
4. Calculate human-human agreement for the ordinal remedy score (weighted kappa; report the number of double-coded cases and uncertainty interval if calculated).
5. Resolve disagreements through documented discussion/adjudication.
6. If using the notebook's optional override, save `human_adjudication.csv` in the project root or results folder with unique columns `slug`, `adjudicated_remedy` (0–3), and `adjudicated_inclusion` (0/1).
7. Keep the completed worksheet as `human_validation_completed.csv` if you want the notebook's human-human kappa check to detect it.

The notebook's built-in completed-file check calculates human-human weighted kappa for the remedy columns when available. It does not automatically provide a full human-versus-automated validation report for every coded variable. Add and test that analysis before claiming full validation. Do not report human validation as completed until the review files are genuinely completed and analysed.

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
