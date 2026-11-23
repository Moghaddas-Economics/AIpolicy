# Stakeholder Power and Governance Asymmetries in AI Policy

**Research focus:** from stakeholder recognition to protection and enforceable accountability in AI-governance policy instruments.

This repository contains the analysis notebook and supporting materials for a structured, rule-based analysis of the AIPolicyTracker open-data tables. It is designed for reproducible execution in Kaggle or a local Python environment.

## Research questions

1. How are different stakeholder groups represented across AI-governance policies and jurisdictions?
2. To what extent do AI policies translate stakeholder recognition into clearly defined rights, protections and obligations?
3. What accountability mechanisms (oversight, enforcement and remedies) are established to support compliance with AI-policy obligations?
4. How do stakeholder protection and enforceable accountability differ across jurisdictions and policy-implementation stages?

## Conceptual sequence

**Recognition → Protection → Accountability → Implementation.**

The framework does not assume that every instrument contains all four stages. Missing information in tracker-derived text is not proof that a protection or mechanism is absent from the full legal instrument.

## Repository contents

```text
.
├── README.md
├── CITATION.cff
├── CHANGELOG.md
├── LICENSE
├── requirements.txt
├── environment.yml
├── .gitignore
├── notebooks/
│   └── aipolicy_analysis.ipynb
├── src/
│   └── aipolicy_analysis.py
├── data/
│   ├── raw/                 # place the four source CSVs here; not committed
│   └── processed/
├── outputs/
│   ├── figures/             # generated PDF figures
│   └── tables/              # generated CSV tables
├── docs/
│   ├── codebook.md
│   ├── data_dictionary.md
│   └── reproducibility.md
└── tests/
    └── test_project_structure.py
```

## Data required

Download the four CSV tables from the [AIPolicyTracker Open Data Portal](https://aipolicytracker.org/open-data):

- `policies` — policy instruments; primary unit of analysis.
- `obligations` — obligation records linked to policy instruments.
- `jurisdictions` — jurisdiction metadata and tier.
- `implementation` — implementation-stage records.

The notebook searches recursively for CSV filenames containing these terms. For local execution, put the four files under `data/raw/`. In Kaggle, attach the dataset containing the four CSV files as an input dataset. The notebook's current preferred folder hint is `AIpolicy_data`; if your Kaggle dataset has a different folder name, update `DATA_FOLDER` in the setup cell. Ensure that there is only one intended CSV match for each table name, because the file locator selects the best matching path.

**Raw data are intentionally not included** in this repository. Check the dataset's current licence and attribution requirements at download time. The notebook records CC BY 4.0 as its configured licence label, but the repository does not independently certify the current legal status of every source record or linked document.

## Quick start

### Local environment

```bash
git clone <YOUR-REPOSITORY-URL>
cd AI-Policy-Governance-Research

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Place the four CSV files in `data/raw/`, then launch:

```bash
jupyter lab notebooks/aipolicy_analysis.ipynb
```

Run all cells from top to bottom. The notebook writes outputs to `aipolicy_results/` (local execution) or `/kaggle/working/aipolicy_results/` (Kaggle) and creates `aipolicy_results.zip`.

### Kaggle

1. Create or open a Kaggle notebook.
2. Attach the dataset containing the four AIPolicyTracker CSV files.
3. Import `notebooks/aipolicy_analysis.ipynb` or paste its cells into a Kaggle notebook.
4. Run all cells from the beginning.
5. Download `aipolicy_results.zip` from the Kaggle Output panel.

## Analysis workflow

The notebook performs data loading and quality checks; merges obligations and jurisdiction metadata; derives instrument format and jurisdictional tier; applies two transparent rule-based text coders; reports inter-method agreement; constructs a conservative reconciled coding; produces stakeholder, remedy, accountability, actor, jurisdiction and implementation summaries; runs guarded exploratory logistic models; performs coding-sensitivity analysis; and exports tables, figures, metadata and a manifest.

## Important methodological qualifications

- The coded corpus is tracker-derived text assembled from policy titles, summaries, applicability fields and linked obligation descriptions. It is not necessarily the complete text of each official instrument.
- Coder A and Coder B are two rule-based procedures. Their Cohen's kappa is **inter-method agreement**, not human inter-coder reliability or independent construct validation.
- The reconciled minimum of the two coder scores is a conservative working specification. It is not a substitute for human review.
- The notebook creates `human_validation_template.csv`. Human validation is not complete until reviewers inspect official sources, independently code the cases, document evidence and analyse agreement/adjudication.
- Document-level remedies are kept separate from stakeholder-linked protections. A remedy somewhere in a document must not be described as an individual stakeholder remedy unless the coding establishes that link.
- Regression estimates are exploratory associations, not causal effects. The notebook uses minimum event-count, convergence and separation checks; models may be withheld where data are insufficient.
- Implementation records are descriptive only; a small number of tracker records does not support claims about implementation effectiveness.
- Actor-reference categories may overlap; counts across actor categories are not mutually exclusive.

## Outputs

The notebook saves CSV tables and vector PDF figures, along with run metadata, a manifest, and a human-validation worksheet. The final cell bundles generated result files into `aipolicy_results.zip`. See `docs/reproducibility.md` for the expected output conventions.

## Citation

See [`CITATION.cff`](CITATION.cff). Cite the AIPolicyTracker dataset separately according to the portal's current citation and licence instructions. Cite any official legal instruments used for manual validation individually.

## License

The repository's original code and documentation are released under the MIT License. This does **not** relicense the AIPolicyTracker data, third-party policy texts, or any generated content derived from them.


## Documentation

- [`docs/codebook.md`](docs/codebook.md): stakeholder groups, score definitions, coder A/B rules, accountability features, actor categories and validation boundaries.
- [`docs/data_dictionary.md`](docs/data_dictionary.md): source tables, keys, joins, constructed text, derived variables and run-specific data checks.
- [`docs/reproducibility.md`](docs/reproducibility.md): environment setup, data placement, execution sequence, human-validation workflow, official-source verification and outputs.

## Tests

Run `pytest` from the repository root. These tests validate repository structure, notebook/script syntax and documentation coverage; they do not execute the empirical analysis or establish the accuracy of coded legal classifications.
