from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_required_repository_files_exist():
    required = [
        "README.md", "requirements.txt", "environment.yml", "CITATION.cff",
        "LICENSE", "CHANGELOG.md", "notebooks/aipolicy_analysis.ipynb",
        "src/aipolicy_analysis.py", "docs/codebook.md",
        "docs/data_dictionary.md", "docs/reproducibility.md",
        "data/raw/.gitkeep", "data/processed/.gitkeep",
        "outputs/figures/.gitkeep", "outputs/tables/.gitkeep",
    ]
    missing = [rel for rel in required if not (ROOT / rel).is_file()]
    assert not missing, f"Missing repository files: {missing}"


def test_expected_repository_directories_exist():
    for rel in ("data/raw", "data/processed", "docs", "notebooks", "outputs/figures", "outputs/tables", "src", "tests"):
        assert (ROOT / rel).is_dir(), rel


def test_notebook_is_valid_json_and_has_expected_sections():
    path = ROOT / "notebooks/aipolicy_analysis.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook.get("nbformat") == 4
    cells = notebook.get("cells", [])
    assert cells
    code_cells = [c for c in cells if c.get("cell_type") == "code"]
    assert code_cells
    source_text = "\n".join("".join(c.get("source", [])) for c in cells)
    for marker in ("RQ1", "RQ2", "RQ3", "RQ4", "Coder A", "Coder B", "human_validation_template.csv"):
        assert marker.lower() in source_text.lower(), marker


def test_notebook_code_cells_parse_without_syntax_errors():
    notebook = json.loads((ROOT / "notebooks/aipolicy_analysis.ipynb").read_text(encoding="utf-8"))
    errors = []
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        try:
            ast.parse(source)
        except SyntaxError as exc:
            errors.append((index, str(exc)))
    assert not errors, f"Notebook syntax errors: {errors}"


def test_exported_python_script_parses():
    script = ROOT / "src/aipolicy_analysis.py"
    ast.parse(script.read_text(encoding="utf-8"), filename=str(script))


def test_documentation_covers_methodological_limitations():
    codebook = (ROOT / "docs/codebook.md").read_text(encoding="utf-8").lower()
    reproducibility = (ROOT / "docs/reproducibility.md").read_text(encoding="utf-8").lower()
    for phrase in ("inter-method agreement", "human coder", "document-level", "stakeholder-linked", "public enforcement"):
        assert phrase in codebook, phrase
    for phrase in ("human validation", "official-source verification", "causally", "tracker-derived"):
        assert phrase in reproducibility, phrase


def test_requirements_include_notebook_analysis_dependencies():
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    for package in ("numpy", "pandas", "matplotlib", "scipy", "scikit-learn", "statsmodels", "jupyterlab", "nbformat"):
        assert package in requirements, package
