#!/usr/bin/env python3
"""Offline structural validation for additive StutterSense 2.0 files."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    "stuttersense_v2/__init__.py",
    "stuttersense_v2/config.py",
    "stuttersense_v2/data.py",
    "stuttersense_v2/modeling.py",
    "stuttersense_v2/evaluation.py",
    "stuttersense_v2/reporting.py",
    "stuttersense_v2/inference.py",
    "stuttersense_v2/experiment.py",
    "stuttersense_v2/cli.py",
    "run_v2.py",
    "scripts/validate_v2.py",
    "tests/test_v2.py",
)
REQUIRED_FILES = SOURCE_FILES + (
    "StutterSense_v2.ipynb",
    "configs/v2_smoke.json",
    "configs/v2_full.json",
    "docs/v2_usage.md",
    "docs/identity_manifest.md",
    "docs/endpointing_research_plan.md",
    ".github/workflows/v2-ci.yml",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    missing = [name for name in REQUIRED_FILES if not (ROOT / name).is_file()]
    require(not missing, f"Missing v2 files: {missing}")
    for name in SOURCE_FILES:
        path = ROOT / name
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    full = json.loads((ROOT / "configs/v2_full.json").read_text(encoding="utf-8"))
    smoke = json.loads((ROOT / "configs/v2_smoke.json").read_text(encoding="utf-8"))
    require(full["training"]["seeds"] == [42, 123, 456], "Unexpected full-run seeds")
    require(full["sampling"]["scan_limit"] is None, "Full config must scan the complete stream")
    require(smoke["mode"] == "smoke" and smoke["sampling"]["scan_limit"], "Invalid smoke config")
    require(full["dataset"]["revision"], "Dataset revision is not pinned")
    require(full["model"]["encoder_revision"], "Model revision is not pinned")
    require(full["output"]["root"] != "results", "V2 must not overwrite the validated baseline")
    require("v2_outputs/" in (ROOT / ".gitignore").read_text(encoding="utf-8"),
            "Generated v2 artifacts must be excluded from Git")

    notebook = nbformat.read(ROOT / "StutterSense_v2.ipynb", as_version=4)
    nbformat.validate(notebook)
    source = "\n".join(cell.source for cell in notebook.cells)
    code_source = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    require("StutterSense.ipynb" not in source, "V2 notebook must not rewrite or execute the baseline notebook")
    require('"-U"' not in code_source and '"--upgrade"' not in code_source,
            "V2 notebook must not force-upgrade Colab packages")
    protected_packages = ("numpy>=", "scipy>=", "pandas>=", "scikit-learn>=", "matplotlib>=")
    require(not any(package in code_source for package in protected_packages),
            "V2 notebook must preserve Colab's preloaded scientific stack")
    require('find_spec("torch") is None' in code_source,
            "V2 notebook may install PyTorch only when it is unavailable")
    self_contained_markers = (
        "reservoir_sample",
        "train_variant",
        "select_f1_thresholds",
        "paired_group_bootstrap_difference",
    )
    require(all(marker in source for marker in self_contained_markers),
            "V2 notebook is missing required self-contained experiment stages")
    require(all(not cell.outputs for cell in notebook.cells if cell.cell_type == "code"),
            "V2 notebook must not contain stored outputs")

    baseline = ROOT / "results" / "dataset_provenance.json"
    require(baseline.is_file(), "Validated baseline results are missing")
    print("StutterSense 2.0 structural validation passed.")
    print("Scientific status: curated results present; experiments not rerun by this validator.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
