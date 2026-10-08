# Contributing

Thanks for improving StutterSense. Contributions should preserve the project's distinction between measured results, hypotheses, and clinical claims.

## Development workflow

1. Create a virtual environment and install `requirements.txt`, or install CPU PyTorch plus the smaller `requirements-test.txt` set for offline validation only.
2. Make notebook changes in `StutterSense.ipynb` with cleared execution outputs.
3. Keep generated checkpoints, embeddings, downloaded data, and archives out of Git.
4. Run `python scripts/validate_repo.py` and `python -m unittest discover -s tests -v`.
5. Explain methodological changes and their effect on comparability in the pull request.

Do not commit fabricated metrics or overwrite the curated reference results unless the complete experiment was rerun. A result update should include its configuration, provenance, split indices, metrics, and figures.

## Reporting issues

Include the operating system, Python version, accelerator, relevant package versions, configuration changes, and the smallest reproducible error trace. Do not include private audio or credentials.
