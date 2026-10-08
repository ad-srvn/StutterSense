.PHONY: setup notebook validate clean

setup:
	python3 -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/python -m pip install -r requirements.txt -r requirements-dev.txt

notebook:
	.venv/bin/jupyter lab StutterSense.ipynb

validate:
	.venv/bin/python scripts/validate_repo.py
	.venv/bin/python -m unittest discover -s tests -v

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .ipynb_checkpoints -prune -exec rm -rf {} +
