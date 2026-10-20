.PHONY: setup notebook validate validate-v2 smoke-v2 full-v2 clean

setup:
	python3 -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/python -m pip install -r requirements.txt -r requirements-dev.txt

notebook:
	.venv/bin/jupyter lab StutterSense.ipynb

validate:
	.venv/bin/python scripts/validate_repo.py
	.venv/bin/python -m unittest discover -s tests -v

validate-v2:
	.venv/bin/python scripts/validate_v2.py
	.venv/bin/python -m unittest discover -s tests -p 'test_v2.py' -v

smoke-v2:
	.venv/bin/python run_v2.py --config configs/v2_smoke.json

full-v2:
	.venv/bin/python run_v2.py --config configs/v2_full.json

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .ipynb_checkpoints -prune -exec rm -rf {} +
