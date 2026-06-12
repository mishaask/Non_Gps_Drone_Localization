.PHONY: install test smoke analyze clean

install:
	python -m pip install --upgrade pip
	pip install -e .
	pip install -r requirements.txt

test:
	pytest -q

smoke: test analyze

analyze:
	python -m gps_ex1.tools.analyze_predictions --prediction-csv data/processed/DJI_0010_predictions_filtered.csv data/processed/DJI_0011_predictions_filtered.csv

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	find . -type d -name .pytest_cache -prune -exec rm -rf {} +
