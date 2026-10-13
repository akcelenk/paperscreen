.PHONY: install test lint typecheck cov check demo docker daily weekly

install:
	pip install -e ".[embed,plots,dev]" && pre-commit install

test:
	python -m pytest

lint:
	ruff check src tests

typecheck:
	mypy src

cov:
	python -m pytest --cov=paperscreen --cov-report=term-missing --cov-fail-under=85

check: lint typecheck cov   ## what CI runs

demo:
	paperscreen demo --workdir demo_run

docker:
	docker compose up --build

daily:
	scripts/daily.sh

weekly:
	scripts/weekly.sh
