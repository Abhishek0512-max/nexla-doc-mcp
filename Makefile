.PHONY: install ingest run eval test lint format fix typecheck check

install:
	poetry install

ingest:
	poetry run python -m app.services.ingestion.service

run:
	poetry run python -m app.server

eval:
	poetry run python -m evals.run_retrieval_eval

lint:
	poetry run ruff check app evals tests

format:
	poetry run ruff format app evals tests

fix:
	poetry run ruff check --fix app evals tests
	poetry run ruff format app evals tests

typecheck:
	poetry run mypy app

test:
	poetry run pytest

check: lint typecheck test
