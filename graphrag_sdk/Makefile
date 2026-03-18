.PHONY: lint format typecheck test test-cov check

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

typecheck:
	mypy src/graphrag_sdk/ --strict

test:
	python -m pytest tests/ -q

test-cov:
	python -m pytest tests/ --cov=src/graphrag_sdk --cov-report=term-missing -q

check: lint typecheck test
	@ruff format --check src/ tests/
	@echo "All checks passed!"
