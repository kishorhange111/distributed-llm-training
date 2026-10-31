# Common developer commands. Run `make help` to list them.
.DEFAULT_GOAL := help
.PHONY: help install install-dev lint format test check clean smoke bench

help:           ## list the available commands
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-12s %s\n", $$1, $$2}'

install:        ## this package + data-preparation extras
	pip install -e ".[data]"

install-dev:    ## lint/test tooling + this package (editable)
	pip install -e ".[dev]"

lint:           ## ruff lint + format check (what CI runs)
	ruff check .
	ruff format --check .

format:         ## auto-fix lint findings and reformat
	ruff check --fix .
	ruff format .

test:           ## run the test suite (CPU only)
	pytest

check: lint test ## everything CI runs

smoke:          ## a few training steps on random tokens (any device)
	python -m disttrain.train --config configs/smoke.yaml

bench:          ## memory / throughput ablation (needs a CUDA GPU)
	python scripts/benchmark.py --preset gpt2-medium --micro_batch 8

clean:          ## remove caches and build artefacts
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov build dist src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
