.PHONY: install lint test eval eval-dry

install:
	uv sync --all-extras

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy

test:
	uv run pytest

# Eval: every case x log, N runs (default 3); writes evals/results/phase1.json.
# Pass options with ARGS, e.g. make eval ARGS="--case case-01 --n 5"
eval:
	uv run python -m evals.run_evals $(ARGS)

# Same with the stub agent (no model call); writes evals/results/dry-run.json.
eval-dry:
	uv run python -m evals.run_evals --dry-run --n 1 $(ARGS)
