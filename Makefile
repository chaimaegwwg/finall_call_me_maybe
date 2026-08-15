install:
	uv sync

run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

clean:
	rm -rf src/__pycache__ data/output .mypy_cache

lint:
	uv run flake8 src 
	uv run python -m mypy src

lint-strict:
	uv run flake8 src
	uv run python -m mypy --strict src