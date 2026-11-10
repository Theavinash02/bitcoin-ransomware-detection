PY ?= python
CLI = $(PY) -m ransomdet.cli

.PHONY: all dev test lint demo

# Every report, in order: download, prepare, each stage, demo export.
all:
	$(CLI) all

dev:
	$(CLI) all --dev

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests app notebooks scripts
	$(PY) -m ruff format --check src tests app notebooks scripts

demo:
	cd app && $(PY) app.py
