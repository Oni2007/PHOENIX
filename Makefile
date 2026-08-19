# Thin convenience wrapper over cmake/uv (TDD §8). Deliberately committed.
#
# The working tree is on a cloud-synced Windows drive, so the build tree and the
# virtualenv live OUTSIDE it, on the WSL2 filesystem. Override with PHOENIX_OUT.
PHOENIX_OUT ?= $(HOME)/.phoenix
BUILD       ?= $(PHOENIX_OUT)/build
VENV        ?= $(PHOENIX_OUT)/venv
PY          ?= $(VENV)/bin/python

export PHOENIX_BUILD_DIR = $(BUILD)
export PYTHONPATH        = $(CURDIR)/python:$(BUILD)

.PHONY: all configure build test test-cpp test-py lint typecheck check clean
all: build

configure:
	$(VENV)/bin/cmake -S . -B $(BUILD) -G Ninja -DCMAKE_BUILD_TYPE=Release \
		-DPython_EXECUTABLE=$(PY)

build: configure
	$(VENV)/bin/cmake --build $(BUILD) -j

test-cpp: build
	$(BUILD)/phoenix_tests

test-py: build
	$(PY) -m pytest -q

test: test-cpp test-py

lint:
	$(VENV)/bin/ruff check python tests
	$(VENV)/bin/ruff format --check python tests

typecheck:
	$(VENV)/bin/mypy

check: lint typecheck test

clean:
	rm -rf $(BUILD)
