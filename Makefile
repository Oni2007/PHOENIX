# Thin convenience wrapper over cmake/uv (TDD 8). Deliberately committed.
BUILD ?= build
PY    ?= .venv/bin/python

.PHONY: configure build test lint typecheck clean all
all: build

configure:
	cmake -S . -B $(BUILD) -G Ninja -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE=$(PWD)/$(PY)

build: configure
	cmake --build $(BUILD) -j

test: build
	cd $(BUILD) && ctest --output-on-failure
	$(PY) -m pytest -q

lint:
	.venv/bin/ruff check python tests
	.venv/bin/ruff format --check python tests

typecheck:
	.venv/bin/mypy

clean:
	rm -rf $(BUILD)
