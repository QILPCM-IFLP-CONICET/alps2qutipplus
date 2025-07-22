# A GNU Makefile to run various tasks - compatibility for us old-timers.

# Note: This makefile include remake-style target comments.
# These comments before the targets start with #:
# remake --tasks to shows the targets and the comments


GIT2CL ?= admin-tools/git2cl
PYTHON ?= python3
PIP ?= pip3
BASH ?= bash
RM  ?= rm
PYTEST_OPTIONS ?=
DOCTEST_OPTIONS ?=
PYTEST_WORKERS ?=


SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SOURCEDIR     = docs
BUILDDIR      = docs/_build

BENCHMARK_FILE := bench_$(shell date +%Y%m%d)_$(shell git rev-parse --short HEAD)

.PHONY: \
    all \
    benchmark \
    benchmark-clean\
    benchmark-expect \
    benchmark-commutators \
    benchmark-gram \
    benchmark-projections \
    benchmark-show \
    benchmark-set-reference \
    check_pre_commit \
    conventions \
    develop \
    docs\
    docs-clean\
    install \
    pytest



all: develop check_pre_commit
	$(PIP) install -e .[dev]

check_pre_commit: conventions pytest

conventions:
	ruff check --fix alpsqutip
	ruff check --fix test
	isort test
	isort alpsqutip
	black test
	black alpsqutip


install:
	$(PYTHON) setup.py install

pytest:
	ALPSQUTIP_ALLTESTS=1 $(PYTHON) -m pytest $(PYTEST_OPTIONS) $(PYTEST_WORKERS) test

cprofile:
	ALPSQUTIP_ALLTESTS=1 $(PYTHON) -m cProfile -o output.stats -m pytest $(PYTEST_OPTIONS) $(PYTEST_WORKERS) test

codespell:
	codespell -L parms,fro,coo,indx,ket test
	codespell -L parms,fro,coo,indx,ket alpsqutip



docs:
	$(SPHINXBUILD) -b html $(SPHINXOPTS) $(SOURCEDIR) $(BUILDDIR)/html

docs-clean:
	rm -rf $(BUILDDIR)


benchmark: benchmark-expect benchmark-commutators benchmark-gram benchmark-projections

benchmark-expect:
	BENCHMARKS=1 ALPSQUTIP_ALLTESTS=1 CHAIN_SIZE=20 pytest -x --benchmark-enable --benchmark-save="expect_$(BENCHMARK_FILE)" --benchmark-columns=min test/states/test_expect_benchmark.py

benchmark-gram:
	BENCHMARKS=1 ALPSQUTIP_ALLTESTS=1 CHAIN_SIZE=20 pytest --benchmark-enable --benchmark-save="gram_$(BENCHMARK_FILE)" --benchmark-columns=min test/scalar_product/test_gram.py

benchmark-commutators:
	BENCHMARKS=1 ALPSQUTIP_ALLTESTS=1 CHAIN_SIZE=20 pytest --benchmark-enable --benchmark-save="commutators_$(BENCHMARK_FILE)" --benchmark-columns=min test/basic_operators/test_operator_functions_benchmarks.py

benchmark-projections:
	BENCHMARKS=1 ALPSQUTIP_ALLTESTS=1 CHAIN_SIZE=20 pytest -s --ff -x --benchmark-enable --benchmark-save="projections_$(BENCHMARK_FILE)" --benchmark-columns=min test/states/test_projections_benchmark.py

benchmark-set-reference:
	python test/set_benchmark_reference.py

benchmark-clean:
	rm -R .benchmarks

benchmark-show:
	python test/compare_benchmarks.py
