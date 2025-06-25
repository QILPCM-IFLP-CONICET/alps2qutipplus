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


.PHONY: \
    all \
    check_pre_commit \
    develop \
    install \
    pytest \
    conventions \

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
