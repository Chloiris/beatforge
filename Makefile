PYTHON ?= python3
TASK := $(PYTHON) scripts/beatforge.py

.PHONY: doctor install install-accurate prepare-model prepare-demucs install-vocal prepare-vocal-models prepare-alignment-models seed dev test e2e lint build evaluate clean-generated

doctor:
	$(TASK) doctor

install:
	$(TASK) install

install-accurate:
	$(TASK) install-accurate

prepare-model prepare-demucs:
	$(TASK) prepare-model

install-vocal:
	$(TASK) install-vocal

prepare-vocal-models:
	$(TASK) prepare-vocal-models

prepare-alignment-models:
	$(TASK) prepare-alignment-models

seed:
	$(TASK) seed

dev:
	$(TASK) dev

test:
	$(TASK) test

e2e:
	$(TASK) e2e

lint:
	$(TASK) lint

build:
	$(TASK) build

evaluate:
	$(TASK) evaluate

clean-generated:
	$(TASK) clean-generated
