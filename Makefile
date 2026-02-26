PYTHON ?= python3
RUNNER := run.py
DASHBOARD := dashboard.py
OUT_DIR := out

# Bible-first defaults
CORPUS ?= data/bible.txt
WORLD_VOCAB ?= 512
WORLD_LEN ?= 200000
STEPS ?= 5000
LOG_EVERY ?= 100
SEED ?= 42
DASH_HOST ?= 127.0.0.1
DASH_PORT ?= 8787
DASH_STEPS_PER_TICK ?= 1
DASH_SLEEP_MS ?= 0
DASH_TOKEN_SPACE ?= gpt2
DASH_GPT2_MODEL ?= gpt2
DASH_CORPUS ?= data/tiny_bible.txt

# Energy ecology defaults (conservative system)
CONSERVE ?= true
STRICT_BUDGET ?= true

STAMP := $(shell date +%Y%m%d_%H%M%S)
OUT_JSON ?= $(OUT_DIR)/bible_$(STAMP).json

COMMON_FLAGS := \
	--seed $(SEED) \
	--corpus-file $(CORPUS) \
	--world-vocab-size $(WORLD_VOCAB) \
	--world-len $(WORLD_LEN) \
	--log-every $(LOG_EVERY) \
	--conserve-total-energy $(CONSERVE) \
	--strict-energy-budget $(STRICT_BUDGET)

.PHONY: help check smoke run run-long full-bible conserve dashboard dashboard-public

help:
	@echo "Emera runner targets"
	@echo ""
	@echo "  make check                 # compile-check Python files"
	@echo "  make smoke                 # short Bible smoke run"
	@echo "  make run                   # default Bible run (STEPS=$(STEPS))"
	@echo "  make run-long              # longer Bible run"
	@echo "  make full-bible            # one pass over full Bible token length"
	@echo "  make conserve              # alias of run with conservative defaults"
	@echo "  make dashboard             # launch HTMX dashboard + background training"
	@echo "  make dashboard-public      # dashboard on 0.0.0.0 for LAN access"
	@echo ""
	@echo "Overrides:"
	@echo "  STEPS=..., LOG_EVERY=..., WORLD_VOCAB=..., WORLD_LEN=..."
	@echo "  OUT_JSON=out/my_run.json"
	@echo "  DASH_HOST=127.0.0.1 DASH_PORT=8787 DASH_STEPS_PER_TICK=1 DASH_SLEEP_MS=0"
	@echo "  DASH_TOKEN_SPACE=gpt2 DASH_GPT2_MODEL=gpt2 DASH_CORPUS=data/tiny_bible.txt"
	@echo ""
	@echo "Example:"
	@echo "  make run STEPS=12000 LOG_EVERY=200"
	@echo "  make dashboard DASH_STEPS_PER_TICK=2 DASH_SLEEP_MS=1"

check:
	$(PYTHON) -m py_compile *.py

smoke: check
	mkdir -p $(OUT_DIR)
	$(PYTHON) $(RUNNER) --steps 300 $(COMMON_FLAGS) --output-json $(OUT_DIR)/smoke_$(STAMP).json

run: check
	mkdir -p $(OUT_DIR)
	$(PYTHON) $(RUNNER) --steps $(STEPS) $(COMMON_FLAGS) --output-json $(OUT_JSON)

conserve: run

run-long: check
	mkdir -p $(OUT_DIR)
	$(PYTHON) $(RUNNER) --steps 25000 $(COMMON_FLAGS) --output-json $(OUT_DIR)/bible_long_$(STAMP).json

full-bible: check
	mkdir -p $(OUT_DIR)
	$(PYTHON) $(RUNNER) \
		--steps 4606954 \
		--world-len 4606954 \
		$(COMMON_FLAGS) \
		--output-json $(OUT_DIR)/bible_full_$(STAMP).json

dashboard: check
	$(PYTHON) $(DASHBOARD) \
		--seed $(SEED) \
		--token-space $(DASH_TOKEN_SPACE) \
		--gpt2-model-name $(DASH_GPT2_MODEL) \
		--corpus-file $(DASH_CORPUS) \
		--world-vocab-size $(WORLD_VOCAB) \
		--world-len $(WORLD_LEN) \
		--host $(DASH_HOST) \
		--port $(DASH_PORT) \
		--steps-per-tick $(DASH_STEPS_PER_TICK) \
		--sleep-ms $(DASH_SLEEP_MS)

dashboard-public: check
	$(PYTHON) $(DASHBOARD) \
		--seed $(SEED) \
		--token-space $(DASH_TOKEN_SPACE) \
		--gpt2-model-name $(DASH_GPT2_MODEL) \
		--corpus-file $(DASH_CORPUS) \
		--world-vocab-size $(WORLD_VOCAB) \
		--world-len $(WORLD_LEN) \
		--host 0.0.0.0 \
		--port $(DASH_PORT) \
		--steps-per-tick $(DASH_STEPS_PER_TICK) \
		--sleep-ms $(DASH_SLEEP_MS)
