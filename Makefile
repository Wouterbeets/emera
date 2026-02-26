PYTHON ?= python3
RUNNER := run.py
DASHBOARD := dashboard.py
ISLANDS_RUNNER := evolve_islands.py
OUT_DIR := out

# Bible-first defaults
CORPUS ?= data/tiny_bible.txt
WORLD_VOCAB ?= 512
WORLD_LEN ?= 200000
STEPS ?= 5000
LOG_EVERY ?= 100
SEED ?= 42
DASH_HOST ?= 127.0.0.1
DASH_PORT ?= 8787
DASH_STEPS_PER_TICK ?= 2
DASH_SLEEP_MS ?= 0
DASH_TOKEN_SPACE ?= gpt2
DASH_GPT2_MODEL ?= gpt2
DASH_CORPUS ?= data/tiny_bible.txt
DASH_WORLD_VOCAB ?= $(WORLD_VOCAB)
DASH_WORLD_LEN ?= 9213908
GAP_READ_BACKEND ?= auto
GAP_READ_BATCH ?= 128
MAX_D_LATENT ?= 128
MAX_GAP_DIM ?= 64
MAX_GAP_LEN ?= 1024
MAX_GAP_BATCH ?= 256
MAX_K_ROUNDS ?= 6
MAX_CHAOS_SUBSTEPS ?= 1
DASH_GAP_READ_BACKEND ?= jax
DASH_GAP_READ_BATCH ?= $(MAX_GAP_BATCH)
DASH_D_LATENT ?= $(MAX_D_LATENT)
DASH_GAP_DIM ?= $(MAX_GAP_DIM)
DASH_GAP_LEN ?= $(MAX_GAP_LEN)
DASH_K_ROUNDS ?= $(MAX_K_ROUNDS)
DASH_CHAOS_SUBSTEPS ?= $(MAX_CHAOS_SUBSTEPS)
XLA_PYTHON_CLIENT_PREALLOCATE ?= true
XLA_PYTHON_CLIENT_MEM_FRACTION ?= 0.92
ISLANDS ?= 0
ISLAND_ROUNDS ?= 4
ISLAND_SEEDS_PER ?= 1
ISLAND_PROMOTE_TOP ?= 8
ISLAND_TRAIN_SMALL ?= 700
ISLAND_TRAIN_FULL ?= 2600
ISLAND_SKIP_FULL ?= false

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
	--gap-read-backend $(GAP_READ_BACKEND) \
	--gap-read-batch-size $(GAP_READ_BATCH) \
	--log-every $(LOG_EVERY) \
	--conserve-total-energy $(CONSERVE) \
	--strict-energy-budget $(STRICT_BUDGET)

.PHONY: help check smoke run run-long run-max full-bible double-bible conserve dashboard dashboard-public islands

help:
	@echo "Emera runner targets"
	@echo ""
	@echo "  make check                 # compile-check Python files"
	@echo "  make smoke                 # short Bible smoke run"
	@echo "  make run                   # default Bible run (STEPS=$(STEPS))"
	@echo "  make run-long              # longer Bible run"
	@echo "  make run-max               # high-load run preset (bigger gap geometry)"
	@echo "  make full-bible            # one pass over full Bible token length"
	@echo "  make double-bible          # two-pass Bible world-length run"
	@echo "  make conserve              # alias of run with conservative defaults"
	@echo "  make dashboard             # launch HTMX dashboard + background training"
	@echo "  make dashboard-public      # dashboard on 0.0.0.0 for LAN access"
	@echo "  make islands               # no-migration island GA hyperparam search"
	@echo ""
	@echo "Overrides:"
	@echo "  STEPS=..., LOG_EVERY=..., WORLD_VOCAB=..., WORLD_LEN=..."
	@echo "  OUT_JSON=out/my_run.json"
	@echo "  DASH_HOST=127.0.0.1 DASH_PORT=8787 DASH_STEPS_PER_TICK=2 DASH_SLEEP_MS=0"
	@echo "  DASH_TOKEN_SPACE=gpt2 DASH_GPT2_MODEL=gpt2 DASH_CORPUS=data/bible.txt"
	@echo "  DASH_WORLD_VOCAB=512 DASH_WORLD_LEN=9213908"
	@echo "  DASH_GAP_READ_BACKEND=jax DASH_GAP_READ_BATCH=256"
	@echo "  DASH_D_LATENT=128 DASH_GAP_DIM=64 DASH_GAP_LEN=1024 DASH_K_ROUNDS=6"
	@echo "  XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.92"
	@echo "  GAP_READ_BACKEND=auto GAP_READ_BATCH=128"
	@echo "  MAX_D_LATENT=128 MAX_GAP_DIM=64 MAX_GAP_LEN=1024 MAX_GAP_BATCH=256"
	@echo "  ISLANDS=0 ISLAND_ROUNDS=4 ISLAND_SEEDS_PER=1 ISLAND_PROMOTE_TOP=8"
	@echo "  ISLAND_TRAIN_SMALL=700 ISLAND_TRAIN_FULL=2600 ISLAND_SKIP_FULL=false"
	@echo ""
	@echo "Example:"
	@echo "  make run STEPS=12000 LOG_EVERY=200"
	@echo "  make run-max PYTHON=.venv/bin/python STEPS=20000"
	@echo "  make dashboard DASH_STEPS_PER_TICK=2 DASH_SLEEP_MS=1"
	@echo "  make islands PYTHON=.venv/bin/python ISLANDS=0 ISLAND_ROUNDS=6"

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

run-max: check
	mkdir -p $(OUT_DIR)
	$(PYTHON) $(RUNNER) \
		--steps $(STEPS) \
		$(COMMON_FLAGS) \
		--d-latent $(MAX_D_LATENT) \
		--gap-dim $(MAX_GAP_DIM) \
		--gap-len $(MAX_GAP_LEN) \
		--gap-read-batch-size $(MAX_GAP_BATCH) \
		--k-rounds $(MAX_K_ROUNDS) \
		--chaos-substeps-per-round $(MAX_CHAOS_SUBSTEPS) \
		--output-json $(OUT_DIR)/bible_max_$(STAMP).json

full-bible: check
	mkdir -p $(OUT_DIR)
	$(PYTHON) $(RUNNER) \
		$(COMMON_FLAGS) \
		--steps 4606954 \
		--world-len 4606954 \
		--output-json $(OUT_DIR)/bible_full_$(STAMP).json

double-bible: check
	mkdir -p $(OUT_DIR)
	$(PYTHON) $(RUNNER) \
		$(COMMON_FLAGS) \
		--steps 9213908 \
		--world-len 9213908 \
		--output-json $(OUT_DIR)/bible_double_$(STAMP).json

dashboard: check
	XLA_PYTHON_CLIENT_PREALLOCATE=$(XLA_PYTHON_CLIENT_PREALLOCATE) \
	XLA_PYTHON_CLIENT_MEM_FRACTION=$(XLA_PYTHON_CLIENT_MEM_FRACTION) \
	$(PYTHON) $(DASHBOARD) \
		--seed $(SEED) \
		--token-space $(DASH_TOKEN_SPACE) \
		--gpt2-model-name $(DASH_GPT2_MODEL) \
		--corpus-file $(DASH_CORPUS) \
		--world-vocab-size $(DASH_WORLD_VOCAB) \
		--world-len $(DASH_WORLD_LEN) \
		--d-latent $(DASH_D_LATENT) \
		--gap-dim $(DASH_GAP_DIM) \
		--gap-len $(DASH_GAP_LEN) \
		--k-rounds $(DASH_K_ROUNDS) \
		--chaos-substeps-per-round $(DASH_CHAOS_SUBSTEPS) \
		--gap-read-backend $(DASH_GAP_READ_BACKEND) \
		--gap-read-batch-size $(DASH_GAP_READ_BATCH) \
		--host $(DASH_HOST) \
		--port $(DASH_PORT) \
		--steps-per-tick $(DASH_STEPS_PER_TICK) \
		--sleep-ms $(DASH_SLEEP_MS)

dashboard-public: check
	XLA_PYTHON_CLIENT_PREALLOCATE=$(XLA_PYTHON_CLIENT_PREALLOCATE) \
	XLA_PYTHON_CLIENT_MEM_FRACTION=$(XLA_PYTHON_CLIENT_MEM_FRACTION) \
	$(PYTHON) $(DASHBOARD) \
		--seed $(SEED) \
		--token-space $(DASH_TOKEN_SPACE) \
		--gpt2-model-name $(DASH_GPT2_MODEL) \
		--corpus-file $(DASH_CORPUS) \
		--world-vocab-size $(DASH_WORLD_VOCAB) \
		--world-len $(DASH_WORLD_LEN) \
		--d-latent $(DASH_D_LATENT) \
		--gap-dim $(DASH_GAP_DIM) \
		--gap-len $(DASH_GAP_LEN) \
		--k-rounds $(DASH_K_ROUNDS) \
		--chaos-substeps-per-round $(DASH_CHAOS_SUBSTEPS) \
		--gap-read-backend $(DASH_GAP_READ_BACKEND) \
		--gap-read-batch-size $(DASH_GAP_READ_BATCH) \
		--host 0.0.0.0 \
		--port $(DASH_PORT) \
		--steps-per-tick $(DASH_STEPS_PER_TICK) \
		--sleep-ms $(DASH_SLEEP_MS)

islands: check
	mkdir -p $(OUT_DIR)
	$(PYTHON) $(ISLANDS_RUNNER) \
		--python $(PYTHON) \
		--islands $(ISLANDS) \
		--rounds $(ISLAND_ROUNDS) \
		--seeds-per-candidate $(ISLAND_SEEDS_PER) \
		--promote-top $(ISLAND_PROMOTE_TOP) \
		--train-steps-small $(ISLAND_TRAIN_SMALL) \
		--train-steps-full $(ISLAND_TRAIN_FULL) \
		--skip-full-pass $(ISLAND_SKIP_FULL) \
		--output-json $(OUT_DIR)/islands_$(STAMP).json
