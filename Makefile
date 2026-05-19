# Flash-MoE — Pure C/Metal MoE inference engine
# Builds from src/ into bin/

CC       = clang
CFLAGS   = -O2 -Wall -Wextra -fobjc-arc -Isrc
LDFLAGS  = -lpthread -lcompression
FW       = -framework Metal -framework Foundation -framework Accelerate
METALC   = xcrun -sdk macosx metal
METALLIB = xcrun -sdk macosx metallib

SRC_DIR  = src
BIN_DIR  = bin

# ── Accelerate BLAS for GatedDeltaNet recurrence ──────────────────────────
ACCELERATE ?= 1
ifeq ($(ACCELERATE),1)
	CFLAGS += -DACCELERATE_NEW_LAPACK
endif

# ── Debug build ───────────────────────────────────────────────────────────
DEBUG ?= 0
ifeq ($(DEBUG),1)
	CFLAGS := -g -O3 -Wall -Wextra -fobjc-arc -Isrc -DDEBUG
endif

# ── Targets ───────────────────────────────────────────────────────────────
.PHONY: all clean metallib help

all: $(BIN_DIR)/infer

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# ── Inference engine ──────────────────────────────────────────────────────
INFER_SRCS = $(SRC_DIR)/main.m
INFER_HDEP = $(SRC_DIR)/util.h \
             $(SRC_DIR)/tensors.h \
             $(SRC_DIR)/vocab.h \
             $(SRC_DIR)/cpu_kernels.h \
             $(SRC_DIR)/metal_setup.h \
             $(SRC_DIR)/gpu_ops.h \
             $(SRC_DIR)/attention.h \
             $(SRC_DIR)/moe_forward.h \
             $(SRC_DIR)/embeddings.h \
             $(SRC_DIR)/expert_io.h \
             $(SRC_DIR)/layer_forward.h \
             $(SRC_DIR)/server.h \
             $(SRC_DIR)/tokenizer.h \
             $(SRC_DIR)/model_config.h \
             $(SRC_DIR)/config.h \
             $(SRC_DIR)/optimization.h \
             $(SRC_DIR)/flag.h

$(BIN_DIR)/infer: $(INFER_SRCS) $(INFER_HDEP) $(SRC_DIR)/shaders.metal | $(BIN_DIR)
	$(CC) $(CFLAGS) $(FW) $(LDFLAGS) $(INFER_SRCS) -o $@

# ── Pre-compiled Metal library ────────────────────────────────────────────
metallib: $(BIN_DIR)/shaders.metallib

$(BIN_DIR)/shaders.metallib: $(SRC_DIR)/shaders.metal | $(BIN_DIR)
	$(METALC) -c $(SRC_DIR)/shaders.metal -o $(BIN_DIR)/shaders.air
	$(METALLIB) $(BIN_DIR)/shaders.air -o $@

# ── Python Cython extension ────────────────────────────────────────────────
.PHONY: python-build python-clean

python-build:
	python setup.py build_ext --inplace

python-clean:
	rm -rf moe_infer_mlx/_core*.so build/

# ── Clean ─────────────────────────────────────────────────────────────────
clean:
	rm -rf $(BIN_DIR)

# ── Help ──────────────────────────────────────────────────────────────────
help:
	@echo "Flash-MoE Makefile"
	@echo ""
	@echo "Build targets:"
	@echo "  make              Build infer"
	@echo "  make metallib     Pre-compile Metal shader library"
	@echo "  make python-build Build Cython extension (moe_infer_mlx)"
	@echo "  make python-clean Remove Python build artifacts"
	@echo "  make clean        Remove bin/"
	@echo ""
	@echo "Build options:"
	@echo "  make DEBUG=1      Debug build"
	@echo "  make ACCELERATE=0 Build without Accelerate BLAS"
