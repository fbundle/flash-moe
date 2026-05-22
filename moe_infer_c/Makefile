.PHONY: all clean

CFLAGS = -O2 -Wall -fobjc-arc
LDFLAGS = -framework Metal -framework Foundation -framework Accelerate -lcompression -lpthread

all: bench

bench: bench.m shaders.metal
	clang $(CFLAGS) $(LDFLAGS) bench.m -o bench

# Stripped model (4 layers, 4 experts) for fast verification
bench-stripped: bench.m shaders.metal
	clang $(CFLAGS) $(LDFLAGS) \
		-DNUM_LAYERS=4 -DNUM_EXPERTS=4 -DNUM_EXPERTS_PER_TOK=4 \
		bench.m -o bench-stripped

clean:
	rm -f bench bench-stripped
