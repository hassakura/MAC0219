OUTPUT=xor
CUDA_OUTPUT=cuda_xor

CC=gcc
CC_OPT=

NVCC=nvcc

.PHONY: all
all: $(OUTPUT) $(CUDA_OUTPUT)

$(OUTPUT): $(OUTPUT).c
	$(CC) -o $(OUTPUT) $(CC_OPT) $(OUTPUT).c

$(CUDA_OUTPUT): $(CUDA_OUTPUT).cu
	$(NVCC) -o $(CUDA_OUTPUT) $(CUDA_OUTPUT).cu

.PHONY: clean
clean:
	rm $(OUTPUT) $(CUDA_OUTPUT)