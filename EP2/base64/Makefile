OUTPUT=base64
CUDA_OUTPUT=cuda_base64

CC=gcc
CC_OPT=

NVCC=nvcc
FLAGS=-ccbin clang-3.8 -Wno-deprecated-gpu-targets

.PHONY: all
all: $(OUTPUT) $(CUDA_OUTPUT)

$(OUTPUT): $(OUTPUT)_test.c $(OUTPUT).c $(OUTPUT).h
	$(CC) -o $(OUTPUT) $(CC_OPT) $(OUTPUT)_test.c $(OUTPUT).c $(OUTPUT).h

$(CUDA_OUTPUT): $(CUDA_OUTPUT).cu
	$(NVCC) $(FLAGS) -o $(CUDA_OUTPUT) $(CUDA_OUTPUT).cu

.PHONY: clean
clean:
	rm $(OUTPUT) $(CUDA_OUTPUT)
