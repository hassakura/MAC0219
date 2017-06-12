OUTPUT=rot-13
CUDA_OUTPUT=cuda_rot13

CC=gcc
CC_OPT=

NVCC=nvcc

.PHONY: all
all: $(OUTPUT) $(CUDA_OUTPUT)

$(OUTPUT): $(OUTPUT)_test.c $(OUTPUT).c $(OUTPUT).h
	$(CC) -o $(OUTPUT) $(CC_OPT) $(OUTPUT)_test.c $(OUTPUT).c $(OUTPUT).h

$(CUDA_OUTPUT): $(CUDA_OUTPUT).cu
	$(NVCC) -o $(CUDA_OUTPUT) $(CUDA_OUTPUT).cu

.PHONY: clean
clean:
	rm $(OUTPUT) $(CUDA_OUTPUT)
