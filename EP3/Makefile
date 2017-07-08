OUTPUT=mandelbrot

IMAGE=.ppm

CC=gcc
MCC=mpicc
CORES=2
CC_OPT=-std=c11

.PHONY: all
all: $(OUTPUT)_seq $(OUTPUT)_mpi

$(OUTPUT)_seq: $(OUTPUT)_seq.c
	$(CC) -o $(OUTPUT)_seq $(CC_OPT) $(OUTPUT)_seq.c

$(OUTPUT)_mpi: $(OUTPUT)_mpi.c
	$(MCC) -o $(OUTPUT)_mpi $(OUTPUT)_mpi.c

.PHONY: clean
clean:
	rm $(OUTPUT)_seq *$(IMAGE)
	rm $(OUTPUT)_mpi *$(IMAGE)
