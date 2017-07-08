#! /bin/bash

set -o xtrace

MEASUREMENTS=3
ITERATIONS=10
INITIAL_SIZE=2048

SIZE=$INITIAL_SIZE

NAME=('mandelbrot_seq')
MPINAME=('mandelbrot_mpi')

make
mkdir results

    mkdir results/$NAME

    for ((i=1; i<=$ITERATIONS; i++)); do
            perf stat -r $MEASUREMENTS ./$NAME -2.5 1.5 -2.0 2.0 $SIZE >> full.log 2>&1
            perf stat -r $MEASUREMENTS mpirun -np 2 $MPINAME -2.5 1.5 -2.0 2.0 $SIZE >> mpifull.log 2>&1
            diff outputmpi.ppm output.ppm >> dif.log 2>&1
    done
    mv *.log results/$NAME
    
    rm output.ppm
    rm outputmpi.ppm
