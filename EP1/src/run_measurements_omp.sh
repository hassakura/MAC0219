#! /bin/bash


set -o xtrace

MEASUREMENTS=10
ITERATIONS=10
INITIAL_SIZE=16
NUM_THREADS=1
SIZE=$INITIAL_SIZE

NAMES=('mandelbrot_omp')

make
mkdir results

for NAME in ${NAMES[@]}; do
    
    mkdir results/$NAME
    
    for ((j=1; j<=6; j++)) do
        
        mkdir results/$NAME/$NUM_THREADS
        export OMP_NUM_THREADS=$NUM_THREADS
        for ((i=1; i<=$ITERATIONS; i++)) do
                perf stat -r $MEASUREMENTS ./$NAME -2.5 1.5 -2.0 2.0 $SIZE >> full.log 2>&1
                perf stat -r $MEASUREMENTS ./$NAME -0.8 -0.7 0.05 0.15 $SIZE >> seahorse.log 2>&1
                perf stat -r $MEASUREMENTS ./$NAME 0.175 0.375 -0.1 0.1 $SIZE >> elephant.log 2>&1
                perf stat -r $MEASUREMENTS ./$NAME -0.188 -0.012 0.554 0.754 $SIZE >> triple_spiral.log 2>&1
                SIZE=$(($SIZE * 2))
        done
        
        mv *.log results/$NAME/$NUM_THREADS

        NUM_THREADS=$(($NUM_THREADS * 2))

        SIZE=$INITIAL_SIZE
    done
done
