#! /bin/bash

set -o xtrace

MEASUREMENTS=10
ITERATIONS=10
INITIAL_SIZE=8192

SIZE=$INITIAL_SIZE

NAME=('mandelbrot_seq')
MPINAME=('mandelbrot_mpi')

make
mkdir results

    mkdir results/$MPINAME

    for ((i=1; i<=$ITERATIONS; i++)); do	  
            perf stat -r $MEASUREMENTS mpirun -np 8 -hostfile hostfile $MPINAME -2.5 1.5 -2.0 2.0 $SIZE >> mpifull.log 2>&1
            perf stat -r $MEASUREMENTS mpirun -np 8 -hostfile hostfile $MPINAME -0.8 -0.7 0.05 0.15 $SIZE >> mpiseahorse.log 2>&1
	    perf stat -r $MEASUREMENTS mpirun -np 8 -hostfile hostfile $MPINAME 0.175 0.375 -0.1 0.1 $SIZE >> mpielephant.log 2>&1
	    perf stat -r $MEASUREMENTS mpirun -np 8 -hostfile hostfile $MPINAME -0.188 -0.012 0.554 0.754 $SIZE >> mpitriple.log 2>&1
    done
    mv *.log results/$MPINAME
    
    rm output.ppm
    rm outputmpi.ppm
