CUDA_HOME=/usr/local/cuda
CUDACC=$(CUDA_HOME)/bin/nvcc
CC=gcc
LD=$(CUDACC)
CFLAGS=-c -O3 -g -I$(CUDA_HOME)/include
CUDACFLAGS= -std=c++11 -c -O3 -lineinfo -arch=sm_70 -Xptxas=-v -I../external/cub
LDFLAGS= -lcurand -lcublas

all: ising_tensorcore

ising_tensorcore: main.o
	$(LD) -o ising_tensorcore main.o $(LDFLAGS)

%.o: %.cu
	nvcc -c $(CUDACFLAGS) $<

clean:
	rm *.o ising_tensorcore
