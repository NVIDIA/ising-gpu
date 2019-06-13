CUDA_HOME=/usr/local/cuda
CUDACC=$(CUDA_HOME)/bin/nvcc
CC=gcc
LD=$(CUDACC)
CFLAGS=-c -O3 -g -I$(CUDA_HOME)/include
CUDACFLAGS=-c -O3 --use_fast_math -lineinfo -arch=sm_70 -Xptxas=-v
LDFLAGS= -Xcompiler=-fopenmp

all: cuIsing

cuIsing: main.o utils.o
	$(LD) -o cuIsing main.o utils.o $(LDFLAGS)

%.o: %.cu
	$(CUDACC) $(CUDACFLAGS) $<

%.o: %.c
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm *.o cuIsing
