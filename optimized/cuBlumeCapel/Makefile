CUDA_HOME = /opt/cuda-12.8.1
CUDACC = $(CUDA_HOME)/bin/nvcc
CC = gcc
LD = $(CUDACC)

CFLAGS = -c -O3 -g -I$(CUDA_HOME)/include

SMS ?= 89
CUDA_ARCH = $(foreach SM,$(SMS),-gencode arch=compute_$(SM),code=sm_$(SM))

CUDACFLAGS = -c -O3 -lineinfo $(CUDA_ARCH) -Xptxas=-v
LDFLAGS    = -Xcompiler=-fopenmp -lcurand

C_SRCS  = utils.c
CU_SRCS = main.cu
                                
ifdef USE_MNNVL
$(info Compiling with MNNVL support...)
MPI_HOME = /project/coreai_hpc_hpc/hpc_sdk/Linux_aarch64/dev/comm_libs/12.8/hpcx/latest/ompi

CUDACFLAGS += -I$(MPI_HOME)/include -DUSE_MNNVL
LDFLAGS    += -L$(MPI_HOME)/lib -lcuda -lmpi
CU_SRCS    += vmm_alloc.cu
endif

C_OBJS = $(patsubst %.c, %.o, $(C_SRCS))
CU_OBJS = $(patsubst %.cu, %.o, $(CU_SRCS))

all: cuBlume

cuBlume: $(CU_OBJS) $(C_OBJS)
	$(LD) -o cuBlume $(CU_OBJS) $(C_OBJS) $(LDFLAGS)

%.o: %.cu
	$(CUDACC) $(CUDACFLAGS) $<

%.o: %.c
	$(CC) $(CFLAGS) $< -o $@

clean:
	-@rm -f *.o cuBlume *.sass &> /dev/null || true
