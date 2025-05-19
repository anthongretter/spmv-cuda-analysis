CC        := nvcc
HFLAGS    := -Wall -Wextra -Wno-unused-parameter -Wno-unused-variable
CFLAGS    := -Xcompiler "$(HFLAGS)" -std=c++17 --gpu-architecture=sm_80 -m64
CPPFLAGS  := -I./include/
LDFLAGS   :=
LDLIBS    := -lcudart

BINS := spmv_cpu_csr spmv_cpu_naive spmv_gpu_csr spmv_gpu_mem
SRCS := $(wildcard src/*.cu)
OBJS := $(patsubst %.cu,%.o,$(SRCS))

$(shell module load CUDA/11.8.0)

all: $(BINS)

spmv_gpu_%: CFLAGS+=-DGPU
spmv_%: SRCS+=src/imp/%.cu
spmv_%: $(OBJS) src/imp/%.o
	$(CC) $(LDFLAGS) $^ -o $@ $(LDLIBS)

%.o: %.cu
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

clean:
	$(RM) $(OBJS) $(wildcard src/imp/*.o) $(BINS) $(DEPS)

.PHONY: all clean
