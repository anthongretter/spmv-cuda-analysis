CC        := nvcc
HFLAGS    := -Wall -Wextra -Wno-unused-parameter -Wno-unused-variable -O3
CFLAGS    := -Xcompiler "$(HFLAGS)" -std=c++17 -rdc=true -arch=sm_80 -m64 -O3
CPPFLAGS  := -I./include/
LDFLAGS   := -arch=sm_80
LDLIBS    :=

IMPS := $(wildcard src/imp/*.cu)
SRCS := $(wildcard src/*.cu)
OBJS := $(patsubst %.cu,%.o,$(SRCS))

#$(shell module load CUDA/11.8.0)
gpu: $(patsubst src/imp/gpu_%.cu,spmv_gpu_%,$(IMPS))
cpu: $(patsubst src/imp/cpu_%.cu,spmv_cpu_%,$(IMPS))

spmv_gpu_%: CFLAGS+=-DGPU
spmv_%: SRCS+=src/imp/%.cu
spmv_%: $(OBJS) src/imp/%.o
	$(CC) $(LDFLAGS) $^ -o $@ $(LDLIBS)

%.o: %.cu
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

clean:
	$(RM) $(wildcard spmv_*) $(OBJS) $(wildcard src/imp/*.o)

.PHONY: all clean
