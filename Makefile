CC := gcc 
CFLAGS := -g
LDFLAGS :=
LOADLIBES :=
LDLIBS :=

CS := $(wildcard *.c)
OBJS := $(CS:.c=.o)
TARGET := spmv

$(TARGET): $(OBJS)

clean:
	$(RM) $(OBJS) $(TARGET)
