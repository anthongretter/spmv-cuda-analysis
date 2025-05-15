CC        := gcc
CFLAGS    := -std=c17 -Wall -Wextra
CPPFLAGS  := -I./include/
LDFLAGS   :=
LDLIBS    :=

PROG := spmv
SRCS := $(wildcard src/*.c)
OBJS := $(patsubst %.c,%.o,$(SRCS))
DEPS := $(OBJS:.o=.d)

-include $(DEPS)


all: $(PROG)

$(PROG): $(OBJS)
	$(CC) $(LDFLAGS) $^ -o $@ $(LDLIBS)

clean:
	$(RM) $(OBJS) $(PROG) $(DEPS)

.PHONY: all clean
