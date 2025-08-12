OBJDIR=build
SRCDIR=source
TRGDIR=examples

##############################################################################
# Flags to compile cuda code
# Arch defined by GPU Tesla V100S-32
ARCH=-arch=compute_70 -code=sm_70
CUDA=nvcc

CUDA_FLAGS=-g -G ${ARCH}# -O3 --use_fast_math # include these to slightly improve, not firmly tested
CUDA_LIBS = -lm -Xcompiler -fopenmp --relocatable-device-code true# -lcudart

CUDA_SOURCES = $(wildcard $(SRCDIR)/*.cu)
CUDA_OBJECTS = $(patsubst $(SRCDIR)/%.cu,$(OBJDIR)/%.o,$(CUDA_SOURCES))

#############################################################################%#
# General stuff
Likwid = -I/usr/local/include -DLIKWID_PWERFMON
L= -L/usr/local/lib -llikwid 

TARGETS_SOURCES = $(wildcard $(TRGDIR)/*.cu)
TARGETS = $(patsubst $(TRGDIR)/%.cu,%,$(TARGETS_SOURCES))

##############################################################################
 # default or empty
default: clean build
_: default

build: $(TARGETS)

# Linker
%: $(CUDA_SOURCES) $(TRGDIR)/%.cu
	$(CUDA) $(CUDA_FLAGS) $^ -o $@ $(CUDA_LIBS)

run:
	@echo "Running main example..."
	./main

.PRECIOUS: $(OBJDIR)/%.o

.PHONY: 
	clean

clean:
	@echo "Cleaning..."
	rm -f $(OBJDIR)/*.o $(TARGETS)

usage: help

help:
	@echo "Usage:"
	@echo "  make [target]"
	@echo "Targets:"
	@echo "            default|_|:	compile all. (default, leave empty)\n"
	@echo "        [example name]:	one of the examples without suffix. Build only the specified target.\n"
	@echo "                   run:	run main example\n"
	@echo "                 clean:	remove compiled files\n"
	@echo "            help|usage:	show this help"
	