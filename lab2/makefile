
.PHONY: all

all: a.out
a.out: counting.cu
	nvcc -std=c++11 -O2 -arch sm_50 -I ./ main.cu counting.cu

run: a.out
	./a.out

clean: 
	-rm -f a.out
