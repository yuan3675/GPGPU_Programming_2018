
.PHONY: all

all: a.out
a.out: lab3.cu
	nvcc -std=c++11 main.cu lab3.cu pgm.cpp

run1: a.out
	./a.out img_background.ppm img_target.ppm img_mask.pgm 130 600 output.ppm

run2: a.out
	./a.out img_background.ppm img_target.ppm img_mask.pgm 130 900 output.ppm

clean: 
	-rm -f a.out
