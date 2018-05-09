#include "counting.h"
#include <cstdio>
#include <cassert>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#define max(a, b)((a)>(b)?(a):(b))
#define odd(a)((a)%2!=0? 1: 0)
__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

struct convert : public thrust::unary_function<char, int> {
	__host__ __device__
	int operator()(char c)
	{
		if (c == '\n') return 0;
		return 1;
	}
};

__device__ int Lowbit(int x) {
	return x&(-x);
}

__global__ void BITAlgo(const char* text,int *temp, int *pos, int text_size){
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = i + 1;
	
	if (i < text_size) {
		temp[i] = 0;
		if (text[i] == '\n') pos[i] = i + 1;
		else pos[i] = -1;
		/* This for loop takes lots of time */
		if(pos[i] != -1) {
			while(j <= text_size) {
				temp[j-1] = max(temp[j-1], pos[i]);
				j += Lowbit(j);
			}
		}
		j = i + 1;
		pos[i] = temp[i];
		while (1) {
			j-=(j&-j);
			if (j <= 0)break;
			pos[i] = max(pos[i], temp[j]);
		}
		pos[i] = i + 1 - pos[i];
		if ( i > 1050 && i < 1100 ){
			if (text[i] != '\n') printf("%d %c %d %d\n", i + 1, text[i], temp[i], pos[i]);
			else printf("%d space %d %d\n", i + 1, temp[i], pos[i]);
		}
	}
}

__global__ void Algo(const char *text,int temp, int *pos, int text_size) {
}

void CountPosition1(const char *text, int *pos, int text_size)
{
	thrust::equal_to<int> binary_pred;
	thrust::plus<int> binary_op;
	thrust::device_ptr<const char> dev_ptr1(text);
	thrust::device_ptr<int> dev_ptr2(pos);
	/*    Replace space to 0, others to 1    */
	thrust::transform(thrust::device, dev_ptr1, dev_ptr1 + text_size, dev_ptr2, convert());
	/*    Segment prefix sum all position    */
	thrust::inclusive_scan_by_key(thrust::device, dev_ptr2, dev_ptr2 + text_size, dev_ptr2, dev_ptr2, binary_pred, binary_op);
}

void CountPosition2(const char *text, int *pos, int text_size)
{
	int *temp;
	cudaMalloc(&temp, sizeof(int)*text_size);
	BITAlgo<<<(text_size/512) + 1, 512>>>(text, temp, pos, text_size);
}
