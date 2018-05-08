#include "counting.h"
#include <cstdio>
#include <cassert>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

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

__global__ void Algo(const char* text,int *temp, int *pos, int text_size){
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < text_size) {
		if (text[i] == '\n')pos[i] = i;
		else pos[i] = -1;
		temp[i] = 0;
		while(i < text_size) {
			
	}
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
	Algo<<<(text_size/512) + 1, 512>>>(text, temp, pos, text_size);
}
