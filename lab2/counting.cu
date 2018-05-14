#include "counting.h"
#include <cstdio>
#include <cassert>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#define max(a, b)((a)>(b)?(a):(b))
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

__global__ void SimpleAlgo(const char* text, int *pos, int text_size) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < text_size) {
		if (text[i] == '\n') {
			pos[i] = 0;
			int j = i + 1;
			int counter = 1;
			while(text[j] != '\n' && j < text_size){
				pos[j] = counter;
				counter ++;
				j ++;
			}
		}
		else if (i == 0 && text[i] != '\n') {
			int j = 0;
			int counter = 1;
			while (text[j] != '\n' && i < text_size) {
				pos[j] = counter;
				counter ++;
				j ++;
			}
		}
	}
}

__global__ void Algo(const char *text,int *pos, int text_size) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int tid = threadIdx.x;
	__shared__ int data[512];

	if (i < text_size) {
		if (text[i] == '\n') data[tid] = 0;
		else data[tid] = 1;
		__syncthreads();
		
		if (data[tid] != 0) {
			for (int j = 0; j < 9; j ++) {
				int look = data[tid];
				if ((tid - look) < 0)break;
				data[tid] = data[tid] + data[tid - look];
				__syncthreads();
			}
		}
		__syncthreads();
		
		pos[i] = data[tid];
	}
}

__global__ void Final(int *pos, int text_size) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (pos[i] != 0 ) {
		int look = pos[i];
		if (i - look >= 0) atomicAdd(&pos[i], pos[i - look]);
	}
}


__global__ void BITAlgo(const char* text, int *pos, int text_size){
	const int tid = threadIdx.x;
	const int bid = blockIdx.x;
	__shared__ int data[512];
	const int i = bid * blockDim.x + tid;

	/* if current thread is out of index, we won't do it */
	if (i < text_size) {
		/* Initialize BIT array */
		pos[i] = 0;
		/* Initialize original sequence */
		if (text[i] == '\n') {
			data[tid] = i + 1;
		}
		else {
			data[tid] = 0;
		}
		__syncthreads();

		/* Construct Max BIT */
		//if(data[i] != -1) {

		// Set recurrent index
		int j = tid + 1;

		int val = data[tid];
		while(j <= 512) {
			atomicMax(&data[j - 1], val);
			j += Lowbit(j);
			__syncthreads();
		}
		
		/* Compute interval max */
		j = tid + 1;
		int ans = 0;
		while (j >= 1) {
			ans = max(data[j-1], ans);
			j -= Lowbit(j);
			__syncthreads();
		}

		pos[i] = ans;
	}
}

__global__ void SumBIT(int *pos, int text_size){
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int bid = blockIdx.x;

	if (i < text_size) {
		if (bid != 0){
			int j = (bid - 1) * blockDim.x + 511;
			atomicMax(&pos[i], pos[j]);
		}
	}	
}

__global__ void ConvertBIT(int *pos, int text_size){
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < text_size)pos[i] = i + 1 - pos[i]; 
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
	//int *temp;
	//cudaMalloc(&temp, sizeof(int) * text_size);
	int blocks = (text_size + 511)/512;
	BITAlgo<<<blocks, 512>>>(text, pos, text_size);
	SumBIT<<<blocks, 512>>>(pos, text_size);
	ConvertBIT<<<blocks, 512>>>(pos, text_size);
}
