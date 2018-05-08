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
}
