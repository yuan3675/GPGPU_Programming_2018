#include "lab1.h"
#include <time.h>
#include <stdlib.h>
static const unsigned W = 640;
static const unsigned H = 480;
static const unsigned NFRAME = 240;
double YUV[3];

struct Lab1VideoGenerator::Impl {
	int t = 0;
};

Lab1VideoGenerator::Lab1VideoGenerator(): impl(new Impl) {
}

Lab1VideoGenerator::~Lab1VideoGenerator() {}

void Lab1VideoGenerator::get_info(Lab1VideoInfo &info) {
	info.w = W;
	info.h = H;
	info.n_frame = NFRAME;
	// fps = 24/1 = 24
	info.fps_n = 24;
	info.fps_d = 1;
};


void Lab1VideoGenerator::Generate(uint8_t *yuv) {

	/*			drawing background			*/
	if ((impl->t) == 0) {
		//setup Y value
		cudaMemset(yuv, 0, W*H);
		//setup U value
		cudaMemset(yuv + W * H, 128, W*H/4);
		//setup V value
		cudaMemset(yuv+W*H+W*H/4, 128, W*H/4);
	}
	
	/*			drawing first layer			*/
	/*			drawing circle			*/
	/*
	int center[2] = {320, 240};
	int radius = impl->t;

	for (int i = (center[0] - radius); i <= (center[0] + radius); i ++) {
		for (int j = (center[1] - radius); j <= (center[1] + radius); j++) {
			int len = (i - center[0]) * (i - center[0]) + (j - center[1]) * (j - center[1]);
			if (len <= (radius * radius)) {
				//setup Y value
				cudaMemset(yuv + (i + (j*W)), 0.299*255, 1*1);
				//setup U value
				cudaMemset(yuv + W*H + (i/2 + (j/2)*(W/2)), -0.169*255 + 128, 1*1);
				//setup V value
				cudaMemset(yuv + W*H + W*H/4 + (i/2 + (j/2)*(W/2)), 0.5*255 + 128, 1*1);
			}
		}
	}
	*/

	int i = 0 + 2 * (impl->t);
	int j = 480 - 2 * (impl->t);
	//setup Y value
	cudaMemset(yuv + (i + (j*W)), 0.299 * 255, 1*1);
	//setup U value
	cudaMemset(yuv + W*H + (i/2 + (j/2)*(W/2)), -0.169 * 255 + 128, 1*1);
	//setup V value
	cudaMemset(yuv + W*H + W*H/4 + (i/2 + (j/2)*(W/2)), 0.5 * 255 + 128, 1*1);

}