#include "lab1.h"
#include <time.h>
#include <stdlib.h>
static const unsigned W = 640;
static const unsigned H = 480;
static const unsigned NFRAME = 360;

struct Lab1VideoGenerator::Impl {
	int t = 0;
};

Lab1VideoGenerator::Lab1VideoGenerator(): impl(new Impl) {
}

Lab1VideoGenerator::~Lab1VideoGenerator() {}

__device__ int getYValue(int R, int G, int B) {
	float Y = 0.299 * R + 0.587 * G + 0.114 * B;
	return Y;
}

__device__ int getUValue(int R, int G, int B) {
	float U = -0.169 * R - 0.331 * G + 0.5 * B + 128;
	return U;
}

__device__ int getVValue(int R, int G, int B) {
	float V = 0.5 * R - 0.419 * G - 0.081 * B + 128;
	return V;
}

__device__ void giveColor (uint8_t* yuv, int R, int G, int B, int i, int j) {
	int Y = getYValue(R, G, B);
	int U = getUValue(R, G, B);
	int V = getVValue(R, G, B);
	//draw Y
	memset(yuv + (i + (j*W)), Y, 1*1);
	//draw U
	memset(yuv + W*H + (i/2 + (j/2)*(W/2)), U, 1*1);
	//draw V
	memset(yuv + W*H + W*H/4 + (i/2 + (j/2)*(W/2)), V, 1*1);	
}

__global__ void drawCircle(uint8_t* yuv, int x, int y, int r, int R, int G, int B) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (i >= (x - r) && i <= (x + r) && j <= (y + r) && j >= (y - r)) { 		
		int len = (i - x) * (i - x) + (j - y) * (j - y);
		if (len <= r * r) {
			giveColor(yuv, R, G, B, i, j);
		}		
	}
}

__global__ void drawRect(uint8_t* yuv, int x1, int x2, int y1, int y2, int R, int G, int B) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i <= x2 && i >= x1 && j <= y2 && j >= y1) {		
		giveColor(yuv, R, G, B, i, j);
	}
}

void drawLieDownStick (uint8_t* yuv, int x, int y) {
	//first brick
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x, x + 19, y, y + 19, 25, 25, 112);
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x + 1, x + 18, y + 1, y + 18, 0, 191, 255);	
	//second brick
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x + 20, x + 39, y, y + 19, 25, 25, 112);
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x + 21, x + 38, y + 1, y + 18, 0, 191, 255);	
	//third brick
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x + 40, x + 59, y, y + 19, 25, 25, 112);
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x + 41, x + 58, y + 1, y + 18, 0, 191, 255);	
	//fourth brick
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x + 60, x + 79, y, y + 19, 25, 25, 112);
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x + 61, x + 78, y + 1, y + 18, 0, 191, 255);	
}

void drawStandUpStick (uint8_t* yuv, int x, int y) {
	//first brick
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x, x + 19, y, y + 19, 25, 25, 112);
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x + 1, x + 18, y + 1, y + 18, 0, 191, 255);	
	//second brick
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x, x + 19, y + 20, y + 39, 25, 25, 112);
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x + 1, x + 18, y + 21, y + 38, 0, 191, 255);	
	//third brick
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x, x + 19, y + 40, y + 59, 25, 25, 112);
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x + 1, x + 18, y + 41, y + 58, 0, 191, 255);	
	//fourth brick
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x, x + 19, y + 60, y + 79, 25, 25, 112);
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x + 1, x + 18, y + 61, y + 78, 0, 191, 255);	
}

void drawSquareBrick (uint8_t* yuv, int x, int y) {
	//first brick
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x, x + 19, y, y + 19, 205, 173, 0);
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x + 1, x + 18, y + 1, y + 18, 238, 201, 0);
	//second brick
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x, x + 19, y + 20, y + 39, 205, 173, 0);
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x + 1, x + 18, y + 21, y + 38, 238, 201, 0);
	//third brick
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x + 20, x + 39, y, y + 19, 205, 173, 0);
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x + 21, x + 38, y + 1, y + 18, 238, 201, 0);
	//foutrh brick
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x + 20, x + 39, y + 20, y + 39, 205, 173, 0);
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x + 21, x + 38, y + 21, y + 38, 238, 201, 0);
}

void drawOppositeZ (uint8_t* yuv, int x, int y) {
	//first brick
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x, x + 19, y, y + 19, 154, 205, 50);
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x + 1, x + 18, y + 1, y + 18, 192, 255, 62);
	//second brick
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x + 20, x + 39, y, y + 19, 154, 205, 50);
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x + 21, x + 38, y + 1, y + 18, 192, 255, 62);
	//third brick
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x, x + 19, y + 20, y + 39, 154, 205, 50);
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x + 1, x + 18, y + 21, y + 38, 192, 255, 62);
	//foutrh brick
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x - 20, x - 1, y + 20, y + 39, 154, 205, 50);
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x - 19, x - 2, y + 21, y + 38, 192, 255, 62);
}

void drawZ (uint8_t* yuv, int x, int y) {
	//first brick
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x, x + 19, y, y + 19, 154, 205, 50);
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x + 1, x + 18, y + 1, y + 18, 192, 255, 62);
	//second brick
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x + 20, x + 39, y, y + 19, 154, 205, 50);
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x + 21, x + 38, y + 1, y + 18, 192, 255, 62);
	//third brick
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x + 20, x + 39, y + 20, y + 39, 154, 205, 50);
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x + 21, x + 38, y + 21, y + 38, 192, 255, 62);
	//foutrh brick
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x + 40, x + 59, y + 20, y + 39, 154, 205, 50);
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x + 41, x + 58, y + 21, y + 38, 192, 255, 62);
}

void drawFlash (uint8_t* yuv, int x, int y) {
	//first brick
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x, x + 19, y, y + 19, 154, 205, 50);
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x + 1, x + 18, y + 1, y + 18, 192, 255, 62);
	//second brick
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x, x + 19, y + 20, y + 39, 154, 205, 50);
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x + 1, x + 18, y + 21, y + 38, 192, 255, 62);
	//third brick
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x + 20, x + 39, y + 20, y + 39, 154, 205, 50);
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x + 21, x + 38, y + 21, y + 38, 192, 255, 62);
	//foutrh brick
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x + 20, x + 39, y + 40, y + 59, 154, 205, 50);
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x + 21, x + 38, y + 41, y + 58, 192, 255, 62);
}

void drawOppositeFlash (uint8_t* yuv, int x, int y) {
	//first brick
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x, x + 19, y, y + 19, 154, 205, 50);
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x + 1, x + 18, y + 1, y + 18, 192, 255, 62);
	//second brick
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x + 20, x + 39, y, y + 19, 154, 205, 50);
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x + 21, x + 38, y + 1, y + 18, 192, 255, 62);
	//third brick
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x, x + 19, y + 20, y + 39, 154, 205, 50);
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x + 1, x + 18, y + 21, y + 38, 192, 255, 62);
	//foutrh brick
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x + 20, x + 39, y - 20 , y - 1, 154, 205, 50);
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x + 20, x + 39, y - 19, y - 2, 192, 255, 62);
}

void drawFinalBrick (uint8_t* yuv, int y) {
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, 160, 179, y, y + 19, 255, 0, 0); 	
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, 161, 178, y + 1, y + 18, 255, 99, 71); 	
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, 180, 199, y, y + 19, 255, 0, 0); 	
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, 181, 198, y + 1, y + 18, 255, 99, 71); 	
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, 160, 179, y + 20, y + 39, 255, 0, 0); 	
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, 161, 178, y + 21, y + 38, 255, 99, 71); 	
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, 180, 199, y + 20, y + 39, 255, 0, 0); 	
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, 181, 198, y + 21, y + 38, 255, 99, 71); 	
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, 200, 219, y, y + 19, 255, 0, 0);
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, 201, 218, y + 1, y + 18, 255, 99, 71); 	
	
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, 260, 279, y, y + 19, 255, 0, 0); 	
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, 261, 278, y + 1, y + 18, 255, 99, 71); 	
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, 280, 299, y, y + 19, 255, 0, 0); 	
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, 281, 298, y + 1, y + 18, 255, 99, 71); 	
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, 300, 319, y, y + 19, 255, 0, 0); 	
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, 301, 318, y + 1, y + 18, 255, 99, 71); 	
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, 320, 339, y, y + 19, 255, 0, 0); 	
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, 321, 338, y + 1, y + 18, 255, 99, 71); 	
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, 340, 359, y, y + 19, 255, 0, 0); 	
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, 341, 358, y + 1, y + 18, 255, 99, 71); 	
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, 360, 379, y, y + 19, 255, 0, 0); 	
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, 361, 378, y + 1, y + 18, 255, 99, 71); 	
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, 280, 299, y + 20, y + 39, 255, 0, 0); 	
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, 281, 298, y + 21, y + 38, 255, 99, 71); 	
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, 300, 319, y + 20, y + 39, 255, 0, 0); 	
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, 301, 318, y + 21, y + 38, 255, 99, 71); 	
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, 320, 339, y + 20, y + 39, 255, 0, 0); 	
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, 321, 338, y + 21, y + 38, 255, 99, 71); 	
	
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, 400, 419, y, y + 19, 255, 0, 0); 	
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, 401, 418, y + 1, y + 18, 255, 99, 71); 	
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, 420, 439, y, y + 19, 255, 0, 0); 	
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, 421, 438, y + 1, y + 18, 255, 99, 71); 	
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, 440, 459, y, y + 19, 255, 0, 0); 	
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, 441, 458, y + 1, y + 18, 255, 99, 71); 	
}

void drawFeet(uint8_t* yuv, int x, int y) {
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x, x + 49, y, y + 59, 0, 0, 139);
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x + 89, x + 139, y, y + 59, 0, 0, 139);
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x - 20, x + 49, y + 60, y + 79, 0, 0, 139);
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x + 89, x + 159, y + 60, y + 79, 0, 0, 139);
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x - 40, x + 49, y + 80, y + 99, 0, 0, 139);
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x + 89, x + 179, y + 80, y + 99, 0, 0, 139);	
}

void drawLegs(uint8_t* yuv, int x, int y) {
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x, x + 109, y, y + 39, 130, 130, 130);
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x, x + 29, y + 40, y + 139, 238, 233, 233);
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x + 80, x + 109, y + 40, y + 139, 238, 233, 233);
}

void drawBody(uint8_t* yuv, int x, int y) {
	for (int i = 0; i < 100; i++) {
		if ( i > 80) {
			drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x + 30, x + 129, y + i, y + i, 130, 130, 130);
		}
		else {
			drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x + i/3, x + 159 - i/3 , y + i, y + i, 130, 130, 130);
		}
	}
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x + 35, x + 124, y + 100, y + 129, 238, 233, 233);
	for (int i = 0; i < 20; i++){
		drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x + 80 - i, x + 80 + i, y + 110 + i, y + 110 + i, 255, 215, 0);
		drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x + 80 - i, x + 80 + i, y + 149 - i, y + 149 - i, 255, 215, 0);
	}
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x - 10, x + 9, y + 28, y + 67, 238, 233, 233);
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x + 149, x + 168, y + 28, y + 67, 238, 233, 233);
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x + 144, x + 173, y + 68, y + 107, 0, 0, 139);
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x - 15, x + 14, y + 68, y + 107, 0, 0, 139);
	
	drawCircle<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x, y + 117, 30, 0, 0, 139);
	drawCircle<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x + 159, y + 117, 30, 0, 0, 139);
	
	drawCircle<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x, y, 30, 130, 130, 130);
	drawCircle<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x + 159, y, 30, 130, 130, 130);
}

void drawHead(uint8_t* yuv, int x, int y) {
	//head
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x, x + 59, y, y + 59, 238, 233, 233);
	//mouth
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x + 10, x + 49, y + 35, y + 54, 130, 130, 130);
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x + 13, x + 21, y + 38, y + 51, 0, 0, 0);
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x + 26, x + 34, y + 38, y + 51, 0, 0, 0);
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x + 38, x + 46, y + 38, y + 51, 0, 0, 0);
	//eyes
	for (int i = 0; i < 10; i++) {
		drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x + i, x + 19 + i, y + 10 + i, y + 10 + i, 255, 255, 0);
		drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x + 40 - i, x + 59 - i, y + 10 + i, y + 10 + i, 255, 255, 0);
	}
	//ears
	for (int i = 0; i < 20; i++) {
		drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x - 20 + i, x - 20 + i, y + 15 - i/4, y + 15 + i/4, 139, 0, 0);
		drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x + 79 - i, x + 79 - i, y + 15 - i/4, y + 15 + i/4, 139, 0, 0);
	}
	//hamlet
	for (int i = 0; i < 20; i++) {
		drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x - 10 + i/2, x + 69 - i/2, y - 20 + i, y - 20 + i, 238, 233, 233);
	}
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x - 10, x + 9, y - 40, y -21, 238, 233, 233);		
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, x + 50, x + 69, y - 40, y -21, 238, 233, 233);		
}

void Lab1VideoGenerator::get_info(Lab1VideoInfo &info) {
	info.w = W;
	info.h = H;
	info.n_frame = NFRAME;
	// fps = 24/1 = 24
	info.fps_n = 24;
	info.fps_d = 1;
};


void Lab1VideoGenerator::Generate(uint8_t *yuv) {
	//////////////////////////	layer1	//////////////////////////////	
	//draw background
	//setup Y value
	cudaMemset(yuv, 0, W*H);
	//setup U value
	cudaMemset(yuv + W * H, 128, W*H/4);
	//setup V value
	cudaMemset(yuv+W*H+W*H/4, 128, W*H/4);
	
	//////////////////////////	layer2	//////////////////////////////
	// draw border	
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, 155, 159, 0, 479, 255, 250, 250);
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, 460, 464, 0, 479, 255, 250, 250);
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, 25, 124, 50, 149, 255, 250, 250);
	drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, 30, 119, 55, 144, 0, 0, 0);
		
	//////////////////////////	layer3	//////////////////////////////
	
	switch (impl->t / 24) {
		case 0: 
			//draw next brick
			drawSquareBrick(yuv, 55, 80);			
			//draw falling brick
			if ((impl->t / 4) == 3) {
				drawLieDownStick(yuv, 220, 40 + (impl->t / 4) * 70);
			}
			else if ((impl->t / 4) > 3) {	
				drawLieDownStick(yuv, 160, 40 + (impl->t / 4) * 70);
			}
			else {	
				drawLieDownStick(yuv, 280, 40 + (impl->t / 4) * 70);
			}
			break;
		case 1:
			// draw existed bricks
			drawLieDownStick(yuv, 160, 460);
			
			//draw next brick
			drawOppositeZ(yuv, 65, 80);

			//draw falling brick
			if ((impl->t / 4) == 8) {
				drawSquareBrick(yuv, 270, 40 + ((impl->t / 4) - 6)  * 70);
			}
			else if ((impl->t / 4) > 8) {	
				drawSquareBrick(yuv, 240, 40 + ((impl->t / 4) - 6) * 70);
			}
			else {	
				drawSquareBrick(yuv, 300, 40 + ((impl->t / 4) - 6) * 70);
			}
			break;
		case 2:
			//draw existed bricks
			drawLieDownStick(yuv, 160, 460);
			drawSquareBrick(yuv, 240, 440);
			
			//draw next brick
			drawStandUpStick(yuv, 65, 60);

			//draw falling brick
			if ((impl->t / 4) == 14) {
				drawOppositeZ(yuv, 265, 40 + ((impl->t / 4) - 12)  * 70);
			}
			else if ((impl->t / 4) > 14) {	
				drawOppositeZ(yuv, 220, 40 + ((impl->t / 4) - 12) * 70);
			}
			else {	
				drawOppositeZ(yuv, 310, 40 + ((impl->t / 4) - 12) * 70);
			}

			break;
		case 3:
			//draw existed bricks
			drawLieDownStick(yuv, 160, 460);
			drawSquareBrick(yuv, 240, 440);
			drawOppositeZ(yuv, 220, 420);

			//draw next brick
			drawZ(yuv, 45, 80);
			
			//draw falling brick
			if ((impl->t / 4) >= 20) {
				drawLieDownStick(yuv, 280, 40 + ((impl->t / 4) - 18) * 70);
			}
			else {	
				drawStandUpStick(yuv, 310, 40 + ((impl->t / 4) - 18) * 70);
			}

			break;
		case 4:
			//draw existed bricks
			drawLieDownStick(yuv, 160, 460);
			drawSquareBrick(yuv, 240, 440);
			drawOppositeZ(yuv, 220, 420);
			drawLieDownStick(yuv, 280, 460);			

			//draw next brick
			drawFlash(yuv, 55, 70);

			//draw falling brick
			if ((impl->t / 4) >= 27) {
				drawZ(yuv, 340, 40 + ((impl->t / 4) - 24)  * 70);
			}
			else {	
				drawZ(yuv, 290, 40 + ((impl->t / 4) - 24) * 70);
			}

			break;
		case 5:
			//draw existed bricks
			drawLieDownStick(yuv, 160, 460);
			drawSquareBrick(yuv, 240, 440);
			drawOppositeZ(yuv, 220, 420);
			drawLieDownStick(yuv, 280, 460);
			drawZ(yuv, 340, 440);

			//draw next brick
			drawSquareBrick(yuv, 55, 80);			
			
			//draw falling brick
			if ((impl->t / 4) == 33) {
				drawFlash(yuv, 340, 40 + ((impl->t / 4) - 30) * 70);
			}
			else if ((impl->t / 4) > 33) {
				drawFlash(yuv, 380, 40 + ((impl->t / 4) - 30) * 70);
			}
			else {	
				drawFlash(yuv, 300, 40 + ((impl->t / 4) - 30) * 70);
			}

			break;
		case 6:
			//draw existed bricks
			drawLieDownStick(yuv, 160, 460);
			drawSquareBrick(yuv, 240, 440);
			drawOppositeZ(yuv, 220, 420);
			drawLieDownStick(yuv, 280, 460);
			drawZ(yuv, 340, 440);
			drawFlash(yuv, 380, 420);			

			//draw next brick
			
			//draw falling brick
			if ((impl->t / 4) == 38) {	
				drawSquareBrick(yuv, 340, 40 + ((impl->t / 4) - 36) * 70);
			}
			else if ((impl->t / 4) == 39) {
				drawSquareBrick(yuv, 380, 40 + ((impl->t / 4) - 36) * 70);
			}
			else if ((impl->t / 4) > 39) {
				drawSquareBrick(yuv, 420, 40 + ((impl->t / 4) - 36) * 70);
			}
			else {	
				drawSquareBrick(yuv, 300, 40 + ((impl->t / 4) - 36) * 70);
			}

			break;
		case 7:
			//draw existed brick
			drawLieDownStick(yuv, 160, 460);
			drawSquareBrick(yuv, 240, 440);
			drawOppositeZ(yuv, 220, 420);
			drawLieDownStick(yuv, 280, 460);
			drawZ(yuv, 340, 440);
			drawFlash(yuv, 380, 420);
			drawSquareBrick(yuv, 420, 440);
	
			//draw next brick
			
			// erase a line
			if ((impl->t % 24 < 6) || ((impl->t % 24 >= 12) && (impl->t % 24 < 18))) {
				drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, 160, 459, 460, 479, 0, 0, 0);
			}
			break;
		case 8:
			//draw existing bricks
			drawOppositeZ(yuv, 220, 440);
			// draw half square
			drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, 240, 240 + 19, 460, 460 + 19, 205, 173, 0);
			drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, 240 + 1, 240 + 18, 460 + 1, 460 + 18, 238, 201, 0);
			drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, 260, 260 + 19, 460, 460 + 19, 205, 173, 0);
			drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, 260 + 1, 260 + 18, 460 + 1, 460 + 18, 238, 201, 0);
			// draw z part
			drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, 400, 400 + 19, 460, 460 + 19, 154, 205, 50);
			drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, 400 + 1, 400 + 18, 460 + 1, 460 + 18, 192, 255, 62);
			drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, 380, 380 + 19, 460, 460 + 19, 154, 205, 50);
			drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, 380 + 1, 380 + 18, 460 + 1, 460 + 18, 192, 255, 62);
			drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, 380, 380 + 19, 440, 440 + 19, 154, 205, 50);
			drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, 380 + 1, 380 + 18, 440 + 1, 440 + 18, 192, 255, 62);
			drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, 360, 360 + 19, 460, 460 + 19, 154, 205, 50);
			drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, 360 + 1, 360 + 18, 460 + 1, 460 + 18, 192, 255, 62);
			drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, 340, 340 + 19, 460, 460 + 19, 154, 205, 50);
			drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, 340 + 1, 340 + 18, 460 + 1, 460 + 18, 192, 255, 62);

			// draw half square
			drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, 420, 420 + 19, 460, 460 + 19, 205, 173, 0);
			drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, 420 + 1, 420 + 18, 460 + 1, 460 + 18, 238, 201, 0);
			drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, 440, 440 + 19, 460 , 460 + 19, 205, 173, 0);
			drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, 440 + 1, 440 + 18, 460 + 1, 460 + 18, 238, 201, 0);

			//draw next brick

			//draw falling brick
			drawFinalBrick(yuv, 40 + ((impl->t / 4) - 48) * 70);
			
			break;
		case 9:
			//draw existing bricks
			drawOppositeZ(yuv, 220, 440);
			// draw half square
			drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, 240, 240 + 19, 460, 460 + 19, 205, 173, 0);
			drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, 240 + 1, 240 + 18, 460 + 1, 460 + 18, 238, 201, 0);
			drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, 260, 260 + 19, 460, 460 + 19, 205, 173, 0);
			drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, 260 + 1, 260 + 18, 460 + 1, 460 + 18, 238, 201, 0);
			// draw z part
			drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, 400, 400 + 19, 460, 460 + 19, 154, 205, 50);
			drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, 400 + 1, 400 + 18, 460 + 1, 460 + 18, 192, 255, 62);
			drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, 380, 380 + 19, 460, 460 + 19, 154, 205, 50);
			drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, 380 + 1, 380 + 18, 460 + 1, 460 + 18, 192, 255, 62);
			drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, 380, 380 + 19, 440, 440 + 19, 154, 205, 50);
			drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, 380 + 1, 380 + 18, 440 + 1, 440 + 18, 192, 255, 62);
			drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, 360, 360 + 19, 460, 460 + 19, 154, 205, 50);
			drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, 360 + 1, 360 + 18, 460 + 1, 460 + 18, 192, 255, 62);
			drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, 340, 340 + 19, 460, 460 + 19, 154, 205, 50);
			drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, 340 + 1, 340 + 18, 460 + 1, 460 + 18, 192, 255, 62);

			// draw half square
			drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, 420, 420 + 19, 460, 460 + 19, 205, 173, 0);
			drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, 420 + 1, 420 + 18, 460 + 1, 460 + 18, 238, 201, 0);
			drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, 440, 440 + 19, 460, 460 + 19, 205, 173, 0);
			drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, 440 + 1, 440 + 18, 460 + 1, 460 + 18, 238, 201, 0);
			
			//draw final brick
			drawFinalBrick(yuv, 440);
			
			//draw next brick

			//erase lines
			if ((impl->t % 24 > 17) || ((impl->t % 24 >= 6) && (impl->t % 24 < 12))) {
				drawRect<<<dim3((W - 1)/16+1, (H-1)/12+1), dim3(16, 12)>>>(yuv, 160, 459, 440, 479, 0, 0, 0);
			}

			break;
		case 10:
			//draw existed bricks
			
			//draw falling brick
			drawFeet(yuv, 240, 40 + ((impl->t / 4) - 60) * 60);
			
			break;
		case 11:
			//draw existed bricks
			drawFeet(yuv, 240, 380);

			//draw falling brick
			drawLegs(yuv, 250, 40 + ((impl->t / 4) - 66) * 40);
			break;
		case 12:
			//draw existed bricks
			drawFeet(yuv, 240, 380);
			drawLegs(yuv, 250, 240);
	
			//draw falling brick
			drawBody(yuv, 225, 0 + ((impl->t / 4) - 72) * 25);

			break;

		case 13:
			//draw existed bricks 		
			drawFeet(yuv, 240, 380);
			drawLegs(yuv, 250, 240);
			drawBody(yuv, 225, 110);
			
			//draw falling brick
			drawHead(yuv, 275, 0 + ((impl->t / 4) - 78) * 10); 
			break;
		default:
			//draw existed bricks 		
			drawFeet(yuv, 240, 380);
			drawLegs(yuv, 250, 240);
			drawBody(yuv, 225, 110);
			drawHead(yuv, 275, 50);
			break; 
	}
	++ (impl->t);
}
