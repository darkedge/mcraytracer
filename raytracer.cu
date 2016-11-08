#include "raytracer.h"

#include <math.h>

#define BLOCK_SIZE 16     // block size

static uchar4* kernelOutputBuffer;
static size_t kernelOutputBufferSize;
static int width;
static int height;

__global__ void Kernel(uchar4* dst, int width, int height) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	float u = x / (float)width;
	float v = y / (float)height;

	int index = (x * width) + y;
	if (index >= width * height) return;

	dst[index] = make_uchar4(u * 255.0f, v * 255.0f, 255.0f, 255.0f);
}

void Resize(JNIEnv* env, int w, int h) {
	width = w;
	height = h;
	cudaError_t err;

	size_t newSize = width * height * sizeof(uchar4);
	if (kernelOutputBufferSize != newSize) {
		// Resize
		if (kernelOutputBuffer) {
			err = cudaFree(kernelOutputBuffer);
			if (err != cudaSuccess) {
				Log(env, std::string("cudaFree failed: ") + std::to_string(err));
			}
		}
		kernelOutputBufferSize = newSize;
		err = cudaMalloc((void**)&kernelOutputBuffer, kernelOutputBufferSize);
		if (err != cudaSuccess) {
			Log(env, std::string("cudaMalloc failed: ") + std::to_string(err));
		}
	}
}

void Raytrace(JNIEnv*, cudaGraphicsResource_t glTexture) {
	unsigned int blocksW = (unsigned int)ceilf(width / (float)BLOCK_SIZE);
	unsigned int blocksH = (unsigned int)ceilf(height / (float)BLOCK_SIZE);
	dim3 gridDim(blocksW, blocksH, 1);
	dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);
	Kernel <<<gridDim, blockDim>>>(kernelOutputBuffer, width, height);

	// Copy CUDA result to OpenGL texture
	cudaGraphicsMapResources(1, &glTexture);
	cudaArray* mappedGLArray;
	cudaGraphicsSubResourceGetMappedArray(&mappedGLArray, glTexture, 0, 0);
	cudaMemcpyToArray(mappedGLArray, 0, 0, kernelOutputBuffer, kernelOutputBufferSize, cudaMemcpyDeviceToDevice);
	cudaGraphicsUnmapResources(1, &glTexture);
}
