#include "raytracer.h"
#include <math.h>

#define BLOCK_SIZE 16     // block size

static uchar4* kernelOutputBuffer;
static int g_screenWidth;
static int g_screenHeight;
static size_t g_bufferPitch;

__global__ void Kernel(uchar4* dst, int width, int height, size_t bufferPitch) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    // Invert Y because OpenGL
    int y = height - ((blockIdx.y * blockDim.y) + threadIdx.y + 1);

    float u = x / (float)width;
    float v = y / (float)height;

    int offset = (y * bufferPitch) + x * sizeof(uchar4);
    if (offset >= bufferPitch * height) return;

    *((uchar4*)(((uchar1*)dst) + offset)) = make_uchar4(u * 256.0f, v * 256.0f, 256.0f, 256.0f);
}

void rtResize(JNIEnv* env, int screenWidth, int screenHeight) {
    g_screenWidth = screenWidth;
    g_screenHeight = screenHeight;

    cudaError_t err;
    
    // Resize
    if (kernelOutputBuffer) {
        err = cudaFree(kernelOutputBuffer);
        if (err != cudaSuccess) {
            Log(env, std::string("cudaFree failed: ") + std::to_string(err));
        }
    }

    err = cudaMallocPitch((void**)&kernelOutputBuffer, &g_bufferPitch, g_screenWidth * sizeof(uchar4), g_screenHeight * sizeof(uchar4));
    if (err != cudaSuccess) {
        Log(env, std::string("cudaMalloc failed: ") + std::to_string(err));
    }
}

void rtRaytrace(JNIEnv*, cudaGraphicsResource_t glTexture, int texHeight) {
    unsigned int blocksW = (unsigned int)ceilf(g_screenWidth / (float)BLOCK_SIZE);
    unsigned int blocksH = (unsigned int)ceilf(g_screenHeight / (float)BLOCK_SIZE);
    dim3 gridDim(blocksW, blocksH, 1);
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);

    // Kernel call
    Kernel<<<gridDim, blockDim>>>(kernelOutputBuffer, g_screenWidth, g_screenHeight, g_bufferPitch);

    // Copy CUDA result to OpenGL texture
    cudaArray* mappedGLArray;
    cudaGraphicsSubResourceGetMappedArray(&mappedGLArray, glTexture, 0, 0);

    int width = g_screenWidth * sizeof(uchar4);
    cudaMemcpy2DToArray(
        mappedGLArray,              // dst
        0,                          // wOffset
        texHeight - g_screenHeight, // hOffset
        kernelOutputBuffer,         // src
        g_bufferPitch,              // spitch
        width,                      // width
        g_screenHeight,             // height
        cudaMemcpyDeviceToDevice    // kind
    );
}
