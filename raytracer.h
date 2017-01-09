#pragma once
#include <cuda_runtime_api.h>
#include <jni.h>
#include <string>

#define MAX_RENDER_DISTANCE 32
#define GRID_DIM (MAX_RENDER_DISTANCE + 1 + MAX_RENDER_DISTANCE)
#define VERTEX_SIZE_BYTES 28
#define DEVICE_PTRS_COUNT (GRID_DIM * GRID_DIM * 16 * 4)

void rtRaytrace(JNIEnv* env, cudaGraphicsResource_t dst, int texHeight);
void rtResize(JNIEnv* env, int w, int h);

void Log(JNIEnv*, const std::string&);


#define CUDA_TRY(x)\
do {\
    cudaError_t err = x;\
    if (err != cudaSuccess) {\
        Log(env, std::string(#x" failed, err = ") + std::to_string(err));\
    }\
} while (0);

