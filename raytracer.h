#pragma once
#include <cuda_runtime_api.h>
#include <jni.h>
#include <string>

#define MAX_RENDER_DISTANCE 32
#define GRID_DIM (MAX_RENDER_DISTANCE + 1 + MAX_RENDER_DISTANCE)
#define VERTEX_SIZE_BYTES 28
#define DEVICE_PTRS_COUNT (GRID_DIM * GRID_DIM * 16)

// This requires a JNIEnv* called env
#ifdef _DEBUG
#define CUDA_TRY(expr)\
do {\
    cudaError err =  expr;\
    if (err != cudaSuccess) {\
        Log(env, std::string(#expr " failed: ") + std::to_string(err) + std::string(": ") + cudaGetErrorString(err));\
    }\
} while (0);
#else
#define CUDA_TRY(expr, ...) expr;
#endif

// sizeof(Vertex) should be VERTEX_SIZE_BYTES
struct Vertex {
    float3 pos;
    unsigned char rgba[4];
    float u0;
    float v0;
    short u1;
    short u2;
};

struct Quad {
    Vertex vertices[4];
};

struct Viewport {
    float3 origin;
    float3 p0; // Top-left
    float3 p1; // Top-right
    float3 p2; // Bottom-left
};

void rtRaytrace(JNIEnv*, cudaGraphicsResource_t glTexture, int texHeight, cudaTextureObject_t devicePointers, cudaTextureObject_t arraySizes, const Viewport &viewport, const float3& viewEntity);
void rtResize(JNIEnv* env, int w, int h);

void Log(JNIEnv*, const std::string&);
