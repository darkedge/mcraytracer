#include "raytracer.h"
#include "helper_math.h"
#include <texture_fetch_functions.h>

// BLOCK_SIZE^2 = max threads per SM / max active blocks
#define BLOCK_SIZE 8 // JOTARO

static uchar4* kernelOutputBuffer;
static int g_screenWidth;
static int g_screenHeight;
static size_t g_bufferPitch;

// TODO: test texture references (because texture objects are compute 3.0 only)
//texture<float, 1, cudaReadModeElementType> tex;

// Calculates t for a line starting from s
// to cross the next integer in terms of ds.
// Assume s = [0..1], ds is [-1..1]
__inline__ __device__ float FindFirstT(float s, float ds) {
    return (ds > 0 ? ceil(s) - s : s - floorf(s)) / fabsf(ds);
}

// Input is in grid coordinates
// Output is [0..1]
__inline__ __device__ float3 NormalizeGridPosition(float3 f) {
    return f - floorf(f);
}

// Transforms a point from world space to grid space [0..1].
__inline__ __device__ float3 WorldToGrid(float3 f) {
    return NormalizeGridPosition(f * (1 / 16.0f));
}

__device__ bool IntersectTriangle(float3* origin, float3* v0, float3* v1, float3* v2, float3* dir, float* out_distance) {
    float3 v0v1 = *v1 - *v0; // e1
    float3 v0v2 = *v2 - *v0; // e2
    float3 pvec = cross(*dir, v0v2); // P
    float det = dot(v0v1, pvec);

    if (det < 0.000001f) return false;

    float invDet = 1 / det;

    float3 tvec = *origin - *v0;
    float u = dot(tvec, pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return false;

    float3 qvec = cross(tvec, v0v1);
    float v = dot(*dir, qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return false;

    *out_distance = dot(v0v2, qvec) * invDet;

    return true;
}

__device__ bool IntersectQuad(float3* origin, float3* dir, Quad* quad, float* out_distance) {
//return false;
#if 1
    float3* v0 = &quad->vertices[0].pos;
    float3* v1 = &quad->vertices[1].pos;
    float3* v2 = &quad->vertices[2].pos;
    float3* v3 = &quad->vertices[3].pos;

    return IntersectTriangle(origin, v0, v1, v2, dir, out_distance) || IntersectTriangle(origin, v0, v2, v3, dir, out_distance);
#endif

#if 0
    // Get plane normal
    float3 normal = normalize(cross(*v1 - *v0, *v3 - *v0));

    //
    float denom = dot(normal, *dir);
    if (fabsf(denom) > 0.000001f) { // TODO: tweak epsilon
        *out_distance = dot(*v0 - *origin, normal) / denom;
        return true;
    }

    return false;
#endif
}

__global__ void Kernel(uchar4* dst, int width, int height, cudaTextureObject_t vertexBuffers, cudaTextureObject_t arraySizes, Viewport viewport, float3 entity, size_t bufferPitch) {
#if 1
    float3 direction;
    {
        int x = (blockIdx.x * blockDim.x) + threadIdx.x;
        // Invert Y because OpenGL
        int y = height - ((blockIdx.y * blockDim.y) + threadIdx.y + 1);
        float u = x / (float)width;
        float v = y / (float)height;

        float3 point = lerp(viewport.p0, viewport.p1, u) + lerp(viewport.p0, viewport.p2, v) - viewport.p0;
        direction = normalize(point - viewport.origin);
    }
#if 1
    // World space
    float3 origin = entity + viewport.origin;

    float distance = FLT_MAX;

    char renderChunkX = MAX_RENDER_DISTANCE;
    char renderChunkY = (char)floorf(origin.y / 16.0f);
    char renderChunkZ = MAX_RENDER_DISTANCE;

    // Transform origin to [0..1]
    // Same for all threads (precompute?)
    origin = WorldToGrid(origin);

    // = (-1/1) signs of vector dir
    char stepX = (direction.x < 0) ? -1 : 1;
    char stepY = (direction.y < 0) ? -1 : 1;
    char stepZ = (direction.z < 0) ? -1 : 1;

    // All positive values
    float tMaxX = FindFirstT(origin.x, direction.x);
    float tMaxY = FindFirstT(origin.y, direction.y);
    float tMaxZ = FindFirstT(origin.z, direction.z);

    // All positive values
    float deltaX = (float)stepX / direction.x;
    float deltaY = (float)stepY / direction.y;
    float deltaZ = (float)stepZ / direction.z;

    unsigned char checks = 0;
    float3 raypos = origin * 16.0f;
#if 1
    do {
        int renderChunk =
            renderChunkX * GRID_DIM * 16 +
            renderChunkZ * 16 +
            renderChunkY;
        
        if (checks < 255) checks += 5;

        // Opaque pass
#if 1
        //Quad* buffer = vertexBuffers[renderChunk];
        uint2 rval = tex1Dfetch<uint2>(vertexBuffers, renderChunk);
        Quad* buffer = (Quad*) (((size_t)rval.y << 32) | rval.x);
        int size = tex1Dfetch<int>(arraySizes, renderChunk);
        for (int j = 0; j < size; j++) {
        //for (int j = 0; j < arraySizes[renderChunk]; j++) {
            // Quads in buffer
            float dist = FLT_MAX;
#if 1
            if (IntersectQuad(&raypos, &direction, &buffer[j], &dist)) {
                if (dist < distance) {
                    // TODO: Remember quad for texturing etc
                    distance = dist;
                }
            }
#endif
        }
#endif

#define TODO_RENDER_DISTANCE 1
        if (tMaxX < tMaxY) {
            if (tMaxX < tMaxZ) {
                renderChunkX += stepX;
                if (renderChunkX < (MAX_RENDER_DISTANCE - TODO_RENDER_DISTANCE) || (renderChunkX > MAX_RENDER_DISTANCE + TODO_RENDER_DISTANCE)) break;
                tMaxX += deltaX;
                raypos = origin + tMaxX * direction;
            }
            else {
                renderChunkZ += stepZ;
                if (renderChunkZ < (MAX_RENDER_DISTANCE - TODO_RENDER_DISTANCE) || (renderChunkZ > MAX_RENDER_DISTANCE + TODO_RENDER_DISTANCE)) break;
                tMaxZ += deltaZ;
                raypos = origin + tMaxZ * direction;
            }
        } else {
            if (tMaxY < tMaxZ) {
                renderChunkY += stepY;
                if (renderChunkY < 0 || renderChunkY >= 16) break;
                tMaxY += deltaY;
                raypos = origin + tMaxY * direction;
            }
            else {
                renderChunkZ += stepZ;
                if (renderChunkZ < (MAX_RENDER_DISTANCE - TODO_RENDER_DISTANCE) || (renderChunkZ > MAX_RENDER_DISTANCE + TODO_RENDER_DISTANCE)) break;
                tMaxZ += deltaZ;
                raypos = origin + tMaxZ * direction;
            }
        }

        raypos = NormalizeGridPosition(raypos) * 16.0f;
    } while (distance == FLT_MAX);
#endif

    unsigned char val = distance != FLT_MAX ? 255 : 0;
#endif

#endif
    {
        int x = (blockIdx.x * blockDim.x) + threadIdx.x;
        // Invert Y because OpenGL
        int y = height - ((blockIdx.y * blockDim.y) + threadIdx.y + 1);
        int offset = (y * bufferPitch) + x * sizeof(uchar4);
        if (offset >= bufferPitch * height) return;
        dst = (uchar4*)(((char*)dst) + offset);
    }

    *dst = make_uchar4(val, checks, 255, 255);
}

void rtResize(JNIEnv* env, int screenWidth, int screenHeight) {
    g_screenWidth = screenWidth;
    g_screenHeight = screenHeight;

    // Resize
    if (kernelOutputBuffer) {
        CUDA_TRY(cudaFree(kernelOutputBuffer));
    }

    CUDA_TRY(cudaMallocPitch((void**)&kernelOutputBuffer, &g_bufferPitch, g_screenWidth * sizeof(uchar4), g_screenHeight * sizeof(uchar4)));
}

void rtRaytrace(JNIEnv* env, cudaGraphicsResource_t glTexture, int texHeight, cudaTextureObject_t devicePointers, cudaTextureObject_t arraySizes, const Viewport &viewport, const float3& viewEntity) {
    unsigned int blocksW = (unsigned int)ceilf(g_screenWidth / (float)BLOCK_SIZE);
    unsigned int blocksH = (unsigned int)ceilf(g_screenHeight / (float)BLOCK_SIZE);
    dim3 gridDim(blocksW, blocksH, 1);
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);

    // Kernel call
#if 0
    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    Kernel<<<gridDim, blockDim>>>(kernelOutputBuffer, g_screenWidth, g_screenHeight, devicePointers, arraySizes, viewport, viewEntity, g_bufferPitch);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    Log(env, std::to_string(time));
#endif
    Kernel<<<gridDim, blockDim>>>(kernelOutputBuffer, g_screenWidth, g_screenHeight, devicePointers, arraySizes, viewport, viewEntity, g_bufferPitch);

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
