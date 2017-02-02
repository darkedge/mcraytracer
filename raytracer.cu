#include "raytracer.h"
#include "helper_math.h"
#include <texture_fetch_functions.h>

// BLOCK_SIZE^2 = max threads per SM / max active blocks
#define BLOCK_SIZE 8 // JOTARO
#define EPSILON 0.000001f

static uchar4* kernelOutputBuffer;
static int g_screenWidth;
static int g_screenHeight;
static size_t g_bufferPitch;

// Calculates t for a line starting from s
// to cross the next integer in terms of ds.
// Assume s = [0..1], ds is [-1..1]
inline __device__ float FindFirstT(float s, float ds) {
    return (ds > 0 ? ceilf(s) - s : s - floorf(s)) / fabsf(ds);
}

// Transforms a point from world space to grid space [0..1].
static float3 WorldToGrid(float3 f) {
    return fracf(f * (1 / 16.0f));
}

__global__ void Kernel(uchar4* dst, int width, int height, size_t bufferPitch, const Quad** __restrict__ vertexBuffers, const int* __restrict__ arraySizes, Viewport viewport, float3 origin, char renderChunkY) {
    __shared__ Quad quads[BLOCK_SIZE * BLOCK_SIZE]; // For caching quads, 64 * 4 * 28 = 7168 bytes
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

    // = (-1/1) signs of vector dir
    char4 step = make_char4(
        (direction.x < 0) ? -1 : 1,
        (direction.y < 0) ? -1 : 1,
        (direction.z < 0) ? -1 : 1,
        0
    );

    // All positive values
    float tMaxX = FindFirstT(origin.x, direction.x);
    float tMaxY = FindFirstT(origin.y, direction.y);
    float tMaxZ = FindFirstT(origin.z, direction.z);

    float3 raypos = origin * 16.0f;

    char4 renderChunk = make_char4(
        MAX_RENDER_DISTANCE,
        renderChunkY,
        MAX_RENDER_DISTANCE,
        threadIdx.y * BLOCK_SIZE + threadIdx.x
    );

    float distance = FLT_MAX;
    while (true) {
        int size;
        {
            const int index =
                (renderChunk.x * GRID_DIM << 4) +
                (renderChunk.z << 4) +
                renderChunk.y;
        
            // Opaque pass
            size = arraySizes[index];
            if (renderChunk.w < size) {
                memcpy(quads + renderChunk.w, vertexBuffers[index] + renderChunk.w, sizeof(Quad));
                #if 0
                Quad q = vertexBuffers[index][renderChunk.w];
                Pos4* dest = &quads[renderChunk.w];
                dest->v0 = q.vertices[0].pos;
                dest->v1 = q.vertices[1].pos;
                dest->v2 = q.vertices[2].pos;
                dest->v3 = q.vertices[3].pos;
                #endif
            }
        }
        __syncthreads();

        for (int j = 0; j < size; j++) {
            //Pos4 q = quads[j];
            // Triangle 1
            float3 v0v1 = quads[j].v1.pos - quads[j].v0.pos; // e1
            float3 v0v2 = quads[j].v2.pos - quads[j].v0.pos; // e2
            float3 pvec = cross(direction, v0v2); // P
            float det = dot(v0v1, pvec);

            if (det < EPSILON) continue; // Ray does not hit front face

            det = 1.0f / det;

            float3 tvec = raypos - quads[j].v0.pos;
            float u = dot(tvec, pvec) * det;
            if (!(u < 0.0f || u > 1.0f)) {
                float3 qvec = cross(tvec, v0v1);
                float v = dot(direction, qvec) * det;

                if (!(v < 0.0f || u + v > 1.0f)) {
                    float dist = dot(v0v2, qvec) * det;

                    if (dist < distance) {
                        distance = dist;
                    }

                    // Found a hit
                    continue;
                }
            }

            // Triangle 2
            // TODO: Optimize this further
            det = -det;
            tvec = raypos - quads[j].v2.pos;
            u = dot(tvec, pvec) * det;

            if (!(u < 0.0f || u > 1.0f)) {

                float3 qvec = cross(tvec, v0v1);
                float v = dot(direction, qvec) * det;

                if (!(v < 0.0f || u + v > 1.0f)) {

                    float dist = dot(v0v2, qvec) * det;

                    if (dist < distance) {
                        distance = dist;
                    }
                }
            }
        }
        if (distance != FLT_MAX) break;

#define TODO_RENDER_DISTANCE 1
        if (tMaxX < tMaxY) {
            if (tMaxX < tMaxZ) {
                renderChunk.x += step.x;
                if (renderChunk.x < (MAX_RENDER_DISTANCE - TODO_RENDER_DISTANCE) || (renderChunk.x > MAX_RENDER_DISTANCE + TODO_RENDER_DISTANCE)) break;
                tMaxX += step.x / direction.x;
                // raypos = tMaxX * direction + origin;
                raypos.x = __fmaf_rn(tMaxX, direction.x, origin.x);
                raypos.y = __fmaf_rn(tMaxX, direction.y, origin.y);
                raypos.z = __fmaf_rn(tMaxX, direction.z, origin.z);
            }
            else {
                renderChunk.z += step.z;
                if (renderChunk.z < (MAX_RENDER_DISTANCE - TODO_RENDER_DISTANCE) || (renderChunk.z > MAX_RENDER_DISTANCE + TODO_RENDER_DISTANCE)) break;
                tMaxZ += step.z / direction.z;
                // raypos = tMaxZ * direction + origin;
                raypos.x = __fmaf_rn(tMaxZ, direction.x, origin.x);
                raypos.y = __fmaf_rn(tMaxZ, direction.y, origin.y);
                raypos.z = __fmaf_rn(tMaxZ, direction.z, origin.z);
            }
        } else {
            if (tMaxY < tMaxZ) {
                renderChunk.y += step.y;
                if (renderChunk.y < 0 || renderChunk.y >= 16) break;
                tMaxY += step.y / direction.y;
                // raypos = tMaxY * direction + origin;
                raypos.x = __fmaf_rn(tMaxY, direction.x, origin.x);
                raypos.y = __fmaf_rn(tMaxY, direction.y, origin.y);
                raypos.z = __fmaf_rn(tMaxY, direction.z, origin.z);
            }
            else {
                renderChunk.z += step.z;
                if (renderChunk.z < (MAX_RENDER_DISTANCE - TODO_RENDER_DISTANCE) || (renderChunk.z > MAX_RENDER_DISTANCE + TODO_RENDER_DISTANCE)) break;
                tMaxZ += step.z / direction.z;
                // raypos = tMaxZ * direction + origin;
                raypos.x = __fmaf_rn(tMaxZ, direction.x, origin.x);
                raypos.y = __fmaf_rn(tMaxZ, direction.y, origin.y);
                raypos.z = __fmaf_rn(tMaxZ, direction.z, origin.z);
            }
        }

        raypos = fracf(raypos) * 16.0f;
    }

    {
        int x = (blockIdx.x * blockDim.x) + threadIdx.x;
        // Invert Y because OpenGL
        int y = height - ((blockIdx.y * blockDim.y) + threadIdx.y + 1);
        int offset = (y * bufferPitch) + (x << 2); // x * sizeof(uchar4)
        if (offset >= bufferPitch * height) return;
        dst = (uchar4*)(((char*)dst) + offset);
    }

    unsigned char val = distance != FLT_MAX ? 255 : 0;
    *dst = make_uchar4(val, 0, 255, 255);
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

void rtRaytrace(JNIEnv*, cudaGraphicsResource_t glTexture, int texHeight, void* devicePointers, void* arraySizes, const Viewport &viewport, const float3& viewEntity) {
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

    // World space
    float3 origin = viewEntity + viewport.origin;
    char renderChunkY = (char)floorf(origin.y / 16.0f);

    // Transform origin to [0..1]
    origin = WorldToGrid(origin);
    Kernel<<<gridDim, blockDim>>>(kernelOutputBuffer, g_screenWidth, g_screenHeight, g_bufferPitch, (const Quad**)devicePointers, (const int*)arraySizes, viewport, origin, renderChunkY);

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
