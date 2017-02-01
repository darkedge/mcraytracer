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

// TODO: test texture references (because texture objects are compute 3.0 only)
//texture<float, 1, cudaReadModeElementType> tex;

// Calculates t for a line starting from s
// to cross the next integer in terms of ds.
// Assume s = [0..1], ds is [-1..1]
__inline__ __device__ float FindFirstT(float s, float ds) {
    return (ds > 0 ? ceilf(s) - s : s - floorf(s)) / fabsf(ds);
}

// Input is in grid coordinates
// Output is [0..1]
__inline__ __host__ __device__ float3 NormalizeGridPosition(float3 f) {
    return f - floorf(f);
}

// Transforms a point from world space to grid space [0..1].
static float3 WorldToGrid(float3 f) {
    return NormalizeGridPosition(f * (1 / 16.0f));
}

#if 0
__device__ bool IntersectTriangle(const float3* const origin, const float3* const v0, const float3* const v1, const float3* const v2, const float3* const dir, float* out_distance) {
    const float3 v0v1 = *v1 - *v0; // e1
    const float3 v0v2 = *v2 - *v0; // e2
    const float3 pvec = cross(*dir, v0v2); // P
    float det = dot(v0v1, pvec);

    if (det < 0.000001f) return false;

    det = 1.0f / det;

    float3 tvec = *origin - *v0;
    const float u = dot(tvec, pvec) * det;
    if (u < 0.0f || u > 1.0f) return false;

    tvec = cross(tvec, v0v1);
    const float v = dot(*dir, tvec) * det;
    if (v < 0.0f || u + v > 1.0f) return false;

    *out_distance = dot(v0v2, tvec) * det;

    return true;
}

__device__ bool IntersectQuad(const float3* const origin, const float3* const dir, const Quad* const quad, float* const out_distance) {
    const float3* const v0 = &quad->vertices[0].pos;
    const float3* const v1 = &quad->vertices[1].pos;
    const float3* const v2 = &quad->vertices[2].pos;
    const float3* const v3 = &quad->vertices[3].pos;

    //return IntersectTriangle(origin, v0, v1, v2, dir, out_distance) || IntersectTriangle(origin, v0, v2, v3, dir, out_distance);
}
#endif

__global__ void Kernel(uchar4* dst, int width, int height, size_t bufferPitch, Quad** vertexBuffers, int* arraySizes, Viewport viewport, float3 origin, char renderChunkY) {
    __shared__ Pos4 quads[BLOCK_SIZE * BLOCK_SIZE]; // For caching quads, TODO: Can fit 438!
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
    int size = 255;
    while (true) {
        //int size;
        {
            const int index =
                (renderChunk.x * GRID_DIM << 4) +
                (renderChunk.z << 4) +
                renderChunk.y;
        
            // Opaque pass
            size = arraySizes[index];
            if (renderChunk.w < size) {
                Quad q = vertexBuffers[index][renderChunk.w];
                quads[renderChunk.w].v0 = q.vertices[0].pos;
                quads[renderChunk.w].v1 = q.vertices[1].pos;
                quads[renderChunk.w].v2 = q.vertices[2].pos;
                quads[renderChunk.w].v3 = q.vertices[3].pos;
            }
        }
        __syncthreads();

        for (int j = 0; j < size; j++) {
            float3 v0v1 = quads[j].v1 - quads[j].v0; // e1
            float3 v0v2 = quads[j].v2 - quads[j].v0; // e2
            float3 pvec = cross(direction, v0v2); // P
            float det = dot(v0v1, pvec);

            if (det < EPSILON) goto next;

            det = 1.0f / det;

            float3 tvec = raypos - quads[j].v0;
            float u = dot(tvec, pvec) * det;
            if (u < 0.0f || u > 1.0f) goto next;

            float3 qvec = cross(tvec, v0v1);
            float v = dot(direction, qvec) * det;
            if (v < 0.0f || u + v > 1.0f) goto next;

            float dist = dot(v0v2, qvec) * det;
            if (dist < distance) {
                distance = dist;
            }
            continue;
next:
            //v0v1 = -v0v1;
            //v0v2 = -v0v2;
            //pvec = -pvec;
            v0v1 = quads[j].v3 - quads[j].v2;
            v0v2 = quads[j].v0 - quads[j].v2;
            pvec = cross(direction, v0v2); // P
            det = dot(v0v1, pvec);

            if (det < EPSILON) continue;

            det = 1.0f / det;

            tvec = raypos - quads[j].v2;
            u = dot(tvec, pvec) * det;
            if (u < 0.0f || u > 1.0f) continue;

            qvec = cross(tvec, v0v1);
            v = dot(direction, qvec) * det;
            if (v < 0.0f || u + v > 1.0f) continue;

            dist = dot(v0v2, qvec) * det;
            if (dist < distance) {
                distance = dist;
            }
        #if 0
            // Triangle 1
            float3 v0v1 = quads[j].v1 - quads[j].v0; // e1
            float3 v0v2 = quads[j].v2 - quads[j].v0; // e2
            float3 pvec = cross(direction, v0v2); // P
            float det = dot(v0v1, pvec);

            if (det < EPSILON) continue; // Ray does not hit front of quad

            det = 1.0f / det;

            
            float3 tvec = raypos - quads[j].v0;
            float u = dot(tvec, pvec) * det;
            if (u < 0.0f || u > 1.0f) goto next;
                
            float3 qvec = cross(tvec, v0v1);
            {
                float v = dot(direction, tvec) * det;
                if (v < 0.0f || u + v > 1.0f) goto next;
            }

            {
                float dist = dot(v0v2, qvec) * det;
                if (dist < distance) {
                    // TODO: Remember quad for texturing etc
                    distance = dist;
                }
            }
            continue;

            // Triangle 2
next:
            v0v1 = quads[j].v3 - quads[j].v0; // e1
            pvec = cross(direction, v0v1); // P
            det = dot(v0v2, pvec); //

            det = 1.0f / det;

            {
                float3 tvec = raypos - quads[j].v0;
                float u = dot(tvec, pvec) * det;
                if (u < 0.0f || u > 1.0f) continue;

                float3 qvec = cross(tvec, v0v2);
                {
                    float v = dot(direction, tvec) * det;
                    if (v < 0.0f || u + v > 1.0f) continue;
                }

                {
                    float dist = dot(v0v1, qvec) * det;
                    if (dist < distance) {
                        // TODO: Remember quad for texturing etc
                        distance = dist;
                    }
                }
            }
            continue;
            #endif
        }
        if (distance != FLT_MAX) break;

#define TODO_RENDER_DISTANCE 1
        if (tMaxX < tMaxY) {
            if (tMaxX < tMaxZ) {
                renderChunk.x += step.x;
                if (renderChunk.x < (MAX_RENDER_DISTANCE - TODO_RENDER_DISTANCE) || (renderChunk.x > MAX_RENDER_DISTANCE + TODO_RENDER_DISTANCE)) break;
                tMaxX += step.x / direction.x;
                raypos = origin + tMaxX * direction;
            }
            else {
                renderChunk.z += step.z;
                if (renderChunk.z < (MAX_RENDER_DISTANCE - TODO_RENDER_DISTANCE) || (renderChunk.z > MAX_RENDER_DISTANCE + TODO_RENDER_DISTANCE)) break;
                tMaxZ += step.z / direction.z;
                raypos = origin + tMaxZ * direction;
            }
        } else {
            if (tMaxY < tMaxZ) {
                renderChunk.y += step.y;
                if (renderChunk.y < 0 || renderChunk.y >= 16) break;
                tMaxY += step.y / direction.y;
                raypos = origin + tMaxY * direction;
            }
            else {
                renderChunk.z += step.z;
                if (renderChunk.z < (MAX_RENDER_DISTANCE - TODO_RENDER_DISTANCE) || (renderChunk.z > MAX_RENDER_DISTANCE + TODO_RENDER_DISTANCE)) break;
                tMaxZ += step.z / direction.z;
                raypos = origin + tMaxZ * direction;
            }
        }

        raypos = NormalizeGridPosition(raypos) * 16.0f;
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
    Kernel<<<gridDim, blockDim>>>(kernelOutputBuffer, g_screenWidth, g_screenHeight, g_bufferPitch, (Quad**)devicePointers, (int*)arraySizes, viewport, origin, renderChunkY);

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
