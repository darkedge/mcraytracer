#include "raytracer.h"
#include "helper_math.h"

#define BLOCK_SIZE 16     // block size

static uchar4* kernelOutputBuffer;
static int g_screenWidth;
static int g_screenHeight;
static size_t g_bufferPitch;

__device__ float IntBound(float s, float ds) {
    return (ds > 0 ? ceil(s) - s : s - floor(s)) / abs(ds);
}

__global__ void Kernel(uchar4* dst, int width, int height, void** devicePointers, int* arraySizes, Viewport viewport, size_t bufferPitch) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    // Invert Y because OpenGL
    int y = height - ((blockIdx.y * blockDim.y) + threadIdx.y + 1);

    float u = x / (float)width;
    float v = y / (float)height;

    int offset = (y * bufferPitch) + x * sizeof(uchar4);
    if (offset >= bufferPitch * height) return;

    float3 point = (lerp(viewport.p0, viewport.p1, u) + lerp(viewport.p0, viewport.p2, v)) * 0.5f;
    float3 direction = normalize(point - viewport.origin);

    int renderChunkX = floor(viewport.origin.x) + MAX_RENDER_DISTANCE;
    int renderChunkY = floor(viewport.origin.y);
    int renderChunkZ = floor(viewport.origin.z) + MAX_RENDER_DISTANCE;

    // = (-1/1) signs of vector dir
    int stepX = (direction.x < 0) ? -1 : 1;
    int stepY = (direction.y < 0) ? -1 : 1;
    int stepZ = (direction.z < 0) ? -1 : 1;

    float tMaxX = IntBound(viewport.origin.x, direction.x);
    float tMaxY = IntBound(viewport.origin.y, direction.y);
    float tMaxZ = IntBound(viewport.origin.z, direction.z);

    float tDeltaX = (float)stepX / direction.x; // length of v between two YZ-boundaries
    float tDeltaY = (float)stepY / direction.y; // length of v between two XZ-boundaries
    float tDeltaZ = (float)stepZ / direction.z; // length of v between two XY-boundaries

    // Range of 5 renderChunks for now
    // Max = MAX_RENDER_DISTANCE - 1
    int radius = 5;
    do {
        if (tMaxX < tMaxY) {
            if (tMaxX < tMaxZ) {
                renderChunkX += stepX;
                if (renderChunkX > radius) break;
                tMaxX += tDeltaX;
            } else {
                renderChunkZ += stepZ;
                if (renderChunkZ > radius) break;
                tMaxZ += tDeltaZ;
            }
        } else {
            if (tMaxY < tMaxZ) {
                renderChunkY += stepY;
                if (renderChunkY > radius) break;
                tMaxY += tDeltaY;
            } else {
                renderChunkZ += stepZ;
                if (renderChunkZ > radius) break;
                tMaxZ += tDeltaZ;
            }
        }

        // TODO: Test all quads in renderchunk at: renderChunkX, renderChunkY, renderChunkZ
        int offset =
            renderChunkX * GRID_DIM * 16 * 4 +
            renderChunkZ * 16 * 4 +
            renderChunkY * 4;
        void** relevantBuffers = devicePointers + offset;
#if 0
        for (int i = 0; i < 4; i++) {
            // Buffers in RenderChunk
            Quad* buffer = (Quad*) relevantBuffers[i];
            if (buffer)
            for (int j = 0; j < arraySizes[offset + i]; j++) {
                // Quads in buffer
                buffer[j];
            }
        }
#endif
        
        // Need to check for closest intersection
    } while (true);

    *((uchar4*)(((uchar1*)dst) + offset)) = make_uchar4(u * 256.0f, v * 256.0f, 255.0f, 255.0f);
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

void rtRaytrace(JNIEnv*, cudaGraphicsResource_t glTexture, int texHeight, void** devicePointers, int* arraySizes, const Viewport &viewport) {
    unsigned int blocksW = (unsigned int)ceilf(g_screenWidth / (float)BLOCK_SIZE);
    unsigned int blocksH = (unsigned int)ceilf(g_screenHeight / (float)BLOCK_SIZE);
    dim3 gridDim(blocksW, blocksH, 1);
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);

    // Kernel call
    Kernel<<<gridDim, blockDim>>>(kernelOutputBuffer, g_screenWidth, g_screenHeight, devicePointers, arraySizes, viewport, g_bufferPitch);

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
