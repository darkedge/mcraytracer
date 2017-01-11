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

__device__ bool IntersectQuad(float3* origin, float3* dir, Quad* quad, float* out_distance) {
    Vertex* v0 = &quad->vertices[0];
    Vertex* v1 = &quad->vertices[0];
    Vertex* v2 = &quad->vertices[0];
    //Vertex* v0 = &quad->vertices[0];

    // Get normal
    float3 normal = normalize(cross(v2->pos - v1->pos, v0->pos - v1->pos));
    float denom = dot(normal, *dir);
    float t = -1.0f;
    if (abs(denom) > 0.0001f) { // TODO: tweak epsilon
        t = dot(v0->pos - *origin, normal) / denom;
        *out_distance = t;
    }

    return t > 0.0001f;
}

__global__ void Kernel(uchar4* dst, int width, int height, Quad** vertexBuffers, int* arraySizes, Viewport viewport, float3 entity, size_t bufferPitch) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    // Invert Y because OpenGL
    int y = height - ((blockIdx.y * blockDim.y) + threadIdx.y + 1);

    {
        int offset = (y * bufferPitch) + x * sizeof(uchar4);
        if (offset >= bufferPitch * height) return;
        dst = (uchar4*)(((char*)dst) + offset);
    }

    float3 direction;
    {
        float u = x / (float)width;
        float v = y / (float)height;

        float3 point = (lerp(viewport.p0, viewport.p1, u) + lerp(viewport.p0, viewport.p2, v)) * 0.5f;
        direction = normalize(point - viewport.origin);
    }
    float3 origin = entity + viewport.origin;

    float distance = FLT_MAX;

    int renderChunkX = MAX_RENDER_DISTANCE;
    int renderChunkY = floor(origin.y) / 16;
    int renderChunkZ = MAX_RENDER_DISTANCE;

    // = (-1/1) signs of vector dir
    int stepX = (direction.x < 0) ? -1 : 1;
    int stepY = (direction.y < 0) ? -1 : 1;
    int stepZ = (direction.z < 0) ? -1 : 1;

    float tMaxX = IntBound(origin.x, direction.x);
    float tMaxY = IntBound(origin.y, direction.y);
    float tMaxZ = IntBound(origin.z, direction.z);

    unsigned char checks = 0;
    do {
        if (tMaxX < tMaxY) {
            if (tMaxX < tMaxZ) {
                renderChunkX += stepX;
                if (renderChunkX < 0 || renderChunkX >= GRID_DIM) break;
                tMaxX += (float)stepX / direction.x;
            }
            else {
                renderChunkZ += stepZ;
                if (renderChunkZ < 0 || renderChunkZ >= GRID_DIM) break;
                tMaxZ += (float)stepZ / direction.z;
            }
        }
        else {
            if (tMaxY < tMaxZ) {
                renderChunkY += stepY;
                if (renderChunkY < 0 || renderChunkY >= 16) break;
                tMaxY += (float)stepY / direction.y;
            }
            else {
                renderChunkZ += stepZ;
                if (renderChunkZ < 0 || renderChunkZ >= GRID_DIM) break;
                tMaxZ += (float)stepZ / direction.z;
            }
        }

        int renderChunk =
            renderChunkX * GRID_DIM * 16 * 4 +
            renderChunkZ * 16 * 4 +
            renderChunkY * 4;

        // Create the ray used for intersection
        float3 ray{
            (tMaxX - (int)tMaxX) * 16,
            (tMaxY - (int)tMaxY) * 16,
            (tMaxZ - (int)tMaxZ) * 16,
        };

        if (renderChunk >= DEVICE_PTRS_COUNT - 4) break;

        for (int pass = 0; pass < 4; pass++) {
            // Buffers in RenderChunk
            Quad* buffer = vertexBuffers[renderChunk + pass];
            if (buffer) {
                for (int j = 0; j < arraySizes[renderChunk + pass]; j++) {
                    if(checks < 255) checks++;
                    #if 1
                    // Quads in buffer
                    float dist = FLT_MAX;
                    if (IntersectQuad(&ray, &direction, &buffer[j], &dist)) {
                        if (dist < distance) {
                            // TODO: Remember quad for texturing etc
                            distance = dist;
                        }
                    }
                    #endif
                }
            }
        }
    } while (distance != FLT_MAX);

    unsigned char val = distance != FLT_MAX ? 255 : 0;

    *dst = make_uchar4(val, checks, 255, 255);
}

void rtResize(JNIEnv* env, int screenWidth, int screenHeight) {
    g_screenWidth = screenWidth;
    g_screenHeight = screenHeight;

    cudaError_t err;
    
    // Resize
    if (kernelOutputBuffer) {
        err = cudaFree(kernelOutputBuffer);
        if (err != cudaSuccess) {
            Log(env, std::string("Error during cudaFree, error code ") + std::to_string(err) + std::string(": ") + std::string(": ") + cudaGetErrorString(err));
        }
    }

    err = cudaMallocPitch((void**)&kernelOutputBuffer, &g_bufferPitch, g_screenWidth * sizeof(uchar4), g_screenHeight * sizeof(uchar4));
    if (err != cudaSuccess) {
        Log(env, std::string("Error during cudaMallocPitch, error code ") + std::to_string(err) + std::string(": ") + cudaGetErrorString(err));
    }
}

void rtRaytrace(JNIEnv*, cudaGraphicsResource_t glTexture, int texHeight, Quad** devicePointers, int* arraySizes, const Viewport &viewport, const float3& viewEntity) {
    unsigned int blocksW = (unsigned int)ceilf(g_screenWidth / (float)BLOCK_SIZE);
    unsigned int blocksH = (unsigned int)ceilf(g_screenHeight / (float)BLOCK_SIZE);
    dim3 gridDim(blocksW, blocksH, 1);
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);

    // Kernel call
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
