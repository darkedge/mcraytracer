#include "raytracer.h"
#include "helper_math.h"

// BLOCK_SIZE^2 = max threads per SM / max active blocks
#define BLOCK_SIZE 8 // JOTARO

static uchar4* kernelOutputBuffer;
static int g_screenWidth;
static int g_screenHeight;
static size_t g_bufferPitch;

// Calculates t for a line starting from s
// to cross the next integer in terms of ds.
// Assume s, ds are [-1..1]
__device__ float FindFirstT(float s, float ds) {
    return (ds > 0 ? ceil(s) - s : s - floor(s)) / abs(ds);
}

// Transforms a point from world space to grid space [-1..1].
__device__ float WorldToGrid(float x) {
    float g = x * (1.0f / 16.0f);
    return g - (int)g;
}

__device__ bool IntersectQuad(float3* origin, float3* dir, Quad* quad, char x, char y, char z, float* out_distance) {
    Vertex* v0 = &quad->vertices[0];
    Vertex* v1 = &quad->vertices[1];
    Vertex* v2 = &quad->vertices[2];

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

static __inline__ __device__ float4 Mul(mat4 mat, float4 vec) {
    return make_float4(
        dot(mat.row0, vec),
        dot(mat.row1, vec),
        dot(mat.row2, vec),
        dot(mat.row3, vec)
    );
}

__global__ void Kernel(uchar4* dst, int width, int height, Quad** vertexBuffers, int* arraySizes, Viewport viewport, float3 entity, size_t bufferPitch, mat4 invViewMatrix, mat4 invProjMatrix) {
    float3 direction;
    {
        int x = (blockIdx.x * blockDim.x) + threadIdx.x;
        // Invert Y because OpenGL
        int y = height - ((blockIdx.y * blockDim.y) + threadIdx.y + 1);
        float u = 2.0f * x / (float)width - 1.0f;
        float v = 2.0f * y / (float)height - 1.0f;
        float4 ray_eye = Mul(invProjMatrix, make_float4(u, v, -1.0f, 1.0f));
        ray_eye = Mul(invViewMatrix, make_float4(ray_eye.x, ray_eye.y, -1.0f, 0.0f));
        direction = normalize(make_float3(ray_eye.x, ray_eye.y, ray_eye.z));
    }
    float3 origin = entity + viewport.origin;

    float distance = FLT_MAX;

    char renderChunkX = MAX_RENDER_DISTANCE;
    char renderChunkY = (unsigned char)floor(origin.y / 16.0f);
    char renderChunkZ = MAX_RENDER_DISTANCE;

    // Transform origin to [-1..1]
    origin.x = WorldToGrid(origin.x);
    origin.y = WorldToGrid(origin.y);
    origin.z = WorldToGrid(origin.z);

    // = (-1/1) signs of vector dir
    int stepX = (direction.x < 0) ? -1 : 1;
    int stepY = (direction.y < 0) ? -1 : 1;
    int stepZ = (direction.z < 0) ? -1 : 1;

    // All positive values
    float tMaxX = FindFirstT(origin.x, direction.x);
    float tMaxY = FindFirstT(origin.y, direction.y);
    float tMaxZ = FindFirstT(origin.z, direction.z);

    // All positive values
    float deltaX = (float)stepX / direction.x;
    float deltaY = (float)stepY / direction.y;
    float deltaZ = (float)stepZ / direction.z;

    unsigned char checks = 0;
    do {
        int renderChunk =
            renderChunkX * GRID_DIM * 16 +
            renderChunkZ * 16 +
            renderChunkY;

        // Ray position from [0..16]
        float3 ray;
        {
            float rx = origin.x + tMaxX * direction.x; rx = (rx - (int)rx) * 16.0f;
            float ry = origin.y + tMaxY * direction.y; ry = (ry - (int)ry) * 16.0f;
            float rz = origin.z + tMaxZ * direction.z; rz = (rz - (int)rz) * 16.0f;
            ray = make_float3(rx, ry, rz);
        }
        
        if (checks < 255) checks += 5;

        // Opaque pass
        Quad* buffer = vertexBuffers[renderChunk];
        for (int j = 0; j < arraySizes[renderChunk]; j++) {
            // Quads in buffer
            float dist = FLT_MAX;
#if 1
            if (IntersectQuad(&ray, &direction, &buffer[j], renderChunkX, renderChunkY, renderChunkZ, &dist)) {
                if (dist < distance) {
                    // TODO: Remember quad for texturing etc
                    distance = dist;
                }
            }
#endif
        }

#define TODO_RENDER_DISTANCE MAX_RENDER_DISTANCE
        if (tMaxX < tMaxY) {
            if (tMaxX < tMaxZ) {
                renderChunkX += stepX;
                if (renderChunkX < (MAX_RENDER_DISTANCE - TODO_RENDER_DISTANCE) || (renderChunkX > MAX_RENDER_DISTANCE + TODO_RENDER_DISTANCE)) break;
                tMaxX += deltaX;
            }
            else {
                renderChunkZ += stepZ;
                if (renderChunkZ < (MAX_RENDER_DISTANCE - TODO_RENDER_DISTANCE) || (renderChunkZ > MAX_RENDER_DISTANCE + TODO_RENDER_DISTANCE)) break;
                tMaxZ += deltaZ;
            }
        } else {
            if (tMaxY < tMaxZ) {
                renderChunkY += stepY;
                if (renderChunkY < 0 || renderChunkY >= 16) break;
                tMaxY += deltaY;
            }
            else {
                renderChunkZ += stepZ;
                if (renderChunkZ < (MAX_RENDER_DISTANCE - TODO_RENDER_DISTANCE) || (renderChunkZ > MAX_RENDER_DISTANCE + TODO_RENDER_DISTANCE)) break;
                tMaxZ += deltaZ;
            }
        }
    } while (distance == FLT_MAX);

    unsigned char val = distance != FLT_MAX ? 255 : 0;

    {
        int x = (blockIdx.x * blockDim.x) + threadIdx.x;
        // Invert Y because OpenGL
        int y = height - ((blockIdx.y * blockDim.y) + threadIdx.y + 1);
        int offset = (y * bufferPitch) + x * sizeof(uchar4);
        if (offset >= bufferPitch * height) return;
        dst = (uchar4*)(((char*)dst) + offset);
    }

    *dst = make_uchar4(val, checks, 255, 255);
    //*dst = make_uchar4(direction.x * 127 + 127, direction.y * 127 + 127, direction.z * 127 + 127, 255);
    //*dst = make_uchar4(direction.x * 256, direction.y * 256, direction.z * 256, 255);
    //*dst = make_uchar4(u * 256, v * 256, 255, 255);
    //*dst = make_uchar4(u * 127 + 127, v * 127 + 127, 255, 255);
    //*dst = make_uchar4(bar.x * 127 + 127, bar.y * 127 + 127, 255, 255);
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

void rtRaytrace(JNIEnv*, cudaGraphicsResource_t glTexture, int texHeight, Quad** devicePointers, int* arraySizes, const Viewport &viewport, const float3& viewEntity, mat4 invViewMatrix, mat4 invProjMatrix) {
    unsigned int blocksW = (unsigned int)ceilf(g_screenWidth / (float)BLOCK_SIZE);
    unsigned int blocksH = (unsigned int)ceilf(g_screenHeight / (float)BLOCK_SIZE);
    dim3 gridDim(blocksW, blocksH, 1);
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);

    // Kernel call
    Kernel<<<gridDim, blockDim>>>(kernelOutputBuffer, g_screenWidth, g_screenHeight, devicePointers, arraySizes, viewport, viewEntity, g_bufferPitch, invViewMatrix, invProjMatrix);

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
