#define NOMINMAX
#define WIN32_LEAN_AND_MEAN

#include <glad/glad.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <vector_types.h>
#include <cuda_gl_interop.h>

#include <assert.h>
#include <stdio.h>
#include <string>
#include <vector>
#include <tuple>
#include <map>
#include <set>

#include "raytracer.h"
#include <jni.h>

#include <vectormath_aos.h>
#include "helper_math.h"

using namespace Vectormath::Aos;

static jint width;
static jint height;
static GLint texWidth;
static GLint texHeight;

static GLuint texture;
static cudaGraphicsResource* gfxResource;

#define MJ_EXPORT __declspec(dllexport)

/*
    "For optimal performance, however, a thread should pass the JNIEnv that it
    received when it was invoked down through the methods it calls, because
    looking it up can require significant work."
    - http://www.ibm.com/developerworks/java/library/j-jni/
*/

extern "C" {
    MJ_EXPORT void Init(JNIEnv*);
    MJ_EXPORT void Destroy(JNIEnv*);
    MJ_EXPORT void Resize(JNIEnv*, jint, jint);
    MJ_EXPORT jint Raytrace(JNIEnv*);
    MJ_EXPORT void SetViewingPlane(JNIEnv*, jobject);
    MJ_EXPORT void SetVertexBuffer(JNIEnv*, jint, jint, jint, jint, jint, jobject, jint);
    MJ_EXPORT void SetViewEntity(JNIEnv*, jdouble, jdouble, jdouble);
    MJ_EXPORT void StopProfiling(JNIEnv*);
}

static jfieldID jni_VertexBuffer_count;
static jfieldID jni_VertexBuffer_glBufferId;

void CacheJNI(JNIEnv* env) {
#if 0
    // System.out.println
    jni_system = env->FindClass("java/lang/System");
    jni_system_out_id = env->GetStaticFieldID(jni_system, "out", "Ljava/io/PrintStream;");
    
    jclass pscls = env->FindClass("java/io/PrintStream");
    jni_println = env->GetMethodID(pscls, "println", "(Ljava/lang/String;)V");
#endif

    // TODO: Obfuscated names
    jclass vertexBuffer = env->FindClass("net/minecraft/client/renderer/vertex/VertexBuffer");
    jni_VertexBuffer_count = env->GetFieldID(vertexBuffer, "count", "I");
    jni_VertexBuffer_glBufferId = env->GetFieldID(vertexBuffer, "glBufferId", "I");
}

void Log(JNIEnv* env, const std::string& stdstr) {
#if 0
    assert(jni_println);

    // For some reason we cannot cache this field
    jobject jni_system_out = env->GetStaticObjectField(jni_system, jni_system_out_id);

    jstring str = env->NewStringUTF((std::string(std::to_string((size_t)env) + ": ") + stdstr).c_str());
    env->CallVoidMethod(jni_system_out, jni_println, str);
#endif

    jclass logmanager = env->FindClass("org/apache/logging/log4j/LogManager");
    jmethodID getLogger = env->GetStaticMethodID(logmanager, "getLogger", "(Ljava/lang/String;)Lorg/apache/logging/log4j/Logger;");
    jclass loggerC = env->FindClass("org/apache/logging/log4j/Logger");
    jmethodID jni_info = env->GetMethodID(loggerC, "info", "(Ljava/lang/String;)V");
    jstring logstr = env->NewStringUTF("native_raytracer");
    jobject jni_logger = env->CallStaticObjectMethod(logmanager, getLogger, logstr);

    // Prepend JNIEnv pointer value to message
    jstring str = env->NewStringUTF((std::string(std::to_string((size_t)env) + ": ") + stdstr).c_str());
    env->CallVoidMethod(jni_logger, jni_info, str);
}

// Used in the kernel
static Viewport viewport;

// Host memory
static void* h_devPtrs[DEVICE_PTRS_COUNT];
static int h_arraySizes[DEVICE_PTRS_COUNT];

// Device memory backing arrays for textures
static void* d_devPtrs;
static void* d_arraySizes;

void Init(JNIEnv* env) {
    if (!gladLoadGL()) {
        Log(env, "Could not load OpenGL functions!");
    }
    CacheJNI(env);
    Log(env, "Init");

#if 0
    err = cudaHostAlloc((void**)&h_devPtrs, DEVICE_PTRS_COUNT * sizeof(void*), cudaHostAllocMapped);
    if (err != cudaSuccess) {
        Log(env, std::string("cudaMalloc failed: ") + std::to_string(err) + std::string(": ") + cudaGetErrorString(err));
    }
    err = cudaHostAlloc(&h_arraySizes, DEVICE_PTRS_COUNT * sizeof(int), cudaHostAllocMapped);
    if (err != cudaSuccess) {
        Log(env, std::string("cudaMalloc failed: ") + std::to_string(err) + std::string(": ") + cudaGetErrorString(err));
    }
    err = cudaHostGetDevicePointer(&cudah_devPtrs, h_devPtrs, 0);
    if (err != cudaSuccess) {
        Log(env, std::string("cudaHostGetDevicePointer failed: ") + std::to_string(err) + std::string(": ") + cudaGetErrorString(err));
    }
    err = cudaHostGetDevicePointer(&cudah_arraySizes, h_arraySizes, 0);
    if (err != cudaSuccess) {
        Log(env, std::string("cudaHostGetDevicePointer failed: ") + std::to_string(err) + std::string(": ") + cudaGetErrorString(err));
    }
#endif

    // Create CUDA arrays
    {
        CUDA_TRY(cudaMalloc(&d_devPtrs, sizeof(h_devPtrs)));
        CUDA_TRY(cudaMalloc(&d_arraySizes, sizeof(h_arraySizes)));
    }

    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
}

void Destroy(JNIEnv* env) {
    Log(env, "Destroy");
    // Unregister CUDA resource
    if (gfxResource) {
        CUDA_TRY(cudaGraphicsUnregisterResource(gfxResource));
        gfxResource = NULL;
    }

    if (texture) {
        glDeleteTextures(1, &texture);
        texture = 0;
    }

    // Free arrays
    if (d_arraySizes) {
        cudaFree(d_arraySizes);
        d_arraySizes = NULL;
    }

    if (d_devPtrs) {
        cudaFree(d_devPtrs);
        d_devPtrs = NULL;
    }
}

// Returns a OpenGL texture handle
void Resize(JNIEnv* env, jint screenWidth, jint screenHeight) {
    Log(env, "Resize");
    // Assume the size is different (already checked in java)
    width = screenWidth;
    height = screenHeight;

    // Round up to nearest power of two
    int tw = (int) pow(2, ceil(log(screenWidth) / log(2)));
    int th = (int) pow(2, ceil(log(screenHeight) / log(2)));

    if (tw != texWidth || th != texHeight) {
        texWidth = tw;
        texHeight = th;

        // Unregister CUDA resource
        if (gfxResource) {
            cudaGraphicsUnmapResources(1, &gfxResource);
            CUDA_TRY(cudaGraphicsUnregisterResource(gfxResource));
        }

        // glTexImage2D supports resizing so we only need to call glGenTextures once
        if (!texture) {
            glGenTextures(1, &texture);
            glBindTexture(GL_TEXTURE_2D, texture);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP); // GL_CLAMP_TO_EDGE
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP); // GL_CLAMP_TO_EDGE
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glBindTexture(GL_TEXTURE_2D, 0);

            Log(env, std::string("OpenGL texture id: ") + std::to_string(texture));
        }

        glBindTexture(GL_TEXTURE_2D, texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, texWidth, texHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glBindTexture(GL_TEXTURE_2D, 0);

        Log(env, std::string("Texture size: ") + std::to_string(texWidth) + std::string(", ") + std::to_string(texHeight));

        // Register CUDA resource
        CUDA_TRY(cudaGraphicsGLRegisterImage(&gfxResource, texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));

        cudaGraphicsMapResources(1, &gfxResource);
    }

    // CUDA does not need to know the texture width
    rtResize(env, screenWidth, screenHeight);
}

// Mapped to [0, GRID_DIM)
struct GfxRes2DevPtr {
    int count;
    int x;
    int y;
    int z;
    //int i;
};

static std::vector<cudaGraphicsResource*> allResources; // Application lifetime
static std::vector<cudaGraphicsResource*> frameResources; // Cleared after every frame
static std::vector<GfxRes2DevPtr> translations;
static float3 viewEntity;

jint Raytrace(JNIEnv* env) {
    // Clear kernel buffers
    memset(h_devPtrs, 0, sizeof(h_devPtrs));
    memset(h_arraySizes, 0, sizeof(h_arraySizes));

    if (!frameResources.empty()) {
        // Map all resources
        CUDA_TRY(cudaGraphicsMapResources((int) frameResources.size(), frameResources.data()));
    
        // Update device pointers
        for (int i = 0; i < translations.size(); i++) {
            GfxRes2DevPtr& t = translations[i];

            size_t bufferSize;
            size_t idx = t.x * GRID_DIM * 16 + t.z * 16 + t.y;
            void* devicePointer;
            cudaError err;
            if ((err = cudaGraphicsResourceGetMappedPointer(&devicePointer, &bufferSize, frameResources[i])) != cudaSuccess) {
                Log(env, std::string("Error during cudaGraphicsResourceGetMappedPointer, error code ") + std::to_string(err) + std::string(": ") + cudaGetErrorString(err));
                continue;
            }
            // FIXME: Some buffers do not pass this check for some reason
            if (bufferSize >= t.count * VERTEX_SIZE_BYTES) {
                h_devPtrs[idx] = devicePointer;
                h_arraySizes[idx] = t.count / 4;
            }
        }
    }

    // memcpy to texture memory
    cudaMemcpy(d_devPtrs, h_devPtrs, sizeof(h_devPtrs), cudaMemcpyHostToDevice);
    cudaMemcpy(d_arraySizes, h_arraySizes, sizeof(h_arraySizes), cudaMemcpyHostToDevice);
    
    rtRaytrace(env, gfxResource, texHeight, d_devPtrs, d_arraySizes, viewport, viewEntity);

    if (!frameResources.empty()) {
        // Unmap all resources
        CUDA_TRY(cudaGraphicsUnmapResources((int) frameResources.size(), frameResources.data()));

        // Clear per-frame data
        frameResources.clear();
        translations.clear();
    }

    return texture;
}

void SetViewingPlane(JNIEnv* env, jobject arr) {
    jfloat* buffer = (jfloat*)env->GetDirectBufferAddress(arr);
    viewport.p0 = make_float3(buffer[0], buffer[1], buffer[2]);
    viewport.p1 = make_float3(buffer[3], buffer[4], buffer[5]);
    viewport.p2 = make_float3(buffer[6], buffer[7], buffer[8]);
    float3 p0p1 = viewport.p1 - viewport.p0;
    float3 p0p2 = viewport.p2 - viewport.p0;

    float originDistance = (length(p0p2) * 0.5f) / tanf(buffer[9] * 0.5f);
    float3 originDir = normalize(cross(p0p1, p0p2));

    viewport.origin = (viewport.p1 + viewport.p2) * 0.5f + originDir * originDistance;
}

#if 0
// Currently only called for the opaque pass
void SetVertexBuffer(JNIEnv* env, jint chunkX, jint chunkY, jint chunkZ, jint, jobject obj) {
    int count = env->GetIntField(obj, jni_VertexBuffer_count);

    // CUDA cannot register empty buffers
    if (count == 0) return;

    int glBufferId = env->GetIntField(obj, jni_VertexBuffer_glBufferId);

    //Log(env, std::to_string(glBufferId) + std::string(" vertex count: ") + std::to_string(count));

    if ((glBufferId + 1) > allResources.size()) {
        allResources.resize((glBufferId + 1), NULL);
    }

    if (!allResources[glBufferId]) {
        // Register buffer in CUDA
        // TODO: Unregister buffer on destroy using cudaGraphicsUnregisterResource
        cudaGraphicsResource* dst = NULL;
        CUDA_TRY(cudaGraphicsGLRegisterBuffer(&dst, glBufferId, cudaGraphicsRegisterFlagsReadOnly));
        allResources[glBufferId] = dst;

#if 0
        // Print buffer for testing
        cudaGraphicsMapResources(1, &dst);
        void* cudaPtr;
        size_t bufferSize;
        cudaGraphicsResourceGetMappedPointer(&cudaPtr, &bufferSize, dst);
        assert(bufferSize);
        Vertex* vertices = (Vertex*) malloc(bufferSize);
        cudaMemcpy(vertices, cudaPtr, bufferSize, cudaMemcpyDeviceToHost);
        _CrtDbgBreak();
        free(vertices);
        cudaGraphicsUnmapResources(1, &dst);
#endif
    }

    int x = (int)((double)chunkX - viewEntity.x) / 16 + MAX_RENDER_DISTANCE;
    int y = chunkY / 16;
    int z = (int)((double)chunkZ - viewEntity.z) / 16 + MAX_RENDER_DISTANCE;
    assert(x >= 0); assert(x < GRID_DIM);
    assert(y >= 0); assert(y < 16);
    assert(z >= 0); assert(z < GRID_DIM);

    GfxRes2DevPtr translation = { 0 };
    translation.count = count;
    translation.x = x;
    translation.y = y;
    translation.z = z;
    translations.push_back(translation);
    frameResources.push_back(allResources[glBufferId]);
}
#endif

static int bufferIndices[DEVICE_PTRS_COUNT];

void ShiftGrid(int shiftX, int shiftZ) {
    if (shiftX != 0 && shiftX < GRID_DIM && shiftX > -GRID_DIM) {
        if (shiftX > 0) {
            // Shift right (copy left to right)
            // Start from right
            memmove(&bufferIndices[shiftX * GRID_DIM * 16], bufferIndices, (GRID_DIM - shiftX) * GRID_DIM * 16 * sizeof(*bufferIndices));
            // Invalidate left side
            for (int x = 0; x < shiftX; x++) {
                for (int i = 0; i < GRID_DIM * 16; i++) {
                    bufferIndices[x * GRID_DIM * 16 + i] = -1;
                }
            }
        } else {
            // Shift left (copy right to left)
            // Start from left
            memmove(bufferIndices, &bufferIndices[-shiftX * GRID_DIM * 16], (GRID_DIM + shiftX) * GRID_DIM * 16 * sizeof(*bufferIndices));
            // Invalidate right side
            for (int x = GRID_DIM + shiftX; x < GRID_DIM; x++) {
                for (int i = 0; i < GRID_DIM * 16; i++) {
                    bufferIndices[x * GRID_DIM * 16 + i] = -1;
                }
            }
        }
    } else {
        // Invalidate everything
        for (int x = 0; x < GRID_DIM; x++) {
            for (int z = 0; z < GRID_DIM; z++) {
                for (int i = 0; i < 16; i++) {
                    bufferIndices[x * GRID_DIM * 16 + z * 16 + i] = -1;
                }
            }
        }
        return;
    }
    if (shiftZ != 0 && shiftZ < GRID_DIM && shiftZ > -GRID_DIM) {
        if (shiftZ > 0) {
            // Shift down (copy top to bottom)
            // Start from bottom
            for (int x = 0; x < GRID_DIM; x++) {
                for (int z = GRID_DIM - 1; z >= shiftZ; z--) {
                    memcpy(&bufferIndices[x * GRID_DIM + z * GRID_DIM * 16], &bufferIndices[x * GRID_DIM + (z - shiftZ) * GRID_DIM * 16], 16 * sizeof(*bufferIndices));
                }
            }
            // Invalidate top side
            for (int x = 0; x < GRID_DIM; x++) {
                for (int z = 0; z < shiftZ; z++) {
                    for (int i = 0; i < 16; i++) {
                        bufferIndices[x * GRID_DIM * 16 + z * 16 + i] = -1;
                    }
                }
            }
        } else {
            // Shift up (copy bottom to top)
            // Start from top
            for (int x = 0; x < GRID_DIM; x++) {
                for (int z = 0; z < -shiftZ; z++) {
                    memcpy(&bufferIndices[x * GRID_DIM + z * GRID_DIM * 16], &bufferIndices[x * GRID_DIM + (z - shiftZ) * GRID_DIM * 16], 16 * sizeof(*bufferIndices));
                }
            }
            // Invalidate bottom side
            for (int x = 0; x < GRID_DIM; x++) {
                for (int z = GRID_DIM + shiftZ; z < GRID_DIM; z++) {
                    for (int i = 0; i < 16; i++) {
                        bufferIndices[x * GRID_DIM * 16 + z * 16 + i] = -1;
                    }
                }
            }
        }
    } else {
        // Invalidate everything
        for (int x = 0; x < GRID_DIM; x++) {
            for (int z = 0; z < GRID_DIM; z++) {
                for (int i = 0; i < 16; i++) {
                    bufferIndices[x * GRID_DIM * 16 + z * 16 + i] = -1;
                }
            }
        }
        return;
    }
}

// https://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/
// Inverse of Part1By2 - "delete" all bits not at positions divisible by 3
int Compact1By2(int x) {
    x &= 0x09249249;                  // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
    x = (x ^ (x >> 2)) & 0x030c30c3; // x = ---- --98 ---- 76-- --54 ---- 32-- --10
    x = (x ^ (x >> 4)) & 0x0300f00f; // x = ---- --98 ---- ---- 7654 ---- ---- 3210
    x = (x ^ (x >> 8)) & 0xff0000ff; // x = ---- --98 ---- ---- ---- ---- 7654 3210
    x = (x ^ (x >> 16)) & 0x000003ff; // x = ---- ---- ---- ---- ---- --98 7654 3210
    return x;
}

int DecodeMorton3X(int code) {
    return Compact1By2(code >> 0);
}

int DecodeMorton3Y(int code) {
    return Compact1By2(code >> 1);
}

int DecodeMorton3Z(int code) {
    return Compact1By2(code >> 2);
}

static std::set<std::tuple<int, int, int, int>> cells;
static std::vector<int> grid[16][16][16];
void BuildOctree(JNIEnv* env, jint id, Quad* quads, int numQuads) {
    int duplicates = 0;
    for (int i = 0; i < numQuads; i++) {
        Quad& q = quads[i];

        
        float3 v0 = q.vertices[0].pos;
        float3 v1 = q.vertices[1].pos;
        float3 v2 = q.vertices[2].pos;
        float3 v3 = q.vertices[3].pos;

        // We assume that input vertices do not have precision errors
        // Check if this quad is aligned on the grid (true for most world-gen blocks)
        if (v0.x == v1.x && v1.x == v2.x && v2.x == v3.x && floorf(v0.x) == v0.x) {
            // Check which side the quad is facing based on the direction of an edge
            if (v2.z - v1.z > 0) {
                // We get the vertex that corresponds with the cell index
                // -x
                int x = (int)floorf(q.vertices[1].pos.x);
                int y = (int)floorf(q.vertices[1].pos.y);
                int z = (int)floorf(q.vertices[1].pos.z);
                cells.insert(std::make_tuple(x, y, z, i));
            } else {
                // On positive sided faces, we get the closest one and shift the correct axis
                // +x
                int x = (int)floorf(q.vertices[2].pos.x) - 1;
                int y = (int)floorf(q.vertices[2].pos.y);
                int z = (int)floorf(q.vertices[2].pos.z);
                cells.insert(std::make_tuple(x, y, z, i));
            }
        } else if (v0.y == v1.y && v1.y == v2.y && v2.y == v3.y && floorf(v0.y) == v0.y) {
            if (v1.z - v0.z > 0) {
                // +y
                int x = (int)floorf(q.vertices[0].pos.x);
                int y = (int)floorf(q.vertices[0].pos.y) - 1;
                int z = (int)floorf(q.vertices[0].pos.z);
                cells.insert(std::make_tuple(x, y, z, i));
            } else {
                // -y
                int x = (int)floorf(q.vertices[1].pos.x);
                int y = (int)floorf(q.vertices[1].pos.y);
                int z = (int)floorf(q.vertices[1].pos.z);
                cells.insert(std::make_tuple(x, y, z, i));
            }
        } else if (v0.z == v1.z && v1.z == v2.z && v2.z == v3.z && floorf(v0.z) == v0.z) {
            if (v2.x - v1.x > 0) {
                // +z
                int x = (int)floorf(q.vertices[1].pos.x);
                int y = (int)floorf(q.vertices[1].pos.y);
                int z = (int)floorf(q.vertices[1].pos.z - 1);
                cells.insert(std::make_tuple(x, y, z, i));
            } else {
                // -z
                int x = (int)floorf(q.vertices[2].pos.x);
                int y = (int)floorf(q.vertices[2].pos.y);
                int z = (int)floorf(q.vertices[2].pos.z);
                cells.insert(std::make_tuple(x, y, z, i));
            }
        } else {
            // Quad does not align with the grid
            // Copy quad to all touching cells
            for (int j = 0; j < 4; j++) {
                int x = (int)floorf(q.vertices[j].pos.x);
                int y = (int)floorf(q.vertices[j].pos.y);
                int z = (int)floorf(q.vertices[j].pos.z);
                cells.insert(std::make_tuple(x, y, z, i));
            }
        }

        // Duplicate quad counter
        if (!cells.empty()) {
            duplicates += (int) cells.size() - 1;
        }

        // Paste quad in every cell
        for (auto& tuple : cells) {
            int x = std::get<0>(tuple);
            int y = std::get<1>(tuple);
            int z = std::get<2>(tuple);
            int j = std::get<3>(tuple);
            grid[x][y][z].push_back(j);
        }
        cells.clear();
    }

    // Voxel-sized cubes do not contain duplicates
    if (duplicates > 0) {
        Log(env, std::string("Buffer ") + std::to_string(id) + std::string(" contains ") + std::to_string(duplicates) + std::string(" duplicate quads."));
    }

    // Grid is filled, create octree
    std::vector<int> builder(8, 0);

    // Walk across grid using z-order curve
    // https://en.wikipedia.org/wiki/Z-order_curve
    // This creates a depth-first traversal of the octree

    // The octree has a fixed depth, so we can just hard-code for each level
    // Used for counting nodes
    int idx1 = 0;
    int idx2 = 0;
    int idx3 = 0;

    bool insertNode1 = true;
    bool insertNode2 = true;
    bool insertNode3 = true;
    
    for (int i = 0; i < 16 * 16 * 16; i++) {
        // TODO: Optimize using masks and shifts if necessary
        // Node indices
        int l0 = i / 512; int l3 = i % 512;
        int l1 = l3 / 64; l3 %= 64;
        int l2 = l3 / 8; l3 %= 8;

        int x = DecodeMorton3X(i);
        int y = DecodeMorton3Y(i);
        int z = DecodeMorton3Z(i);

        if (!grid[x][y][z].empty()) {
            // Check if new nodes need to be created
            if (insertNode1) {
                insertNode1 = false;
                // Store new node index in parent
                builder[l0] = (int)builder.size();
                // Construct new node
                builder.insert(builder.end(), { 0, 0, 0, 0, 0, 0, 0, 0 });
            }
            if (insertNode2) {
                insertNode2 = false;
                builder[builder[l0] + l1] = (int)builder.size();
                builder.insert(builder.end(), { 0, 0, 0, 0, 0, 0, 0, 0 });
            }
            if (insertNode3) {
                insertNode3 = false;
                builder[builder[builder[l0] + l1] + l2] = (int)builder.size();
                builder.insert(builder.end(), { 0, 0, 0, 0, 0, 0, 0, 0 });
            }

            // Insert quads
            builder[builder[builder[builder[l0] + l1] + l2] + l3] = (int)builder.size();
            builder.push_back((int)grid[x][y][z].size());
            builder.insert(builder.end(), grid[x][y][z].begin(), grid[x][y][z].end());
        }

        // Check if we crossed a node boundary
        idx3++;
        if (idx3 == 8) {
            insertNode3 = true;
            idx3 = 0;
            idx2++;
            if (idx2 == 8) {
                insertNode2 = true;
                idx2 = 0;
                idx1++;
                if (idx1 == 8) {
                    insertNode1 = true;
                    idx1 = 0;
                }
            }
        }
    }

    // Clear grid
    for (int x = 0; x < 16; x++) {
        for (int y = 0; y < 16; y++) {
            for (int z = 0; z < 16; z++) {
                grid[x][y][z].clear();
            }
        }
    }
}

struct VertexBuffer {
    int x;
    int y;
    int z;
    int layer;
    int numTris;
    int index; // Index in quad or index (TBD) array
};

static int totalTriangles;
static VertexBuffer counts[17424]; // 33 * 33 * 16 * 1
static int2 playerChunkPosition;

void SetVertexBuffer(JNIEnv* env, jint id, jint x, jint y, jint z, jint layer, jobject data, jint size) {
    // TODO: Just using the Opaque layer for now
    assert(layer == 0);

    // Input ID = 0..17423
    // TODO: Probably breaks when changing draw distance due to a static counter in CppVertexBuffer.java
    totalTriangles -= counts[id].numTris;
    if (counts[id].index == -1) {
        if (counts[id].x != x || counts[id].y != y || counts[id].z != z) {
            //Log(env, std::string("Buffer ") + std::to_string(id) + std::string(" location changed!"));
        }
    }
    counts[id] = {};
    counts[id].index = -1; // TODO
    counts[id].x = x;
    counts[id].y = y;
    counts[id].z = z;
    counts[id].layer = layer;
    counts[id].numTris = size / VERTEX_SIZE_BYTES / 4 * 2;
    totalTriangles += counts[id].numTris;

    int chunkX = x / 16 - playerChunkPosition.x + MAX_RENDER_DISTANCE;
    int chunkY = y / 16;
    int chunkZ = z / 16 - playerChunkPosition.y + MAX_RENDER_DISTANCE;

    bufferIndices[chunkX * GRID_DIM * 16 + chunkZ * 16 + chunkY] = id;

    Quad* buf = (Quad*) env->GetDirectBufferAddress(data);
    BuildOctree(env, id, buf, size / VERTEX_SIZE_BYTES / 4);
    
    //Log(env, std::string("Buffer ") + std::to_string(id) + std::string(" now contains ") + std::to_string(counts[id].numTris) + std::string(" triangles, total: ") + std::to_string(totalTriangles));
}

// This is called before SetVertexBuffer in order to translate the renderChunks.
// Note: I hope this is called before new vertex buffers are being added...
void SetViewEntity(JNIEnv* env, jdouble x, jdouble y, jdouble z) {
    viewEntity = float3{(float)x, (float)y, (float)z};
    int chunkX = (int) floor(x / 16);
    int chunkZ = (int) floor(z / 16);
    if (chunkX != playerChunkPosition.x || chunkZ != playerChunkPosition.y) {
        ShiftGrid(chunkX - playerChunkPosition.x, chunkZ - playerChunkPosition.y);
        //Log(env, std::string("Shifted grid (") + std::to_string(chunkX - playerChunkPosition.x) + std::string(", ") + std::to_string(chunkZ - playerChunkPosition.y) + std::string("), new chunk positon: (") + std::to_string(chunkX) + std::string(", ") + std::to_string(chunkZ) + std::string(")"));
    }
    playerChunkPosition.x = chunkX;
    playerChunkPosition.y = chunkZ;
}

void StopProfiling(JNIEnv*) {
    cudaDeviceSynchronize();
    cudaProfilerStop();
}