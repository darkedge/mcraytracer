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

void InsertQuads(Quad* quads, int numQuads) {

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
static VertexBuffer counts[69696];
static int4 bufferIndices[DEVICE_PTRS_COUNT]; // TODO: Build something like this?

void SetVertexBuffer(JNIEnv* env, jint id, jint x, jint y, jint z, jint layer, jobject data, jint size) {
    totalTriangles -= counts[id].numTris;
    counts[id] = {};
    counts[id].index = -1; // TODO
    counts[id].x = x;
    counts[id].y = y;
    counts[id].z = z;
    counts[id].layer = layer;
    counts[id].numTris = size / VERTEX_SIZE_BYTES / 4 * 2;
    totalTriangles += counts[id].numTris;

    //Quad* buf = (Quad*) env->GetDirectBufferAddress(data);
    //InsertQuads(buf, size / VERTEX_SIZE_BYTES / 4);
    
    Log(env, std::string("Updated ") + std::to_string(counts[id].numTris) + std::string(" triangles, total: ") + std::to_string(totalTriangles));
}

// This is called before SetVertexBuffer in order to translate the renderChunks.
void SetViewEntity(JNIEnv*, jdouble x, jdouble y, jdouble z) {
    viewEntity = float3{(float)x, (float)y, (float)z};
}

void StopProfiling(JNIEnv*) {
    cudaDeviceSynchronize();
    cudaProfilerStop();
}