#define NOMINMAX
#define WIN32_LEAN_AND_MEAN

#include <glad/glad.h>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <cuda_gl_interop.h>

#include <assert.h>
#include <stdio.h>
#include <string>
#include <vector>

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
static cudaGraphicsResource_t gfxResource;

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
    MJ_EXPORT void SetViewingPlane(JNIEnv*, jobject, jobject);
    MJ_EXPORT void SetVertexBuffer(JNIEnv*, jint, jint, jint, jint, jobject);
    MJ_EXPORT void SetViewEntity(JNIEnv*, jdouble, jdouble, jdouble);
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

void Init(JNIEnv* env) {
    if (!gladLoadGL()) {
        Log(env, "Could not load OpenGL functions!");
    }
    CacheJNI(env);
    Log(env, "Init");
}

void Destroy(JNIEnv* env) {
    Log(env, "Destroy");
    // Unregister CUDA resource
    if (gfxResource) {
        cudaError_t err = cudaGraphicsUnregisterResource(gfxResource);
        if (err != cudaSuccess) {
            Log(env, std::string("cudaGraphicsUnregisterResource failed: ") + std::to_string(err));
        }
        gfxResource = 0;
    }

    if (texture) {
        glDeleteTextures(1, &texture);
        texture = 0;
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

        cudaError_t err;

        // Unregister CUDA resource
        if (gfxResource) {
            cudaGraphicsUnmapResources(1, &gfxResource);
            err = cudaGraphicsUnregisterResource(gfxResource);
            if (err != cudaSuccess) {
                Log(env, std::string("cudaGraphicsUnregisterResource failed: ") + std::to_string(err));
            }
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
        err = cudaGraphicsGLRegisterImage(&gfxResource, texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
        if (err != cudaSuccess) {
            Log(env, std::string("cudaGraphicsGLRegisterImage failed: ") + std::to_string(err));
        }

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
    int i;
};

static std::vector<cudaGraphicsResource*> allResources; // Application lifetime
static std::vector<cudaGraphicsResource*> frameResources; // Cleared after every frame
static std::vector<GfxRes2DevPtr> translations;
static double viewEntityX, viewEntityY, viewEntityZ;

// Used in the kernel
static void* devicePointers[DEVICE_PTRS_COUNT];
static int arraySizes[DEVICE_PTRS_COUNT];
static Viewport viewport;

jint Raytrace(JNIEnv* env) {
    // Clear kernel buffers
    memset(devicePointers, 0, sizeof(devicePointers));
    memset(arraySizes, 0, sizeof(arraySizes));

    cudaError err;
    // Map all resources
    err = cudaGraphicsMapResources((int) frameResources.size(), frameResources.data());
    if (err != cudaSuccess) {
        Log(env, std::string("Error during cudaGraphicsMapResources, error code ") + std::to_string(err));
    }
    
    // Update device pointers
    for (int i = 0; i < translations.size(); i++) {
        GfxRes2DevPtr& t = translations[i];

        size_t bufferSize;
        size_t idx = t.x * GRID_DIM * 16 * 4 + t.z * 16 * 4 + t.y * 4 + t.i;
        if ((err = cudaGraphicsResourceGetMappedPointer(&devicePointers[idx], &bufferSize, frameResources[i])) != cudaSuccess) {
            Log(env, std::string("Error during cudaGraphicsResourceGetMappedPointer, error code ") + std::to_string(err));
            continue;
        }
        assert(bufferSize >= t.count * VERTEX_SIZE_BYTES);
        arraySizes[idx] = t.count;
    }

    rtRaytrace(env, gfxResource, texHeight, devicePointers, arraySizes, viewport);

    // Unmap all resources
    err = cudaGraphicsUnmapResources((int) frameResources.size(), frameResources.data());
    if (err != cudaSuccess) {
        Log(env, std::string("Error during cudaGraphicsUnmapResources, error code ") + std::to_string(err));
    }

    // Clear per-frame data
    frameResources.clear();
    translations.clear();

    return texture;
}

void SetViewingPlane(JNIEnv* env, jobject, jobject arr) {
    jfloat* buffer = (jfloat*)env->GetDirectBufferAddress(arr);
    viewport.p0 = float3{buffer[0], buffer[1], buffer[2]};
    viewport.p1 = float3{buffer[3], buffer[4], buffer[5]};
    viewport.p2 = float3{buffer[6], buffer[7], buffer[8]};
    float3 p0p1 = viewport.p1 - viewport.p0;
    float3 p0p2 = viewport.p2 - viewport.p0;

    float originDistance = (length(p0p2) * 0.5f) / tanf(buffer[9] * 0.5f);
    float3 originDir = normalize(cross(p0p1, p0p2));

    viewport.origin = (viewport.p1 + viewport.p2) * 0.5f + originDir * originDistance;
}

void SetVertexBuffer(JNIEnv* env, jint chunkX, jint chunkY, jint chunkZ, jint pass, jobject obj) {
    int count = env->GetIntField(obj, jni_VertexBuffer_count);

    // CUDA cannot register empty buffers
    if (count == 0) return;

    int glBufferId = env->GetIntField(obj, jni_VertexBuffer_glBufferId);

    chunkX; chunkY; chunkZ; pass; glBufferId;
    if ((glBufferId + 1) > allResources.size()) {
        allResources.resize((glBufferId + 1), NULL);
    }

    if (!allResources[glBufferId]) {
        // Register buffer in CUDA
        // TODO: Unregister buffer on destroy using cudaGraphicsUnregisterResource
        cudaGraphicsResource* dst = 0;
        cudaError err = cudaGraphicsGLRegisterBuffer(&dst, glBufferId, cudaGraphicsRegisterFlagsReadOnly);
        assert(err == cudaSuccess);
        assert(dst);
        allResources[glBufferId] = dst;
    }

    int x = (int)((double)chunkX - viewEntityX) / 16 + MAX_RENDER_DISTANCE;
    int y = chunkY / 16;
    int z = (int)((double)chunkZ - viewEntityZ) / 16 + MAX_RENDER_DISTANCE;
    assert(x >= 0); assert(x < GRID_DIM);
    assert(y >= 0); assert(y < 16);
    assert(z >= 0); assert(z < GRID_DIM);

    GfxRes2DevPtr translation = { 0 };
    translation.count = count;
    translation.i = pass;
    translation.x = x;
    translation.y = y;
    translation.z = z;
    translations.push_back(translation);
    frameResources.push_back(allResources[glBufferId]);
}

// This is called before SetVertexBuffer in order to translate the renderChunks.
void SetViewEntity(JNIEnv* env, jdouble x, jdouble y, jdouble z) {
    env;
    viewEntityX = x;
    viewEntityY = y;
    viewEntityZ = z;
}
