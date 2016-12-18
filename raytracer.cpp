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
    MJ_EXPORT void SetVertexBuffer(JNIEnv*, jint, jint, jint, jobject);
    MJ_EXPORT void SetViewEntity(JNIEnv*, jdouble, jdouble, jdouble);
}

static jmethodID jni_println;
static jmethodID jni_ebs_get;
static jmethodID jni_ibs_getBlock;
static jmethodID jni_block_getIdFromBlock;
static jmethodID jni_chunk_getBlockStorageArray;
static jclass jni_system;
static jfieldID jni_system_out_id;

void CacheJNI(JNIEnv* env) {
#if 0
    // System.out.println
    jni_system = env->FindClass("java/lang/System");
    jni_system_out_id = env->GetStaticFieldID(jni_system, "out", "Ljava/io/PrintStream;");
    
    jclass pscls = env->FindClass("java/io/PrintStream");
    jni_println = env->GetMethodID(pscls, "println", "(Ljava/lang/String;)V");
#endif

    // Chunk.getBlockStorageArray
    jclass chunk = env->FindClass("net/minecraft/world/chunk/Chunk");
    jni_chunk_getBlockStorageArray = env->GetMethodID(chunk, "getBlockStorageArray", "()[Lnet/minecraft/world/chunk/storage/ExtendedBlockStorage;");
    jclass ebs = env->FindClass("net/minecraft/world/chunk/storage/ExtendedBlockStorage");
    jni_ebs_get = env->GetMethodID(ebs, "get", "(III)Lnet/minecraft/block/state/IBlockState;");
    jclass ibs = env->FindClass("net/minecraft/block/state/IBlockState");
    jni_ibs_getBlock = env->GetMethodID(ibs, "getBlock", "()Lnet/minecraft/block/Block;");
    jclass block = env->FindClass("net/minecraft/block/Block");
    jni_block_getIdFromBlock = env->GetStaticMethodID(block, "getIdFromBlock", "(Lnet/minecraft/block/Block;)I");
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
    }

    // CUDA does not need to know the texture width
    rtResize(env, screenWidth, screenHeight);
}

static std::vector<cudaGraphicsResource*> allResources; // Application lifetime
static std::vector<cudaGraphicsResource*> frameResources; // Cleared after every frame
static double viewEntityX, viewEntityY, viewEntityZ;

jint Raytrace(JNIEnv* env) {
    cudaGraphicsMapResources((int) frameResources.size(), frameResources.data());
    rtRaytrace(env, gfxResource, texHeight);
    cudaGraphicsUnmapResources((int) frameResources.size(), frameResources.data());
    frameResources.clear();

    return texture;
}

void SetViewingPlane(JNIEnv* env, jobject, jobject arr) {
    jfloat* buffer = (jfloat*)env->GetDirectBufferAddress(arr);
    Vector3 p0(buffer[0], buffer[1], buffer[2]);
    Vector3 p1(buffer[3], buffer[4], buffer[5]);
    Vector3 p2(buffer[6], buffer[7], buffer[8]);
    Vector3 p0p1 = p1 - p0;
    Vector3 p0p2 = p2 - p0;

    float originDistance = (length(p0p2) * 0.5f) / tanf(buffer[9] * 0.5f);
    Vector3 originDir = normalize(cross(p0p1, p0p2));
    Vector3 origin = (p1 + p2) * 0.5f + originDir * originDistance;
}

// TODO: Add layer index
void SetVertexBuffer(JNIEnv* env, jint chunkX, jint chunkY, jint chunkZ, jobject obj) {
    jclass cl = env->GetObjectClass(obj);
    int count = env->GetIntField(obj, env->GetFieldID(cl, "count", "I"));

    // CUDA cannot register empty buffers
    if (count == 0) return;

    int glBufferId = env->GetIntField(obj, env->GetFieldID(cl, "glBufferId", "I"));

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

    int x = (int)((double)chunkX - viewEntityX) / 16 + 12;
    int y = chunkY / 16;
    int z = (int)((double)chunkZ - viewEntityZ) / 16 + 12;
    assert(x >= 0); assert(x <= 24);
    assert(y >= 0); assert(y <= 16);
    assert(z >= 0); assert(z <= 24);

    frameResources.push_back(allResources[glBufferId]);
}

// This is called before SetVertexBuffer.
void SetViewEntity(JNIEnv* env, jdouble x, jdouble y, jdouble z) {
    env;
    viewEntityX = x;
    viewEntityY = y;
    viewEntityZ = z;
}
