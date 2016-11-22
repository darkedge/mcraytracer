#include <glad/glad.h>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <cuda_gl_interop.h>

#include <assert.h>
#include <stdio.h>
#include <string>

#include "raytracer.h"
#include <jni.h>

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
	MJ_EXPORT void LoadChunk(JNIEnv*, jobject);
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

        // TODO: Debug code
        uchar4* debugPixels = new uchar4[texWidth * texHeight];
        memset(debugPixels, 0xFF, texWidth * texHeight * sizeof(uchar4));
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, texWidth, texHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glBindTexture(GL_TEXTURE_2D, 0);
        delete[] debugPixels;

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

jint Raytrace(JNIEnv* env) {
    rtRaytrace(env, gfxResource, texHeight);

	return texture;
}

void LoadChunk(JNIEnv* env, jobject ) { 
	Log(env, "LoadChunk");
	/*
	jclass cls = env->GetObjectClass(obj); 

	// First get the class object
	jmethodID mid = env->GetMethodID(cls, "getClass", "()Ljava/lang/Class;");
	jobject clsObj = env->CallObjectMethod(obj, mid);

	// Now get the class object's class descriptor
	cls = env->GetObjectClass(clsObj);

	// Find the getName() method on the class object
	mid = env->GetMethodID(cls, "getName", "()Ljava/lang/String;");

	// Call the getName() to get a jstring object back
	jstring strObj = (jstring)env->CallObjectMethod(clsObj, mid);

	env->CallVoidMethod(jni_system_out, jni_println, strObj);
	

	//jobjectArray arr = (jobjectArray) env->CallObjectMethod(chunk, jni_chunk_getBlockStorageArray);
	for (int i = 0; i < 16; i++) {
		//jobject section = env->GetObjectArrayElement(arr, i);
		//jobject iblockstate = env->CallObjectMethod(section, jni_ebs_get, 0, 0, 0); // TODO
		//jobject block = env->CallObjectMethod(iblockstate, jni_ibs_getBlock);
		//block;
	}
	*/
}
