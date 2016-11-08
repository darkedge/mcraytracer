#include <glad/glad.h>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <cuda_gl_interop.h>

#include <assert.h>
#include <stdio.h>
#include <string>

#include "raytracer.h"
#include "raytracer_jni.h"

static jint width;
static jint height;
static GLint texWidth;
static GLint texHeight;

static GLuint texture;
static cudaGraphicsResource_t gfxResource;

JNIEXPORT void JNICALL Java_com_marcojonkers_mcraytracer_Raytracer_init(JNIEnv* env, jobject) {
    if (!gladLoadGL()) {
        Log(env, "Could not load OpenGL functions!");
    }
}

JNIEXPORT jint JNICALL Java_com_marcojonkers_mcraytracer_Raytracer_resize(JNIEnv* env, jobject, jint screenWidth, jint screenHeight) {
    // Assume the size is different (already checked in java)
    width = screenWidth;
    height = screenHeight;

    // power of two check
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
    Resize(env, screenWidth, screenHeight);

    return texture;
}

JNIEXPORT void JNICALL Java_com_marcojonkers_mcraytracer_Raytracer_raytrace(JNIEnv* env, jobject) {
    assert(texture);

    Raytrace(env, gfxResource, texHeight);
}
