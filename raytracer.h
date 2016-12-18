#pragma once
#include <cuda_runtime_api.h>
#include <jni.h>
#include <string>

void rtRaytrace(JNIEnv* env, cudaGraphicsResource_t dst, int texHeight);
void rtResize(JNIEnv* env, int w, int h);

void Log(JNIEnv*, const std::string&);


#define CUDA_TRY(x)\
do {\
	cudaError_t err = x;\
	if (err != cudaSuccess) {\
		Log(env, std::string(#x" failed, err = ") + std::to_string(err));\
	}\
} while (0);

