#pragma once
#include <jni.h>
#include <cuda_runtime_api.h>
#include <string>

void rtRaytrace(JNIEnv* env, cudaGraphicsResource_t dst, int texHeight);
void rtResize(JNIEnv* env, int w, int h);

static void Log(JNIEnv* env, const std::string& stdstr) {
	jclass syscls = env->FindClass("java/lang/System");
	jfieldID fid = env->GetStaticFieldID(syscls, "out", "Ljava/io/PrintStream;");
	jobject out = env->GetStaticObjectField(syscls, fid);
	jclass pscls = env->FindClass("java/io/PrintStream");
	jmethodID mid = env->GetMethodID(pscls, "println", "(Ljava/lang/String;)V");
	jstring str = env->NewStringUTF(stdstr.c_str());
	env->CallVoidMethod(out, mid, str);
}

/*
#define CUDA_TRY(x)\
do {\
	cudaError_t err = x;\
	if (err != cudaSuccess) {\
		Log(env, #x" failed");\
	}\
} while (0);
*/
