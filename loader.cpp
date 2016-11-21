// Purpose of this loader:
// - Copy raytracer_native.dll to raytracer_native_temp.dll
// - Load temp DLL and symbols
// - Pass JNI function calls to temp DLL
// - Load temp DLL if raytracer_native.dll changed
#include <string>
#include <cassert>
#include <windows.h>

#include "loader_jni.h"

// New functions:
// - Add #define/typedef pair
// - Add function pointer to Raytracer struct
// - Load function in Load() + validness checks
// - Implement function at bottom of file
// - Add corresponding function in raytracer.cpp

#define RT_INIT(name) void name(JNIEnv*)
typedef RT_INIT(InitFunc);

#define RT_DESTROY(name) void name(JNIEnv*)
typedef RT_DESTROY(DestroyFunc);

#define RT_RESIZE(name) void name(JNIEnv*, jint, jint)
typedef RT_RESIZE(ResizeFunc);

#define RT_RAYTRACE(name) jint name(JNIEnv*)
typedef RT_RAYTRACE(RaytraceFunc);

#define RT_LOAD_CHUNK(name) void name(JNIEnv*, jobject)
typedef RT_LOAD_CHUNK(LoadChunkFunc);

struct Raytracer {
	InitFunc* Init;
	DestroyFunc* Destroy;
	RaytraceFunc* Raytrace;
	ResizeFunc* Resize;
	LoadChunkFunc* LoadChunk;

	FILETIME DLLLastWriteTime;
	bool valid;
	HMODULE dll;
};

static const char* g_dllName = "../x64/Debug/raytracer_native.dll";
static const char* g_tempName = "raytracer_native_temp.dll";
static LARGE_INTEGER g_lastLoadTime;
static Raytracer g_raytracer;
static jint g_width;
static jint g_height;

static jobject jni_system_out;
static jmethodID jni_println;

void CacheJNI(JNIEnv* env) {
	// System.out.println
	jclass syscls = env->FindClass("java/lang/System");
	jfieldID fid = env->GetStaticFieldID(syscls, "out", "Ljava/io/PrintStream;");
	jni_system_out = env->GetStaticObjectField(syscls, fid);
	jclass pscls = env->FindClass("java/io/PrintStream");
	jni_println = env->GetMethodID(pscls, "println", "(Ljava/lang/String;)V");
}

void Log(JNIEnv* env, const std::string& stdstr) {
	assert(jni_system_out);
	assert(jni_println);

	jstring str = env->NewStringUTF((std::string(std::to_string((size_t)env) + ": ") + stdstr).c_str());
	env->CallVoidMethod(jni_system_out, jni_println, str);
}

inline FILETIME mjGetLastWriteTime(char *filename) {
	FILETIME lastWriteTime = {};

	WIN32_FILE_ATTRIBUTE_DATA data;
	if (GetFileAttributesExA(filename, GetFileExInfoStandard, &data)) {
		lastWriteTime = data.ftLastWriteTime;
	}

	return lastWriteTime;
}

static Raytracer Load(JNIEnv* env) {
	Raytracer raytracer = {};

	WIN32_FILE_ATTRIBUTE_DATA foo;
	if (!GetFileAttributesExA("lock.tmp", GetFileExInfoStandard, &foo))
	{
		raytracer.DLLLastWriteTime = mjGetLastWriteTime((char*)g_dllName);
		QueryPerformanceCounter(&g_lastLoadTime);
		CopyFile(g_dllName, g_tempName, FALSE);

		raytracer.dll = LoadLibrary(g_tempName);
		if (raytracer.dll) {
			raytracer.Init = (InitFunc*) GetProcAddress(raytracer.dll, "Init");
			raytracer.Destroy = (DestroyFunc*)GetProcAddress(raytracer.dll, "Destroy");
			raytracer.Resize = (ResizeFunc*)GetProcAddress(raytracer.dll, "Resize");
			raytracer.Raytrace = (RaytraceFunc*)GetProcAddress(raytracer.dll, "Raytrace");
			raytracer.LoadChunk = (LoadChunkFunc*)GetProcAddress(raytracer.dll, "LoadChunk");

			raytracer.valid = 
				raytracer.Init &&
				raytracer.Destroy &&
				raytracer.Resize &&
				raytracer.Raytrace &&
				raytracer.LoadChunk;

			if (raytracer.valid) {
				Log(env, "Successfully (re)loaded Raytracer DLL.");
			} else {
				Log(env, "Failed to (re)load Raytracer DLL!");
			}
		}
	}

	// If any functions fail to load, set all to 0
	if (!raytracer.valid) {
		raytracer.Init = NULL;
		raytracer.Destroy = NULL;
		raytracer.Resize = NULL;
		raytracer.Raytrace = NULL;
		raytracer.LoadChunk = NULL;
	}

	return raytracer;
}

static void Unload(JNIEnv* env) {
	if (g_raytracer.dll) {
		g_raytracer.Destroy(env);
		FreeLibrary(g_raytracer.dll);
	}

	g_raytracer = {};
}

// Returns true if the DLL was reloaded.
static bool ReloadIfNecessary(JNIEnv* env) {
	Log(env, "Loader: Reload");
	// Load DLL
	FILETIME NewDLLWriteTime = mjGetLastWriteTime((char*)g_dllName);

	if (CompareFileTime(&NewDLLWriteTime, &g_raytracer.DLLLastWriteTime) != 0) {
		Unload(env);
		for (int i = 0; !g_raytracer.valid && (i < 100); i++) {
			g_raytracer = Load(env);
			Sleep(100);
		}

		// Reload static memory
		Log(env, "Loader: Calling Raytracer::Init");
		g_raytracer.Init(env);

		return true;
	}

	return false;
}

JNIEXPORT void JNICALL Java_com_marcojonkers_mcraytracer_Raytracer_init
(JNIEnv* env, jobject) {
	CacheJNI(env);
	Log(env, "Loader: Init");
	ReloadIfNecessary(env);
	//g_raytracer.Init(); // Redundant
}

JNIEXPORT void JNICALL Java_com_marcojonkers_mcraytracer_Raytracer_resize
(JNIEnv* env, jobject, jint width, jint height) {
	Log(env, "Loader: Resize");
	ReloadIfNecessary(env);
	// Cache width and height
	g_width = width;
	g_height = height;
	g_raytracer.Resize(env, width, height);
}

JNIEXPORT jint JNICALL Java_com_marcojonkers_mcraytracer_Raytracer_raytrace
(JNIEnv* env, jobject) {
	// Main place where DLL is reloaded
	if (ReloadIfNecessary(env)) {
		g_raytracer.Resize(env, g_width, g_height);
	}
	return g_raytracer.Raytrace(env);
}

// TODO: This is probably called on a different thread -> problem!
JNIEXPORT void JNICALL Java_com_marcojonkers_mcraytracer_Raytracer_loadChunk
(JNIEnv* env, jobject, jobject obj) {
	Log(env, "Loader: LoadChunk");
	g_raytracer.LoadChunk(env, obj);
}
