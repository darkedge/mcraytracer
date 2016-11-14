// Purpose of this loader:
// - Copy raytracer_native.dll to raytracer_native_temp.dll
// - Load temp DLL and symbols
// - Pass JNI function calls to temp DLL
// - Load temp DLL if raytracer_native.dll changed

#include <windows.h>
#include "loader_jni.h"
#include <string>

#define RT_INIT(name) void name(JNIEnv*)
typedef RT_INIT(InitFunc);

#define RT_DESTROY(name) void name(JNIEnv*)
typedef RT_DESTROY(DestroyFunc);

#define RT_RESIZE(name) void name(JNIEnv*, jint, jint)
typedef RT_RESIZE(ResizeFunc);

#define RT_RAYTRACE(name) jint name(JNIEnv*)
typedef RT_RAYTRACE(RaytraceFunc);

struct Raytracer {
	InitFunc* Init;
	DestroyFunc* Destroy;
	RaytraceFunc* Raytrace;
	ResizeFunc* Resize;

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

inline void Log(JNIEnv* env, const std::string& stdstr) {
	jclass syscls = env->FindClass("java/lang/System");
	jfieldID fid = env->GetStaticFieldID(syscls, "out", "Ljava/io/PrintStream;");
	jobject out = env->GetStaticObjectField(syscls, fid);
	jclass pscls = env->FindClass("java/io/PrintStream");
	jmethodID mid = env->GetMethodID(pscls, "println", "(Ljava/lang/String;)V");
	jstring str = env->NewStringUTF(stdstr.c_str());
	env->CallVoidMethod(out, mid, str);
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

			raytracer.valid = 
				raytracer.Init &&
				raytracer.Destroy &&
				raytracer.Resize &&
				raytracer.Raytrace;

			if (raytracer.valid) {
				Log(env, "Successfully (re)loaded Raytracer DLL.");
			}
		}
	}

	// If any functions fail to load, set all to 0
	if (!raytracer.valid) {
		raytracer.Init = NULL;
		raytracer.Destroy = NULL;
		raytracer.Resize = NULL;
		raytracer.Raytrace = NULL;
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
	// Load DLL
	FILETIME NewDLLWriteTime = mjGetLastWriteTime((char*)g_dllName);

	if (CompareFileTime(&NewDLLWriteTime, &g_raytracer.DLLLastWriteTime) != 0) {
		Unload(env);
		for (int i = 0; !g_raytracer.valid && (i < 100); i++) {
			g_raytracer = Load(env);
			Sleep(100);
		}

		// Reload static memory
		g_raytracer.Init(env);

		return true;
	}

	return false;
}

JNIEXPORT void JNICALL Java_com_marcojonkers_mcraytracer_Raytracer_init
(JNIEnv* env, jobject) {
	ReloadIfNecessary(env);
	//g_raytracer.Init(); // Redundant
}

JNIEXPORT void JNICALL Java_com_marcojonkers_mcraytracer_Raytracer_resize
(JNIEnv* env, jobject, jint width, jint height) {
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
