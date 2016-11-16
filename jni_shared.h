#pragma once
#include <jni.h>
#include <string>

void CacheJNI(JNIEnv*);
void Log(JNIEnv*, const std::string&);
