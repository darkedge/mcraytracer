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

	jstring str = env->NewStringUTF(stdstr.c_str());
	env->CallVoidMethod(jni_system_out, jni_println, str);
}
