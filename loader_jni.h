/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class com_marcojonkers_mcraytracer_Raytracer */

#ifndef _Included_com_marcojonkers_mcraytracer_Raytracer
#define _Included_com_marcojonkers_mcraytracer_Raytracer
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     com_marcojonkers_mcraytracer_Raytracer
 * Method:    init
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_com_marcojonkers_mcraytracer_Raytracer_init
  (JNIEnv *, jobject);

/*
 * Class:     com_marcojonkers_mcraytracer_Raytracer
 * Method:    resize
 * Signature: (II)V
 */
JNIEXPORT void JNICALL Java_com_marcojonkers_mcraytracer_Raytracer_resize
  (JNIEnv *, jobject, jint, jint);

/*
 * Class:     com_marcojonkers_mcraytracer_Raytracer
 * Method:    raytrace
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_com_marcojonkers_mcraytracer_Raytracer_raytrace
  (JNIEnv *, jobject);

/*
 * Class:     com_marcojonkers_mcraytracer_Raytracer
 * Method:    setViewingPlane
 * Signature: (Ljava/nio/FloatBuffer;)V
 */
JNIEXPORT void JNICALL Java_com_marcojonkers_mcraytracer_Raytracer_setViewingPlane
  (JNIEnv *, jobject, jobject);

/*
 * Class:     com_marcojonkers_mcraytracer_Raytracer
 * Method:    setViewEntity
 * Signature: (DDD)V
 */
JNIEXPORT void JNICALL Java_com_marcojonkers_mcraytracer_Raytracer_setViewEntity
  (JNIEnv *, jobject, jdouble, jdouble, jdouble);

/*
 * Class:     com_marcojonkers_mcraytracer_Raytracer
 * Method:    stopProfiling
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_com_marcojonkers_mcraytracer_Raytracer_stopProfiling
  (JNIEnv *, jobject);

/*
 * Class:     com_marcojonkers_mcraytracer_Raytracer
 * Method:    setVertexBuffer
 * Signature: (IIILjava/nio/ByteBuffer;I)V
 */
JNIEXPORT void JNICALL Java_com_marcojonkers_mcraytracer_Raytracer_setVertexBuffer
  (JNIEnv *, jobject, jint, jint, jint, jobject, jint);

#ifdef __cplusplus
}
#endif
#endif
