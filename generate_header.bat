@echo off
where /q javah.exe
set var=%PATH%;%JAVA_HOME%\bin
IF ERRORLEVEL 1 (
    set "PATH=%var%"
)
::javac -classpath ..\..\..\..\..\..\build\libs\* Raytracer.java
javah -o C:\Users\Marco\Desktop\raytracer_native\raytracer_jni.h -classpath build\classes\production\minecraft_raytracer_main;build\tmp\recompileMc\compiled com.marcojonkers.mcraytracer.Raytracer
