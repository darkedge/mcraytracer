@echo off
where /q javah.exe
set var=%PATH%;%JAVA_HOME%\bin
IF ERRORLEVEL 1 (
    set "PATH=%var%"
)
:: Would be nice to get the path to the forgeSrc jar file somehow
javah -o loader_jni.h -classpath build\classes\production\mcraytracer_main;C:\Users\Marco\.gradle\caches\minecraft\net\minecraftforge\forge\1.10.2-12.18.2.2099\snapshot\20160518\forgeSrc-1.10.2-12.18.2.2099.jar com.marcojonkers.mcraytracer.Raytracer
