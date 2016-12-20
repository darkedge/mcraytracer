@echo off
where /q javah.exe
set var=%PATH%;%JAVA_HOME%\bin
IF ERRORLEVEL 1 (
    set "PATH=%var%"
)
:: TODO: Get path to forgeSrc jar dynamically
:: otherwise this file is useless to the public
:: Note: loader_jni.h is in version control
javah -o loader_jni.h -classpath build\classes\production\mcraytracer_main;C:\Users\Marco\.gradle\caches\minecraft\net\minecraftforge\forge\1.10.2-12.18.2.2099\snapshot\20160518\forgeSrc-1.10.2-12.18.2.2099.jar com.marcojonkers.mcraytracer.Raytracer
