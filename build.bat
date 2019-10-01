@echo off
setlocal ENABLEDELAYEDEXPANSION

for %%V in (14,12,11,10) do if exist "!VS%%V0COMNTOOLS!" call "!VS%%V0COMNTOOLS!..\..\VC\vcvarsall.bat" amd64 && goto compile
echo Unable to detect Visual Studio path!
goto error

:compile

::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
pushd src

cl /MP /DNDEBUG main.cpp timer.cpp poisson_disk_wrapper/utils_sampling.cpp poisson_disk_wrapper/vcg_mesh.cpp lbfgs\lbfgs.c video\video.cpp video\md5.c video\ffmpeg-fas\ffmpeg_fas.cpp video\ffmpeg-fas\seek_indices.cpp gui.lib glew32.lib avcodec.lib avdevice.lib avfilter.lib avformat.lib avutil.lib postproc.lib swresample.lib swscale.lib opengl32.lib glu32.lib /D_SILENCE_STDEXT_HASH_DEPRECATION_WARNINGS /I"." /I"poisson_disk_wrapper" /I"vcglib" /I"vcglib/eigenlib" /I"chromium" /I"eigen" /I"gui\include" /I"lbfgs" /I"video" /I"video\ffmpeg-fas" /I"video\libav\include" /Zi /O2 /Ox /Oy /Gy /fp:fast /openmp /EHsc /wd4018 /wd4996 /wd4244 /wd4305 /wd4312 /Fe"..\bin\meshtrack.exe" /link /LIBPATH:"gui\lib" /LIBPATH:"video\libav\lib" /MACHINE:X64 || goto error

popd
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

goto :EOF

:error
popd
echo FAILED
@%COMSPEC% /C exit 1 >nul
