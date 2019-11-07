solution ("meshtrack-".._ACTION)
  configurations { "Debug", "Profiling", "Release" }
  platforms { "x64" }

  project ("meshtrack-".._ACTION)
    kind "ConsoleApp"
    language "C++"
    files { "src/*.h", "src/*.cpp", "src/lbfgs/*.c", "src/ffmpeg-fas/*.cpp", "src/gui/*.cpp", "src/gui/*.h", "src/video/video.cpp", "src/video/md5.c", "src/poisson_disk_wrapper/*.cpp", "src/video/ffmpeg-fas/*.cpp" }
    includedirs { "src", "src/vcglib", "src/vcglib/eigenlib","src/gui/include", "src/lbfgs", "src/video", "src/video/ffmpeg-fas", "src/video/libav/include", "src/video", "src/poisson_disk_wrapper", "src/chromium/", "src/eigen/", "src/fbx/include" }
    libdirs { "src", "src/gui/lib", "src/video/libav/lib" }
    links { "gui", "glew32", "avcodec", "avdevice", "avfilter", "avformat", "avutil", "postproc", "swresample", "swscale", "opengl32", "glu32", "libfbxsdk-mt", "libxml2-mt", "zlib-mt.lib"}
    buildoptions { "/wd4018","/wd4996","/wd4244","/wd4305","/wd4312" }
    objdir "tmp/meshtrack"
    targetdir "bin"
    targetname "meshtrack"
    debugargs { "C:/Users/Petul/.work/kostilam/data/stepanka ./calibration/gph0%d.pkrc 1 8 ./takes/take0/gph0%d_4.mp4 ./takes/take0/bkg0%d.png stepanka_v4.skin stepanka.trsa" }
	characterset "MBCS"

  configuration "Debug"
    defines { "DEBUG", "_SILENCE_STDEXT_HASH_DEPRECATION_WARNINGS" }
    flags { "Symbols", "StaticRuntime", "EnableSSE2", "FloatFast", "NoPCH" }
	libdirs { "src/fbx/lib/vs2017/x64/debug" }
	symbols "On"
    buildoptions { "/openmp" }

  configuration "Profiling"
    defines { "NDEBUG", "_SILENCE_STDEXT_HASH_DEPRECATION_WARNINGS" }
    flags { "Symbols", "StaticRuntime", "OptimizeSpeed", "EnableSSE2", "FloatFast", "NoPCH" }
	symbols "On"
	libdirs { "src/fbx/lib/vs2017/x64/release" }

  configuration "Release"
    defines { "NDEBUG", "_SILENCE_STDEXT_HASH_DEPRECATION_WARNINGS" }
    flags { "StaticRuntime", "OptimizeSpeed", "EnableSSE2", "FloatFast", "NoFramePointer", "NoPCH" }
	libdirs { "src/fbx/lib/vs2017/x64/release" }
    buildoptions { "/openmp" }
