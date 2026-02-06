@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat" > nul 2>&1

set CMAKE="C:\Program Files\Microsoft Visual Studio\2022\Enterprise\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
set BUILD_DIR=build_msvc
set LOG=build_log.txt

echo === CMAKE VERSION === > %LOG% 2>&1
%CMAKE% --version >> %LOG% 2>&1

if exist %BUILD_DIR% rmdir /s /q %BUILD_DIR% >> %LOG% 2>&1

echo. >> %LOG% 2>&1
echo === CONFIGURING (WebGPU ON) === >> %LOG% 2>&1
%CMAKE% -B %BUILD_DIR% -G "Visual Studio 17 2022" -A x64 -DGEMMA3_USE_WEBGPU=ON -DGEMMA3_USE_THREADS=ON "%~dp0" >> %LOG% 2>&1
echo Configure exit: %ERRORLEVEL% >> %LOG% 2>&1

echo. >> %LOG% 2>&1
echo === BUILDING === >> %LOG% 2>&1
%CMAKE% --build %BUILD_DIR% --config Release >> %LOG% 2>&1
echo Build exit: %ERRORLEVEL% >> %LOG% 2>&1
