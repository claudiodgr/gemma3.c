@echo off
setlocal enabledelayedexpansion

call "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat" > nul 2>&1

set "CMAKE=C:\Program Files\Microsoft Visual Studio\2022\Enterprise\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
set "SRC=C:\Users\Authority\Desktop\C++ projects\gemma3.c"
set "BUILD=C:\Users\Authority\Desktop\C++ projects\gemma3.c\build_msvc"
set "LOG=C:\Users\Authority\Desktop\C++ projects\gemma3.c\build_output.txt"

echo === CONFIGURE === > "%LOG%"
"%CMAKE%" -B "%BUILD%" -G "Visual Studio 17 2022" -A x64 -DGEMMA3_USE_WEBGPU=ON -DGEMMA3_USE_THREADS=ON "%SRC%" >> "%LOG%" 2>&1
echo EXIT_CONFIGURE=!ERRORLEVEL! >> "%LOG%"
if !ERRORLEVEL! neq 0 (
    echo Configure FAILED. See build_output.txt
    exit /b 1
)

echo. >> "%LOG%"
echo === BUILD === >> "%LOG%"
"%CMAKE%" --build "%BUILD%" --config Release --clean-first >> "%LOG%" 2>&1
echo EXIT_BUILD=!ERRORLEVEL! >> "%LOG%"
if !ERRORLEVEL! neq 0 (
    echo Build FAILED. See build_output.txt
    exit /b 1
)

echo Build OK.
endlocal
