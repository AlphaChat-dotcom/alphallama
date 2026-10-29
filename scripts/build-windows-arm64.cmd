@echo off
rem Build AlphaLlama natively on Windows on ARM (Snapdragon X etc.), CPU backend.
rem
rem Needs: VS 2022 Build Tools (C++ workload + "MSVC ARM64 build tools" + Windows SDK + CMake)
rem        and LLVM/clang for Windows ARM64 (https://github.com/llvm/llvm-project/releases, *-woa64.exe).
rem ggml-cpu refuses MSVC for ARM, so clang compiles everything; MSVC supplies the linker, SDK and CRT.
rem
rem Usage: scripts\build-windows-arm64.cmd [jobs] [target] [static]
rem        defaults: jobs=3 (8 parallel clang jobs exhaust 16 GB of RAM), target=alphallama, shared DLLs
rem        static: one self-contained exe per tool in build-win-arm64-static (BUILD_SHARED_LIBS=OFF). Use it when
rem                Windows Smart App Control is on: it refuses some freshly built unsigned DLLs (seen with
rem                llama-common.dll and mtmd.dll, exit 0xC0E90002 before main), and fewer new files means fewer
rem                refusals. It can still refuse an exe; that one then has to be rebuilt or run elsewhere.
rem Output: build-win-arm64[-static]\bin\<target>.exe (+ ggml/llama DLLs unless static, and libomp140.aarch64.dll)

setlocal
set "JOBS=%~1"
if "%JOBS%"=="" set "JOBS=3"
set "TARGET=%~2"
if "%TARGET%"=="" set "TARGET=alphallama"
set "BDIR=build-win-arm64"
set "SHARED=ON"
if /i "%~3"=="static" (
  set "BDIR=build-win-arm64-static"
  set "SHARED=OFF"
)

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -version [17.0^,18.0^) -requires Microsoft.VisualStudio.Component.VC.Tools.ARM64 -property installationPath`) do set "VSDIR=%%i"
if not defined VSDIR (
  echo error: no VS 2022 install with the ARM64 C++ tools found
  exit /b 1
)
call "%VSDIR%\VC\Auxiliary\Build\vcvarsall.bat" arm64 >nul || exit /b 1

if exist "%ProgramFiles%\LLVM\bin\clang.exe" set "PATH=%ProgramFiles%\LLVM\bin;%PATH%"
where clang >nul 2>&1 || (
  echo error: clang not found; install LLVM for Windows ARM64 or the VS "C++ Clang Compiler" component
  exit /b 1
)

cd /d "%~dp0.."
cmake -S . -B %BDIR% -G Ninja ^
  -DCMAKE_TOOLCHAIN_FILE=cmake/arm64-windows-llvm.cmake ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DGGML_CUDA=OFF -DALPHA_FLASHINFER=OFF ^
  -DLLAMA_BUILD_TESTS=OFF -DBUILD_SHARED_LIBS=%SHARED% || exit /b 1
cmake --build %BDIR% -j %JOBS% --target %TARGET% || exit /b 1
echo.
echo Built %BDIR%\bin\%TARGET%.exe
