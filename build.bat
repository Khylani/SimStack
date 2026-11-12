@echo off
setlocal
REM Simstack unity build (Windows). Run from a "x64 Native Tools Command Prompt"
REM for MSVC, or set CXX=clang-cl.
REM
REM   build.bat [debug|release|ship] [simd0] [simonly]
REM
REM Produces build\simstack.exe and build\simstack_sim.dll. build\lock.tmp exists
REM while the DLL is being written so the running exe never loads a half-linked
REM file, and each DLL build gets a random PDB name so the debugger does not pin
REM the previous one (the Handmade Hero trick).
REM
REM STATUS: not yet exercised on a Windows machine; see docs/BUILD.md.

set MODE=debug
set SIMD=1
set SIMONLY=0
for %%a in (%*) do (
    if "%%a"=="debug"   set MODE=debug
    if "%%a"=="release" set MODE=release
    if "%%a"=="ship"    set MODE=ship
    if "%%a"=="simd0"   set SIMD=0
    if "%%a"=="simonly" set SIMONLY=1
)
if not defined CXX set CXX=cl

REM i9-12900KF: AVX2 + FMA, no AVX-512. /fp:precise does not contract a*b+c, so the
REM scalar f32x8 fallback stays bit-identical to the AVX2 path.
set COMMON=/nologo /std:c++20 /EHs-c- /GR- /arch:AVX2 /fp:precise /W4 /WX- /FC /Isrc ^
    /wd4100 /wd4189 /wd4201 /wd4127 /wd4505 /wd4324 /D_CRT_SECURE_NO_WARNINGS /DSIM_SIMD=%SIMD%

if "%MODE%"=="debug"   set CFG=/Od /Zi /RTC1 /MTd /DSIM_INTERNAL=1 /DSIM_SLOW=1
if "%MODE%"=="release" set CFG=/O2 /Zi /MT /DSIM_INTERNAL=1 /DSIM_SLOW=0
if "%MODE%"=="ship"    set CFG=/O2 /MT /DSIM_INTERNAL=0 /DSIM_SLOW=0 /DNDEBUG

if not exist build mkdir build
echo [build] %MODE%  cxx=%CXX%  simd=%SIMD%

REM --- sim module -------------------------------------------------------------
del /q build\simstack_sim_*.pdb >nul 2>nul
echo lock > build\lock.tmp
%CXX% %COMMON% %CFG% /LD src\sim\simstack.cpp /Fo:build\simstack_sim.obj /Fe:build\simstack_sim.dll ^
    /link /INCREMENTAL:NO /PDB:build\simstack_sim_%random%.pdb
set ERR=%errorlevel%
del build\lock.tmp
if not "%ERR%"=="0" exit /b %ERR%
echo [build] simstack_sim.dll

REM --- platform exe -----------------------------------------------------------
if "%SIMONLY%"=="1" exit /b 0
%CXX% %COMMON% %CFG% src\platform\win32_simstack.cpp /Fo:build\simstack.obj /Fe:build\simstack.exe ^
    /link /INCREMENTAL:NO /SUBSYSTEM:CONSOLE user32.lib gdi32.lib opengl32.lib
if not "%errorlevel%"=="0" exit /b %errorlevel%
echo [build] simstack.exe
