# Building and running Simstack

Simstack is a unity build: one compiler invocation per target, no build system.
Two targets exist in milestone 1, a third is optional:

| Target | Source root | Output (Linux / Windows) |
| --- | --- | --- |
| platform executable | `src/platform/linux_simstack.cpp` / `src/platform/win32_simstack.cpp` | `build/simstack` / `build\simstack.exe` |
| sim module (hot-reloadable) | `src/sim/simstack.cpp` | `build/simstack_sim.so` / `build\simstack_sim.dll` |
| gpu module (optional, needs CUDA 12.x) | `src/gpu/kernels.cu` | `build/simstack_gpu.so` / not wired on Windows yet |

Every other `.cpp` under `src/sim/` and `src/platform/` is `#include`d by its
root file. Headers in `src/base/` and `src/sim_platform.h` are shared.

## Requirements

- C++20 compiler: clang 17+ or gcc 13+ on Linux; MSVC 2022 (`cl`) or `clang-cl` on Windows.
- An x86-64 CPU with AVX2 + FMA (the build targets the i9-12900KF; no AVX-512 is used).
- Linux: Xlib and OpenGL development headers (`libx11-dev`, `libgl-dev`, `libglx-dev` on Debian/Ubuntu). Any GL 3.3+ core driver works; 4.5 is requested first. Mesa `llvmpipe` is enough for the windowed demo.
- Windows: Windows SDK (comes with Visual Studio); `opengl32.lib`, `user32.lib`, `gdi32.lib` are part of it.
- Optional: CUDA toolkit 12.x or 13.x with `nvcc` for the gpu module (13.4 confirmed on an RTX 3060 12 GB). If `nvcc` is not on `PATH` (e.g. `/opt/cuda/bin`), set `NVCC=/opt/cuda/bin/nvcc` or `CUDA_HOME=/opt/cuda`; `build.sh --cuda` also probes `/opt/cuda` and `/usr/local/cuda`.

No third-party libraries. No C++ runtime library is linked on Linux (`-nostdlib++`):
there are no exceptions, no RTTI, no `new`/`delete`, no STL.

## Linux

```sh
./build.sh                 # debug: -O0 -g, asserts on, SIM_INTERNAL=1
./build.sh release         # -O2 -g, asserts off, profiler on
./build.sh ship            # -O3, SIM_INTERNAL=0 (profiler compiled out)
./build.sh release --simd=0    # scalar f32x8 fallback (determinism check)
./build.sh debug --asan        # AddressSanitizer + UBSan
./build.sh release --sim-only  # rebuild only simstack_sim.so (hot reload)
./build.sh release --cuda      # also build simstack_gpu.so with nvcc
CUDA_HOME=/opt/cuda ./build.sh release --cuda   # nvcc off PATH (or NVCC=/opt/cuda/bin/nvcc)
CXX=g++ ./build.sh release     # use gcc instead of clang
```

A full debug build of both targets takes well under a second on a modern
desktop; the sim module alone is ~0.3 s, which is what makes hot reload useful.

Run:

```sh
./build/simstack                          # windowed, 65536 particles
./build/simstack --particles 262144 --size 1920x1080
./build/simstack --headless 600           # no window: 600 frames at a fixed 1/60 s, timings on stdout
./build/simstack --frames 240 --screenshot out.ppm   # windowed, quit after 240 frames, dump the last frame
./build/simstack --help
```

Keys in the window: `Space`/`P` pause, `R` reset particles, `F1` toggle camera
auto-orbit, arrow keys orbit, `Esc` quit. The window title shows ms/frame; the
profiler table and arena usage are printed to stdout every 120 frames unless
`--quiet` is given.

### Hot reload

Leave `./build/simstack` running, edit anything under `src/sim/` (for example
`PARTICLE_SWIRL` in `src/sim/simstack.cpp`), then run
`./build.sh release --sim-only`. The platform polls the module's mtime every
frame, waits while `build/lock.tmp` exists (the build script holds it while
linking), copies the new `.so` to a temporary name, `dlopen`s it, and calls
`sim_on_reload`. All state lives in the memory block the platform owns, so the
particle positions and step counter carry over; the title bar shows
`[reloaded]` and stdout prints the preserved step count. Reload also works in
`--headless` mode.

### Headless output

```
[platform] file io selftest: ok
[platform] memory 384 MB at 0x200000000000, sim /.../build/simstack_sim.so, 2400006 cycles/ms, simd=1
[headless] 600 frames in 0.410 s (0.682 ms/frame avg, 0.628 min, 2.051 max), 0 reloads
[headless] 10000 fixed steps (dt 0.0010 s), sim time 10.000 s, 65536 particles, 1601.2 M particle-steps/s
[headless] state hash ad345be205e7478d
  arenas: permanent 1.5/256 MB  frame 0.00/64 MB  scratch high-water 32 B/64 MB
  timed block                  hits/frame cycles/frame   ms/frame
  sim_update                         1.00      1636757      0.682   (src/sim/simstack.cpp:129)
  ...
```

The state hash is an FNV-1a over the particle positions after the last frame.
It must be identical across runs, across `--simd=0` / `--simd=1`, across
debug / release / ship, and across clang / gcc for the same particle count and
seed (`-ffp-contract=off` and an explicit `fmadd` in both SIMD paths keep the
arithmetic bit-exact). Reference values for the defaults (65536 particles,
seed 1): `--headless 120` gives 2000 steps and hash `d0dd6ee26223a1ec`;
`--headless 600` gives 10000 steps and hash `ad345be205e7478d`. The step
accumulator is `f64`, so N frames of the fixed 1/60 s headless dt always
produce exactly the expected number of 1 ms steps.

## Windows

From an "x64 Native Tools Command Prompt for VS 2022" (or any shell where `cl`
is on `PATH`):

```bat
build.bat                 REM debug
build.bat release
build.bat ship
build.bat release simd0   REM scalar fallback
build.bat release simonly REM rebuild only simstack_sim.dll
set CXX=clang-cl && build.bat release
```

Run `build\simstack.exe` with the same arguments as on Linux. The executable is
built as a console program so timings go to the console.

Hot reload on Windows uses the same scheme as Linux (copy to a temp DLL,
`LoadLibrary`, poll mtime, honour `build\lock.tmp`) plus the PDB trick: every
DLL build gets a random PDB file name so a debugger that locked the previous PDB
does not block the linker.

The Win32 layer and `build.bat` were written against the shared interface but
have not yet been compiled on a Windows machine; expect to fix small things on
the first build.

## Build flags and modes

| Mode | Optimisation | `SIM_INTERNAL` | `SIM_SLOW` (asserts) | Notes |
| --- | --- | --- | --- | --- |
| debug | `-O0 -g` / `/Od /Zi /RTC1` | 1 | 1 | `--asan` adds ASan + UBSan on Linux |
| release | `-O2 -g` / `/O2 /Zi` | 1 | 0 | profiler and debug table on |
| ship | `-O3` / `/O2` | 0 | 0 | `TIMED_BLOCK` compiles to nothing |

Architecture flags: `-mavx2 -mfma -mbmi2 -mf16c -ffp-contract=off` (Linux) and
`/arch:AVX2 /fp:precise` (Windows). `-DSIM_SIMD=0` selects the scalar `f32x8`.

## Layout

```
build.sh  build.bat
src/
  sim_platform.h            platform <-> sim contract: SimMemory, SimInput, RenderCommands, PlatformAPI, exports
  base/base_types.h         u8..f64, Assert, Kilobytes, hash, xorshift RNG
  base/base_arena.h         Arena, push_struct/push_array, TempMemory, arena_split, SoA helpers
  base/base_math.h          v2/v3/v4/m4
  base/base_simd.h          f32x8 (AVX2 or scalar)
  base/base_debug.h         TIMED_BLOCK cycle counters, DebugTable
  platform/linux_simstack.cpp   Xlib/GLX exe: window, input, timing, file IO, mmap, dlopen hot reload, headless
  platform/win32_simstack.cpp   Win32/WGL exe, same interface (untested)
  platform/platform_common.cpp  args, memory sizes, profiler printing (included by both)
  platform/render_gl.cpp        GL loader + SoA point renderer + PPM screenshot (included by both)
  sim/simstack.cpp          sim module root: fixed-step loop, camera, exports
  sim/particles.cpp         SoA particle system stepped with f32x8
  gpu/gpu_api.h             C ABI of the optional CUDA module
  gpu/kernels.cu            nvcc TU: GPU bump arena + point kernel (optional)
docs/BUILD.md               this file
```
