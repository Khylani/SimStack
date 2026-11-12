#ifndef BASE_TYPES_H
#define BASE_TYPES_H

// Base types and macros shared by every translation unit (platform, sim, gpu).
// Plain C headers only: no STL anywhere in the sim layer.

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <float.h>

typedef int8_t   s8;
typedef int16_t  s16;
typedef int32_t  s32;
typedef int64_t  s64;
typedef uint8_t  u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef float    f32;
typedef double   f64;
typedef s32      b32;
typedef size_t   umm;
typedef intptr_t smm;

#define U32_MAX 0xFFFFFFFFu
#define U64_MAX 0xFFFFFFFFFFFFFFFFull
#define F32_MAX FLT_MAX
#define F32_EPSILON FLT_EPSILON

#define internal      static
#define local_persist static
// `global` is not defined in CUDA translation units: nvcc expands __global__
// through __attribute__((global)), which this macro would turn into
// __attribute__((static)) and silently demote every kernel to a host function.
// CUDA code uses plain `static` for file-scope state.
#ifndef __CUDACC__
#define global        static
#endif

// Build configuration (set by build.sh / build.bat, defaults for stray TUs):
//   SIM_INTERNAL  1 = developer build (debug tables, asserts stay on)
//   SIM_SLOW      1 = expensive asserts on
//   SIM_SIMD      1 = f32x8 is AVX2, 0 = scalar fallback
#ifndef SIM_INTERNAL
#define SIM_INTERNAL 1
#endif
#ifndef SIM_SLOW
#define SIM_SLOW 1
#endif
#ifndef SIM_SIMD
#define SIM_SIMD 1
#endif

#if defined(_WIN32)
#define SIM_OS_WINDOWS 1
#define SIM_OS_LINUX 0
#else
#define SIM_OS_WINDOWS 0
#define SIM_OS_LINUX 1
#endif

#if defined(_MSC_VER)
#define SIM_COMPILER_MSVC 1
#define SIM_COMPILER_CLANG 0
#define SIM_COMPILER_GCC 0
#elif defined(__clang__)
#define SIM_COMPILER_MSVC 0
#define SIM_COMPILER_CLANG 1
#define SIM_COMPILER_GCC 0
#else
#define SIM_COMPILER_MSVC 0
#define SIM_COMPILER_CLANG 0
#define SIM_COMPILER_GCC 1
#endif

#if SIM_COMPILER_MSVC
#define SIM_TRAP() __debugbreak()
#define SIM_FORCE_INLINE __forceinline
#define SIM_ALIGNED(n) __declspec(align(n))
#else
#define SIM_TRAP() __builtin_trap()
#define SIM_FORCE_INLINE inline __attribute__((always_inline))
#define SIM_ALIGNED(n) __attribute__((aligned(n)))
#endif

#if SIM_SLOW
#define Assert(expr) do { if (!(expr)) { SIM_TRAP(); } } while (0)
#else
#define Assert(expr) ((void)0)
#endif
#define InvalidCodePath Assert(!"Invalid code path")
#define InvalidDefaultCase default: { InvalidCodePath; } break
#define StaticAssert(expr, name) typedef char static_assert_##name[(expr) ? 1 : -1]

#define Kilobytes(n) ((u64)(n) * 1024ull)
#define Megabytes(n) (Kilobytes(n) * 1024ull)
#define Gigabytes(n) (Megabytes(n) * 1024ull)

#define ArrayCount(a) (sizeof(a) / sizeof((a)[0]))
#define Minimum(a, b) ((a) < (b) ? (a) : (b))
#define Maximum(a, b) ((a) > (b) ? (a) : (b))
#define Clamp(x, lo, hi) Minimum(Maximum(x, lo), hi)
#define Swap(T, a, b) do { T swap_tmp_ = (a); (a) = (b); (b) = swap_tmp_; } while (0)

#define AlignPow2(x, a) (((x) + ((a) - 1)) & ~((u64)(a) - 1))
#define IsPow2(x) (((x) & ((x) - 1)) == 0)
#define OffsetOf(T, member) ((umm)&(((T *)0)->member))

#define ZeroStruct(s) memset(&(s), 0, sizeof(s))
#define ZeroArray(a, count) memset((a), 0, sizeof((a)[0]) * (count))
#define CopyArray(dst, src, count) memcpy((dst), (src), sizeof((dst)[0]) * (count))

#define CONCAT_(a, b) a##b
#define CONCAT(a, b) CONCAT_(a, b)
#define STRINGIFY_(x) #x
#define STRINGIFY(x) STRINGIFY_(x)

// Simple deterministic hash used for state checksums and replay verification.
SIM_FORCE_INLINE u64 hash_fnv1a_64(const void *data, u64 size, u64 seed = 14695981039346656037ull) {
    const u8 *p = (const u8 *)data;
    u64 h = seed;
    for (u64 i = 0; i < size; ++i) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

// xorshift32: deterministic, good enough for seeding particles.
struct RandomSeries {
    u32 state;
};

SIM_FORCE_INLINE u32 random_next_u32(RandomSeries *series) {
    u32 x = series->state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    series->state = x;
    return x;
}

SIM_FORCE_INLINE f32 random_unilateral(RandomSeries *series) {
    return (f32)(random_next_u32(series) >> 8) * (1.0f / 16777216.0f);
}

SIM_FORCE_INLINE f32 random_bilateral(RandomSeries *series) {
    return 2.0f * random_unilateral(series) - 1.0f;
}

#endif // BASE_TYPES_H
