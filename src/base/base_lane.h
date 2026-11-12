#ifndef BASE_LANE_H
#define BASE_LANE_H

// Lane vocabulary for code that is written once and compiled three ways:
//
//   V = f64     scalar: CPU reference path, and one CUDA thread per lane
//   V = f64x4   AVX2: four lanes per register (defined in base_simd.h)
//
// Templates written against these functions (src/sim/kinetics/kin_solver.h)
// must use only the operations below so every instantiation performs the
// same IEEE operations in the same order. Together with -ffp-contract=off
// (clang/gcc) and -fmad=false (nvcc) that makes the scalar, AVX2 and CUDA
// results bit-identical. That is a test, not a hope: see src/tests/.
//
// Rules for lane code:
//   - no libm calls (exp/pow/cbrt differ between CPU libm and CUDA);
//     sqrt and division are IEEE correctly rounded everywhere, so they are allowed
//   - fused multiply-add only through lane_fmadd / lane_fnmadd
//   - min/max through lane_min / lane_max (defined as "a < b ? a : b" so NaN
//     and signed-zero behaviour matches _mm256_min_pd / _mm256_max_pd exactly)
//
// This header has no intrinsics and is safe to include from nvcc.

#include "base_types.h"

#if defined(__CUDACC__)
#define SIM_HD __host__ __device__ __forceinline__
#else
#define SIM_HD SIM_FORCE_INLINE
#endif

#if defined(__CUDACC__)
#define LANE_FMA(a, b, c) fma((a), (b), (c))
#define LANE_SQRT(a) sqrt(a)
#else
#define LANE_FMA(a, b, c) __builtin_fma((a), (b), (c))
#define LANE_SQRT(a) __builtin_sqrt(a)
#endif

// Width and mask type of a lane vector. Specialised for f64x4 in base_simd.h.
template <class V> struct LaneTraits;
template <> struct LaneTraits<f64> {
    typedef bool Mask;
    enum { width = 1 };
};

SIM_HD f64  lane_splat(f64 *, f64 a) { return a; }          // tag-dispatched: lane_splat((V *)0, x)
SIM_HD f64  lane_load(const f64 *p, f64 *) { return *p; }
SIM_HD void lane_store(f64 *p, f64 a) { *p = a; }

SIM_HD f64  lane_fmadd(f64 a, f64 b, f64 c) { return LANE_FMA(a, b, c); }     // a*b + c, one rounding
SIM_HD f64  lane_fnmadd(f64 a, f64 b, f64 c) { return LANE_FMA(-a, b, c); }   // c - a*b, one rounding
SIM_HD f64  lane_min(f64 a, f64 b) { return a < b ? a : b; }
SIM_HD f64  lane_max(f64 a, f64 b) { return a > b ? a : b; }
SIM_HD f64  lane_abs(f64 a) { return a < 0.0 ? -a : a; }
SIM_HD f64  lane_sqrt(f64 a) { return LANE_SQRT(a); }

SIM_HD bool lane_lt(f64 a, f64 b) { return a < b; }
SIM_HD bool lane_le(f64 a, f64 b) { return a <= b; }
SIM_HD bool lane_gt(f64 a, f64 b) { return a > b; }
SIM_HD bool lane_ge(f64 a, f64 b) { return a >= b; }
SIM_HD bool lane_eq(f64 a, f64 b) { return a == b; }
SIM_HD bool lane_and(bool a, bool b) { return a && b; }
SIM_HD bool lane_or(bool a, bool b) { return a || b; }
SIM_HD bool lane_not(bool a) { return !a; }
SIM_HD bool lane_any(bool a) { return a; }
SIM_HD bool lane_all(bool a) { return a; }
SIM_HD bool lane_mask_splat(bool *, bool a) { return a; }
SIM_HD f64  lane_select(bool m, f64 a, f64 b) { return m ? a : b; }
SIM_HD bool lane_select_mask(bool m, bool a, bool b) { return m ? a : b; }
// a is finite (not NaN, not +-inf). x - x is 0 for finite x and NaN otherwise.
SIM_HD bool lane_is_finite(f64 a) { f64 d = a - a; return d == d; }

// Storage is f64 for every lane precision; these convert on the way in/out.
SIM_HD f64  lane_load_f64(const f64 *p, f64 *) { return *p; }
SIM_HD void lane_store_f64(f64 *p, f64 a) { *p = a; }
SIM_HD void lane_accumulate_f64(f64 *p, f64 a) { *p += a; }

SIM_HD f64  lane_get(f64 a, u32) { return a; }
SIM_HD void lane_set(f64 *a, u32, f64 x) { *a = x; }

// x^(1/3) for x in [1e-4, 1e4], from arithmetic only (no libm), identical on
// every path. Used by step-size control, where the argument is clamped first.
// x^(1/4) * x^(1/16) * x^(1/64) = x^(21/64) is within 5 % of the cube root
// on this range (6 square roots); two Newton steps take it to ~1e-6.
template <class V>
SIM_HD V lane_cbrt_clamped(V x) {
    V lo = lane_splat((V *)0, 1e-4), hi = lane_splat((V *)0, 1e4);
    x = lane_min(lane_max(x, lo), hi);
    V r4 = lane_sqrt(lane_sqrt(x));
    V r16 = lane_sqrt(lane_sqrt(r4));
    V r64 = lane_sqrt(lane_sqrt(r16));
    V g = r4 * r16 * r64;
    V two = lane_splat((V *)0, 2.0), third = lane_splat((V *)0, 1.0 / 3.0);
    g = (two * g + x / (g * g)) * third;
    g = (two * g + x / (g * g)) * third;
    return g;
}

#endif // BASE_LANE_H
