#ifndef BASE_SIMD_H
#define BASE_SIMD_H

// f32x8: eight f32 lanes. AVX2+FMA when SIM_SIMD=1, plain loops otherwise.
// Hot loops are written on this type explicitly; nothing relies on
// auto-vectorisation. The scalar fallback uses fmaf() for fmadd so that both
// paths produce identical results when the compiler is told not to contract
// (build.sh passes -ffp-contract=off).

#include "base_types.h"
#include <math.h>

#if SIM_SIMD
#include <immintrin.h>

struct f32x8 { __m256 v; };

SIM_FORCE_INLINE f32x8 f32x8_set1(f32 a) { f32x8 r; r.v = _mm256_set1_ps(a); return r; }
SIM_FORCE_INLINE f32x8 f32x8_zero() { f32x8 r; r.v = _mm256_setzero_ps(); return r; }
SIM_FORCE_INLINE f32x8 f32x8_load(const f32 *p) { f32x8 r; r.v = _mm256_load_ps(p); return r; }
SIM_FORCE_INLINE f32x8 f32x8_loadu(const f32 *p) { f32x8 r; r.v = _mm256_loadu_ps(p); return r; }
SIM_FORCE_INLINE void f32x8_store(f32 *p, f32x8 a) { _mm256_store_ps(p, a.v); }
SIM_FORCE_INLINE void f32x8_storeu(f32 *p, f32x8 a) { _mm256_storeu_ps(p, a.v); }
SIM_FORCE_INLINE f32x8 f32x8_iota() { f32x8 r; r.v = _mm256_setr_ps(0, 1, 2, 3, 4, 5, 6, 7); return r; }

SIM_FORCE_INLINE f32x8 operator+(f32x8 a, f32x8 b) { f32x8 r; r.v = _mm256_add_ps(a.v, b.v); return r; }
SIM_FORCE_INLINE f32x8 operator-(f32x8 a, f32x8 b) { f32x8 r; r.v = _mm256_sub_ps(a.v, b.v); return r; }
SIM_FORCE_INLINE f32x8 operator*(f32x8 a, f32x8 b) { f32x8 r; r.v = _mm256_mul_ps(a.v, b.v); return r; }
SIM_FORCE_INLINE f32x8 operator/(f32x8 a, f32x8 b) { f32x8 r; r.v = _mm256_div_ps(a.v, b.v); return r; }
SIM_FORCE_INLINE f32x8 operator-(f32x8 a) { f32x8 r; r.v = _mm256_sub_ps(_mm256_setzero_ps(), a.v); return r; }

// a*b + c in one rounding
SIM_FORCE_INLINE f32x8 fmadd(f32x8 a, f32x8 b, f32x8 c) { f32x8 r; r.v = _mm256_fmadd_ps(a.v, b.v, c.v); return r; }
// c - a*b
SIM_FORCE_INLINE f32x8 fnmadd(f32x8 a, f32x8 b, f32x8 c) { f32x8 r; r.v = _mm256_fnmadd_ps(a.v, b.v, c.v); return r; }

SIM_FORCE_INLINE f32x8 f32x8_min(f32x8 a, f32x8 b) { f32x8 r; r.v = _mm256_min_ps(a.v, b.v); return r; }
SIM_FORCE_INLINE f32x8 f32x8_max(f32x8 a, f32x8 b) { f32x8 r; r.v = _mm256_max_ps(a.v, b.v); return r; }
SIM_FORCE_INLINE f32x8 f32x8_sqrt(f32x8 a) { f32x8 r; r.v = _mm256_sqrt_ps(a.v); return r; }
SIM_FORCE_INLINE f32x8 f32x8_abs(f32x8 a) {
    f32x8 r; r.v = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), a.v); return r;
}

// Comparisons yield all-ones / all-zero lane masks (as f32x8), consumed by select.
SIM_FORCE_INLINE f32x8 f32x8_lt(f32x8 a, f32x8 b) { f32x8 r; r.v = _mm256_cmp_ps(a.v, b.v, _CMP_LT_OQ); return r; }
SIM_FORCE_INLINE f32x8 f32x8_gt(f32x8 a, f32x8 b) { f32x8 r; r.v = _mm256_cmp_ps(a.v, b.v, _CMP_GT_OQ); return r; }
SIM_FORCE_INLINE f32x8 f32x8_le(f32x8 a, f32x8 b) { f32x8 r; r.v = _mm256_cmp_ps(a.v, b.v, _CMP_LE_OQ); return r; }
SIM_FORCE_INLINE f32x8 f32x8_ge(f32x8 a, f32x8 b) { f32x8 r; r.v = _mm256_cmp_ps(a.v, b.v, _CMP_GE_OQ); return r; }
SIM_FORCE_INLINE f32x8 f32x8_and(f32x8 a, f32x8 b) { f32x8 r; r.v = _mm256_and_ps(a.v, b.v); return r; }
SIM_FORCE_INLINE f32x8 f32x8_or(f32x8 a, f32x8 b) { f32x8 r; r.v = _mm256_or_ps(a.v, b.v); return r; }
SIM_FORCE_INLINE f32x8 f32x8_andnot(f32x8 mask, f32x8 a) { f32x8 r; r.v = _mm256_andnot_ps(mask.v, a.v); return r; }
// mask ? a : b
SIM_FORCE_INLINE f32x8 f32x8_select(f32x8 mask, f32x8 a, f32x8 b) {
    f32x8 r; r.v = _mm256_blendv_ps(b.v, a.v, mask.v); return r;
}
SIM_FORCE_INLINE b32 f32x8_any(f32x8 mask) { return _mm256_movemask_ps(mask.v) != 0; }
SIM_FORCE_INLINE b32 f32x8_all(f32x8 mask) { return _mm256_movemask_ps(mask.v) == 0xFF; }

// Horizontal sum, fixed order within the lane so it is deterministic.
SIM_FORCE_INLINE f32 f32x8_hsum(f32x8 a) {
    __m128 lo = _mm256_castps256_ps128(a.v);
    __m128 hi = _mm256_extractf128_ps(a.v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 0x55));
    return _mm_cvtss_f32(s);
}

SIM_FORCE_INLINE f32 f32x8_lane(f32x8 a, u32 i) {
    SIM_ALIGNED(32) f32 tmp[8];
    _mm256_store_ps(tmp, a.v);
    return tmp[i];
}

#else // ---------------------------------------------------------- scalar fallback

struct f32x8 { f32 e[8]; };

#define F32X8_LANE_OP(expr) f32x8 r; for (int i = 0; i < 8; ++i) { r.e[i] = (expr); } return r

SIM_FORCE_INLINE f32x8 f32x8_set1(f32 a) { F32X8_LANE_OP(a); }
SIM_FORCE_INLINE f32x8 f32x8_zero() { F32X8_LANE_OP(0.0f); }
SIM_FORCE_INLINE f32x8 f32x8_load(const f32 *p) { F32X8_LANE_OP(p[i]); }
SIM_FORCE_INLINE f32x8 f32x8_loadu(const f32 *p) { F32X8_LANE_OP(p[i]); }
SIM_FORCE_INLINE void f32x8_store(f32 *p, f32x8 a) { for (int i = 0; i < 8; ++i) p[i] = a.e[i]; }
SIM_FORCE_INLINE void f32x8_storeu(f32 *p, f32x8 a) { f32x8_store(p, a); }
SIM_FORCE_INLINE f32x8 f32x8_iota() { F32X8_LANE_OP((f32)i); }

SIM_FORCE_INLINE f32x8 operator+(f32x8 a, f32x8 b) { F32X8_LANE_OP(a.e[i] + b.e[i]); }
SIM_FORCE_INLINE f32x8 operator-(f32x8 a, f32x8 b) { F32X8_LANE_OP(a.e[i] - b.e[i]); }
SIM_FORCE_INLINE f32x8 operator*(f32x8 a, f32x8 b) { F32X8_LANE_OP(a.e[i] * b.e[i]); }
SIM_FORCE_INLINE f32x8 operator/(f32x8 a, f32x8 b) { F32X8_LANE_OP(a.e[i] / b.e[i]); }
SIM_FORCE_INLINE f32x8 operator-(f32x8 a) { F32X8_LANE_OP(0.0f - a.e[i]); }

SIM_FORCE_INLINE f32x8 fmadd(f32x8 a, f32x8 b, f32x8 c) { F32X8_LANE_OP(fmaf(a.e[i], b.e[i], c.e[i])); }
SIM_FORCE_INLINE f32x8 fnmadd(f32x8 a, f32x8 b, f32x8 c) { F32X8_LANE_OP(fmaf(-a.e[i], b.e[i], c.e[i])); }

SIM_FORCE_INLINE f32x8 f32x8_min(f32x8 a, f32x8 b) { F32X8_LANE_OP(b.e[i] < a.e[i] ? b.e[i] : a.e[i]); }
SIM_FORCE_INLINE f32x8 f32x8_max(f32x8 a, f32x8 b) { F32X8_LANE_OP(b.e[i] > a.e[i] ? b.e[i] : a.e[i]); }
SIM_FORCE_INLINE f32x8 f32x8_sqrt(f32x8 a) { F32X8_LANE_OP(sqrtf(a.e[i])); }
SIM_FORCE_INLINE f32x8 f32x8_abs(f32x8 a) { F32X8_LANE_OP(fabsf(a.e[i])); }

SIM_FORCE_INLINE f32 f32x8_mask_bits_(b32 c) { u32 bits = c ? 0xFFFFFFFFu : 0u; f32 f; memcpy(&f, &bits, 4); return f; }
SIM_FORCE_INLINE u32 f32x8_bits_(f32 f) { u32 bits; memcpy(&bits, &f, 4); return bits; }
SIM_FORCE_INLINE f32 f32x8_from_bits_(u32 bits) { f32 f; memcpy(&f, &bits, 4); return f; }

SIM_FORCE_INLINE f32x8 f32x8_lt(f32x8 a, f32x8 b) { F32X8_LANE_OP(f32x8_mask_bits_(a.e[i] < b.e[i])); }
SIM_FORCE_INLINE f32x8 f32x8_gt(f32x8 a, f32x8 b) { F32X8_LANE_OP(f32x8_mask_bits_(a.e[i] > b.e[i])); }
SIM_FORCE_INLINE f32x8 f32x8_le(f32x8 a, f32x8 b) { F32X8_LANE_OP(f32x8_mask_bits_(a.e[i] <= b.e[i])); }
SIM_FORCE_INLINE f32x8 f32x8_ge(f32x8 a, f32x8 b) { F32X8_LANE_OP(f32x8_mask_bits_(a.e[i] >= b.e[i])); }
SIM_FORCE_INLINE f32x8 f32x8_and(f32x8 a, f32x8 b) { F32X8_LANE_OP(f32x8_from_bits_(f32x8_bits_(a.e[i]) & f32x8_bits_(b.e[i]))); }
SIM_FORCE_INLINE f32x8 f32x8_or(f32x8 a, f32x8 b) { F32X8_LANE_OP(f32x8_from_bits_(f32x8_bits_(a.e[i]) | f32x8_bits_(b.e[i]))); }
SIM_FORCE_INLINE f32x8 f32x8_andnot(f32x8 mask, f32x8 a) { F32X8_LANE_OP(f32x8_from_bits_(~f32x8_bits_(mask.e[i]) & f32x8_bits_(a.e[i]))); }
SIM_FORCE_INLINE f32x8 f32x8_select(f32x8 mask, f32x8 a, f32x8 b) { F32X8_LANE_OP((f32x8_bits_(mask.e[i]) & 0x80000000u) ? a.e[i] : b.e[i]); }
SIM_FORCE_INLINE b32 f32x8_any(f32x8 mask) { for (int i = 0; i < 8; ++i) if (f32x8_bits_(mask.e[i]) & 0x80000000u) return 1; return 0; }
SIM_FORCE_INLINE b32 f32x8_all(f32x8 mask) { for (int i = 0; i < 8; ++i) if (!(f32x8_bits_(mask.e[i]) & 0x80000000u)) return 0; return 1; }

// Same association order as the AVX2 version: (0+4)+(2+6) then +((1+5)+(3+7)).
SIM_FORCE_INLINE f32 f32x8_hsum(f32x8 a) {
    f32 s0 = a.e[0] + a.e[4], s1 = a.e[1] + a.e[5], s2 = a.e[2] + a.e[6], s3 = a.e[3] + a.e[7];
    f32 t0 = s0 + s2, t1 = s1 + s3;
    return t0 + t1;
}

SIM_FORCE_INLINE f32 f32x8_lane(f32x8 a, u32 i) { return a.e[i]; }

#undef F32X8_LANE_OP
#endif // SIM_SIMD

// Derived helpers, identical for both paths.
SIM_FORCE_INLINE f32x8 operator*(f32x8 a, f32 s) { return a * f32x8_set1(s); }
SIM_FORCE_INLINE f32x8 operator*(f32 s, f32x8 a) { return f32x8_set1(s) * a; }
SIM_FORCE_INLINE f32x8 operator+(f32x8 a, f32 s) { return a + f32x8_set1(s); }
SIM_FORCE_INLINE f32x8 operator-(f32x8 a, f32 s) { return a - f32x8_set1(s); }
SIM_FORCE_INLINE f32x8 &operator+=(f32x8 &a, f32x8 b) { a = a + b; return a; }
SIM_FORCE_INLINE f32x8 &operator-=(f32x8 &a, f32x8 b) { a = a - b; return a; }
SIM_FORCE_INLINE f32x8 &operator*=(f32x8 &a, f32x8 b) { a = a * b; return a; }
SIM_FORCE_INLINE f32x8 f32x8_clamp(f32x8 x, f32x8 lo, f32x8 hi) { return f32x8_min(f32x8_max(x, lo), hi); }
SIM_FORCE_INLINE f32x8 f32x8_lerp(f32x8 a, f32x8 t, f32x8 b) { return fmadd(t, b - a, a); }

#endif // BASE_SIMD_H
