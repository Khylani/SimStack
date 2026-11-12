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


// ----------------------------------------------------------------------------
// f64x4: four f64 lanes (AVX2) for the stiff kinetics solver, plus its mask
// type b64x4. Exposes the lane_* vocabulary of base_lane.h so templates
// written for V = f64 also compile for V = f64x4 with identical arithmetic.

#include "base_lane.h"

#if SIM_SIMD

struct f64x4 { __m256d v; };
struct b64x4 { __m256d v; };   // all-ones / all-zeros per lane

SIM_FORCE_INLINE f64x4 lane_splat(f64x4 *, f64 a) { f64x4 r; r.v = _mm256_set1_pd(a); return r; }
SIM_FORCE_INLINE f64x4 lane_load(const f64 *p, f64x4 *) { f64x4 r; r.v = _mm256_load_pd(p); return r; }
SIM_FORCE_INLINE void  lane_store(f64 *p, f64x4 a) { _mm256_store_pd(p, a.v); }

SIM_FORCE_INLINE f64x4 operator+(f64x4 a, f64x4 b) { f64x4 r; r.v = _mm256_add_pd(a.v, b.v); return r; }
SIM_FORCE_INLINE f64x4 operator-(f64x4 a, f64x4 b) { f64x4 r; r.v = _mm256_sub_pd(a.v, b.v); return r; }
SIM_FORCE_INLINE f64x4 operator*(f64x4 a, f64x4 b) { f64x4 r; r.v = _mm256_mul_pd(a.v, b.v); return r; }
SIM_FORCE_INLINE f64x4 operator/(f64x4 a, f64x4 b) { f64x4 r; r.v = _mm256_div_pd(a.v, b.v); return r; }
SIM_FORCE_INLINE f64x4 operator-(f64x4 a) { f64x4 r; r.v = _mm256_xor_pd(a.v, _mm256_set1_pd(-0.0)); return r; }

SIM_FORCE_INLINE f64x4 lane_fmadd(f64x4 a, f64x4 b, f64x4 c) { f64x4 r; r.v = _mm256_fmadd_pd(a.v, b.v, c.v); return r; }
SIM_FORCE_INLINE f64x4 lane_fnmadd(f64x4 a, f64x4 b, f64x4 c) { f64x4 r; r.v = _mm256_fnmadd_pd(a.v, b.v, c.v); return r; }
// _mm256_min_pd(a, b) returns b unless a < b: same as the scalar "a < b ? a : b".
SIM_FORCE_INLINE f64x4 lane_min(f64x4 a, f64x4 b) { f64x4 r; r.v = _mm256_min_pd(a.v, b.v); return r; }
SIM_FORCE_INLINE f64x4 lane_max(f64x4 a, f64x4 b) { f64x4 r; r.v = _mm256_max_pd(a.v, b.v); return r; }
SIM_FORCE_INLINE f64x4 lane_abs(f64x4 a) {
    // scalar path is "a < 0 ? -a : a", which keeps -0.0; match that rather than clearing the sign bit
    __m256d neg = _mm256_cmp_pd(a.v, _mm256_setzero_pd(), _CMP_LT_OQ);
    f64x4 r; r.v = _mm256_blendv_pd(a.v, _mm256_xor_pd(a.v, _mm256_set1_pd(-0.0)), neg); return r;
}
SIM_FORCE_INLINE f64x4 lane_sqrt(f64x4 a) { f64x4 r; r.v = _mm256_sqrt_pd(a.v); return r; }

SIM_FORCE_INLINE b64x4 lane_lt(f64x4 a, f64x4 b) { b64x4 r; r.v = _mm256_cmp_pd(a.v, b.v, _CMP_LT_OQ); return r; }
SIM_FORCE_INLINE b64x4 lane_le(f64x4 a, f64x4 b) { b64x4 r; r.v = _mm256_cmp_pd(a.v, b.v, _CMP_LE_OQ); return r; }
SIM_FORCE_INLINE b64x4 lane_gt(f64x4 a, f64x4 b) { b64x4 r; r.v = _mm256_cmp_pd(a.v, b.v, _CMP_GT_OQ); return r; }
SIM_FORCE_INLINE b64x4 lane_ge(f64x4 a, f64x4 b) { b64x4 r; r.v = _mm256_cmp_pd(a.v, b.v, _CMP_GE_OQ); return r; }
SIM_FORCE_INLINE b64x4 lane_eq(f64x4 a, f64x4 b) { b64x4 r; r.v = _mm256_cmp_pd(a.v, b.v, _CMP_EQ_OQ); return r; }
SIM_FORCE_INLINE b64x4 lane_and(b64x4 a, b64x4 b) { b64x4 r; r.v = _mm256_and_pd(a.v, b.v); return r; }
SIM_FORCE_INLINE b64x4 lane_or(b64x4 a, b64x4 b) { b64x4 r; r.v = _mm256_or_pd(a.v, b.v); return r; }
SIM_FORCE_INLINE b64x4 lane_not(b64x4 a) {
    b64x4 r; r.v = _mm256_xor_pd(a.v, _mm256_castsi256_pd(_mm256_set1_epi64x(-1))); return r;
}
SIM_FORCE_INLINE b32 lane_any(b64x4 a) { return _mm256_movemask_pd(a.v) != 0; }
SIM_FORCE_INLINE b32 lane_all(b64x4 a) { return _mm256_movemask_pd(a.v) == 0xF; }
SIM_FORCE_INLINE b64x4 lane_mask_splat(b64x4 *, bool a) {
    b64x4 r; r.v = _mm256_castsi256_pd(_mm256_set1_epi64x(a ? -1 : 0)); return r;
}
// m ? a : b
SIM_FORCE_INLINE f64x4 lane_select(b64x4 m, f64x4 a, f64x4 b) { f64x4 r; r.v = _mm256_blendv_pd(b.v, a.v, m.v); return r; }
SIM_FORCE_INLINE b64x4 lane_select_mask(b64x4 m, b64x4 a, b64x4 b) { b64x4 r; r.v = _mm256_blendv_pd(b.v, a.v, m.v); return r; }
SIM_FORCE_INLINE b64x4 lane_is_finite(f64x4 a) { f64x4 d = a - a; return lane_eq(d, d); }

SIM_FORCE_INLINE f64 lane_get(f64x4 a, u32 i) { SIM_ALIGNED(32) f64 t[4]; _mm256_store_pd(t, a.v); return t[i]; }
SIM_FORCE_INLINE void lane_set(f64x4 *a, u32 i, f64 x) {
    SIM_ALIGNED(32) f64 t[4]; _mm256_store_pd(t, a->v); t[i] = x; a->v = _mm256_load_pd(t);
}

#else // ---------------------------------------------------------- scalar fallback

struct f64x4 { f64 e[4]; };
struct b64x4 { bool e[4]; };

#define F64X4_OP(expr) f64x4 r; for (int i = 0; i < 4; ++i) { r.e[i] = (expr); } return r
#define B64X4_OP(expr) b64x4 r; for (int i = 0; i < 4; ++i) { r.e[i] = (expr); } return r

SIM_FORCE_INLINE f64x4 lane_splat(f64x4 *, f64 a) { F64X4_OP(a); }
SIM_FORCE_INLINE f64x4 lane_load(const f64 *p, f64x4 *) { F64X4_OP(p[i]); }
SIM_FORCE_INLINE void  lane_store(f64 *p, f64x4 a) { for (int i = 0; i < 4; ++i) p[i] = a.e[i]; }

SIM_FORCE_INLINE f64x4 operator+(f64x4 a, f64x4 b) { F64X4_OP(a.e[i] + b.e[i]); }
SIM_FORCE_INLINE f64x4 operator-(f64x4 a, f64x4 b) { F64X4_OP(a.e[i] - b.e[i]); }
SIM_FORCE_INLINE f64x4 operator*(f64x4 a, f64x4 b) { F64X4_OP(a.e[i] * b.e[i]); }
SIM_FORCE_INLINE f64x4 operator/(f64x4 a, f64x4 b) { F64X4_OP(a.e[i] / b.e[i]); }
SIM_FORCE_INLINE f64x4 operator-(f64x4 a) { F64X4_OP(-a.e[i]); }

SIM_FORCE_INLINE f64x4 lane_fmadd(f64x4 a, f64x4 b, f64x4 c) { F64X4_OP(lane_fmadd(a.e[i], b.e[i], c.e[i])); }
SIM_FORCE_INLINE f64x4 lane_fnmadd(f64x4 a, f64x4 b, f64x4 c) { F64X4_OP(lane_fnmadd(a.e[i], b.e[i], c.e[i])); }
SIM_FORCE_INLINE f64x4 lane_min(f64x4 a, f64x4 b) { F64X4_OP(lane_min(a.e[i], b.e[i])); }
SIM_FORCE_INLINE f64x4 lane_max(f64x4 a, f64x4 b) { F64X4_OP(lane_max(a.e[i], b.e[i])); }
SIM_FORCE_INLINE f64x4 lane_abs(f64x4 a) { F64X4_OP(lane_abs(a.e[i])); }
SIM_FORCE_INLINE f64x4 lane_sqrt(f64x4 a) { F64X4_OP(lane_sqrt(a.e[i])); }

SIM_FORCE_INLINE b64x4 lane_lt(f64x4 a, f64x4 b) { B64X4_OP(a.e[i] < b.e[i]); }
SIM_FORCE_INLINE b64x4 lane_le(f64x4 a, f64x4 b) { B64X4_OP(a.e[i] <= b.e[i]); }
SIM_FORCE_INLINE b64x4 lane_gt(f64x4 a, f64x4 b) { B64X4_OP(a.e[i] > b.e[i]); }
SIM_FORCE_INLINE b64x4 lane_ge(f64x4 a, f64x4 b) { B64X4_OP(a.e[i] >= b.e[i]); }
SIM_FORCE_INLINE b64x4 lane_eq(f64x4 a, f64x4 b) { B64X4_OP(a.e[i] == b.e[i]); }
SIM_FORCE_INLINE b64x4 lane_and(b64x4 a, b64x4 b) { B64X4_OP(a.e[i] && b.e[i]); }
SIM_FORCE_INLINE b64x4 lane_or(b64x4 a, b64x4 b) { B64X4_OP(a.e[i] || b.e[i]); }
SIM_FORCE_INLINE b64x4 lane_not(b64x4 a) { B64X4_OP(!a.e[i]); }
SIM_FORCE_INLINE b32 lane_any(b64x4 a) { return a.e[0] || a.e[1] || a.e[2] || a.e[3]; }
SIM_FORCE_INLINE b32 lane_all(b64x4 a) { return a.e[0] && a.e[1] && a.e[2] && a.e[3]; }
SIM_FORCE_INLINE b64x4 lane_mask_splat(b64x4 *, bool a) { B64X4_OP(a); }
SIM_FORCE_INLINE f64x4 lane_select(b64x4 m, f64x4 a, f64x4 b) { F64X4_OP(m.e[i] ? a.e[i] : b.e[i]); }
SIM_FORCE_INLINE b64x4 lane_select_mask(b64x4 m, b64x4 a, b64x4 b) { B64X4_OP(m.e[i] ? a.e[i] : b.e[i]); }
SIM_FORCE_INLINE b64x4 lane_is_finite(f64x4 a) { B64X4_OP(lane_is_finite(a.e[i])); }

SIM_FORCE_INLINE f64 lane_get(f64x4 a, u32 i) { return a.e[i]; }
SIM_FORCE_INLINE void lane_set(f64x4 *a, u32 i, f64 x) { a->e[i] = x; }

#undef F64X4_OP
#undef B64X4_OP
#endif // SIM_SIMD

SIM_FORCE_INLINE f64x4 lane_load_f64(const f64 *p, f64x4 *) { return lane_load(p, (f64x4 *)0); }
SIM_FORCE_INLINE void  lane_store_f64(f64 *p, f64x4 a) { lane_store(p, a); }
SIM_FORCE_INLINE void  lane_accumulate_f64(f64 *p, f64x4 a) { lane_store(p, lane_load(p, (f64x4 *)0) + a); }

template <> struct LaneTraits<f64x4> {
    typedef b64x4 Mask;
    enum { width = 4 };
};

// ----------------------------------------------------------------------------
// f32x8 as a lane type (eight f32 lanes), with b32x8 masks: the same lane_*
// vocabulary, so the kinetics templates also instantiate at f32x8. Storage
// stays f64: lane_load_f64 / lane_store_f64 convert.

#if SIM_SIMD

struct b32x8 { __m256 v; };

SIM_FORCE_INLINE f32x8 lane_splat(f32x8 *, f64 a) { return f32x8_set1((f32)a); }
SIM_FORCE_INLINE f32x8 lane_load_f32(const f32 *p, f32x8 *) { return f32x8_load(p); }
SIM_FORCE_INLINE void  lane_store_f32(f32 *p, f32x8 a) { f32x8_store(p, a); }
SIM_FORCE_INLINE f32x8 lane_load_f64(const f64 *p, f32x8 *) {
    __m128 lo = _mm256_cvtpd_ps(_mm256_load_pd(p));
    __m128 hi = _mm256_cvtpd_ps(_mm256_load_pd(p + 4));
    f32x8 r; r.v = _mm256_set_m128(hi, lo); return r;
}
SIM_FORCE_INLINE void lane_store_f64(f64 *p, f32x8 a) {
    _mm256_store_pd(p, _mm256_cvtps_pd(_mm256_castps256_ps128(a.v)));
    _mm256_store_pd(p + 4, _mm256_cvtps_pd(_mm256_extractf128_ps(a.v, 1)));
}
SIM_FORCE_INLINE void lane_accumulate_f64(f64 *p, f32x8 a) {
    _mm256_store_pd(p, _mm256_add_pd(_mm256_load_pd(p), _mm256_cvtps_pd(_mm256_castps256_ps128(a.v))));
    _mm256_store_pd(p + 4, _mm256_add_pd(_mm256_load_pd(p + 4), _mm256_cvtps_pd(_mm256_extractf128_ps(a.v, 1))));
}
SIM_FORCE_INLINE f32x8 lane_fmadd(f32x8 a, f32x8 b, f32x8 c) { return fmadd(a, b, c); }
SIM_FORCE_INLINE f32x8 lane_fnmadd(f32x8 a, f32x8 b, f32x8 c) { return fnmadd(a, b, c); }
SIM_FORCE_INLINE f32x8 lane_min(f32x8 a, f32x8 b) { return f32x8_min(a, b); }
SIM_FORCE_INLINE f32x8 lane_max(f32x8 a, f32x8 b) { return f32x8_max(a, b); }
SIM_FORCE_INLINE f32x8 lane_abs(f32x8 a) {
    __m256 neg = _mm256_cmp_ps(a.v, _mm256_setzero_ps(), _CMP_LT_OQ);
    f32x8 r; r.v = _mm256_blendv_ps(a.v, _mm256_xor_ps(a.v, _mm256_set1_ps(-0.0f)), neg); return r;
}
SIM_FORCE_INLINE f32x8 lane_sqrt(f32x8 a) { return f32x8_sqrt(a); }
SIM_FORCE_INLINE b32x8 lane_lt(f32x8 a, f32x8 b) { b32x8 r; r.v = _mm256_cmp_ps(a.v, b.v, _CMP_LT_OQ); return r; }
SIM_FORCE_INLINE b32x8 lane_le(f32x8 a, f32x8 b) { b32x8 r; r.v = _mm256_cmp_ps(a.v, b.v, _CMP_LE_OQ); return r; }
SIM_FORCE_INLINE b32x8 lane_gt(f32x8 a, f32x8 b) { b32x8 r; r.v = _mm256_cmp_ps(a.v, b.v, _CMP_GT_OQ); return r; }
SIM_FORCE_INLINE b32x8 lane_ge(f32x8 a, f32x8 b) { b32x8 r; r.v = _mm256_cmp_ps(a.v, b.v, _CMP_GE_OQ); return r; }
SIM_FORCE_INLINE b32x8 lane_eq(f32x8 a, f32x8 b) { b32x8 r; r.v = _mm256_cmp_ps(a.v, b.v, _CMP_EQ_OQ); return r; }
SIM_FORCE_INLINE b32x8 lane_and(b32x8 a, b32x8 b) { b32x8 r; r.v = _mm256_and_ps(a.v, b.v); return r; }
SIM_FORCE_INLINE b32x8 lane_or(b32x8 a, b32x8 b) { b32x8 r; r.v = _mm256_or_ps(a.v, b.v); return r; }
SIM_FORCE_INLINE b32x8 lane_not(b32x8 a) { b32x8 r; r.v = _mm256_xor_ps(a.v, _mm256_castsi256_ps(_mm256_set1_epi32(-1))); return r; }
SIM_FORCE_INLINE b32 lane_any(b32x8 a) { return _mm256_movemask_ps(a.v) != 0; }
SIM_FORCE_INLINE b32 lane_all(b32x8 a) { return _mm256_movemask_ps(a.v) == 0xFF; }
SIM_FORCE_INLINE b32x8 lane_mask_splat(b32x8 *, bool a) { b32x8 r; r.v = _mm256_castsi256_ps(_mm256_set1_epi32(a ? -1 : 0)); return r; }
SIM_FORCE_INLINE f32x8 lane_select(b32x8 m, f32x8 a, f32x8 b) { f32x8 r; r.v = _mm256_blendv_ps(b.v, a.v, m.v); return r; }
SIM_FORCE_INLINE b32x8 lane_select_mask(b32x8 m, b32x8 a, b32x8 b) { b32x8 r; r.v = _mm256_blendv_ps(b.v, a.v, m.v); return r; }
SIM_FORCE_INLINE b32x8 lane_is_finite(f32x8 a) { f32x8 d = a - a; return lane_eq(d, d); }
SIM_FORCE_INLINE f32 lane_get(f32x8 a, u32 i) { return f32x8_lane(a, i); }

#else

struct b32x8 { bool e[8]; };
#define F32X8_LOP(expr) f32x8 r; for (int i = 0; i < 8; ++i) { r.e[i] = (expr); } return r
#define B32X8_LOP(expr) b32x8 r; for (int i = 0; i < 8; ++i) { r.e[i] = (expr); } return r
SIM_FORCE_INLINE f32x8 lane_splat(f32x8 *, f64 a) { F32X8_LOP((f32)a); }
SIM_FORCE_INLINE f32x8 lane_load_f32(const f32 *p, f32x8 *) { return f32x8_load(p); }
SIM_FORCE_INLINE void  lane_store_f32(f32 *p, f32x8 a) { f32x8_store(p, a); }
SIM_FORCE_INLINE f32x8 lane_load_f64(const f64 *p, f32x8 *) { F32X8_LOP((f32)p[i]); }
SIM_FORCE_INLINE void lane_store_f64(f64 *p, f32x8 a) { for (int i = 0; i < 8; ++i) p[i] = (f64)a.e[i]; }
SIM_FORCE_INLINE void lane_accumulate_f64(f64 *p, f32x8 a) { for (int i = 0; i < 8; ++i) p[i] += (f64)a.e[i]; }
SIM_FORCE_INLINE f32x8 lane_fmadd(f32x8 a, f32x8 b, f32x8 c) { F32X8_LOP(lane_fmadd(a.e[i], b.e[i], c.e[i])); }
SIM_FORCE_INLINE f32x8 lane_fnmadd(f32x8 a, f32x8 b, f32x8 c) { F32X8_LOP(lane_fnmadd(a.e[i], b.e[i], c.e[i])); }
SIM_FORCE_INLINE f32x8 lane_min(f32x8 a, f32x8 b) { F32X8_LOP(lane_min(a.e[i], b.e[i])); }
SIM_FORCE_INLINE f32x8 lane_max(f32x8 a, f32x8 b) { F32X8_LOP(lane_max(a.e[i], b.e[i])); }
SIM_FORCE_INLINE f32x8 lane_abs(f32x8 a) { F32X8_LOP(lane_abs(a.e[i])); }
SIM_FORCE_INLINE f32x8 lane_sqrt(f32x8 a) { F32X8_LOP(lane_sqrt(a.e[i])); }
SIM_FORCE_INLINE b32x8 lane_lt(f32x8 a, f32x8 b) { B32X8_LOP(a.e[i] < b.e[i]); }
SIM_FORCE_INLINE b32x8 lane_le(f32x8 a, f32x8 b) { B32X8_LOP(a.e[i] <= b.e[i]); }
SIM_FORCE_INLINE b32x8 lane_gt(f32x8 a, f32x8 b) { B32X8_LOP(a.e[i] > b.e[i]); }
SIM_FORCE_INLINE b32x8 lane_ge(f32x8 a, f32x8 b) { B32X8_LOP(a.e[i] >= b.e[i]); }
SIM_FORCE_INLINE b32x8 lane_eq(f32x8 a, f32x8 b) { B32X8_LOP(a.e[i] == b.e[i]); }
SIM_FORCE_INLINE b32x8 lane_and(b32x8 a, b32x8 b) { B32X8_LOP(a.e[i] && b.e[i]); }
SIM_FORCE_INLINE b32x8 lane_or(b32x8 a, b32x8 b) { B32X8_LOP(a.e[i] || b.e[i]); }
SIM_FORCE_INLINE b32x8 lane_not(b32x8 a) { B32X8_LOP(!a.e[i]); }
SIM_FORCE_INLINE b32 lane_any(b32x8 a) { for (int i = 0; i < 8; ++i) if (a.e[i]) return 1; return 0; }
SIM_FORCE_INLINE b32 lane_all(b32x8 a) { for (int i = 0; i < 8; ++i) if (!a.e[i]) return 0; return 1; }
SIM_FORCE_INLINE b32x8 lane_mask_splat(b32x8 *, bool a) { B32X8_LOP(a); }
SIM_FORCE_INLINE f32x8 lane_select(b32x8 m, f32x8 a, f32x8 b) { F32X8_LOP(m.e[i] ? a.e[i] : b.e[i]); }
SIM_FORCE_INLINE b32x8 lane_select_mask(b32x8 m, b32x8 a, b32x8 b) { B32X8_LOP(m.e[i] ? a.e[i] : b.e[i]); }
SIM_FORCE_INLINE b32x8 lane_is_finite(f32x8 a) { B32X8_LOP(lane_is_finite(a.e[i])); }
SIM_FORCE_INLINE f32 lane_get(f32x8 a, u32 i) { return a.e[i]; }
#undef F32X8_LOP
#undef B32X8_LOP

#endif // SIM_SIMD

template <> struct LaneTraits<f32x8> {
    typedef b32x8 Mask;
    enum { width = 8 };
};

#endif // BASE_SIMD_H
