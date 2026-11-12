#ifndef BASE_MATH_H
#define BASE_MATH_H

// f32 vector and matrix types. Column-major m4 (OpenGL convention).

#include "base_types.h"
#include <math.h>

#define PI32 3.14159265358979323846f
#define TAU32 6.28318530717958647692f

union v2 {
    struct { f32 x, y; };
    f32 e[2];
};

union v3 {
    struct { f32 x, y, z; };
    struct { f32 r, g, b; };
    struct { v2 xy; f32 ignored_; };
    f32 e[3];
};

union v4 {
    struct { f32 x, y, z, w; };
    struct { f32 r, g, b, a; };
    struct { v3 xyz; f32 ignored_; };
    f32 e[4];
};

// Column-major: e[col][row]; cols[c] is a column vector.
union m4 {
    f32 e[4][4];
    v4 cols[4];
};

// -- constructors --------------------------------------------------------------

SIM_FORCE_INLINE v2 V2(f32 x, f32 y) { v2 r; r.x = x; r.y = y; return r; }
SIM_FORCE_INLINE v3 V3(f32 x, f32 y, f32 z) { v3 r; r.x = x; r.y = y; r.z = z; return r; }
SIM_FORCE_INLINE v3 V3(f32 s) { return V3(s, s, s); }
SIM_FORCE_INLINE v4 V4(f32 x, f32 y, f32 z, f32 w) { v4 r; r.x = x; r.y = y; r.z = z; r.w = w; return r; }
SIM_FORCE_INLINE v4 V4(v3 v, f32 w) { return V4(v.x, v.y, v.z, w); }

// -- scalar helpers -------------------------------------------------------------

SIM_FORCE_INLINE f32 square(f32 a) { return a * a; }
SIM_FORCE_INLINE f32 lerp_f32(f32 a, f32 t, f32 b) { return (1.0f - t) * a + t * b; }
SIM_FORCE_INLINE f32 clamp01(f32 x) { return Clamp(x, 0.0f, 1.0f); }
SIM_FORCE_INLINE f32 safe_ratio0(f32 n, f32 d) { return d != 0.0f ? n / d : 0.0f; }
SIM_FORCE_INLINE f32 radians(f32 deg) { return deg * (PI32 / 180.0f); }

// -- v2 -------------------------------------------------------------------------

SIM_FORCE_INLINE v2 operator+(v2 a, v2 b) { return V2(a.x + b.x, a.y + b.y); }
SIM_FORCE_INLINE v2 operator-(v2 a, v2 b) { return V2(a.x - b.x, a.y - b.y); }
SIM_FORCE_INLINE v2 operator*(v2 a, f32 s) { return V2(a.x * s, a.y * s); }
SIM_FORCE_INLINE v2 operator*(f32 s, v2 a) { return a * s; }
SIM_FORCE_INLINE f32 dot(v2 a, v2 b) { return a.x * b.x + a.y * b.y; }

// -- v3 -------------------------------------------------------------------------

SIM_FORCE_INLINE v3 operator+(v3 a, v3 b) { return V3(a.x + b.x, a.y + b.y, a.z + b.z); }
SIM_FORCE_INLINE v3 operator-(v3 a, v3 b) { return V3(a.x - b.x, a.y - b.y, a.z - b.z); }
SIM_FORCE_INLINE v3 operator-(v3 a) { return V3(-a.x, -a.y, -a.z); }
SIM_FORCE_INLINE v3 operator*(v3 a, f32 s) { return V3(a.x * s, a.y * s, a.z * s); }
SIM_FORCE_INLINE v3 operator*(f32 s, v3 a) { return a * s; }
SIM_FORCE_INLINE v3 operator/(v3 a, f32 s) { f32 inv = 1.0f / s; return a * inv; }
SIM_FORCE_INLINE v3 &operator+=(v3 &a, v3 b) { a = a + b; return a; }
SIM_FORCE_INLINE v3 &operator-=(v3 &a, v3 b) { a = a - b; return a; }
SIM_FORCE_INLINE v3 &operator*=(v3 &a, f32 s) { a = a * s; return a; }
SIM_FORCE_INLINE v3 hadamard(v3 a, v3 b) { return V3(a.x * b.x, a.y * b.y, a.z * b.z); }
SIM_FORCE_INLINE f32 dot(v3 a, v3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
SIM_FORCE_INLINE v3 cross(v3 a, v3 b) {
    return V3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
SIM_FORCE_INLINE f32 length_sq(v3 a) { return dot(a, a); }
SIM_FORCE_INLINE f32 length(v3 a) { return sqrtf(dot(a, a)); }
SIM_FORCE_INLINE v3 normalize(v3 a) { return a * (1.0f / length(a)); }
SIM_FORCE_INLINE v3 normalize_or_zero(v3 a) {
    f32 len_sq = length_sq(a);
    return len_sq > square(0.0001f) ? a * (1.0f / sqrtf(len_sq)) : V3(0.0f);
}
SIM_FORCE_INLINE v3 lerp_v3(v3 a, f32 t, v3 b) { return (1.0f - t) * a + t * b; }

// -- v4 -------------------------------------------------------------------------

SIM_FORCE_INLINE v4 operator+(v4 a, v4 b) { return V4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
SIM_FORCE_INLINE v4 operator-(v4 a, v4 b) { return V4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w); }
SIM_FORCE_INLINE v4 operator*(v4 a, f32 s) { return V4(a.x * s, a.y * s, a.z * s, a.w * s); }
SIM_FORCE_INLINE v4 operator*(f32 s, v4 a) { return a * s; }
SIM_FORCE_INLINE f32 dot(v4 a, v4 b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }

// -- m4 -------------------------------------------------------------------------

SIM_FORCE_INLINE m4 m4_identity() {
    m4 r = {};
    r.e[0][0] = r.e[1][1] = r.e[2][2] = r.e[3][3] = 1.0f;
    return r;
}

SIM_FORCE_INLINE m4 operator*(m4 a, m4 b) {
    m4 r;
    for (int c = 0; c < 4; ++c) {
        for (int row = 0; row < 4; ++row) {
            r.e[c][row] = a.e[0][row] * b.e[c][0] + a.e[1][row] * b.e[c][1] +
                          a.e[2][row] * b.e[c][2] + a.e[3][row] * b.e[c][3];
        }
    }
    return r;
}

SIM_FORCE_INLINE v4 operator*(m4 m, v4 v) {
    return m.cols[0] * v.x + m.cols[1] * v.y + m.cols[2] * v.z + m.cols[3] * v.w;
}

SIM_FORCE_INLINE m4 m4_translation(v3 t) {
    m4 r = m4_identity();
    r.e[3][0] = t.x; r.e[3][1] = t.y; r.e[3][2] = t.z;
    return r;
}

SIM_FORCE_INLINE m4 m4_scale(v3 s) {
    m4 r = m4_identity();
    r.e[0][0] = s.x; r.e[1][1] = s.y; r.e[2][2] = s.z;
    return r;
}

SIM_FORCE_INLINE m4 m4_rotation_y(f32 angle) {
    f32 c = cosf(angle), s = sinf(angle);
    m4 r = m4_identity();
    r.e[0][0] = c;  r.e[0][2] = -s;
    r.e[2][0] = s;  r.e[2][2] = c;
    return r;
}

SIM_FORCE_INLINE m4 m4_perspective(f32 fov_y_radians, f32 aspect, f32 near_z, f32 far_z) {
    f32 f = 1.0f / tanf(0.5f * fov_y_radians);
    m4 r = {};
    r.e[0][0] = f / aspect;
    r.e[1][1] = f;
    r.e[2][2] = (far_z + near_z) / (near_z - far_z);
    r.e[2][3] = -1.0f;
    r.e[3][2] = (2.0f * far_z * near_z) / (near_z - far_z);
    return r;
}

SIM_FORCE_INLINE m4 m4_look_at(v3 eye, v3 target, v3 up) {
    v3 f = normalize(target - eye);
    v3 s = normalize(cross(f, up));
    v3 u = cross(s, f);
    m4 r = m4_identity();
    r.e[0][0] = s.x; r.e[1][0] = s.y; r.e[2][0] = s.z;
    r.e[0][1] = u.x; r.e[1][1] = u.y; r.e[2][1] = u.z;
    r.e[0][2] = -f.x; r.e[1][2] = -f.y; r.e[2][2] = -f.z;
    r.e[3][0] = -dot(s, eye);
    r.e[3][1] = -dot(u, eye);
    r.e[3][2] = dot(f, eye);
    return r;
}

#endif // BASE_MATH_H
