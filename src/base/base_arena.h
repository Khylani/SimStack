#ifndef BASE_ARENA_H
#define BASE_ARENA_H

// Bump arenas. The platform reserves the memory blocks; the sim carves arenas
// out of them on first update. Nothing here calls malloc.
//
//   Arena        : linear allocator over a fixed block
//   TempMemory   : begin_temp/end_temp scope that rolls the arena back
//   arena_split  : carve a child arena out of a parent (permanent -> sub-pools)
//   Soa*         : helpers for lane-padded structure-of-arrays allocations

#include "base_types.h"

#define ARENA_DEFAULT_ALIGN 16
#define SIMD_LANE_COUNT 8      // AVX2 f32 width; SoA counts are padded to this
#define SIMD_ALIGN 32          // bytes

struct Arena {
    u8 *base;
    u64 size;
    u64 used;
    u32 temp_count;    // outstanding begin_temp scopes (debug check)
};

struct TempMemory {
    Arena *arena;
    u64 used;
};

enum ArenaPushFlags {
    ArenaPush_Zero = 0x1,
};

SIM_FORCE_INLINE Arena arena_make(void *base, u64 size) {
    Arena arena = {};
    arena.base = (u8 *)base;
    arena.size = size;
    return arena;
}

SIM_FORCE_INLINE u64 arena_remaining(Arena *arena) {
    return arena->size - arena->used;
}

SIM_FORCE_INLINE u64 arena_alignment_offset(Arena *arena, u64 align) {
    Assert(IsPow2(align));
    u64 ptr = (u64)(arena->base + arena->used);
    u64 mask = align - 1;
    return (ptr & mask) ? (align - (ptr & mask)) : 0;
}

SIM_FORCE_INLINE void *push_size_(Arena *arena, u64 size, u64 align, u32 flags) {
    u64 offset = arena_alignment_offset(arena, align);
    u64 total = size + offset;
    Assert(arena->used + total <= arena->size);
    if (arena->used + total > arena->size) {
        return 0;
    }
    u8 *result = arena->base + arena->used + offset;
    arena->used += total;
    if (flags & ArenaPush_Zero) {
        memset(result, 0, size);
    }
    return result;
}

// Zero-initialised by default; *_nozero variants for buffers you fully overwrite.
#define push_size(arena, size)                  push_size_((arena), (size), ARENA_DEFAULT_ALIGN, ArenaPush_Zero)
#define push_size_aligned(arena, size, align)   push_size_((arena), (size), (align), ArenaPush_Zero)
#define push_struct(arena, T)                   (T *)push_size_((arena), sizeof(T), alignof(T), ArenaPush_Zero)
#define push_array(arena, T, count)             (T *)push_size_((arena), sizeof(T) * (u64)(count), alignof(T), ArenaPush_Zero)
#define push_array_aligned(arena, T, count, al) (T *)push_size_((arena), sizeof(T) * (u64)(count), (al), ArenaPush_Zero)
#define push_array_nozero(arena, T, count)      (T *)push_size_((arena), sizeof(T) * (u64)(count), alignof(T), 0)

SIM_FORCE_INLINE char *push_string_copy(Arena *arena, const char *src) {
    u64 len = strlen(src);
    char *dst = (char *)push_size_(arena, len + 1, 1, 0);
    memcpy(dst, src, len + 1);
    return dst;
}

SIM_FORCE_INLINE void arena_clear(Arena *arena) {
    Assert(arena->temp_count == 0);
    arena->used = 0;
}

SIM_FORCE_INLINE Arena arena_split(Arena *parent, u64 size, u64 align = SIMD_ALIGN) {
    void *base = push_size_(parent, size, align, 0);
    return arena_make(base, size);
}

SIM_FORCE_INLINE TempMemory begin_temp(Arena *arena) {
    TempMemory temp;
    temp.arena = arena;
    temp.used = arena->used;
    ++arena->temp_count;
    return temp;
}

SIM_FORCE_INLINE void end_temp(TempMemory temp) {
    Arena *arena = temp.arena;
    Assert(arena->used >= temp.used);
    Assert(arena->temp_count > 0);
    arena->used = temp.used;
    --arena->temp_count;
}

// RAII helper so a scratch scope cannot forget end_temp. Still no exceptions,
// just a destructor.
struct TempScope {
    TempMemory temp;
    TempScope(Arena *arena) : temp(begin_temp(arena)) {}
    ~TempScope() { end_temp(temp); }
};
#define SCRATCH_SCOPE(arena) TempScope CONCAT(scratch_scope_, __LINE__)(arena)

// ----------------------------------------------------------------------------
// SoA helpers. Hot arrays are padded to a multiple of SIMD_LANE_COUNT and
// aligned to SIMD_ALIGN so f32x8 loads never need a tail loop or an unaligned
// path. Padding lanes are zero and must be kept finite by the kernels.

#define SoaPadCount(n) ((u32)AlignPow2((u64)(n), SIMD_LANE_COUNT))
#define SoaPush(arena, T, count) push_array_aligned((arena), T, SoaPadCount(count), SIMD_ALIGN)
#define SoaForEachLane(i, padded_count) for (u32 i = 0; i < (padded_count); i += SIMD_LANE_COUNT)
#define SoaIsAligned(ptr) ((((u64)(ptr)) & (SIMD_ALIGN - 1)) == 0)

#endif // BASE_ARENA_H
