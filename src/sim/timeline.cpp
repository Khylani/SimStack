// Timeline: keyframes of the whole scene for playback, scrubbing and baking.
//
// Every scene allocates all of its state from SimState::permanent at reset and
// nothing afterwards, at fixed addresses. A keyframe is therefore one memcpy of
// permanent[0, used), and restoring it is a memcpy back: every pointer inside
// stays valid. Keyframes live in the platform's timeline block (reserved once,
// committed by the kernel on first touch), never allocated at runtime.
//
//   live      the scene simulates; a keyframe is recorded every `interval`
//   playback  the playhead is inside the recorded range: keyframes are restored
//             as it moves (nothing is simulated), so baked high-precision runs
//             play back at any speed
//   bake      simulate as fast as the frame budget allows to `bake_end`,
//             recording; then play back from the start
//
// When the block is full the keyframes are thinned (every other one dropped,
// interval doubled), so a run is always scrubbable end to end. An edit while
// in the past forks: the recorded future is dropped and the scene goes live.
// The sim is deterministic, so playing on without edits reproduces the future.

#define TIMELINE_MAX_KEYS 65536

struct TimelineKey {
    f64 sim_time;
    f64 accumulator;      // fixed-step remainder, so a restore steps exactly like the original
    u64 step_count;
    u64 offset;           // into Timeline::data
};

struct Timeline {
    u8 *data;
    u64 data_size;
    TimelineKey *keys;
    u32 key_count, capacity;   // capacity: keyframes that fit
    u64 frame_bytes;           // bytes per keyframe (the scene's arena size)
    u32 generation;            // scene generation the keys belong to
    f64 interval, next_key_t;
    b32 playback;              // the playhead is in the recorded range
    f64 playhead;              // sim time shown in playback
    u32 shown_key;             // keyframe currently restored
    b32 baking;
    f64 bake_start, bake_end;
    u32 bake_keys;
    u64 params_hash;           // edit detection for forking
};

internal void timeline_init(Timeline *tl, SimMemory *memory) {
    memset(tl, 0, sizeof(*tl));
    if (!memory->timeline_base || memory->timeline_size < Megabytes(64)) return;
    tl->keys = (TimelineKey *)memory->timeline_base;
    u64 table = AlignPow2((u64)sizeof(TimelineKey) * TIMELINE_MAX_KEYS, 4096);
    tl->data = (u8 *)memory->timeline_base + table;
    tl->data_size = memory->timeline_size - table;
}

internal b32 timeline_available(Timeline *tl) { return tl->data != 0; }

// Fresh timeline for the current scene (called on every scene reset).
internal void timeline_reset(Timeline *tl, Arena *scene, u32 generation, f64 interval) {
    if (!tl->data) return;
    tl->key_count = 0;
    tl->frame_bytes = AlignPow2(scene->used, 64);
    tl->capacity = tl->frame_bytes ? (u32)Minimum((u64)TIMELINE_MAX_KEYS, tl->data_size / tl->frame_bytes) : 0;
    tl->generation = generation;
    tl->interval = interval;
    tl->next_key_t = 0.0;
    tl->playback = 0;
    tl->baking = 0;
    tl->shown_key = 0;
}

internal f64 timeline_start(Timeline *tl) { return tl->key_count ? tl->keys[0].sim_time : 0.0; }
internal f64 timeline_end(Timeline *tl) { return tl->key_count ? tl->keys[tl->key_count - 1].sim_time : 0.0; }

// Drops every other keyframe (keeping the first and the last), doubles the interval.
internal void timeline_thin(Timeline *tl) {
    u32 n = 0;
    for (u32 i = 0; i < tl->key_count; ++i) {
        if (i % 2 == 1 && i != tl->key_count - 1) continue;
        if (n != i) {
            memmove(tl->data + (u64)n * tl->frame_bytes, tl->data + (u64)i * tl->frame_bytes, tl->frame_bytes);
            tl->keys[n] = tl->keys[i];
            tl->keys[n].offset = (u64)n * tl->frame_bytes;
        }
        ++n;
    }
    tl->key_count = n;
    tl->interval *= 2.0;
}

internal void timeline_record(Timeline *tl, Arena *scene, f64 sim_time, u64 step_count, f64 accumulator = 0.0) {
    if (!tl->capacity) return;
    if (tl->key_count == tl->capacity) timeline_thin(tl);
    TimelineKey *k = tl->keys + tl->key_count;
    k->sim_time = sim_time;
    k->accumulator = accumulator;
    k->step_count = step_count;
    k->offset = (u64)tl->key_count * tl->frame_bytes;
    memcpy(tl->data + k->offset, scene->base, scene->used);
    ++tl->key_count;
    tl->next_key_t = sim_time + tl->interval;
}

// Last keyframe at or before t (0 if none).
internal u32 timeline_find(Timeline *tl, f64 t) {
    u32 lo = 0, hi = tl->key_count;
    while (hi - lo > 1) {
        u32 mid = (lo + hi) / 2;
        if (tl->keys[mid].sim_time <= t + 1e-9) lo = mid; else hi = mid;
    }
    return lo;
}

internal void timeline_restore(Timeline *tl, Arena *scene, u32 key, f64 *sim_time, u64 *step_count, f64 *accumulator) {
    TimelineKey *k = tl->keys + key;
    memcpy(scene->base, tl->data + k->offset, scene->used);
    *sim_time = k->sim_time;
    *accumulator = k->accumulator;
    *step_count = k->step_count;
    tl->shown_key = key;
}

// Drops keyframes after `key` (fork on edit).
internal void timeline_truncate_after(Timeline *tl, u32 key) {
    if (key + 1 < tl->key_count) tl->key_count = key + 1;
    tl->next_key_t = tl->keys[key].sim_time + tl->interval;
}
