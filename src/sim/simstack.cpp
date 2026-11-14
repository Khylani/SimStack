// Simstack sim module: unity build root. Compiled into simstack_sim.{so,dll}.
// Every other sim .cpp is #included below; the platform never sees them.
//
// No OS calls, no malloc, no mutable globals except g_debug_table (re-patched
// every update) so a hot reload preserves all state in SimMemory. Threads and
// the GPU come from the platform through SimMemory (work_queue, gpu).

// Third-party headers first: base_types.h defines function-like macros
// (Swap, Clamp, Minimum, ...) and `internal` that collide with their identifiers.
#include "imgui.h"
#include "imgui_internal.h"
#include "implot.h"

#define DEBUG_RECORD_BASE DEBUG_RECORDS_PER_TU
#include "../sim_platform.h"

#include "particles.cpp"
#include "dbd.cpp"
#include "nft.cpp"          // (paw.cpp inside: the dbd scene's skid preset uses it)
#include "dbd_scene.cpp"
#include "nft_scene.cpp"
#include "timeline.cpp"
#include "../ui/ui_simstack.cpp"

#define SIM_DT_FIXED (1.0 / 1000.0)     // 1 ms master step (f64; cast to f32 for the kernels)
#define SIM_MAX_STEPS_PER_FRAME 64      // spiral-of-death guard
#define SIM_CHEM_FRAME_BUDGET_MS 12.0   // windowed chemistry scenes: wall time per frame for master steps

// Tunables applied every update, so editing one here and rebuilding the sim
// module shows the change through hot reload with the particle state intact.
#define PARTICLE_SPRING  4.0f
#define PARTICLE_SWIRL   0.6f
#define PARTICLE_DAMPING 0.02f
#define PARTICLE_BOUND   1.5f
#define POINT_SIZE       2.0f
#define POINT_COLOR      V4(0.55f, 0.85f, 1.0f, 1.0f)

struct Camera {
    f32 yaw;        // radians about +y
    f32 pitch;
    f32 distance;
    b32 auto_orbit;
    v3 target;      // orbit / zoom centre (pan moves it)
    b32 frame_request;          // fit the scene's bounds on the next render
    v3 bounds_min, bounds_max;  // last framed scene bounds
    f32 fov;        // radians, vertical
    f32 framed_aspect;          // view aspect at the last framing
    b32 user_moved;             // the user orbited / panned / zoomed since: keep their view
};

// Camera basis for a view rect of the given aspect.
struct CameraBasis { v3 eye, forward, right, up; f32 tan_half, aspect; };
internal CameraBasis camera_basis(Camera *cam, f32 aspect) {
    CameraBasis b;
    f32 cp = cosf(cam->pitch), sp = sinf(cam->pitch);
    v3 offset = V3(cam->distance * cp * sinf(cam->yaw), cam->distance * sp, cam->distance * cp * cosf(cam->yaw));
    b.eye = cam->target + offset;
    b.forward = normalize(cam->target - b.eye);
    b.right = normalize(cross(b.forward, V3(0.0f, 1.0f, 0.0f)));
    b.up = cross(b.right, b.forward);
    b.tan_half = tanf(0.5f * cam->fov);
    b.aspect = aspect;
    return b;
}

internal void camera_reset(Camera *cam, u32 scene) {
    cam->yaw = scene == SimScene_Nft ? 0.55f : 0.0f;
    cam->pitch = scene == SimScene_Nft ? 0.62f : 0.35f;
    cam->distance = scene == SimScene_Particles ? 3.5f : 4.2f;
    cam->target = V3(0.0f);
    cam->fov = radians(45.0f);
    cam->frame_request = scene != SimScene_Particles;
    cam->user_moved = 0;
}

struct SimState {
    Arena permanent;   // the rest of the permanent block, after SimState itself
    Arena frame;
    Arena scratch;
    u64 scratch_high_water;

    f64 dt_fixed;
    f64 accumulator;   // f64 so N frames of a fixed dt yield exactly the expected step count
    u64 step_count;
    f64 sim_time;
    b32 paused;
    u32 reload_count;

    u32 scene;         // SimScene
    RandomSeries rng;
    Particles particles;
    DbdScene *dbd;     // dbd / sweep scenes (in the permanent arena)
    NftScene *nft;     // nft scene (in the permanent arena)
    NftParams nft_params;   // survive scene rebuilds (structure edits in the UI)
    b32 nft_params_set;
    b32 dbd_defaults_next;  // next reset of a DBD scene drops the UI's settings
    Camera camera;
    Timeline timeline; // keyframes (SimMemory::timeline_base); not part of any keyframe itself
    u32 scene_generation;
    b32 timeline_recording;   // the chemistry job records keyframes this frame
    UiState ui;        // windowed runs only; its heap lives in SimMemory::ui_base
    SimStats last_stats;

    // Windowed chemistry runs as one background job per frame, overlapping the
    // platform's render and swap; joined at the top of the next update.
    struct {
        SimMemory *memory;
        f64 budget_ms;
        b32 in_flight;
        u32 steps;
        f32 ms;
    } chem_job;
};

// Resetting rolls the permanent arena back to just past the SimState header
// and rebuilds the scene from the config.
// DBD settings the UI edits; a reset (restart) keeps them, Defaults drops them.
struct DbdSettings {
    DbdOperating op;
    AirSourceModel source;
    KinControl control;
    u32 method_id, precision, mode, chem_steps, backend;
    f64 residence_s;
};

internal void sim_reset_scene(SimState *state, SimMemory *memory) {
    DbdSettings keep = {};
    b32 keep_dbd = state->dbd && !state->dbd_defaults_next && state->scene != SimScene_Particles && state->scene != SimScene_Nft &&
                   state->dbd->chained == (state->scene == SimScene_Dbd);
    state->dbd_defaults_next = 0;
    if (keep_dbd) {
        DbdScene *d = state->dbd;
        keep.op = d->op; keep.source = d->source; keep.control = d->control;
        keep.method_id = d->method_id; keep.precision = d->precision; keep.mode = d->mode;
        keep.chem_steps = d->chem_steps; keep.backend = d->backend; keep.residence_s = d->residence_s;
    }
    if (state->dbd && state->dbd->hybrid.registered && memory->gpu.available)
        memory->gpu.gpu_host_unregister(state->dbd->hybrid.registered_block);   // the block is about to be reused
    state->permanent.used = 0;
    if (memory->gpu.available) memory->gpu.gpu_arena_reset();   // every device buffer belonged to the old scene
    state->step_count = 0;
    state->sim_time = 0.0;
    state->accumulator = 0.0;
    state->dbd = 0;
    state->nft = 0;
    state->rng.state = memory->config.seed ? memory->config.seed : 0x9E3779B9u;
    if (state->scene == SimScene_Particles) {
        particles_init(&state->particles, &state->permanent, memory->config.particle_count, &state->rng,
                       sqrtf(PARTICLE_SPRING));
    } else if (state->scene == SimScene_Nft) {
        if (!state->nft_params_set) {
            state->nft_params = nft_preset(memory->config.preset);
            if (memory->config.water) nft_set_water(&state->nft_params, memory->config.water);
            if (memory->config.zone_count) state->nft_params.rails = memory->config.zone_count;
            if (memory->config.channel_length_m > 0) state->nft_params.rail_length_m = memory->config.channel_length_m;
            if (memory->config.plants_per_channel) state->nft_params.plants_per_rail = memory->config.plants_per_channel;
            state->nft_params_set = 1;
        }
        state->nft = push_struct(&state->permanent, NftScene);
        nft_scene_init(state->nft, &state->permanent, &memory->config, &state->nft_params);
    } else {
        state->dbd = push_struct(&state->permanent, DbdScene);
        dbd_scene_init(state->dbd, &state->permanent, &memory->config, state->scene == SimScene_Sweep);
        if (state->dbd->model.error) memory->platform.log("[dbd] mechanism error: %s\n", state->dbd->model.error_text);
        if (keep_dbd) {
            DbdScene *d = state->dbd;
            d->op = keep.op; d->source = keep.source; d->control = keep.control;
            d->method_id = (KinMethodId)keep.method_id; d->precision = (DbdPrecision)keep.precision; d->mode = (DbdDischargeMode)keep.mode;
            d->chem_steps = keep.chem_steps; d->backend = keep.backend; d->residence_s = keep.residence_s;
            dbd_scene_on_reload(d);          // method pointer from method_id, GPU re-upload
            dbd_scene_apply_operating(d);
            dbd_zones_reset(&d->zones, &d->sp);
        }
    }
    ++state->scene_generation;
}

// --- timeline glue -------------------------------------------------------------------

internal f64 sim_default_key_interval(SimState *state) {
    return state->nft ? 3600.0 : state->dbd ? 0.05 : 0.05;
}

// Parameters that change the simulation's future (edit -> fork).
internal u64 sim_params_hash(SimState *state) {
    if (state->nft) return hash_fnv1a_64(&state->nft->sys.params, sizeof(NftParams));
    if (state->dbd) {
        DbdScene *d = state->dbd;
        u64 h = hash_fnv1a_64(&d->op, sizeof(d->op));
        h = hash_fnv1a_64(&d->source, sizeof(d->source), h);
        h = hash_fnv1a_64(&d->control, sizeof(d->control), h);
        h = hash_fnv1a_64(&d->method_id, sizeof(d->method_id), h);
        h = hash_fnv1a_64(&d->precision, sizeof(d->precision), h);
        h = hash_fnv1a_64(&d->mode, sizeof(d->mode), h);
        h = hash_fnv1a_64(&d->chem_steps, sizeof(d->chem_steps), h);
        h = hash_fnv1a_64(&d->residence_s, sizeof(d->residence_s), h);
        return h;
    }
    return 0;
}

// Inside the chemistry job: a keyframe whenever the simulation crosses the next one.
internal void sim_timeline_maybe_record(SimState *state) {
    Timeline *tl = &state->timeline;
    if (state->timeline_recording && state->sim_time >= tl->next_key_t - 1e-9)
        timeline_record(tl, &state->permanent, state->sim_time, state->step_count, state->accumulator);
}

internal void sim_timeline_restore(SimState *state, u32 key) {
    Timeline *tl = &state->timeline;
    timeline_restore(tl, &state->permanent, key, &state->sim_time, &state->step_count, &state->accumulator);
    if (state->dbd) dbd_scene_on_reload(state->dbd);   // the keyframe may predate a hot reload; re-uploads the GPU mechanism
    tl->params_hash = sim_params_hash(state);
}

// After the UI frame (its requests are in UiState), before stepping. Returns
// 1 when the scene is in playback (no simulation this frame).
internal b32 sim_timeline_control(SimState *state, SimMemory *memory, SimInput *input) {
    Timeline *tl = &state->timeline;
    UiState *ui = &state->ui;
    state->timeline_recording = 0;
    if (!timeline_available(tl)) { ui->scrub_t = -1.0; ui->step_back = 0; return 0; }
    if (tl->generation != state->scene_generation || tl->frame_bytes != AlignPow2(state->permanent.used, 64)) {
        timeline_reset(tl, &state->permanent, state->scene_generation, sim_default_key_interval(state));
        timeline_record(tl, &state->permanent, state->sim_time, state->step_count, state->accumulator);
        tl->params_hash = sim_params_hash(state);
    }
    // an edit while in the past forks: drop the recorded future, go live from here
    u64 h = sim_params_hash(state);
    if (h != tl->params_hash) {
        if (tl->playback && tl->key_count) { timeline_truncate_after(tl, tl->shown_key); tl->playback = 0; }
        tl->params_hash = h;
    }
    // bake
    if (ui->bake_go && !tl->baking && tl->key_count) {
        ui->bake_go = 0;
        if (tl->playback) { timeline_truncate_after(tl, tl->shown_key); tl->playback = 0; }
        else timeline_truncate_after(tl, timeline_find(tl, state->sim_time));
        if (ui->bake_precise && state->dbd) {   // reference-grade solver settings for the bake
            DbdScene *d = state->dbd;
            d->precision = DbdPrecision_F64;
            d->control.rtol = Minimum(d->control.rtol, 1e-5);
            d->chem_steps = 1;
            d->gpu_ready = 0;
        }
        tl->params_hash = sim_params_hash(state);
        f64 dur = Maximum((f64)ui->bake_duration, 1e-3);
        tl->interval = dur / (f64)Clamp(ui->bake_keys, 10, (s32)TIMELINE_MAX_KEYS / 2);
        tl->bake_start = state->sim_time;
        tl->bake_end = state->sim_time + dur;
        if (timeline_end(tl) < state->sim_time - 1e-9) timeline_record(tl, &state->permanent, state->sim_time, state->step_count, state->accumulator);
        tl->next_key_t = state->sim_time + tl->interval;
        tl->baking = 1;
        ui->bake_saved_max = ui->max_speed;
        ui->max_speed = 1;
        state->paused = 0;
    }
    if (tl->baking && (ui->bake_cancel || state->sim_time >= tl->bake_end - 1e-9)) {
        b32 done = !ui->bake_cancel;
        ui->bake_cancel = 0;
        tl->baking = 0;
        ui->max_speed = ui->bake_saved_max;
        state->paused = 1;
        if (done) {   // play the bake back from its start
            tl->playback = 1;
            tl->playhead = tl->bake_start;
            sim_timeline_restore(state, timeline_find(tl, tl->bake_start));
        }
    }
    // scrub / step through keyframes
    if (ui->scrub_t >= 0.0 && tl->key_count && !tl->baking) {
        if (ui->scrub_t > timeline_end(tl) + 1e-9) {
            // past the recording: go live (from the last keyframe if we were in playback)
            if (tl->playback) sim_timeline_restore(state, tl->key_count - 1);
            tl->playback = 0;
        } else {
            f64 t = Clamp(ui->scrub_t, timeline_start(tl), timeline_end(tl));
            u32 k = timeline_find(tl, t);
            if (k != tl->shown_key || !tl->playback) sim_timeline_restore(state, k);
            tl->playback = 1;
            tl->playhead = t;
        }
    }
    ui->scrub_t = -1.0;
    if (ui->step_back && tl->key_count && !tl->baking) {
        u32 k = tl->playback ? (tl->shown_key ? tl->shown_key - 1 : 0) : timeline_find(tl, state->sim_time - 1e-9);
        if (!tl->playback && tl->keys[k].sim_time >= state->sim_time - 1e-9 && k > 0) --k;
        sim_timeline_restore(state, k);
        tl->playback = 1;
        tl->playhead = tl->keys[k].sim_time;
        state->paused = 1;
    }
    ui->step_back = 0;
    if (tl->playback) {
        if (ui->step_once) {   // step forward one keyframe
            ui->step_once = 0;
            if (tl->shown_key + 1 < tl->key_count) {
                sim_timeline_restore(state, tl->shown_key + 1);
                tl->playhead = tl->keys[tl->shown_key].sim_time;
            } else {
                tl->playback = 0;
            }
        } else if (!state->paused) {
            f64 speed = state->nft ? (f64)ui->nft_speed : (f64)ui->speed;
            if (ui->max_speed) speed *= 100.0;
            tl->playhead += (f64)input->dt_frame * speed;
            if (tl->playhead >= timeline_end(tl)) {   // end of the recording: go live from the last keyframe
                if (tl->shown_key != tl->key_count - 1) sim_timeline_restore(state, tl->key_count - 1);
                tl->playback = 0;
            } else {
                u32 k = timeline_find(tl, tl->playhead);
                if (k != tl->shown_key) sim_timeline_restore(state, k);
            }
        }
    }
    if (!tl->playback) state->timeline_recording = 1;
    (void)memory;
    return tl->playback;
}

internal void sim_initialize(SimMemory *memory) {
    Arena whole = arena_make(memory->permanent_base, memory->permanent_size);
    SimState *state = push_struct(&whole, SimState);
    state->permanent = arena_split(&whole, arena_remaining(&whole) - SIMD_ALIGN);
    state->frame = arena_make(memory->frame_base, memory->frame_size);
    state->scratch = arena_make(memory->scratch_base, memory->scratch_size);

    state->dt_fixed = SIM_DT_FIXED;
    timeline_init(&state->timeline, memory);
    state->scene = memory->config.scene;
    camera_reset(&state->camera, state->scene);
    state->camera.auto_orbit = state->scene == SimScene_Particles || state->scene == SimScene_Sweep;

    sim_reset_scene(state, memory);
    memory->is_initialized = 1;
}

internal void sim_handle_input(SimState *state, SimInput *input) {
    if (was_pressed(input->keys[SimKey_Space]) || was_pressed(input->keys[SimKey_P])) {
        state->paused = !state->paused;
    }
    if (was_pressed(input->keys[SimKey_F1])) {
        state->camera.auto_orbit = !state->camera.auto_orbit;
    }
    if (was_pressed(input->keys[SimKey_F])) { state->camera.frame_request = 1; state->camera.user_moved = 0; }
    if (was_pressed(input->keys[SimKey_Home])) { camera_reset(&state->camera, state->scene); state->camera.auto_orbit = 0; }
    f32 orbit_speed = 1.5f * input->dt_frame;
    if (is_down(input->keys[SimKey_Left]))  state->camera.yaw -= orbit_speed;
    if (is_down(input->keys[SimKey_Right])) state->camera.yaw += orbit_speed;
    if (is_down(input->keys[SimKey_Up]))    state->camera.pitch = Minimum(state->camera.pitch + orbit_speed, 1.4f);
    if (is_down(input->keys[SimKey_Down]))  state->camera.pitch = Maximum(state->camera.pitch - orbit_speed, -1.4f);
    if (state->camera.auto_orbit) {
        state->camera.yaw += 0.25f * input->dt_frame;
    }
}

// Bounds of everything in the point batches.
internal b32 render_bounds(RenderCommands *render, v3 *lo, v3 *hi) {
    b32 any = 0;
    *lo = V3(1e30f); *hi = V3(-1e30f);
    for (u32 b = 0; b < render->batch_count; ++b) {
        RenderPointBatch *pb = render->batches + b;
        for (u32 i = 0; i < pb->count; ++i) {
            lo->x = Minimum(lo->x, pb->px[i]); hi->x = Maximum(hi->x, pb->px[i]);
            lo->y = Minimum(lo->y, pb->py[i]); hi->y = Maximum(hi->y, pb->py[i]);
            lo->z = Minimum(lo->z, pb->pz[i]); hi->z = Maximum(hi->z, pb->pz[i]);
            any = 1;
        }
    }
    return any;
}

// A floor grid under the scene: 1-2-5 spacing, about a dozen lines a side,
// fading toward its edge; major lines every 5.
internal void render_floor_grid(RenderCommands *render, Arena *frame, v3 lo, v3 hi) {
    f32 ext = Maximum(hi.x - lo.x, hi.z - lo.z) * 0.75f + 0.5f;
    f32 raw = ext / 6.0f, p10 = powf(10.0f, floorf(log10f(raw)));
    f32 step = raw / p10 < 2.0f ? p10 : raw / p10 < 5.0f ? 2.0f * p10 : 5.0f * p10;
    v3 c = 0.5f * (lo + hi);
    f32 y = lo.y - 0.04f * Maximum(hi.y - lo.y, 0.25f);
    s32 n = (s32)ceilf(ext / step);
    u32 lines = (u32)(2 * (2 * n + 1));
    f32 *xyz = push_array_nozero(frame, f32, lines * 6);
    u32 *rgba = push_array_nozero(frame, u32, lines * 2);
    f32 cx = floorf(c.x / step) * step, cz = floorf(c.z / step) * step;
    u32 v = 0;
    for (s32 axis = 0; axis < 2; ++axis)
        for (s32 i = -n; i <= n; ++i) {
            f32 t = (f32)i * step;
            b32 major = (i % 5) == 0;
            f32 fade = 1.0f - (f32)(i < 0 ? -i : i) / (f32)(n + 1);
            u32 a = (u32)((major ? 70.0f : 34.0f) * fade);
            u32 col = (a << 24) | (0xA0u << 16) | (0x90u << 8) | 0x80u;
            f32 x0 = axis == 0 ? cx + t : cx - n * step, x1 = axis == 0 ? cx + t : cx + n * step;
            f32 z0 = axis == 0 ? cz - n * step : cz + t, z1 = axis == 0 ? cz + n * step : cz + t;
            f32 *p = xyz + v * 3;
            p[0] = x0; p[1] = y; p[2] = z0; p[3] = x1; p[4] = y; p[5] = z1;
            rgba[v] = col; rgba[v + 1] = col;
            v += 2;
        }
    RenderLineBatch *lb = render->line_batches + render->line_batch_count++;
    *lb = RenderLineBatch{};
    lb->vertex_count = v;
    lb->xyz = xyz;
    lb->rgba = rgba;
    lb->width = 1.0f;
}

internal void sim_fill_render(SimState *state, SimInput *input, RenderCommands *render) {
    TIMED_FUNCTION();
    Camera *cam = &state->camera;
    if (cam->fov <= 0.0f) camera_reset(cam, state->scene);
    render->clear_color = V4(0.030f, 0.034f, 0.046f, 1.0f);
    if (state->dbd) dbd_scene_fill_render(state->dbd, render, (u32)state->ui.view_species);
    else if (state->nft) nft_scene_fill_render(state->nft, render, (u32)state->ui.nft_layer);
    else {
        render->batch_count = 1;
        RenderPointBatch *b = render->batches;
        b->point_size = POINT_SIZE;
        b->color = POINT_COLOR;
        b->count = state->particles.count;
        b->px = state->particles.x;
        b->py = state->particles.y;
        b->pz = state->particles.z;
        b->rgba = 0;
    }
    f32 vw = render->scene_w > 0 ? render->scene_w : (f32)input->window_width;
    f32 vh = render->scene_h > 0 ? render->scene_h : (f32)input->window_height;
    f32 aspect = vh > 0 ? vw / vh : 1.0f;
    // frame the scene's bounds when asked (scene change, F, menu): the bounding
    // sphere fits the narrower of the vertical and horizontal fields of view
    if (!cam->user_moved && cam->framed_aspect > 0.0f && fabsf(aspect / cam->framed_aspect - 1.0f) > 0.05f)
        cam->frame_request = 1;   // the view was resized (e.g. the dock layout appeared) before the user took over
    if (cam->frame_request) {
        cam->framed_aspect = aspect;
        v3 lo, hi;
        if (render_bounds(render, &lo, &hi)) {
            cam->bounds_min = lo; cam->bounds_max = hi;
            cam->target = 0.5f * (lo + hi);
            f32 radius = 0.5f * length(hi - lo);
            f32 half = Minimum(0.5f * cam->fov, atanf(tanf(0.5f * cam->fov) * aspect));
            cam->distance = Clamp(radius / sinf(half) * 0.92f, 0.05f, 400.0f);
        }
        cam->frame_request = 0;
    }
    CameraBasis cb = camera_basis(cam, aspect);
    f32 near_z = Maximum(0.01f, cam->distance * 0.01f), far_z = cam->distance * 20.0f + 50.0f;
    m4 proj = m4_perspective(cam->fov, aspect, near_z, far_z);
    m4 view = m4_look_at(cb.eye, cam->target, V3(0.0f, 1.0f, 0.0f));
    render->view_proj = proj * view;
    render->point_ref_w = 4.2f;   // attenuated points have their nominal size at the default framing distance
    if (state->ui.show_grid && state->scene != SimScene_Particles) {
        v3 lo = cam->bounds_min, hi = cam->bounds_max;
        if (hi.x < lo.x) render_bounds(render, &lo, &hi);
        render_floor_grid(render, &state->frame, lo, hi);
    }
    // scene element visibility (after framing, so frame-all still sees everything)
    if (state->nft && render->batch_count >= 2) {
        if (!state->ui.show_cells) render->batches[0].count = 0;
        if (!state->ui.show_plants) render->batches[1].count = 0;
    }
    // hover and selection markers, drawn over the scene
    UiPick *marks[2] = { &state->ui.selected, &state->ui.hover };
    for (u32 i = 0; i < 2 && render->batch_count < RENDER_MAX_BATCHES; ++i) {
        if (!marks[i]->valid) continue;
        f32 *xyz = push_array_nozero(&state->frame, f32, 3);
        xyz[0] = marks[i]->pos.x; xyz[1] = marks[i]->pos.y; xyz[2] = marks[i]->pos.z;
        RenderPointBatch *b = render->batches + render->batch_count++;
        *b = {};
        b->count = 1; b->px = xyz; b->py = xyz + 1; b->pz = xyz + 2; b->rgba = 0;
        b->color = i == 0 ? V4(1.0f, 0.85f, 0.30f, 0.9f) : V4(1.0f, 1.0f, 1.0f, 0.55f);
        b->point_size = i == 0 ? 18.0f : 14.0f;
        b->on_top = 1;
    }
}

SIM_API SIM_ON_RELOAD(sim_on_reload) {
#if SIM_INTERNAL
    g_debug_table = memory->debug;
#endif
    if (memory->is_initialized) {
        SimState *state = (SimState *)memory->permanent_base;
        state->reload_count += 1;
        // Re-derive anything that points into this module's image.
        if (state->dbd) dbd_scene_on_reload(state->dbd);
        ui_on_reload(&state->ui);
#if SIM_INTERNAL
        // Profiler records keep pointers to their name / __FILE__ literals, which
        // lived in the old module. Forget them; the next hit re-registers.
        if (memory->debug) {
            for (u32 i = DEBUG_RECORDS_PER_TU; i < 2 * DEBUG_RECORDS_PER_TU && i < DEBUG_MAX_RECORDS; ++i) {
                DebugRecord *r = memory->debug->records + i;
                r->name = 0;
                r->file = 0;
                r->last_hits = 0;
                r->last_cycles = 0;
            }
        }
#endif
        memory->platform.log("[sim] reloaded (count %u), step %llu preserved\n",
                             state->reload_count, (unsigned long long)state->step_count);
    }
}

SIM_API SIM_BEFORE_UNLOAD(sim_before_unload) {
    if (memory->is_initialized) {
        SimState *state = (SimState *)memory->permanent_base;
        ui_before_unload(&state->ui);
    }
}

#include "../ui/ui_panels.cpp"

// Consumes the accumulator in master steps. budget_ms > 0 (windowed): stop
// when the next step would overrun the wall budget and drop the remaining
// accumulated time, so a slow solver slows simulated time, not the frame
// rate. budget_ms == 0 (headless): the accumulator alone decides, so runs are
// deterministic.
internal u32 sim_run_chemistry(SimState *state, SimMemory *memory, f64 budget_ms) {
    u32 steps = 0;
    u64 start_ns = memory->platform.now_ns();
    f64 per_step_ms = state->dbd->last_step_wall_ms;
    // One chemistry advance covers chem_steps master steps (the coupling interval).
    u32 k = Maximum(state->dbd->chem_steps, 1u);
    f64 dt_chem = state->dt_fixed * (f64)k;
    while (state->accumulator >= dt_chem - 1e-12 && !state->dbd->finished) {
        f64 spent_ms = (f64)(memory->platform.now_ns() - start_ns) * 1e-6;
        if (budget_ms > 0.0 && steps > 0 && spent_ms + per_step_ms > budget_ms) {
            state->accumulator = 0.0;
            break;
        }
        dbd_scene_step(state->dbd, memory, &state->frame, dt_chem);
        per_step_ms = state->dbd->last_step_wall_ms;
        state->accumulator -= dt_chem;
        state->step_count += k;
        state->sim_time += dt_chem;
        steps += k;
        sim_timeline_maybe_record(state);
    }
    if (state->dbd->finished) state->accumulator = 0.0;
    state->ui.chem_ms_frame = (f32)((f64)(memory->platform.now_ns() - start_ns) * 1e-6);
    return steps;
}

// NFT: whole coupling windows (60 s of plant time each) while the accumulator
// (in simulated seconds) holds one; same budget rules as the chemistry.
internal u32 sim_run_nft(SimState *state, SimMemory *memory, f64 budget_ms) {
    NftScene *sc = state->nft;
    u32 windows = 0;
    u64 start_ns = memory->platform.now_ns();
    f64 per_ms = sc->last_window_wall_ms;
    while (state->accumulator >= sc->sys.window_s - 1e-9) {
        f64 spent_ms = (f64)(memory->platform.now_ns() - start_ns) * 1e-6;
        if (budget_ms > 0.0 && windows > 0 && spent_ms + per_ms > budget_ms) {
            state->accumulator = 0.0;
            break;
        }
        nft_scene_window(sc, memory, &state->frame);
        per_ms = sc->last_window_wall_ms;
        state->accumulator -= sc->sys.window_s;
        state->step_count += 1;
        state->sim_time = sc->sys.t;
        ++windows;
        sim_timeline_maybe_record(state);
    }
    state->ui.chem_ms_frame = (f32)((f64)(memory->platform.now_ns() - start_ns) * 1e-6);
    return windows;
}

internal PLATFORM_WORK_QUEUE_CALLBACK(sim_chemistry_job) {
    SimState *state = (SimState *)data;
    if (state->nft) state->chem_job.steps = sim_run_nft(state, state->chem_job.memory, state->chem_job.budget_ms);
    else state->chem_job.steps = sim_run_chemistry(state, state->chem_job.memory, state->chem_job.budget_ms);
}

SIM_API SIM_UPDATE(sim_update) {
#if SIM_INTERNAL
    g_debug_table = memory->debug;
#endif
    TIMED_FUNCTION();

    if (!memory->is_initialized) {
        sim_initialize(memory);
    }
    SimState *state = (SimState *)memory->permanent_base;
    f32 join_ms = 0.0f;
    if (state->chem_job.in_flight) {
        u64 j0 = memory->platform.now_ns();
        memory->platform.complete_all_work(memory->background_queue);
        join_ms = (f32)((f64)(memory->platform.now_ns() - j0) * 1e-6);
        state->chem_job.in_flight = 0;
    }
    state->ui.join_ms = join_ms;
    arena_clear(&state->frame);

    // Windowed: the UI runs first so its edits apply to this frame's steps.
    b32 ui_on = !input->headless && input->window_width > 0;
    b32 ui_keyboard = 0;
    *render = RenderCommands{};
    if (ui_on) {
        ui_frame_begin(state, memory, input, render, &state->last_stats);
        ui_keyboard = ImGui::GetIO().WantCaptureKeyboard;
    }
    if (!ui_keyboard) {
        sim_handle_input(state, input);
        if (was_pressed(input->keys[SimKey_R])) sim_reset_scene(state, memory);
        if (ui_on && was_pressed(input->keys[SimKey_Period])) { state->ui.step_once = 1; state->paused = 1; }
        if (ui_on && was_pressed(input->keys[SimKey_Comma])) state->ui.step_back = 1;
        if (ui_on && is_down(input->keys[SimKey_Ctrl]) && was_pressed(input->keys[SimKey_Q])) render->request_quit = 1;
        if (ui_on && was_pressed(input->keys[SimKey_F2])) state->ui.open_shortcuts = 1;
    }
    b32 in_playback = ui_on ? sim_timeline_control(state, memory, input) : 0;
    if (input->timeline_op && timeline_available(&state->timeline)) {   // headless --timeline-check
        Timeline *tl = &state->timeline;
        if (input->timeline_op == 1) {
            timeline_reset(tl, &state->permanent, state->scene_generation, 1.0);
            timeline_record(tl, &state->permanent, state->sim_time, state->step_count, state->accumulator);
        } else if (tl->key_count) {
            sim_timeline_restore(state, 0);
        }
    }

    // Fixed timestep: accumulate wall time, step deterministically.
    u32 steps = 0;
    b32 async_chem = ui_on && (state->dbd || state->nft) && memory->background_queue;
    b32 kick_chem = 0;
    b32 step_once = ui_on && state->ui.step_once;
    state->ui.step_once = 0;
    if (in_playback) {
        // playback: the timeline restored the keyframe to show; nothing to simulate
    } else if (state->nft && (!state->paused || step_once)) {
        // NFT runs on its own clock: simulated seconds per wall second.
        f64 speed = ui_on ? (f64)state->ui.nft_speed : (memory->config.speed > 0.0 ? memory->config.speed : 3600.0);
        if (ui_on && state->ui.max_speed) speed = 1e12;
        state->accumulator += (f64)input->dt_frame * speed;
        f64 window = state->nft->sys.window_s;
        if (step_once) state->accumulator = window;
        if (ui_on) state->accumulator = Minimum(state->accumulator, window * 100000.0);
        TIMED_BLOCK("nft_windows");
        if (async_chem) {
            state->chem_job.budget_ms = ui_chemistry_budget(&state->ui, input);
            kick_chem = 1;
        } else {
            steps = sim_run_nft(state, memory, 0.0);
        }
    } else if (!state->paused || step_once) {
        f64 speed = (ui_on && state->dbd) ? (f64)state->ui.speed : 1.0;
        if (ui_on && state->dbd && state->ui.max_speed) speed = 1e9;
        state->accumulator += (f64)input->dt_frame * speed;
        u32 max_steps = (ui_on && state->dbd) ? 1000000 : SIM_MAX_STEPS_PER_FRAME;   // windowed chemistry: wall budget instead
        if (state->dbd) max_steps = Maximum(max_steps, state->dbd->chem_steps);   // room for one chemistry advance
        f64 max_accum = state->dt_fixed * max_steps;
        if (state->accumulator > max_accum) {
            state->accumulator = max_accum;
        }
        if (step_once) state->accumulator = state->dt_fixed * (state->dbd ? (f64)Maximum(state->dbd->chem_steps, 1u) : 1.0);
        TIMED_BLOCK("fixed_steps");
        if (state->dbd) {
            if (async_chem) {
                state->chem_job.budget_ms = ui_chemistry_budget(&state->ui, input);
                kick_chem = 1;   // after the render data is filled, below
            } else {
                steps = sim_run_chemistry(state, memory, input->headless ? 0.0 : SIM_CHEM_FRAME_BUDGET_MS);
            }
        } else {
            ParticleParams params;
            params.spring = PARTICLE_SPRING;
            params.swirl = PARTICLE_SWIRL;
            params.damping = PARTICLE_DAMPING;
            params.bound = PARTICLE_BOUND;
            f32 dt_step = (f32)state->dt_fixed;
            while (state->accumulator >= state->dt_fixed) {
                particles_step(&state->particles, params, dt_step);
                state->accumulator -= state->dt_fixed;
                state->step_count += 1;
                state->sim_time += state->dt_fixed;
                ++steps;
            }
        }
    }

    if (!state->dbd && !state->nft) {
        // Example scratch use: a transient reduction buffer that is rolled back.
        SCRATCH_SCOPE(&state->scratch);
        f32 *tmp = push_array_nozero(&state->scratch, f32, 8);
        tmp[0] = particles_kinetic_energy(&state->particles);
        state->scratch_high_water = Maximum(state->scratch_high_water, state->scratch.used);
    }

    sim_fill_render(state, input, render);
    if (ui_on) ui_frame_end(state, input, render);
    if (async_chem) steps = state->chem_job.steps;   // last job's count, for the stats below

    stats->step_count = state->step_count;
    stats->steps_this_frame = steps;
    stats->sim_time = state->sim_time;
    stats->dt_fixed = (f32)state->dt_fixed;
    stats->particle_count = state->dbd ? state->dbd->zones.count : state->nft ? state->nft->sys.rails : state->particles.count;
    stats->state_hash = 0;
    if (input->compute_state_hash) {
        stats->state_hash = state->dbd ? dbd_scene_hash(state->dbd)
                          : state->nft ? nft_scene_hash(state->nft) : particles_hash(&state->particles);
    }
    stats->thread_count = memory->worker_count;
    stats->summary[0] = 0;
    if (state->dbd) dbd_scene_summary(state->dbd, stats->summary, sizeof(stats->summary), memory->worker_count);
    if (state->nft) nft_scene_summary(state->nft, stats->summary, sizeof(stats->summary));
    stats->permanent_used = sizeof(SimState) + state->permanent.used;
    stats->permanent_size = memory->permanent_size;
    stats->frame_used = state->frame.used;
    stats->frame_size = memory->frame_size;
    stats->scratch_high_water = state->scratch_high_water;
    stats->scratch_size = memory->scratch_size;
    stats->paused = state->paused;
    stats->reload_count = state->reload_count;
    state->last_stats = *stats;

    // Everything the platform reads (render data, stats) is filled; the job only
    // touches chemistry state and pushes temporaries above them in the frame arena.
    if (kick_chem) {
        state->chem_job.memory = memory;
        state->chem_job.in_flight = 1;
        memory->platform.add_work_entry(memory->background_queue, sim_chemistry_job, state);
    }
}
