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
#include "dbd_scene.cpp"
#include "nft.cpp"
#include "nft_scene.cpp"
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
};

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
    Camera camera;
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
internal void sim_reset_scene(SimState *state, SimMemory *memory) {
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
            state->nft_params = nft_default_params();
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
    }
}

internal void sim_initialize(SimMemory *memory) {
    Arena whole = arena_make(memory->permanent_base, memory->permanent_size);
    SimState *state = push_struct(&whole, SimState);
    state->permanent = arena_split(&whole, arena_remaining(&whole) - SIMD_ALIGN);
    state->frame = arena_make(memory->frame_base, memory->frame_size);
    state->scratch = arena_make(memory->scratch_base, memory->scratch_size);

    state->dt_fixed = SIM_DT_FIXED;
    state->scene = memory->config.scene;
    state->camera.distance = state->scene == SimScene_Particles ? 3.5f : 4.2f;
    state->camera.pitch = 0.35f;
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
    f32 orbit_speed = 1.5f * input->dt_frame;
    if (is_down(input->keys[SimKey_Left]))  state->camera.yaw -= orbit_speed;
    if (is_down(input->keys[SimKey_Right])) state->camera.yaw += orbit_speed;
    if (is_down(input->keys[SimKey_Up]))    state->camera.pitch = Minimum(state->camera.pitch + orbit_speed, 1.4f);
    if (is_down(input->keys[SimKey_Down]))  state->camera.pitch = Maximum(state->camera.pitch - orbit_speed, -1.4f);
    if (state->camera.auto_orbit) {
        state->camera.yaw += 0.25f * input->dt_frame;
    }
}

internal void sim_fill_render(SimState *state, SimInput *input, RenderCommands *render) {
    TIMED_FUNCTION();
    Camera *cam = &state->camera;
    f32 cp = cosf(cam->pitch), sp = sinf(cam->pitch);
    v3 eye = V3(cam->distance * cp * sinf(cam->yaw), cam->distance * sp, cam->distance * cp * cosf(cam->yaw));
    f32 vw = render->scene_w > 0 ? render->scene_w : (f32)input->window_width;
    f32 vh = render->scene_h > 0 ? render->scene_h : (f32)input->window_height;
    f32 aspect = vh > 0 ? vw / vh : 1.0f;
    m4 proj = m4_perspective(radians(50.0f), aspect, 0.05f, 100.0f);
    m4 view = m4_look_at(eye, V3(0.0f), V3(0.0f, 1.0f, 0.0f));

    render->clear_color = V4(0.035f, 0.04f, 0.055f, 1.0f);
    render->view_proj = proj * view;
    if (state->dbd) {
        dbd_scene_fill_render(state->dbd, render, (u32)state->ui.view_species);
        return;
    }
    if (state->nft) {
        nft_scene_fill_render(state->nft, render, (u32)state->ui.nft_layer);
        return;
    }
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
    }

    // Fixed timestep: accumulate wall time, step deterministically.
    u32 steps = 0;
    b32 async_chem = ui_on && (state->dbd || state->nft) && memory->background_queue;
    b32 kick_chem = 0;
    b32 step_once = ui_on && state->ui.step_once;
    state->ui.step_once = 0;
    if (state->nft && (!state->paused || step_once)) {
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
