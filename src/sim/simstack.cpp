// Simstack sim module: unity build root. Compiled into simstack_sim.{so,dll}.
// Every other sim .cpp is #included below; the platform never sees them.
//
// No OS calls, no malloc, no mutable globals except g_debug_table (re-patched
// every update) so a hot reload preserves all state in SimMemory. Threads and
// the GPU come from the platform through SimMemory (work_queue, gpu).

#define DEBUG_RECORD_BASE DEBUG_RECORDS_PER_TU
#include "../sim_platform.h"

#include "particles.cpp"
#include "dbd.cpp"
#include "dbd_scene.cpp"

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
    Camera camera;
};

// Resetting rolls the permanent arena back to just past the SimState header
// and rebuilds the scene from the config.
internal void sim_reset_scene(SimState *state, SimMemory *memory) {
    state->permanent.used = 0;
    state->step_count = 0;
    state->sim_time = 0.0;
    state->accumulator = 0.0;
    state->dbd = 0;
    state->rng.state = memory->config.seed ? memory->config.seed : 0x9E3779B9u;
    if (state->scene == SimScene_Particles) {
        particles_init(&state->particles, &state->permanent, memory->config.particle_count, &state->rng,
                       sqrtf(PARTICLE_SPRING));
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
    state->camera.auto_orbit = state->scene != SimScene_Dbd;

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
    f32 aspect = input->window_height ? (f32)input->window_width / (f32)input->window_height : 1.0f;
    m4 proj = m4_perspective(radians(50.0f), aspect, 0.05f, 100.0f);
    m4 view = m4_look_at(eye, V3(0.0f), V3(0.0f, 1.0f, 0.0f));

    render->clear_color = V4(0.05f, 0.06f, 0.08f, 1.0f);
    render->view_proj = proj * view;
    if (state->dbd) {
        dbd_scene_fill_render(state->dbd, render);
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
        memory->platform.log("[sim] reloaded (count %u), step %llu preserved\n",
                             state->reload_count, (unsigned long long)state->step_count);
    }
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
    arena_clear(&state->frame);

    sim_handle_input(state, input);
    if (was_pressed(input->keys[SimKey_R])) {
        sim_reset_scene(state, memory);
    }

    // Fixed timestep: accumulate wall time, step deterministically.
    u32 steps = 0;
    if (!state->paused) {
        state->accumulator += (f64)input->dt_frame;
        f64 max_accum = state->dt_fixed * SIM_MAX_STEPS_PER_FRAME;
        if (state->accumulator > max_accum) {
            state->accumulator = max_accum;
        }
        TIMED_BLOCK("fixed_steps");
        if (state->dbd) {
            // Chemistry master steps. Headless: deterministic (the accumulator
            // alone decides). Windowed: also stop when the frame's wall budget
            // is spent, so a slow solver slows sim time instead of the frame rate.
            u64 start_ns = memory->platform.now_ns();
            while (state->accumulator >= state->dt_fixed) {
                if (!input->headless && steps > 0 &&
                    (f64)(memory->platform.now_ns() - start_ns) * 1e-6 > SIM_CHEM_FRAME_BUDGET_MS) {
                    state->accumulator = 0.0;
                    break;
                }
                dbd_scene_step(state->dbd, memory, &state->frame, state->dt_fixed);
                state->accumulator -= state->dt_fixed;
                state->step_count += 1;
                state->sim_time += state->dt_fixed;
                ++steps;
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

    if (!state->dbd) {
        // Example scratch use: a transient reduction buffer that is rolled back.
        SCRATCH_SCOPE(&state->scratch);
        f32 *tmp = push_array_nozero(&state->scratch, f32, 8);
        tmp[0] = particles_kinetic_energy(&state->particles);
        state->scratch_high_water = Maximum(state->scratch_high_water, state->scratch.used);
    }

    sim_fill_render(state, input, render);

    stats->step_count = state->step_count;
    stats->steps_this_frame = steps;
    stats->sim_time = state->sim_time;
    stats->dt_fixed = (f32)state->dt_fixed;
    stats->particle_count = state->dbd ? state->dbd->zones.count : state->particles.count;
    stats->state_hash = 0;
    if (input->compute_state_hash) {
        stats->state_hash = state->dbd ? dbd_scene_hash(state->dbd) : particles_hash(&state->particles);
    }
    stats->thread_count = memory->worker_count;
    stats->summary[0] = 0;
    if (state->dbd) dbd_scene_summary(state->dbd, stats->summary, sizeof(stats->summary), memory->worker_count);
    stats->permanent_used = sizeof(SimState) + state->permanent.used;
    stats->permanent_size = memory->permanent_size;
    stats->frame_used = state->frame.used;
    stats->frame_size = memory->frame_size;
    stats->scratch_high_water = state->scratch_high_water;
    stats->scratch_size = memory->scratch_size;
    stats->paused = state->paused;
    stats->reload_count = state->reload_count;
}
