// DBD scenes on top of dbd.cpp:
//
//   reactor  a coaxial DBD tube split into N axial zones (plug flow). Humid
//            air enters zone 0; each zone is discharged and its afterglow
//            chemistry integrated; the outlet is the last zone. Outlet
//            mole fractions over time go to the CSV.
//   sweep    N independent batch reactors over a power x humidity grid.
//            A batch reactor at time t is a plug-flow reactor at residence
//            time t, so after `residence_s` each zone is an outlet
//            composition. Written to the CSV once, then the scene holds.
//
// Included by simstack.cpp (unity build).

#define DBD_PLOT_SPECIES 8
#define DBD_SERIES_CAPACITY 65536   // outlet rows kept for the reactor CSV
#define DBD_TUBE_RING       28      // points per axial ring in the 3D reactor view

struct DbdPlotSpecies {
    const char *name;
    v4 color;
};

global const DbdPlotSpecies dbd_plot_species[DBD_PLOT_SPECIES] = {
    { "O3",   { 0.40f, 0.75f, 1.00f, 1.0f } },
    { "NO",   { 1.00f, 0.85f, 0.30f, 1.0f } },
    { "NO2",  { 1.00f, 0.50f, 0.20f, 1.0f } },
    { "N2O5", { 0.80f, 0.40f, 1.00f, 1.0f } },
    { "HNO2", { 0.40f, 1.00f, 0.55f, 1.0f } },
    { "HNO3", { 1.00f, 0.35f, 0.55f, 1.0f } },
    { "H2O2", { 0.90f, 0.90f, 0.90f, 1.0f } },
    { "N2O",  { 0.55f, 0.55f, 0.70f, 1.0f } },
};

struct DbdOperating {
    f64 T_K, p_Pa, x_H2O, power_W_cm3, freq_Hz;
};

struct DbdScene {
    KinModel model;
    AirSpecies sp;
    AirSourceModel source;
    KinMethodId method_id;
    KinMethod method;          // re-derived on reload (holds a pointer to a string literal)
    KinControl control;
    DbdDischargeMode mode;
    b32 chained;               // reactor: plug flow; sweep: independent
    u32 chem_steps;            // master steps per chemistry advance (coupling interval = chem_steps * dt)
    f64 residence_s;           // reactor: total gas residence; sweep: batch duration
    u32 sweep_nx, sweep_ny;    // sweep grid (power x humidity)

    DbdZones zones;
    DbdGpu gpu;
    b32 generated;             // straight-line kernels match this mechanism
    u32 precision;             // DbdPrecision
    b32 use_gpu;
    b32 gpu_ready;             // mechanism/method/control uploaded

    f64 t;                     // chemistry time [s]
    u64 master_steps;
    b32 finished;              // sweep reached residence_s
    f64 last_step_wall_ms;
    f64 wall_ms_total;
    u32 plot_index[DBD_PLOT_SPECIES];
    // 3D view (allocated once at init): reactor = tube of rings, sweep = surface
    u32 view_count;
    f32 *view_x, *view_y, *view_z;
    u32 *view_rgba;
    f64 view_lo, view_hi;      // log10 ppm range of the coloured species (for the colour bar)

    DbdOperating op;           // operating point the UI edits (sweep: power/humidity are the grid axes)

    // outlet time series, ring of DBD_SERIES_CAPACITY rows [t, ppm...]: oldest at series_head
    // when full. Reactor: outlet zone. Sweep: highest-power, wettest corner.
    f64 *series;
    u32 series_count, series_head;
    u32 series_every;          // master steps per sample
    f64 next_csv_t;
    const char *csv_path;      // points into the platform's argv (stable)
};

internal DbdOperating dbd_operating_defaults() {
    DbdOperating op;
    op.T_K = 310.0;             // gas warms a little in the discharge gap
    op.p_Pa = 101325.0;
    op.x_H2O = 0.02;            // ~65 % RH at 25 C
    op.power_W_cm3 = 0.1;       // 0.2 J/cm^3 = 200 J/L specific energy over a 2 s residence
    op.freq_Hz = 10000.0;
    return op;
}

// Writes the operating point into every zone (sweep keeps its power x
// humidity grid) and recomputes rate constants. Composition is untouched, so
// the chemistry responds live.
internal void dbd_scene_apply_operating(DbdScene *scene) {
    DbdZones *z = &scene->zones;
    DbdOperating *op = &scene->op;
    for (u32 i = 0; i < z->count; ++i) {
        z->T_K[i] = op->T_K;
        z->p_Pa[i] = op->p_Pa;
        z->freq_Hz[i] = op->freq_Hz;
        if (scene->chained) {
            z->x_H2O[i] = op->x_H2O;
            z->power_W_cm3[i] = op->power_W_cm3;
        } else {
            u32 ix = i % scene->sweep_nx, iy = i / scene->sweep_nx;
            f64 fx = scene->sweep_nx > 1 ? (f64)ix / (f64)(scene->sweep_nx - 1) : 0.0;
            f64 fy = scene->sweep_ny > 1 ? (f64)iy / (f64)(scene->sweep_ny - 1) : 0.0;
            z->power_W_cm3[i] = 0.01 * pow(50.0, fx);   // 0.01 .. 0.5 W/cm^3, log spaced
            z->x_H2O[i] = 0.04 * fy;                    // dry .. ~saturated at 30 C
        }
    }
    dbd_zones_fill_padding(z);
    dbd_zones_prepare_rates(z, &scene->model);
}

internal void dbd_scene_init(DbdScene *scene, Arena *arena, SimConfig *config, b32 sweep) {
    air_mech_build(&scene->model, &scene->sp);
    scene->generated = dbd_generated_matches(&scene->model);
    scene->precision = config->precision == 1 ? DbdPrecision_F64 : DbdPrecision_F32;
    scene->source = air_source_defaults();
    scene->method_id = KinMethod_Rodas3;
    scene->method = kin_method(scene->method_id);
    scene->control = dbd_default_control();
    scene->mode = config->averaged ? DbdDischarge_Averaged : DbdDischarge_Pulsed;
    scene->chained = !sweep;
    scene->chem_steps = config->chem_steps ? config->chem_steps : (sweep ? 20 : 1);
    scene->residence_s = 2.0;
    scene->use_gpu = config->backend == SimBackend_Gpu;
    scene->csv_path = config->csv_path;

    u32 count = config->zone_count;
    if (sweep) {
        u32 side = 1;
        while ((side + 1) * (side + 1) <= count) ++side;
        scene->sweep_nx = side;
        scene->sweep_ny = side;
        count = side * side;
    } else if (count == 0) {
        count = 16;
    }
    dbd_zones_alloc(&scene->zones, arena, &scene->model, count);
    DbdZones *z = &scene->zones;
    scene->op = dbd_operating_defaults();
    dbd_scene_apply_operating(scene);
    dbd_zones_reset(z, &scene->sp);

    for (u32 p = 0; p < DBD_PLOT_SPECIES; ++p) {
        s32 s = kin_find_species(&scene->model, dbd_plot_species[p].name);
        Assert(s >= 0);
        scene->plot_index[p] = (u32)s;
    }
    scene->view_count = sweep ? count : count * DBD_TUBE_RING;
    scene->view_x = SoaPush(arena, f32, scene->view_count);
    scene->view_y = SoaPush(arena, f32, scene->view_count);
    scene->view_z = SoaPush(arena, f32, scene->view_count);
    scene->view_rgba = push_array_aligned(arena, u32, SoaPadCount(scene->view_count), SIMD_ALIGN);
    scene->series = push_array(arena, f64, (u64)DBD_SERIES_CAPACITY * (1 + DBD_PLOT_SPECIES));
    scene->series_every = 10;
    scene->next_csv_t = 1.0;
}

internal void dbd_scene_on_reload(DbdScene *scene) {
    scene->method = kin_method(scene->method_id);
    scene->gpu_ready = 0;   // re-upload: the method struct changed identity
}

// --- output -----------------------------------------------------------------

internal void dbd_scene_write_csv(DbdScene *scene, PlatformAPI *api, Arena *scratch) {
    if (!scene->csv_path) return;
    TempMemory temp = begin_temp(scratch);
    DbdZones *z = &scene->zones;
    u64 cap = Megabytes(16);
    char *buf = (char *)push_size_(scratch, cap, 16, 0);
    u64 len = 0;
#define CSV(...) len += (u64)snprintf(buf + len, len < cap ? cap - len : 0, __VA_ARGS__)
    if (scene->chained) {
        CSV("# simstack dbd reactor: %u zones, residence %.3f s, %s discharge, outlet mole fractions [ppm]\n",
            z->count, scene->residence_s, scene->mode == DbdDischarge_Pulsed ? "pulsed" : "averaged");
        CSV("t_s");
        for (u32 p = 0; p < DBD_PLOT_SPECIES; ++p) CSV(",%s", dbd_plot_species[p].name);
        CSV("\n");
        for (u32 r = 0; r < scene->series_count && len < cap - 512; ++r) {
            u32 slot = (scene->series_head + r) % DBD_SERIES_CAPACITY;   // oldest first
            f64 *row = scene->series + (u64)slot * (1 + DBD_PLOT_SPECIES);
            CSV("%.4f", row[0]);
            for (u32 p = 0; p < DBD_PLOT_SPECIES; ++p) CSV(",%.6e", row[1 + p]);
            CSV("\n");
        }
    } else {
        CSV("# simstack dbd sweep: %u x %u batch reactors after %.3f s, %s discharge, mole fractions [ppm]\n",
            scene->sweep_nx, scene->sweep_ny, scene->t, scene->mode == DbdDischarge_Pulsed ? "pulsed" : "averaged");
        CSV("power_W_cm3,x_H2O,energy_J_L");
        for (u32 p = 0; p < DBD_PLOT_SPECIES; ++p) CSV(",%s", dbd_plot_species[p].name);
        CSV(",steps,failed\n");
        for (u32 i = 0; i < z->count && len < cap - 512; ++i) {
            CSV("%.5f,%.4f,%.2f", z->power_W_cm3[i], z->x_H2O[i], z->power_W_cm3[i] * scene->t * 1000.0);
            for (u32 p = 0; p < DBD_PLOT_SPECIES; ++p) CSV(",%.6e", dbd_ppm(z, scene->plot_index[p], i));
            CSV(",%.0f,%.0f\n", z->steps[i], z->failed[i]);
        }
    }
#undef CSV
    if (len >= cap) len = cap - 1;
    if (!api->write_entire_file(scene->csv_path, buf, len)) api->log("[dbd] cannot write %s\n", scene->csv_path);
    end_temp(temp);
}

// --- step -----------------------------------------------------------------

// One master step of the chemistry. Returns 0 on success.
internal s32 dbd_scene_step(DbdScene *scene, SimMemory *memory, Arena *frame, f64 dt) {
    TIMED_FUNCTION();
    if (scene->finished) return 0;
    DbdZones *z = &scene->zones;
    u64 w0 = memory->platform.now_ns();

    f64 t0 = scene->t, t1 = scene->t + dt;
    dbd_zones_prepare_source(z, &scene->sp, &scene->source, scene->mode, t0, t1);
    s32 err = 0;
    if (scene->use_gpu && memory->gpu.available) {
        if (!scene->gpu_ready) {
            err = memory->gpu.gpu_kin_set_mech(&scene->model.mech, &scene->method, &scene->control);
            if (err) {
                memory->platform.log("[dbd] gpu_kin_set_mech: %s; falling back to the CPU\n", memory->gpu.gpu_error_string(err));
                scene->use_gpu = 0;
            }
            scene->gpu_ready = 1;
        }
    }
    b32 done_on_gpu = 0;
    if (scene->use_gpu && memory->gpu.available) {
        err = dbd_advance_gpu(z, &memory->gpu, &scene->gpu, dt, scene->precision);
        if (err) {
            memory->platform.log("[dbd] gpu advance failed: %s; falling back to the CPU\n", memory->gpu.gpu_error_string(err));
            scene->use_gpu = 0;
        } else {
            done_on_gpu = 1;
        }
    }
    if (!done_on_gpu) {
        dbd_advance_threads(z, &scene->model.mech, &scene->method, &scene->control, dt, scene->generated, scene->precision,
                            &memory->platform, memory->work_queue, memory->worker_count, frame);
    }
    dbd_zones_commit_pulses(z);
    if (scene->chained) dbd_flow_exchange(z, &scene->sp, scene->residence_s, dt);
    scene->t = t1;
    ++scene->master_steps;

    u64 w1 = memory->platform.now_ns();
    scene->last_step_wall_ms = (f64)(w1 - w0) * 1e-6;
    scene->wall_ms_total += scene->last_step_wall_ms;

    // outlet (reactor) / corner (sweep) sample into the ring
    if (scene->master_steps % Maximum(scene->series_every / scene->chem_steps, 1u) == 0) {
        u32 slot;
        if (scene->series_count < DBD_SERIES_CAPACITY) {
            slot = (scene->series_head + scene->series_count++) % DBD_SERIES_CAPACITY;
        } else {
            slot = scene->series_head;
            scene->series_head = (scene->series_head + 1) % DBD_SERIES_CAPACITY;
        }
        f64 *row = scene->series + (u64)slot * (1 + DBD_PLOT_SPECIES);
        row[0] = scene->t;
        for (u32 p = 0; p < DBD_PLOT_SPECIES; ++p) row[1 + p] = dbd_ppm(z, scene->plot_index[p], z->count - 1);
    }
    if (scene->chained) {
        if (scene->t >= scene->next_csv_t - 1e-9) {
            dbd_scene_write_csv(scene, &memory->platform, frame);
            scene->next_csv_t += 1.0;
        }
    } else if (scene->t >= scene->residence_s - 1e-9) {
        scene->finished = 1;
        dbd_scene_write_csv(scene, &memory->platform, frame);
        memory->platform.log("[dbd] sweep finished at t = %.3f s%s%s\n", scene->t,
                             scene->csv_path ? ", wrote " : "", scene->csv_path ? scene->csv_path : "");
    }
    return err;
}

// --- render -----------------------------------------------------------------

// Perceptual colormap (viridis control points), t in [0, 1] -> 0xAABBGGRR.
internal u32 dbd_colormap(f32 t) {
    static const f32 cm[6][3] = {
        {0.267f, 0.005f, 0.329f}, {0.231f, 0.322f, 0.545f}, {0.128f, 0.567f, 0.551f},
        {0.267f, 0.749f, 0.441f}, {0.741f, 0.873f, 0.150f}, {0.993f, 0.906f, 0.144f},
    };
    t = Clamp(t, 0.0f, 1.0f) * 5.0f;
    u32 i = (u32)t;
    if (i > 4) i = 4;
    f32 f = t - (f32)i;
    u32 rgb[3];
    for (u32 c = 0; c < 3; ++c) rgb[c] = (u32)(255.0f * (cm[i][c] + f * (cm[i + 1][c] - cm[i][c])) + 0.5f);
    return 0xFF000000u | (rgb[2] << 16) | (rgb[1] << 8) | rgb[0];
}

// reactor: a tube along x, one ring of points per zone, coloured by the chosen
//          species' log10 mole fraction (inlet at -x).
// sweep:   x = log power, z = humidity, y and colour = log10 mole fraction.
internal void dbd_scene_fill_render(DbdScene *scene, RenderCommands *render, u32 view_species) {
    TIMED_FUNCTION();
    DbdZones *z = &scene->zones;
    u32 s = scene->plot_index[view_species % DBD_PLOT_SPECIES];
    f64 lo = 1e300, hi = -1e300;
    for (u32 i = 0; i < z->count; ++i) {
        f64 ppm = dbd_ppm(z, s, i);
        f64 v = log10(ppm > 1e-6 ? ppm : 1e-6);
        lo = Minimum(lo, v); hi = Maximum(hi, v);
    }
    if (hi - lo < 0.5) { f64 mid = 0.5 * (hi + lo); lo = mid - 0.25; hi = mid + 0.25; }
    scene->view_lo = lo;
    scene->view_hi = hi;
    f32 inv = (f32)(1.0 / (hi - lo));
    if (scene->chained) {
        for (u32 i = 0; i < z->count; ++i) {
            f64 ppm = dbd_ppm(z, s, i);
            f32 t = ((f32)log10(ppm > 1e-6 ? ppm : 1e-6) - (f32)lo) * inv;
            u32 rgba = dbd_colormap(t);
            f32 x = z->count > 1 ? -1.6f + 3.2f * (f32)i / (f32)(z->count - 1) : 0.0f;
            for (u32 k = 0; k < DBD_TUBE_RING; ++k) {
                f32 a = TAU32 * (f32)k / (f32)DBD_TUBE_RING;
                u32 j = i * DBD_TUBE_RING + k;
                scene->view_x[j] = x;
                scene->view_y[j] = 0.38f * cosf(a);
                scene->view_z[j] = 0.38f * sinf(a);
                scene->view_rgba[j] = rgba;
            }
        }
    } else {
        for (u32 i = 0; i < z->count; ++i) {
            u32 ix = i % scene->sweep_nx, iy = i / scene->sweep_nx;
            f64 ppm = dbd_ppm(z, s, i);
            f32 t = ((f32)log10(ppm > 1e-6 ? ppm : 1e-6) - (f32)lo) * inv;
            scene->view_x[i] = -1.4f + 2.8f * (f32)ix / (f32)Maximum(scene->sweep_nx - 1, 1u);
            scene->view_z[i] = -1.4f + 2.8f * (f32)iy / (f32)Maximum(scene->sweep_ny - 1, 1u);
            scene->view_y[i] = -0.6f + 1.2f * t;
            scene->view_rgba[i] = dbd_colormap(t);
        }
    }
    render->batch_count = 1;
    RenderPointBatch *b = render->batches;
    b->count = scene->view_count;
    b->px = scene->view_x; b->py = scene->view_y; b->pz = scene->view_z;
    b->rgba = scene->view_rgba;
    b->color = V4(1, 1, 1, 1);
    b->point_size = scene->chained ? (z->count <= 64 ? 9.0f : 6.0f) : (z->count <= 1024 ? 7.0f : 4.0f);
}

internal void dbd_scene_summary(DbdScene *scene, char *out, u32 size, u32 threads) {
    DbdZones *z = &scene->zones;
    u32 at = z->count - 1;   // reactor: outlet; sweep: highest power, wettest corner
    f64 steps = 0.0, failed = 0.0;
    for (u32 i = 0; i < z->count; ++i) { steps += z->steps[i]; failed += z->failed[i]; }
    f64 ms = scene->t * 1000.0;
    snprintf(out, size,
             "%s %u zones %s %s %s | t %.3f s | %s O3 %.1f ppm NO %.2f NO2 %.2f N2O5 %.2f HNO2 %.3f HNO3 %.3f ppm | "
             "%.0f steps/zone/ms, %.2f ms wall/step, %u failed",
             scene->chained ? "reactor" : "sweep", z->count, scene->mode == DbdDischarge_Pulsed ? "pulsed" : "averaged",
             scene->precision == DbdPrecision_F32 ? "f32" : "f64",
             scene->use_gpu ? "gpu" : (threads > 1 ? "cpu-mt" : "cpu"), scene->t,
             scene->chained ? "outlet" : "corner",
             dbd_ppm(z, scene->sp.O3, at), dbd_ppm(z, scene->sp.NO, at), dbd_ppm(z, scene->sp.NO2, at),
             dbd_ppm(z, scene->sp.N2O5, at), dbd_ppm(z, scene->sp.HNO2, at), dbd_ppm(z, scene->sp.HNO3, at),
             ms > 0.0 ? steps / (f64)z->count / ms : 0.0,
             scene->master_steps ? scene->wall_ms_total / (f64)scene->master_steps : 0.0, (u32)failed);
}

internal u64 dbd_scene_hash(DbdScene *scene) {
    DbdZones *z = &scene->zones;
    return hash_fnv1a_64(z->y, sizeof(f64) * (u64)z->n_species * z->padded);
}
