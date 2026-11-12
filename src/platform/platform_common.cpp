// Platform code shared by linux_simstack.cpp and win32_simstack.cpp:
// memory layout, command-line parsing, timing/profiler printing.
// Included into each platform TU (unity build); uses only C stdio.

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

// Fixed memory block sizes for milestone 1. The reservation uses a fixed base
// address in internal builds so pointers are stable across runs (useful for
// debugging and later for replay snapshots).
#define PLATFORM_PERMANENT_SIZE Megabytes(256)
#define PLATFORM_FRAME_SIZE     Megabytes(64)
#define PLATFORM_SCRATCH_SIZE   Megabytes(64)
#define PLATFORM_TOTAL_SIZE     (PLATFORM_PERMANENT_SIZE + PLATFORM_FRAME_SIZE + PLATFORM_SCRATCH_SIZE)
#if SIM_INTERNAL
#define PLATFORM_MEMORY_BASE_ADDRESS ((void *)0x200000000000ull)
#else
#define PLATFORM_MEMORY_BASE_ADDRESS ((void *)0)
#endif

#define PLATFORM_HEADLESS_DT (1.0f / 60.0f)
#define PLATFORM_MAX_FRAME_DT 0.1f
#define PLATFORM_TIMING_PRINT_INTERVAL 120   // frames, windowed mode

struct PlatformArgs {
    b32 headless;
    u32 headless_frames;
    u32 scene;            // SimScene
    u32 particle_count;
    u32 zone_count;
    u32 threads;          // work queue threads including the main thread
    b32 gpu;
    u32 gpu_mem_mb;
    b32 averaged;
    u32 precision;        // 0 f32, 1 f64 (SimConfig::precision)
    const char *csv_path;
    u32 seed;
    u32 window_width;
    u32 window_height;
    b32 quiet;            // suppress periodic timing prints in windowed mode
    u32 windowed_frames;  // 0 = run until closed
    const char *screenshot_path;   // write a PPM of the last windowed frame
};

internal void platform_print_usage(const char *exe) {
    printf("usage: %s [--scene particles|dbd|sweep] [--headless N] [--zones N] [--threads N] [--gpu]\n"
           "          [--gpu-mem MB] [--averaged] [--precision f32|f64] [--csv file] [--particles N] [--seed N] [--size WxH]\n"
           "          [--quiet] [--frames N] [--screenshot file.ppm]\n"
           "  --scene S      particles (M1 demo, default), dbd (plug-flow DBD reactor), sweep (power x humidity grid)\n"
           "  --headless N   run N frames without a window at a fixed 1/60 s frame dt, print timings\n"
           "  --zones N      dbd: axial zones (default 64); sweep: grid points (rounded down to a square, default 256)\n"
           "  --threads N    worker threads including the main thread (default 8, one per P-core)\n"
           "  --gpu          load build/simstack_gpu.so and run the chemistry on CUDA\n"
           "  --gpu-mem MB   device arena size (default 512)\n"
           "  --averaged     continuous discharge source instead of one pulse per half-cycle\n"
           "  --precision P  chemistry lanes: f32 (default: 8 zones per AVX2 register, CUDA FP32 units)\n"
           "                 or f64 (reference: exact atom bookkeeping for long runs)\n"
           "  --csv F        dbd: outlet time series (rewritten every simulated second); sweep: final grid\n"
           "  --frames N     windowed: quit after N frames\n"
           "  --screenshot F windowed: write the last frame as a binary PPM to F\n"
           "  --particles N  particles scene: count (default 65536)\n"
           "  --seed N       particle seed (default 1)\n"
           "  --size WxH     window size (default 1280x720)\n"
           "  --quiet        no periodic timing dump in windowed mode\n"
           "keys: Space/P pause, R reset, F1 toggle auto-orbit, arrows orbit, Esc quit\n", exe);
}

internal PlatformArgs platform_parse_args(int argc, char **argv) {
    PlatformArgs args = {};
    args.particle_count = 65536;
    args.threads = 8;
    args.gpu_mem_mb = 512;
    args.seed = 1;
    args.window_width = 1280;
    args.window_height = 720;
    for (int i = 1; i < argc; ++i) {
        const char *a = argv[i];
        const char *next = (i + 1 < argc) ? argv[i + 1] : 0;
        if (strcmp(a, "--headless") == 0) {
            args.headless = 1;
            args.headless_frames = 600;
            if (next && next[0] != '-') { args.headless_frames = (u32)atoi(next); ++i; }
        } else if (strcmp(a, "--scene") == 0 && next) {
            if (strcmp(next, "particles") == 0) args.scene = SimScene_Particles;
            else if (strcmp(next, "dbd") == 0) args.scene = SimScene_Dbd;
            else if (strcmp(next, "sweep") == 0) args.scene = SimScene_Sweep;
            else { printf("unknown scene %s\n", next); exit(1); }
            ++i;
        } else if (strcmp(a, "--zones") == 0 && next) {
            args.zone_count = (u32)atoi(next); ++i;
        } else if (strcmp(a, "--threads") == 0 && next) {
            args.threads = (u32)atoi(next); ++i;
        } else if (strcmp(a, "--gpu") == 0) {
            args.gpu = 1;
        } else if (strcmp(a, "--gpu-mem") == 0 && next) {
            args.gpu_mem_mb = (u32)atoi(next); ++i;
        } else if (strcmp(a, "--precision") == 0 && next) {
            if (strcmp(next, "f32") == 0) args.precision = 0;
            else if (strcmp(next, "f64") == 0) args.precision = 1;
            else { printf("unknown precision %s (f32 | f64)\n", next); exit(1); }
            ++i;
        } else if (strcmp(a, "--averaged") == 0) {
            args.averaged = 1;
        } else if (strcmp(a, "--csv") == 0 && next) {
            args.csv_path = next; ++i;
        } else if (strcmp(a, "--particles") == 0 && next) {
            args.particle_count = (u32)atoi(next); ++i;
        } else if (strcmp(a, "--seed") == 0 && next) {
            args.seed = (u32)atoi(next); ++i;
        } else if (strcmp(a, "--size") == 0 && next) {
            u32 w = 0, h = 0;
            if (sscanf(next, "%ux%u", &w, &h) == 2 && w && h) { args.window_width = w; args.window_height = h; }
            ++i;
        } else if (strcmp(a, "--quiet") == 0) {
            args.quiet = 1;
        } else if (strcmp(a, "--frames") == 0 && next) {
            args.windowed_frames = (u32)atoi(next); ++i;
        } else if (strcmp(a, "--screenshot") == 0 && next) {
            args.screenshot_path = next; ++i;
        } else {
            platform_print_usage(argv[0]);
            exit(a[0] == '-' && (strcmp(a, "--help") == 0 || strcmp(a, "-h") == 0) ? 0 : 1);
        }
    }
    if (args.particle_count == 0) args.particle_count = 8;
    if (args.zone_count == 0) args.zone_count = args.scene == SimScene_Sweep ? 256 : 64;
    if (args.threads == 0) args.threads = 1;
    if (args.threads > 64) args.threads = 64;
    if (args.gpu_mem_mb == 0) args.gpu_mem_mb = 512;
    return args;
}

// Standard-C logger used for PlatformAPI::log on every platform.
internal PLATFORM_LOG(platform_log_stdout) {
    va_list va;
    va_start(va, fmt);
    vfprintf(stdout, fmt, va);
    va_end(va);
}

internal void platform_print_memory_stats(SimStats *stats) {
    printf("  arenas: permanent %.1f/%.0f MB  frame %.2f/%.0f MB  scratch high-water %llu B/%.0f MB\n",
           (f64)stats->permanent_used / (1024.0 * 1024.0), (f64)stats->permanent_size / (1024.0 * 1024.0),
           (f64)stats->frame_used / (1024.0 * 1024.0), (f64)stats->frame_size / (1024.0 * 1024.0),
           (unsigned long long)stats->scratch_high_water, (f64)stats->scratch_size / (1024.0 * 1024.0));
}

// Prints the profiler table. `frames` is how many frames the totals cover;
// `cycles_per_ms` converts rdtsc ticks (measured by the caller) to time.
internal void platform_print_debug_table(DebugTable *table, u64 frames, f64 cycles_per_ms, b32 use_totals) {
#if SIM_INTERNAL
    if (!table || frames == 0) return;
    u32 order[DEBUG_MAX_RECORDS];
    u32 n = 0;
    for (u32 i = 0; i < DEBUG_MAX_RECORDS; ++i) {
        DebugRecord *r = table->records + i;
        u64 cycles = use_totals ? r->total_cycles : r->cycles;
        if (r->name && cycles) {
            // insertion sort by cycles, descending
            u32 j = n++;
            while (j > 0) {
                DebugRecord *prev = table->records + order[j - 1];
                u64 prev_cycles = use_totals ? prev->total_cycles : prev->cycles;
                if (prev_cycles >= cycles) break;
                order[j] = order[j - 1];
                --j;
            }
            order[j] = i;
        }
    }
    printf("  %-28s %10s %12s %10s\n", "timed block", "hits/frame", "cycles/frame", "ms/frame");
    for (u32 k = 0; k < n; ++k) {
        DebugRecord *r = table->records + order[k];
        u64 cycles = use_totals ? r->total_cycles : r->cycles;
        u64 hits = use_totals ? r->total_hits : r->hit_count;
        f64 cpf = (f64)cycles / (f64)frames;
        printf("  %-28s %10.2f %12.0f %10.3f   (%s:%u)\n", r->name, (f64)hits / (f64)frames, cpf,
               cycles_per_ms > 0 ? cpf / cycles_per_ms : 0.0, r->file, r->line);
    }
#else
    (void)table; (void)frames; (void)cycles_per_ms; (void)use_totals;
#endif
}

// Round-trips a small file through the PlatformAPI to prove file IO works.
internal b32 platform_file_io_selftest(PlatformAPI *api, Arena *scratch, const char *path) {
    const char payload[] = "simstack file io selftest\n";
    if (!api->write_entire_file(path, payload, sizeof(payload) - 1)) return 0;
    TempMemory temp = begin_temp(scratch);
    PlatformFileContents read = api->read_entire_file(scratch, path);
    b32 ok = read.data && read.size == sizeof(payload) - 1 && memcmp(read.data, payload, read.size) == 0;
    end_temp(temp);
    return ok;
}
