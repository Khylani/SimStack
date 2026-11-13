// Simstack Linux platform layer: raw Xlib + GLX window, input, timing, file
// IO, memory reservation, and hot reload of simstack_sim.so. One TU.
//
//   ./build/simstack                 windowed
//   ./build/simstack --headless 600  no window, fixed 1/60 s frame dt, timings on stdout

// OS headers first: base_types.h defines `internal`/`global` macros that
// collide with field names in X11 headers (XKBstr.h has a member `internal`).
#include <X11/Xlib.h>
#include <X11/XKBlib.h>
#include <X11/keysym.h>
#define GL_GLEXT_LEGACY
#define GLX_GLXEXT_LEGACY
#include <GL/gl.h>
#include <GL/glx.h>
#include <dlfcn.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <errno.h>
#include <limits.h>
#include <pthread.h>
#include <semaphore.h>
#include <sched.h>

#define DEBUG_RECORD_BASE 0
#include "../sim_platform.h"
#include "platform_common.cpp"
#include "render_gl.cpp"
#include "linux_work_queue.cpp"
#include "linux_gpu.cpp"

#ifndef GLX_CONTEXT_MAJOR_VERSION_ARB
#define GLX_CONTEXT_MAJOR_VERSION_ARB     0x2091
#define GLX_CONTEXT_MINOR_VERSION_ARB     0x2092
#define GLX_CONTEXT_PROFILE_MASK_ARB      0x9126
#define GLX_CONTEXT_CORE_PROFILE_BIT_ARB  0x00000001
#endif
typedef GLXContext (*glXCreateContextAttribsARB_fn)(Display *, GLXFBConfig, GLXContext, Bool, const int *);
typedef void (*glXSwapIntervalEXT_fn)(Display *, GLXDrawable, int);

// ----------------------------------------------------------------------------
// Timing

internal PLATFORM_NOW_NS(linux_now_ns) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (u64)ts.tv_sec * 1000000000ull + (u64)ts.tv_nsec;
}

internal f64 linux_seconds_between(u64 start_ns, u64 end_ns) {
    return (f64)(end_ns - start_ns) * 1e-9;
}

// ----------------------------------------------------------------------------
// File IO

internal PLATFORM_READ_ENTIRE_FILE(linux_read_entire_file) {
    PlatformFileContents result = {};
    int fd = open(path, O_RDONLY);
    if (fd < 0) return result;
    struct stat st;
    if (fstat(fd, &st) == 0 && st.st_size >= 0) {
        u64 size = (u64)st.st_size;
        u8 *data = (u8 *)push_size_(arena, size + 1, 16, 0);   // +1: always NUL terminated
        if (data) {
            u64 total = 0;
            while (total < size) {
                ssize_t n = read(fd, data + total, size - total);
                if (n <= 0) break;
                total += (u64)n;
            }
            if (total == size) {
                data[size] = 0;
                result.data = data;
                result.size = size;
            }
        }
    }
    close(fd);
    return result;
}

internal PLATFORM_WRITE_ENTIRE_FILE(linux_write_entire_file) {
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return 0;
    const u8 *p = (const u8 *)data;
    u64 total = 0;
    while (total < size) {
        ssize_t n = write(fd, p + total, size - total);
        if (n <= 0) break;
        total += (u64)n;
    }
    close(fd);
    return total == size;
}

internal b32 linux_file_exists(const char *path) {
    struct stat st;
    return stat(path, &st) == 0;
}

internal u64 linux_file_mtime_ns(const char *path) {
    struct stat st;
    if (stat(path, &st) != 0) return 0;
    return (u64)st.st_mtim.tv_sec * 1000000000ull + (u64)st.st_mtim.tv_nsec;
}

internal b32 linux_copy_file(const char *src, const char *dst) {
    int in = open(src, O_RDONLY);
    if (in < 0) return 0;
    int out = open(dst, O_WRONLY | O_CREAT | O_TRUNC, 0755);
    if (out < 0) { close(in); return 0; }
    char buf[65536];
    b32 ok = 1;
    for (;;) {
        ssize_t n = read(in, buf, sizeof(buf));
        if (n < 0) { ok = 0; break; }
        if (n == 0) break;
        ssize_t off = 0;
        while (off < n) {
            ssize_t w = write(out, buf + off, (size_t)(n - off));
            if (w <= 0) { ok = 0; break; }
            off += w;
        }
        if (!ok) break;
    }
    close(in);
    close(out);
    return ok;
}

// ----------------------------------------------------------------------------
// Sim module hot reload. The .so is copied to a unique temp name before
// dlopen so the build can overwrite the original at any time; build.sh holds
// lock.tmp while linking and we wait until it disappears.

struct LinuxSimCode {
    void *handle;
    sim_update_fn *update;
    sim_on_reload_fn *on_reload;
    sim_before_unload_fn *before_unload;   // optional
    u64 loaded_mtime_ns;
    u32 load_index;
    b32 is_valid;
};

struct LinuxPaths {
    char exe_dir[PATH_MAX];
    char sim_so[PATH_MAX];
    char lock[PATH_MAX];
    char selftest[PATH_MAX];
    char gpu_so[PATH_MAX];
};

internal void linux_build_paths(LinuxPaths *paths) {
    ssize_t len = readlink("/proc/self/exe", paths->exe_dir, sizeof(paths->exe_dir) - 1);
    if (len <= 0) { strcpy(paths->exe_dir, "."); len = 1; }
    paths->exe_dir[len] = 0;
    char *slash = strrchr(paths->exe_dir, '/');
    if (slash) *slash = 0; else strcpy(paths->exe_dir, ".");
    snprintf(paths->sim_so, sizeof(paths->sim_so), "%s/simstack_sim.so", paths->exe_dir);
    snprintf(paths->lock, sizeof(paths->lock), "%s/lock.tmp", paths->exe_dir);
    snprintf(paths->selftest, sizeof(paths->selftest), "%s/selftest.txt", paths->exe_dir);
    snprintf(paths->gpu_so, sizeof(paths->gpu_so), "%s/simstack_gpu.so", paths->exe_dir);
}

internal void linux_unload_sim_code(LinuxSimCode *code) {
    if (code->handle) dlclose(code->handle);
    code->handle = 0;
    code->update = 0;
    code->on_reload = 0;
    code->is_valid = 0;
}

internal LinuxSimCode linux_load_sim_code(LinuxPaths *paths, u32 load_index) {
    LinuxSimCode code = {};
    code.load_index = load_index;
    code.loaded_mtime_ns = linux_file_mtime_ns(paths->sim_so);
    if (linux_file_exists(paths->lock)) return code;   // build in progress, retry next frame

    char temp_path[PATH_MAX];
    snprintf(temp_path, sizeof(temp_path), "%s/simstack_sim_%d_%u.so", paths->exe_dir, (int)getpid(), load_index);
    if (!linux_copy_file(paths->sim_so, temp_path)) {
        fprintf(stderr, "[platform] cannot copy %s: %s\n", paths->sim_so, strerror(errno));
        return code;
    }
    code.handle = dlopen(temp_path, RTLD_NOW | RTLD_LOCAL);
    unlink(temp_path);   // mapping stays alive; no litter in build/
    if (!code.handle) {
        fprintf(stderr, "[platform] dlopen failed: %s\n", dlerror());
        return code;
    }
    code.update = (sim_update_fn *)dlsym(code.handle, SIM_UPDATE_EXPORT_NAME);
    code.on_reload = (sim_on_reload_fn *)dlsym(code.handle, SIM_ON_RELOAD_EXPORT_NAME);
    code.before_unload = (sim_before_unload_fn *)dlsym(code.handle, SIM_BEFORE_UNLOAD_EXPORT_NAME);
    code.is_valid = code.update && code.on_reload;
    if (!code.is_valid) {
        fprintf(stderr, "[platform] missing exports in %s\n", paths->sim_so);
        linux_unload_sim_code(&code);
    }
    return code;
}

// Returns 1 if the code was (re)loaded this call.
internal b32 linux_check_reload(LinuxSimCode *code, LinuxPaths *paths, SimMemory *memory) {
    u64 mtime = linux_file_mtime_ns(paths->sim_so);
    if (code->is_valid && mtime == code->loaded_mtime_ns) return 0;
    if (mtime == 0 || linux_file_exists(paths->lock)) return 0;
    LinuxSimCode fresh = linux_load_sim_code(paths, code->load_index + 1);
    if (!fresh.is_valid) return 0;
    if (memory->background_queue) linux_complete_all_work(memory->background_queue);   // no job may run across the swap
    if (code->is_valid && code->before_unload && memory->is_initialized) code->before_unload(memory);
    linux_unload_sim_code(code);
    *code = fresh;
    code->on_reload(memory);
    return 1;
}

// ----------------------------------------------------------------------------
// Input

internal void linux_process_button(SimButtonState *button, b32 is_down) {
    if (button->ended_down != is_down) {
        button->ended_down = is_down;
        ++button->half_transition_count;
    }
}

internal void linux_begin_input_frame(SimInput *input) {
    for (u32 i = 0; i < SimKey_Count; ++i) input->keys[i].half_transition_count = 0;
    for (u32 i = 0; i < ArrayCount(input->mouse_buttons); ++i) input->mouse_buttons[i].half_transition_count = 0;
    input->mouse_wheel = 0.0f;
    input->text[0] = 0;
}

internal s32 linux_keysym_to_simkey(KeySym sym) {
    switch (sym) {
        case XK_Escape: return SimKey_Escape;
        case XK_space:  return SimKey_Space;
        case XK_r: case XK_R: return SimKey_R;
        case XK_p: case XK_P: return SimKey_P;
        case XK_F1:     return SimKey_F1;
        case XK_Left:   return SimKey_Left;
        case XK_Right:  return SimKey_Right;
        case XK_Up:     return SimKey_Up;
        case XK_Down:   return SimKey_Down;
        case XK_Tab: case XK_ISO_Left_Tab: return SimKey_Tab;
        case XK_Return: case XK_KP_Enter: return SimKey_Enter;
        case XK_BackSpace: return SimKey_Backspace;
        case XK_Delete: return SimKey_Delete;
        case XK_Home:   return SimKey_Home;
        case XK_End:    return SimKey_End;
        case XK_Page_Up: return SimKey_PageUp;
        case XK_Page_Down: return SimKey_PageDown;
        case XK_Control_L: case XK_Control_R: return SimKey_Ctrl;
        case XK_Shift_L: case XK_Shift_R: return SimKey_Shift;
        case XK_Alt_L: case XK_Alt_R: return SimKey_Alt;
        case XK_a: case XK_A: return SimKey_A;
        case XK_c: case XK_C: return SimKey_C;
        case XK_v: case XK_V: return SimKey_V;
        case XK_x: case XK_X: return SimKey_X;
        case XK_y: case XK_Y: return SimKey_Y;
        case XK_z: case XK_Z: return SimKey_Z;
        case XK_F2:     return SimKey_F2;
        case XK_f: case XK_F: return SimKey_F;
        case XK_comma:  return SimKey_Comma;
        case XK_period: return SimKey_Period;
        case XK_q: case XK_Q: return SimKey_Q;
        default:        return -1;
    }
}

// ----------------------------------------------------------------------------
// Window + GL context

struct LinuxWindow {
    Display *display;
    Window window;
    GLXContext gl_context;
    Atom wm_delete_window;
    u32 width, height;
    b32 running;
};

global b32 g_x_error_occurred;
internal int linux_x_error_handler(Display *, XErrorEvent *) { g_x_error_occurred = 1; return 0; }

internal void *linux_gl_get_proc(const char *name) {
    return (void *)glXGetProcAddressARB((const GLubyte *)name);
}

internal b32 linux_create_window(LinuxWindow *win, u32 width, u32 height) {
    win->display = XOpenDisplay(0);
    if (!win->display) {
        fprintf(stderr, "[platform] cannot open X display (set DISPLAY or use --headless)\n");
        return 0;
    }
    int screen = DefaultScreen(win->display);
    int fb_attribs[] = {
        GLX_X_RENDERABLE, True, GLX_DRAWABLE_TYPE, GLX_WINDOW_BIT, GLX_RENDER_TYPE, GLX_RGBA_BIT,
        GLX_X_VISUAL_TYPE, GLX_TRUE_COLOR, GLX_RED_SIZE, 8, GLX_GREEN_SIZE, 8, GLX_BLUE_SIZE, 8,
        GLX_ALPHA_SIZE, 8, GLX_DEPTH_SIZE, 24, GLX_DOUBLEBUFFER, True, None
    };
    int config_count = 0;
    GLXFBConfig *configs = glXChooseFBConfig(win->display, screen, fb_attribs, &config_count);
    if (!configs || config_count == 0) {
        fprintf(stderr, "[platform] no matching GLX framebuffer config\n");
        return 0;
    }
    GLXFBConfig fb_config = configs[0];
    XFree(configs);
    XVisualInfo *visual = glXGetVisualFromFBConfig(win->display, fb_config);

    Window root = RootWindow(win->display, screen);
    XSetWindowAttributes swa = {};
    swa.colormap = XCreateColormap(win->display, root, visual->visual, AllocNone);
    swa.event_mask = KeyPressMask | KeyReleaseMask | ButtonPressMask | ButtonReleaseMask |
                     PointerMotionMask | StructureNotifyMask | ExposureMask | EnterWindowMask | LeaveWindowMask;
    win->window = XCreateWindow(win->display, root, 0, 0, width, height, 0, visual->depth, InputOutput,
                                visual->visual, CWColormap | CWEventMask, &swa);
    XFree(visual);
    XStoreName(win->display, win->window, "simstack");
    win->wm_delete_window = XInternAtom(win->display, "WM_DELETE_WINDOW", False);
    XSetWMProtocols(win->display, win->window, &win->wm_delete_window, 1);
    XkbSetDetectableAutoRepeat(win->display, True, 0);   // no synthetic releases on key repeat
    XMapWindow(win->display, win->window);

    glXCreateContextAttribsARB_fn create_context =
        (glXCreateContextAttribsARB_fn)glXGetProcAddressARB((const GLubyte *)"glXCreateContextAttribsARB");
    int (*old_handler)(Display *, XErrorEvent *) = XSetErrorHandler(linux_x_error_handler);
    if (create_context) {
        int versions[][2] = { {4, 5}, {3, 3} };
        for (u32 i = 0; i < ArrayCount(versions) && !win->gl_context; ++i) {
            int ctx_attribs[] = {
                GLX_CONTEXT_MAJOR_VERSION_ARB, versions[i][0], GLX_CONTEXT_MINOR_VERSION_ARB, versions[i][1],
                GLX_CONTEXT_PROFILE_MASK_ARB, GLX_CONTEXT_CORE_PROFILE_BIT_ARB, None
            };
            g_x_error_occurred = 0;
            win->gl_context = create_context(win->display, fb_config, 0, True, ctx_attribs);
            XSync(win->display, False);
            if (g_x_error_occurred) win->gl_context = 0;
        }
    }
    if (!win->gl_context) {
        win->gl_context = glXCreateNewContext(win->display, fb_config, GLX_RGBA_TYPE, 0, True);
    }
    XSync(win->display, False);
    XSetErrorHandler(old_handler);
    if (!win->gl_context) {
        fprintf(stderr, "[platform] cannot create GL context\n");
        return 0;
    }
    glXMakeCurrent(win->display, win->window, win->gl_context);
    glXSwapIntervalEXT_fn swap_interval = (glXSwapIntervalEXT_fn)glXGetProcAddressARB((const GLubyte *)"glXSwapIntervalEXT");
    if (swap_interval) swap_interval(win->display, win->window, 1);

    win->width = width;
    win->height = height;
    win->running = 1;
    return gl_load_functions(linux_gl_get_proc);
}

internal void linux_process_events(LinuxWindow *win, SimInput *input) {
    while (XPending(win->display)) {
        XEvent ev;
        XNextEvent(win->display, &ev);
        switch (ev.type) {
            case KeyPress:
            case KeyRelease: {
                KeySym sym = XkbKeycodeToKeysym(win->display, (KeyCode)ev.xkey.keycode, 0, 0);
                s32 key = linux_keysym_to_simkey(sym);
                if (key >= 0) linux_process_button(input->keys + key, ev.type == KeyPress);
                if (ev.type == KeyPress && !(ev.xkey.state & ControlMask)) {
                    char buf[16];
                    KeySym ignored;
                    int n = XLookupString(&ev.xkey, buf, sizeof(buf), &ignored, 0);
                    u32 len = (u32)strlen(input->text);
                    for (int i = 0; i < n; ++i) {
                        u8 c = (u8)buf[i];
                        if (c >= 32 && c != 127 && len + 1 < sizeof(input->text)) input->text[len++] = (char)c;
                    }
                    input->text[len] = 0;
                }
            } break;
            case ButtonPress:
            case ButtonRelease: {
                u32 b = ev.xbutton.button;
                // X11 buttons: 1 left, 2 middle, 3 right, 4/5 wheel. SimInput: 0 left, 1 right, 2 middle.
                if (b == 1) linux_process_button(input->mouse_buttons + 0, ev.type == ButtonPress);
                if (b == 3) linux_process_button(input->mouse_buttons + 1, ev.type == ButtonPress);
                if (b == 2) linux_process_button(input->mouse_buttons + 2, ev.type == ButtonPress);
                if (ev.type == ButtonPress && b == 4) input->mouse_wheel += 1.0f;
                if (ev.type == ButtonPress && b == 5) input->mouse_wheel -= 1.0f;
            } break;
            case MotionNotify: {
                input->mouse_x = (f32)ev.xmotion.x;
                input->mouse_y = (f32)ev.xmotion.y;
                input->mouse_inside = 1;
            } break;
            case EnterNotify: input->mouse_inside = 1; break;
            case LeaveNotify: input->mouse_inside = 0; break;
            case ConfigureNotify: {
                win->width = (u32)ev.xconfigure.width;
                win->height = (u32)ev.xconfigure.height;
            } break;
            case ClientMessage: {
                if ((Atom)ev.xclient.data.l[0] == win->wm_delete_window) win->running = 0;
            } break;
            default: break;
        }
    }
    input->window_width = win->width;
    input->window_height = win->height;
}

// ----------------------------------------------------------------------------

internal f64 linux_measure_cycles_per_ms() {
    u64 t0 = linux_now_ns();
    u64 c0 = read_cpu_timer();
    while (linux_now_ns() - t0 < 20000000ull) {}   // 20 ms
    u64 t1 = linux_now_ns();
    u64 c1 = read_cpu_timer();
    return (f64)(c1 - c0) / ((f64)(t1 - t0) * 1e-6);
}

int main(int argc, char **argv) {
    PlatformArgs args = platform_parse_args(argc, argv);
    LinuxPaths paths;
    linux_build_paths(&paths);

    // One reservation for every block; committed lazily by the kernel.
    void *block = mmap(PLATFORM_MEMORY_BASE_ADDRESS, PLATFORM_TOTAL_SIZE, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (block == MAP_FAILED) {
        fprintf(stderr, "[platform] mmap of %llu MB failed: %s\n",
                (unsigned long long)(PLATFORM_TOTAL_SIZE >> 20), strerror(errno));
        return 1;
    }
    SimMemory memory = {};
    memory.permanent_size = PLATFORM_PERMANENT_SIZE;
    memory.permanent_base = block;
    memory.frame_size = PLATFORM_FRAME_SIZE;
    memory.frame_base = (u8 *)block + PLATFORM_PERMANENT_SIZE;
    memory.scratch_size = PLATFORM_SCRATCH_SIZE;
    memory.scratch_base = (u8 *)block + PLATFORM_PERMANENT_SIZE + PLATFORM_FRAME_SIZE;
    memory.ui_size = PLATFORM_UI_SIZE;
    memory.ui_base = (u8 *)memory.scratch_base + PLATFORM_SCRATCH_SIZE;
    if (!args.headless || args.timeline_check) {
        // Timeline keyframes: a separate lazy reservation right after the main block
        // (fixed address, so keyframes of pointer-holding scene state stay valid).
        void *want = PLATFORM_MEMORY_BASE_ADDRESS ? (u8 *)block + PLATFORM_TOTAL_SIZE : 0;
        void *tl = mmap(want, PLATFORM_TIMELINE_SIZE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (tl != MAP_FAILED) { memory.timeline_base = tl; memory.timeline_size = PLATFORM_TIMELINE_SIZE; }
    }
    memory.config.scene = args.scene;
    memory.config.particle_count = args.particle_count;
    memory.config.zone_count = args.zone_count;
    memory.config.backend = args.hybrid ? SimBackend_Hybrid : args.gpu ? SimBackend_Gpu : SimBackend_Cpu;
    memory.config.averaged = args.averaged;
    memory.config.precision = args.precision;
    memory.config.chem_steps = args.chem_steps;
    memory.config.speed = args.speed;
    memory.config.channel_length_m = args.channel_length_m;
    memory.config.plants_per_channel = args.plants_per_channel;
    memory.config.view_layer = args.view_layer;
    memory.config.has_pick = args.has_pick;
    memory.config.bake_s = args.bake_s;
    memory.config.pick_x = args.pick_x;
    memory.config.pick_y = args.pick_y;
    memory.config.preset = args.preset;
    memory.config.csv_path = args.csv_path;
    memory.config.seed = args.seed;
    memory.platform.read_entire_file = linux_read_entire_file;
    memory.platform.write_entire_file = linux_write_entire_file;
    memory.platform.now_ns = linux_now_ns;
    memory.platform.log = platform_log_stdout;
    memory.platform.add_work_entry = linux_add_work_entry;
    memory.platform.complete_all_work = linux_complete_all_work;

    static PlatformWorkQueue work_queue, background_queue;
    linux_make_work_queue(&work_queue, args.threads - 1, 2, 2);
    linux_make_work_queue(&background_queue, 1, 1, 1);
    memory.work_queue = &work_queue;
    memory.worker_count = args.threads;
    memory.background_queue = &background_queue;

    if (args.gpu) {
        if (linux_load_gpu_module(paths.gpu_so, Megabytes((u64)args.gpu_mem_mb), &memory.gpu)) {
            GpuInfo *gi = &memory.gpu.info;
            printf("[gpu] %s (sm_%u%u, %u SMs), %.0f of %.0f MB free, arena %u MB\n", gi->name, gi->sm_major, gi->sm_minor,
                   gi->multiprocessors, (f64)gi->free_bytes / 1048576.0, (f64)gi->total_bytes / 1048576.0, args.gpu_mem_mb);
        } else {
            printf("[gpu] unavailable, running on the CPU\n");
        }
    }

    DebugTable debug_table = {};
#if SIM_INTERNAL
    memory.debug = &debug_table;
    g_debug_table = &debug_table;
#endif

    {
        Arena platform_scratch = arena_make(memory.scratch_base, memory.scratch_size);
        printf("[platform] file io selftest: %s\n",
               platform_file_io_selftest(&memory.platform, &platform_scratch, paths.selftest) ? "ok" : "FAILED");
        unlink(paths.selftest);
    }

    LinuxSimCode sim = linux_load_sim_code(&paths, 0);
    if (!sim.is_valid) {
        fprintf(stderr, "[platform] cannot load %s\n", paths.sim_so);
        return 1;
    }
    sim.on_reload(&memory);
    f64 cycles_per_ms = linux_measure_cycles_per_ms();
    debug_table.cycles_per_ms = cycles_per_ms;
    printf("[platform] memory %llu MB at %p, sim %s, %.0f cycles/ms, simd=%d, threads %u\n",
           (unsigned long long)(PLATFORM_TOTAL_SIZE >> 20), block, paths.sim_so, cycles_per_ms, SIM_SIMD, args.threads);

    SimInput input = {};
    RenderCommands render = {};
    SimStats stats = {};

    if (args.headless) {
        input.headless = 1;
        input.window_width = args.window_width;
        input.window_height = args.window_height;
        u64 t_start = linux_now_ns();
        u64 min_ns = U64_MAX, max_ns = 0, sum_ns = 0;
        u32 reloads = 0;
        u32 half = args.headless_frames / 2;
        for (u32 frame = 0; frame < args.headless_frames; ++frame) {
            reloads += (u32)linux_check_reload(&sim, &paths, &memory);
            linux_begin_input_frame(&input);
            input.dt_frame = PLATFORM_HEADLESS_DT;
            input.compute_state_hash = (frame + 1 == args.headless_frames);
            input.timeline_op = args.timeline_check && frame == half ? 1 : 0;
            u64 f0 = linux_now_ns();
            if (sim.is_valid) sim.update(&memory, &input, &render, &stats);
            u64 f1 = linux_now_ns();
            u64 dt = f1 - f0;
            min_ns = Minimum(min_ns, dt); max_ns = Maximum(max_ns, dt); sum_ns += dt;
            debug_table_end_frame(&debug_table);
        }
        f64 wall = linux_seconds_between(t_start, linux_now_ns());
        printf("[headless] %u frames in %.3f s (%.3f ms/frame avg, %.3f min, %.3f max), %u reloads\n",
               args.headless_frames, wall, (f64)sum_ns * 1e-6 / (f64)args.headless_frames,
               (f64)min_ns * 1e-6, (f64)max_ns * 1e-6, reloads);
        if (args.scene == SimScene_Nft) {
            printf("[headless] %.2f simulated days in %.3f s wall (%.0fx real time)\n", stats.sim_time / 86400.0,
                   (f64)sum_ns * 1e-9, stats.sim_time / ((f64)sum_ns * 1e-9));
            printf("[headless] %s\n", stats.summary);
        } else if (args.scene == SimScene_Particles) {
            printf("[headless] %llu fixed steps (dt %.4f s), sim time %.3f s, %u particles, "
                   "%.1f M particle-steps/s\n",
                   (unsigned long long)stats.step_count, stats.dt_fixed, stats.sim_time, stats.particle_count,
                   (f64)stats.step_count * (f64)stats.particle_count / ((f64)sum_ns * 1e-9) * 1e-6);
        } else {
            printf("[headless] %llu master steps (dt %.4f s), sim time %.3f s, %u zones, %.2f sim-s per wall-s\n",
                   (unsigned long long)stats.step_count, stats.dt_fixed, stats.sim_time, stats.particle_count,
                   stats.sim_time / ((f64)sum_ns * 1e-9));
            printf("[headless] %s\n", stats.summary);
        }
        printf("[headless] state hash %016llx\n", (unsigned long long)stats.state_hash);
        if (args.timeline_check) {
            // restore the half-way keyframe and run the second half again: must be identical
            u64 first = stats.state_hash;
            for (u32 frame = half; frame < args.headless_frames; ++frame) {
                linux_begin_input_frame(&input);
                input.dt_frame = PLATFORM_HEADLESS_DT;
                input.compute_state_hash = (frame + 1 == args.headless_frames);
                input.timeline_op = frame == half ? 2 : 0;
                if (sim.is_valid) sim.update(&memory, &input, &render, &stats);
            }
            b32 ok = stats.state_hash == first;
            printf("[timeline] restored frame %u and re-ran: state hash %016llx %s\n", half,
                   (unsigned long long)stats.state_hash, ok ? "(identical)" : "MISMATCH");
            if (!ok) return 1;
        }
        platform_print_memory_stats(&stats);
        platform_print_debug_table(&debug_table, args.headless_frames, cycles_per_ms, 1);
        return 0;
    }

    LinuxWindow win = {};
    if (!linux_create_window(&win, args.window_width, args.window_height)) return 1;
    GLRenderer renderer = {};
    if (!gl_renderer_init(&renderer)) return 1;

    u64 last_ns = linux_now_ns();
    u64 window_start_ns = last_ns;
    u32 frames_in_window = 0;
    u64 frame_index = 0;
    while (win.running) {
        if (linux_check_reload(&sim, &paths, &memory)) gl_drop_ui_textures(&renderer);

        linux_begin_input_frame(&input);
        linux_process_events(&win, &input);

        u64 now_ns = linux_now_ns();
        input.dt_frame = Clamp((f32)linux_seconds_between(last_ns, now_ns), 0.0f, PLATFORM_MAX_FRAME_DT);
        last_ns = now_ns;

        u64 work_start_ns = linux_now_ns();
        if (sim.is_valid) sim.update(&memory, &input, &render, &stats);
        if (render.request_quit) win.running = 0;
        b32 shot_now = render.screenshot_path != 0;
        if (!sim.is_valid && was_pressed(input.keys[SimKey_Escape])) win.running = 0;   // no sim module: Escape still exits
        {
            TIMED_BLOCK("gl_render_frame");
            gl_render_frame(&renderer, &render, win.width, win.height);
        }
        if (shot_now) {   // File > Save screenshot: this frame's back buffer, before the swap
            Arena platform_scratch = arena_make(memory.scratch_base, memory.scratch_size);
            printf("[platform] screenshot %s: %s\n", render.screenshot_path,
                   gl_write_screenshot_ppm(&memory.platform, &platform_scratch, render.screenshot_path, win.width, win.height) ? "ok" : "failed");
        }
        debug_table_end_frame(&debug_table);
        ++frames_in_window;
        ++frame_index;
        if (args.windowed_frames && frame_index >= args.windowed_frames) win.running = 0;
        if (!win.running && args.screenshot_path) {   // back buffer, before the swap
            Arena platform_scratch = arena_make(memory.scratch_base, memory.scratch_size);
            printf("[platform] screenshot %s: %s\n", args.screenshot_path,
                   gl_write_screenshot_ppm(&memory.platform, &platform_scratch, args.screenshot_path, win.width, win.height) ? "ok" : "FAILED");
        }
        input.last_work_ms = (f32)((f64)(linux_now_ns() - work_start_ns) * 1e-6);
        {
            TIMED_BLOCK("glXSwapBuffers");
            glXSwapBuffers(win.display, win.window);
        }

        if (frames_in_window == PLATFORM_TIMING_PRINT_INTERVAL) {
            u64 t = linux_now_ns();
            f64 ms_per_frame = linux_seconds_between(window_start_ns, t) * 1000.0 / (f64)frames_in_window;
            char title[256];
            snprintf(title, sizeof(title), "simstack  %.2f ms/frame  step %llu  t %.2f s  %u %s%s%s", ms_per_frame,
                     (unsigned long long)stats.step_count, stats.sim_time, stats.particle_count,
                     args.scene == SimScene_Particles ? "particles" : "zones",
                     stats.paused ? "  [paused]" : "", stats.reload_count ? "  [reloaded]" : "");
            XStoreName(win.display, win.window, title);
            if (!args.quiet) {
                printf("[frame %llu] %.2f ms/frame, %u steps/frame, sim t=%.2f s, reloads %u\n",
                       (unsigned long long)frame_index, ms_per_frame, stats.steps_this_frame, stats.sim_time,
                       stats.reload_count);
                if (stats.summary[0]) printf("  %s\n", stats.summary);
                platform_print_memory_stats(&stats);
                platform_print_debug_table(&debug_table, frames_in_window, cycles_per_ms, 0);
            }
            frames_in_window = 0;
            window_start_ns = t;
            debug_table_reset_window(&debug_table);
        }
    }

    glXMakeCurrent(win.display, None, 0);
    glXDestroyContext(win.display, win.gl_context);
    XDestroyWindow(win.display, win.window);
    XCloseDisplay(win.display);
    return 0;
}
