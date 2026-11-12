// Simstack Win32 platform layer: window + WGL context, input, timing, file IO,
// memory reservation, and hot reload of simstack_sim.dll. One TU.
//
// STATUS: written against the same interface as linux_simstack.cpp but NOT
// yet compiled or run on Windows (no Windows toolchain was available when
// milestone 1 was built). Expect small fixes on first build.
//
//   build\simstack.exe                 windowed
//   build\simstack.exe --headless 600  no window, fixed 1/60 s frame dt, timings on stdout

// OS headers first: base_types.h defines `internal`/`global` macros.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <GL/gl.h>

#define DEBUG_RECORD_BASE 0
#include "../sim_platform.h"
#include "platform_common.cpp"
#include "render_gl.cpp"

#define WGL_CONTEXT_MAJOR_VERSION_ARB     0x2091
#define WGL_CONTEXT_MINOR_VERSION_ARB     0x2092
#define WGL_CONTEXT_PROFILE_MASK_ARB      0x9126
#define WGL_CONTEXT_CORE_PROFILE_BIT_ARB  0x00000001
typedef HGLRC (WINAPI *wglCreateContextAttribsARB_fn)(HDC, HGLRC, const int *);
typedef BOOL (WINAPI *wglSwapIntervalEXT_fn)(int);

// ----------------------------------------------------------------------------
// Timing

global s64 g_perf_frequency;

internal PLATFORM_NOW_NS(win32_now_ns) {
    LARGE_INTEGER c;
    QueryPerformanceCounter(&c);
    s64 sec = c.QuadPart / g_perf_frequency;
    s64 rem = c.QuadPart % g_perf_frequency;
    return (u64)sec * 1000000000ull + (u64)((rem * 1000000000ll) / g_perf_frequency);
}

internal f64 win32_seconds_between(u64 start_ns, u64 end_ns) {
    return (f64)(end_ns - start_ns) * 1e-9;
}

// ----------------------------------------------------------------------------
// File IO

internal PLATFORM_READ_ENTIRE_FILE(win32_read_entire_file) {
    PlatformFileContents result = {};
    HANDLE file = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, 0, OPEN_EXISTING, 0, 0);
    if (file == INVALID_HANDLE_VALUE) return result;
    LARGE_INTEGER size_li;
    if (GetFileSizeEx(file, &size_li)) {
        u64 size = (u64)size_li.QuadPart;
        u8 *data = (u8 *)push_size_(arena, size + 1, 16, 0);   // +1: always NUL terminated
        if (data) {
            u64 total = 0;
            while (total < size) {
                DWORD chunk = (DWORD)Minimum(size - total, (u64)0x40000000);
                DWORD got = 0;
                if (!ReadFile(file, data + total, chunk, &got, 0) || got == 0) break;
                total += got;
            }
            if (total == size) {
                data[size] = 0;
                result.data = data;
                result.size = size;
            }
        }
    }
    CloseHandle(file);
    return result;
}

internal PLATFORM_WRITE_ENTIRE_FILE(win32_write_entire_file) {
    HANDLE file = CreateFileA(path, GENERIC_WRITE, 0, 0, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, 0);
    if (file == INVALID_HANDLE_VALUE) return 0;
    const u8 *p = (const u8 *)data;
    u64 total = 0;
    while (total < size) {
        DWORD chunk = (DWORD)Minimum(size - total, (u64)0x40000000);
        DWORD wrote = 0;
        if (!WriteFile(file, p + total, chunk, &wrote, 0) || wrote == 0) break;
        total += wrote;
    }
    CloseHandle(file);
    return total == size;
}

internal b32 win32_file_exists(const char *path) {
    return GetFileAttributesA(path) != INVALID_FILE_ATTRIBUTES;
}

internal u64 win32_file_mtime(const char *path) {
    WIN32_FILE_ATTRIBUTE_DATA data;
    if (!GetFileAttributesExA(path, GetFileExInfoStandard, &data)) return 0;
    ULARGE_INTEGER t;
    t.LowPart = data.ftLastWriteTime.dwLowDateTime;
    t.HighPart = data.ftLastWriteTime.dwHighDateTime;
    return t.QuadPart;
}

// ----------------------------------------------------------------------------
// Sim module hot reload. The DLL is copied to a unique temp name before
// LoadLibrary so the build can overwrite the original; build.bat holds
// lock.tmp while linking and gives each build a random PDB name so the
// debugger does not pin the old one.

struct Win32SimCode {
    HMODULE handle;
    sim_update_fn *update;
    sim_on_reload_fn *on_reload;
    u64 loaded_mtime;
    u32 load_index;
    b32 is_valid;
    char temp_path[MAX_PATH];
};

struct Win32Paths {
    char exe_dir[MAX_PATH];
    char sim_dll[MAX_PATH];
    char lock[MAX_PATH];
    char selftest[MAX_PATH];
};

internal void win32_build_paths(Win32Paths *paths) {
    DWORD len = GetModuleFileNameA(0, paths->exe_dir, sizeof(paths->exe_dir));
    paths->exe_dir[len] = 0;
    char *slash = strrchr(paths->exe_dir, '\\');
    if (slash) *slash = 0; else strcpy(paths->exe_dir, ".");
    snprintf(paths->sim_dll, sizeof(paths->sim_dll), "%s\\simstack_sim.dll", paths->exe_dir);
    snprintf(paths->lock, sizeof(paths->lock), "%s\\lock.tmp", paths->exe_dir);
    snprintf(paths->selftest, sizeof(paths->selftest), "%s\\selftest.txt", paths->exe_dir);
}

internal void win32_unload_sim_code(Win32SimCode *code) {
    if (code->handle) FreeLibrary(code->handle);
    if (code->temp_path[0]) DeleteFileA(code->temp_path);
    code->handle = 0;
    code->update = 0;
    code->on_reload = 0;
    code->is_valid = 0;
    code->temp_path[0] = 0;
}

internal Win32SimCode win32_load_sim_code(Win32Paths *paths, u32 load_index) {
    Win32SimCode code = {};
    code.load_index = load_index;
    code.loaded_mtime = win32_file_mtime(paths->sim_dll);
    if (win32_file_exists(paths->lock)) return code;   // build in progress, retry next frame

    snprintf(code.temp_path, sizeof(code.temp_path), "%s\\simstack_sim_%lu_%u.dll", paths->exe_dir,
             (unsigned long)GetCurrentProcessId(), load_index);
    if (!CopyFileA(paths->sim_dll, code.temp_path, FALSE)) {
        fprintf(stderr, "[platform] cannot copy %s (error %lu)\n", paths->sim_dll, (unsigned long)GetLastError());
        code.temp_path[0] = 0;
        return code;
    }
    code.handle = LoadLibraryA(code.temp_path);
    if (!code.handle) {
        fprintf(stderr, "[platform] LoadLibrary failed (error %lu)\n", (unsigned long)GetLastError());
        DeleteFileA(code.temp_path);
        code.temp_path[0] = 0;
        return code;
    }
    code.update = (sim_update_fn *)GetProcAddress(code.handle, SIM_UPDATE_EXPORT_NAME);
    code.on_reload = (sim_on_reload_fn *)GetProcAddress(code.handle, SIM_ON_RELOAD_EXPORT_NAME);
    code.is_valid = code.update && code.on_reload;
    if (!code.is_valid) {
        fprintf(stderr, "[platform] missing exports in %s\n", paths->sim_dll);
        win32_unload_sim_code(&code);
    }
    return code;
}

internal b32 win32_check_reload(Win32SimCode *code, Win32Paths *paths, SimMemory *memory) {
    u64 mtime = win32_file_mtime(paths->sim_dll);
    if (code->is_valid && mtime == code->loaded_mtime) return 0;
    if (mtime == 0 || win32_file_exists(paths->lock)) return 0;
    Win32SimCode fresh = win32_load_sim_code(paths, code->load_index + 1);
    if (!fresh.is_valid) return 0;
    win32_unload_sim_code(code);
    *code = fresh;
    code->on_reload(memory);
    return 1;
}

// ----------------------------------------------------------------------------
// Input

internal void win32_process_button(SimButtonState *button, b32 is_down) {
    if (button->ended_down != is_down) {
        button->ended_down = is_down;
        ++button->half_transition_count;
    }
}

internal void win32_begin_input_frame(SimInput *input) {
    for (u32 i = 0; i < SimKey_Count; ++i) input->keys[i].half_transition_count = 0;
    for (u32 i = 0; i < ArrayCount(input->mouse_buttons); ++i) input->mouse_buttons[i].half_transition_count = 0;
}

internal s32 win32_vk_to_simkey(WPARAM vk) {
    switch (vk) {
        case VK_ESCAPE: return SimKey_Escape;
        case VK_SPACE:  return SimKey_Space;
        case 'R':       return SimKey_R;
        case 'P':       return SimKey_P;
        case VK_F1:     return SimKey_F1;
        case VK_LEFT:   return SimKey_Left;
        case VK_RIGHT:  return SimKey_Right;
        case VK_UP:     return SimKey_Up;
        case VK_DOWN:   return SimKey_Down;
        default:        return -1;
    }
}

// ----------------------------------------------------------------------------
// Window + GL context

struct Win32Window {
    HWND hwnd;
    HDC dc;
    HGLRC gl_context;
    u32 width, height;
    b32 running;
};

global Win32Window *g_win32_window;

internal LRESULT CALLBACK win32_window_proc(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam) {
    switch (msg) {
        case WM_CLOSE:
        case WM_DESTROY: {
            if (g_win32_window) g_win32_window->running = 0;
            return 0;
        }
        case WM_SIZE: {
            if (g_win32_window) {
                g_win32_window->width = (u32)LOWORD(lparam);
                g_win32_window->height = (u32)HIWORD(lparam);
            }
            return 0;
        }
        case WM_PAINT: {
            PAINTSTRUCT ps;
            BeginPaint(hwnd, &ps);
            EndPaint(hwnd, &ps);
            return 0;
        }
        default: return DefWindowProcA(hwnd, msg, wparam, lparam);
    }
}

internal void *win32_gl_get_proc(const char *name) {
    void *p = (void *)wglGetProcAddress(name);
    // wglGetProcAddress returns sentinel values for failure on some drivers.
    smm sentinel = (smm)p;
    if (sentinel == 0 || sentinel == 1 || sentinel == 2 || sentinel == 3 || sentinel == -1) {
        local_persist HMODULE opengl32 = LoadLibraryA("opengl32.dll");
        p = (void *)GetProcAddress(opengl32, name);
    }
    return p;
}

internal b32 win32_create_window(Win32Window *win, u32 width, u32 height) {
    HINSTANCE instance = GetModuleHandleA(0);
    WNDCLASSA wc = {};
    wc.style = CS_OWNDC | CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc = win32_window_proc;
    wc.hInstance = instance;
    wc.hCursor = LoadCursorA(0, IDC_ARROW);
    wc.lpszClassName = "SimstackWindowClass";
    if (!RegisterClassA(&wc)) return 0;

    RECT rect = { 0, 0, (LONG)width, (LONG)height };
    AdjustWindowRect(&rect, WS_OVERLAPPEDWINDOW, FALSE);
    g_win32_window = win;
    win->hwnd = CreateWindowExA(0, wc.lpszClassName, "simstack", WS_OVERLAPPEDWINDOW | WS_VISIBLE,
                                CW_USEDEFAULT, CW_USEDEFAULT, rect.right - rect.left, rect.bottom - rect.top,
                                0, 0, instance, 0);
    if (!win->hwnd) return 0;
    win->dc = GetDC(win->hwnd);

    PIXELFORMATDESCRIPTOR pfd = {};
    pfd.nSize = sizeof(pfd);
    pfd.nVersion = 1;
    pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
    pfd.iPixelType = PFD_TYPE_RGBA;
    pfd.cColorBits = 32;
    pfd.cDepthBits = 24;
    pfd.iLayerType = PFD_MAIN_PLANE;
    int format = ChoosePixelFormat(win->dc, &pfd);
    if (!format || !SetPixelFormat(win->dc, format, &pfd)) return 0;

    // Legacy context first to reach wglCreateContextAttribsARB, then a core one.
    HGLRC legacy = wglCreateContext(win->dc);
    if (!legacy) return 0;
    wglMakeCurrent(win->dc, legacy);
    wglCreateContextAttribsARB_fn create_context =
        (wglCreateContextAttribsARB_fn)wglGetProcAddress("wglCreateContextAttribsARB");
    if (create_context) {
        int versions[][2] = { {4, 5}, {3, 3} };
        for (u32 i = 0; i < ArrayCount(versions) && !win->gl_context; ++i) {
            int attribs[] = {
                WGL_CONTEXT_MAJOR_VERSION_ARB, versions[i][0], WGL_CONTEXT_MINOR_VERSION_ARB, versions[i][1],
                WGL_CONTEXT_PROFILE_MASK_ARB, WGL_CONTEXT_CORE_PROFILE_BIT_ARB, 0
            };
            win->gl_context = create_context(win->dc, 0, attribs);
        }
    }
    if (win->gl_context) {
        wglMakeCurrent(0, 0);
        wglDeleteContext(legacy);
        wglMakeCurrent(win->dc, win->gl_context);
    } else {
        win->gl_context = legacy;
    }
    wglSwapIntervalEXT_fn swap_interval = (wglSwapIntervalEXT_fn)wglGetProcAddress("wglSwapIntervalEXT");
    if (swap_interval) swap_interval(1);

    win->width = width;
    win->height = height;
    win->running = 1;
    return gl_load_functions(win32_gl_get_proc);
}

internal void win32_process_messages(Win32Window *win, SimInput *input) {
    MSG msg;
    while (PeekMessageA(&msg, 0, 0, 0, PM_REMOVE)) {
        switch (msg.message) {
            case WM_QUIT: win->running = 0; break;
            case WM_SYSKEYDOWN:
            case WM_SYSKEYUP:
            case WM_KEYDOWN:
            case WM_KEYUP: {
                b32 was_down = (msg.lParam & (1 << 30)) != 0;
                b32 is_down = (msg.lParam & (1u << 31)) == 0;
                if (was_down != is_down) {
                    s32 key = win32_vk_to_simkey(msg.wParam);
                    if (key >= 0) win32_process_button(input->keys + key, is_down);
                }
                if (msg.message == WM_SYSKEYDOWN && msg.wParam == VK_F4) win->running = 0;
            } break;
            default:
                TranslateMessage(&msg);
                DispatchMessageA(&msg);
                break;
        }
    }
    POINT mouse;
    GetCursorPos(&mouse);
    ScreenToClient(win->hwnd, &mouse);
    input->mouse_x = (f32)mouse.x;
    input->mouse_y = (f32)mouse.y;
    win32_process_button(input->mouse_buttons + 0, (GetKeyState(VK_LBUTTON) & 0x8000) != 0);
    win32_process_button(input->mouse_buttons + 1, (GetKeyState(VK_MBUTTON) & 0x8000) != 0);
    win32_process_button(input->mouse_buttons + 2, (GetKeyState(VK_RBUTTON) & 0x8000) != 0);
    input->window_width = win->width;
    input->window_height = win->height;
}

// ----------------------------------------------------------------------------

internal f64 win32_measure_cycles_per_ms() {
    u64 t0 = win32_now_ns();
    u64 c0 = read_cpu_timer();
    while (win32_now_ns() - t0 < 20000000ull) {}   // 20 ms
    u64 t1 = win32_now_ns();
    u64 c1 = read_cpu_timer();
    return (f64)(c1 - c0) / ((f64)(t1 - t0) * 1e-6);
}

int main(int argc, char **argv) {
    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);
    g_perf_frequency = freq.QuadPart;

    PlatformArgs args = platform_parse_args(argc, argv);
    Win32Paths paths;
    win32_build_paths(&paths);

    void *block = VirtualAlloc(PLATFORM_MEMORY_BASE_ADDRESS, (SIZE_T)PLATFORM_TOTAL_SIZE,
                               MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (!block) {
        // Fixed base taken: let the OS choose.
        block = VirtualAlloc(0, (SIZE_T)PLATFORM_TOTAL_SIZE, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    }
    if (!block) {
        fprintf(stderr, "[platform] VirtualAlloc of %llu MB failed (error %lu)\n",
                (unsigned long long)(PLATFORM_TOTAL_SIZE >> 20), (unsigned long)GetLastError());
        return 1;
    }
    SimMemory memory = {};
    memory.permanent_size = PLATFORM_PERMANENT_SIZE;
    memory.permanent_base = block;
    memory.frame_size = PLATFORM_FRAME_SIZE;
    memory.frame_base = (u8 *)block + PLATFORM_PERMANENT_SIZE;
    memory.scratch_size = PLATFORM_SCRATCH_SIZE;
    memory.scratch_base = (u8 *)block + PLATFORM_PERMANENT_SIZE + PLATFORM_FRAME_SIZE;
    memory.config.particle_count = args.particle_count;
    memory.config.seed = args.seed;
    memory.platform.read_entire_file = win32_read_entire_file;
    memory.platform.write_entire_file = win32_write_entire_file;
    memory.platform.now_ns = win32_now_ns;
    memory.platform.log = platform_log_stdout;

    DebugTable debug_table = {};
#if SIM_INTERNAL
    memory.debug = &debug_table;
    g_debug_table = &debug_table;
#endif

    {
        Arena platform_scratch = arena_make(memory.scratch_base, memory.scratch_size);
        printf("[platform] file io selftest: %s\n",
               platform_file_io_selftest(&memory.platform, &platform_scratch, paths.selftest) ? "ok" : "FAILED");
        DeleteFileA(paths.selftest);
    }

    Win32SimCode sim = win32_load_sim_code(&paths, 0);
    if (!sim.is_valid) {
        fprintf(stderr, "[platform] cannot load %s\n", paths.sim_dll);
        return 1;
    }
    sim.on_reload(&memory);
    f64 cycles_per_ms = win32_measure_cycles_per_ms();
    printf("[platform] memory %llu MB at %p, sim %s, %.0f cycles/ms, simd=%d\n",
           (unsigned long long)(PLATFORM_TOTAL_SIZE >> 20), block, paths.sim_dll, cycles_per_ms, SIM_SIMD);

    SimInput input = {};
    RenderCommands render = {};
    SimStats stats = {};

    if (args.headless) {
        input.headless = 1;
        input.window_width = args.window_width;
        input.window_height = args.window_height;
        u64 t_start = win32_now_ns();
        u64 min_ns = U64_MAX, max_ns = 0, sum_ns = 0;
        u32 reloads = 0;
        for (u32 frame = 0; frame < args.headless_frames; ++frame) {
            reloads += (u32)win32_check_reload(&sim, &paths, &memory);
            win32_begin_input_frame(&input);
            input.dt_frame = PLATFORM_HEADLESS_DT;
            input.compute_state_hash = (frame + 1 == args.headless_frames);
            u64 f0 = win32_now_ns();
            if (sim.is_valid) sim.update(&memory, &input, &render, &stats);
            u64 f1 = win32_now_ns();
            u64 dt = f1 - f0;
            min_ns = Minimum(min_ns, dt); max_ns = Maximum(max_ns, dt); sum_ns += dt;
            debug_table_end_frame(&debug_table);
        }
        f64 wall = win32_seconds_between(t_start, win32_now_ns());
        printf("[headless] %u frames in %.3f s (%.3f ms/frame avg, %.3f min, %.3f max), %u reloads\n",
               args.headless_frames, wall, (f64)sum_ns * 1e-6 / (f64)args.headless_frames,
               (f64)min_ns * 1e-6, (f64)max_ns * 1e-6, reloads);
        printf("[headless] %llu fixed steps (dt %.4f s), sim time %.3f s, %u particles, "
               "%.1f M particle-steps/s\n",
               (unsigned long long)stats.step_count, stats.dt_fixed, stats.sim_time, stats.particle_count,
               (f64)stats.step_count * (f64)stats.particle_count / ((f64)sum_ns * 1e-9) * 1e-6);
        printf("[headless] state hash %016llx\n", (unsigned long long)stats.state_hash);
        platform_print_memory_stats(&stats);
        platform_print_debug_table(&debug_table, args.headless_frames, cycles_per_ms, 1);
        win32_unload_sim_code(&sim);
        return 0;
    }

    Win32Window win = {};
    if (!win32_create_window(&win, args.window_width, args.window_height)) {
        fprintf(stderr, "[platform] cannot create window / GL context\n");
        return 1;
    }
    GLRenderer renderer = {};
    if (!gl_renderer_init(&renderer)) return 1;

    u64 last_ns = win32_now_ns();
    u64 window_start_ns = last_ns;
    u32 frames_in_window = 0;
    u64 frame_index = 0;
    while (win.running) {
        win32_check_reload(&sim, &paths, &memory);

        win32_begin_input_frame(&input);
        win32_process_messages(&win, &input);
        if (was_pressed(input.keys[SimKey_Escape])) win.running = 0;

        u64 now_ns = win32_now_ns();
        input.dt_frame = Clamp((f32)win32_seconds_between(last_ns, now_ns), 0.0f, PLATFORM_MAX_FRAME_DT);
        last_ns = now_ns;

        if (sim.is_valid) sim.update(&memory, &input, &render, &stats);
        {
            TIMED_BLOCK("gl_render_frame");
            gl_render_frame(&renderer, &render, win.width, win.height);
        }
        ++frames_in_window;
        ++frame_index;
        if (args.windowed_frames && frame_index >= args.windowed_frames) win.running = 0;
        if (!win.running && args.screenshot_path) {   // back buffer, before the swap
            Arena platform_scratch = arena_make(memory.scratch_base, memory.scratch_size);
            printf("[platform] screenshot %s: %s\n", args.screenshot_path,
                   gl_write_screenshot_ppm(&memory.platform, &platform_scratch, args.screenshot_path, win.width, win.height) ? "ok" : "FAILED");
        }
        {
            TIMED_BLOCK("SwapBuffers");
            SwapBuffers(win.dc);
        }

        if (frames_in_window == PLATFORM_TIMING_PRINT_INTERVAL) {
            u64 t = win32_now_ns();
            f64 ms_per_frame = win32_seconds_between(window_start_ns, t) * 1000.0 / (f64)frames_in_window;
            char title[256];
            snprintf(title, sizeof(title), "simstack  %.2f ms/frame  step %llu  %u particles%s%s", ms_per_frame,
                     (unsigned long long)stats.step_count, stats.particle_count,
                     stats.paused ? "  [paused]" : "", stats.reload_count ? "  [reloaded]" : "");
            SetWindowTextA(win.hwnd, title);
            if (!args.quiet) {
                printf("[frame %llu] %.2f ms/frame, %u steps/frame, sim t=%.2f s, reloads %u\n",
                       (unsigned long long)frame_index, ms_per_frame, stats.steps_this_frame, stats.sim_time,
                       stats.reload_count);
                platform_print_memory_stats(&stats);
                platform_print_debug_table(&debug_table, frames_in_window, cycles_per_ms, 0);
            }
            frames_in_window = 0;
            window_start_ns = t;
            debug_table_end_frame(&debug_table);
        }
    }

    wglMakeCurrent(0, 0);
    wglDeleteContext(win.gl_context);
    ReleaseDC(win.hwnd, win.dc);
    DestroyWindow(win.hwnd);
    win32_unload_sim_code(&sim);
    return 0;
}
