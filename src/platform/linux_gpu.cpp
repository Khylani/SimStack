// Loads the optional GPU module (build/simstack_gpu.so) and fills a GpuAPI
// table. Shared by the Linux platform layer and the test executable.
// Needs <dlfcn.h>, <stdio.h> and gpu_api.h already included.

// Returns 1 when the module loaded and gpu_init succeeded. On failure
// api->available stays 0 and a reason is printed; the caller runs CPU-only.
internal b32 linux_load_gpu_module(const char *path, u64 arena_bytes, GpuAPI *api) {
    memset(api, 0, sizeof(*api));
    void *handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        fprintf(stderr, "[gpu] cannot load %s: %s (build with ./build.sh --cuda)\n", path, dlerror());
        return 0;
    }
    b32 complete = 1;
#define GPU_RESOLVE(fn) \
    api->fn = (fn##_fn *)dlsym(handle, #fn); \
    if (!api->fn) { fprintf(stderr, "[gpu] %s: missing export %s\n", path, #fn); complete = 0; }
    GPU_EXPORT_NAMES(GPU_RESOLVE)
#undef GPU_RESOLVE
    if (!complete) { dlclose(handle); memset(api, 0, sizeof(*api)); return 0; }

    s32 err = api->gpu_init(0, arena_bytes, &api->info);
    if (err != 0) {
        fprintf(stderr, "[gpu] gpu_init failed: %s (%d)\n", api->gpu_error_string(err), err);
        dlclose(handle);
        memset(api, 0, sizeof(*api));
        return 0;
    }
    api->available = 1;
    return 1;   // the handle is kept for the life of the process
}
