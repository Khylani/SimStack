// Simstack UI: Dear ImGui + ImPlot driven from the sim module, rendered by the
// platform from plain render commands (no GL in here). Included by
// simstack.cpp (unity build); the libraries themselves are compiled once into
// build/ui_third_party_*.o and linked into the sim module.
//
// Memory: nothing is allocated from the OS at runtime. The platform hands the
// sim a fixed UI block (SimMemory::ui_base). Its first part holds the font
// files (loaded once, kept across hot reloads); the rest is a size-class heap
// that ImGui/ImPlot allocate from: power-of-two blocks bumped from the arena,
// freed blocks go to their class's free list, so usage is bounded by the
// peak. On a hot reload the heap is reset wholesale (the old contexts are
// simply forgotten), which is the arena's no-leak guarantee.
//
// Hot reload: ImGui/ImPlot globals live in this module, and their contexts
// hold function pointers into it. sim_before_unload saves the layout to
// UiState::ini while the old code is still mapped; after the reload the heap
// is reset and fresh contexts are created from the saved layout.

// imgui.h / imgui_internal.h / implot.h are included at the top of
// simstack.cpp, before base_types.h (macro collisions).

#define UI_FONT_REGION      Megabytes(32)
#define UI_HEAP_CLASSES     32          // block sizes 2^5 .. 2^31
#define UI_HEAP_HEADER      16
#define UI_HISTORY          600         // frames of performance history
#define UI_INI_CAPACITY     Kilobytes(64)

struct UiHeap {
    Arena arena;
    u8 *free_list[UI_HEAP_CLASSES];
    u64 live_bytes, peak_bytes;
    u64 alloc_count;
};

struct UiState {
    b32 memory_ready;
    Arena fonts;                // persistent part of the UI block
    UiHeap heap;
    u8 *font_ui;   u64 font_ui_size;
    u8 *font_mono; u64 font_mono_size;

    ImGuiContext *imgui;        // 0 after a reload: recreated on the next frame
    ImPlotContext *implot;
    ImFont *mono;
    b32 layout_built;
    char ini[UI_INI_CAPACITY];
    u32 ini_len;

    // performance history (ring)
    f32 hist_frame_ms[UI_HISTORY], hist_chem_ms[UI_HISTORY], hist_rate[UI_HISTORY];
    u32 hist_head, hist_count;
    f64 last_sim_time;

    // controls
    f32 speed;                  // simulated seconds per wall second (windowed)
    b32 max_speed;              // ignore speed, spend the whole chemistry budget
    b32 step_once;
    s32 heat_species;
    s32 view_species;           // coloured in the 3D view
    s32 profile_x_time;         // axial profile x axis: 0 = zone index, 1 = residence time
    char export_path[256];
    b32 dragging;
    f32 chem_ms_frame;          // chemistry wall time this frame (set by simstack.cpp)
    f32 chem_budget_ms;         // wall time the chemistry may use this frame
    f32 frame_ms_smooth;        // measured frame interval (EMA), for display
    f32 target_fps;             // the budget aims to keep this frame rate
    f32 join_ms;                // this frame's wait for last frame's chemistry job
    b32 manual_fps;             // 0: target follows the display (vsync) rate
};

// Chemistry budget for this frame's background job: 90 % of the target frame
// time minus the main thread's own work (UI, plots, render submit), which runs
// before the job is posted. Predictable, with no feedback through the measured
// frame time: if the compositor shows the window slower than the target,
// lower the target to give the solver the idle time.
internal f32 ui_chemistry_budget(UiState *ui, SimInput *input) {
    f32 dt_ms = input->dt_frame * 1000.0f;
    ui->frame_ms_smooth = ui->frame_ms_smooth > 0.0f ? ui->frame_ms_smooth + 0.05f * (dt_ms - ui->frame_ms_smooth) : dt_ms;
    if (ui->target_fps < 10.0f) ui->target_fps = 60.0f;
    // Auto: when last frame's job finished before we needed it (no join wait)
    // and frames are still slower than the target, the display/compositor is
    // the limit (vsync): adopt its rate so the solver gets the idle time.
    // When frames run clearly faster than the target, raise it again.
    if (!ui->manual_fps && ui->frame_ms_smooth > 0.0f) {
        f32 target_ms = 1000.0f / ui->target_fps;
        if (ui->join_ms < 0.5f && ui->frame_ms_smooth > 1.2f * target_ms) ui->target_fps = 1000.0f / ui->frame_ms_smooth;
        else if (ui->frame_ms_smooth < 0.8f * target_ms) ui->target_fps = 1000.0f / ui->frame_ms_smooth;
        ui->target_fps = Clamp(ui->target_fps, 10.0f, 360.0f);
    }
    // The chemistry job overlaps render + swap; what it must leave room for is
    // the UI work on the main thread (last frame's work minus its join wait).
    f32 other_ms = input->last_work_ms - ui->join_ms;
    if (other_ms < 0.0f) other_ms = 0.0f;
    f32 budget = 1000.0f / ui->target_fps * 0.9f - other_ms;
    ui->chem_budget_ms = Clamp(budget, 1.0f, 200.0f);
    return ui->chem_budget_ms;
}

// --- heap ---------------------------------------------------------------------

internal void *ui_heap_alloc(size_t size, void *user) {
    UiHeap *h = (UiHeap *)user;
    u64 need = (u64)size + UI_HEAP_HEADER;
    u32 cls = 5;
    while (((u64)1 << cls) < need) ++cls;
    if (cls >= UI_HEAP_CLASSES) return 0;
    u8 *block = h->free_list[cls];
    if (block) {
        h->free_list[cls] = *(u8 **)(block + UI_HEAP_HEADER);
    } else {
        u64 bytes = (u64)1 << cls;
        if (arena_remaining(&h->arena) < bytes + 16) return 0;   // UI block exhausted: raise PLATFORM_UI_SIZE
        block = (u8 *)push_size_(&h->arena, bytes, 16, 0);
    }
    *(u64 *)block = cls;
    h->live_bytes += (u64)1 << cls;
    if (h->live_bytes > h->peak_bytes) h->peak_bytes = h->live_bytes;
    ++h->alloc_count;
    return block + UI_HEAP_HEADER;
}

internal void ui_heap_free(void *ptr, void *user) {
    if (!ptr) return;
    UiHeap *h = (UiHeap *)user;
    u8 *block = (u8 *)ptr - UI_HEAP_HEADER;
    u64 cls = *(u64 *)block;
    Assert(cls < UI_HEAP_CLASSES);
    *(u8 **)(block + UI_HEAP_HEADER) = h->free_list[cls];
    h->free_list[cls] = block;
    h->live_bytes -= (u64)1 << cls;
}

internal void ui_heap_reset(UiHeap *h) {
    h->arena.used = 0;
    for (u32 i = 0; i < UI_HEAP_CLASSES; ++i) h->free_list[i] = 0;
    h->live_bytes = 0;
}

// --- lifetime -------------------------------------------------------------------

internal void ui_init_memory(UiState *ui, SimMemory *memory) {
    if (ui->memory_ready || !memory->ui_base) return;
    Arena whole = arena_make(memory->ui_base, memory->ui_size);
    ui->fonts = arena_split(&whole, UI_FONT_REGION, 64);
    ui->heap.arena = arena_split(&whole, arena_remaining(&whole) - 64, 64);
    // Fonts are optional: ImGui's built-in font is the fallback.
    const char *ui_fonts[] = { "/usr/share/fonts/noto/NotoSans-Regular.ttf", "/usr/share/fonts/TTF/DejaVuSans.ttf" };
    const char *mono_fonts[] = { "/usr/share/fonts/TTF/JetBrainsMonoNerdFont-Regular.ttf", "/usr/share/fonts/TTF/DejaVuSansMono.ttf" };
    for (u32 i = 0; i < ArrayCount(ui_fonts) && !ui->font_ui; ++i) {
        PlatformFileContents f = memory->platform.read_entire_file(&ui->fonts, ui_fonts[i]);
        ui->font_ui = (u8 *)f.data; ui->font_ui_size = f.size;
    }
    for (u32 i = 0; i < ArrayCount(mono_fonts) && !ui->font_mono; ++i) {
        PlatformFileContents f = memory->platform.read_entire_file(&ui->fonts, mono_fonts[i]);
        ui->font_mono = (u8 *)f.data; ui->font_mono_size = f.size;
    }
    ui->speed = 1.0f;
    ui->heat_species = 0;
    ui->profile_x_time = 1;
    snprintf(ui->export_path, sizeof(ui->export_path), "dbd_export.csv");
    ui->memory_ready = 1;
}

internal void ui_style() {
    ImGuiStyle &st = ImGui::GetStyle();
    ImGui::StyleColorsDark(&st);
    st.WindowRounding = 6.0f;
    st.ChildRounding = 4.0f;
    st.FrameRounding = 4.0f;
    st.GrabRounding = 4.0f;
    st.PopupRounding = 4.0f;
    st.TabRounding = 4.0f;
    st.ScrollbarRounding = 6.0f;
    st.WindowBorderSize = 0.0f;
    st.FramePadding = ImVec2(8, 4);
    st.ItemSpacing = ImVec2(8, 6);
    st.WindowPadding = ImVec2(10, 10);
    ImVec4 *c = st.Colors;
    ImVec4 bg = ImVec4(0.075f, 0.085f, 0.105f, 0.94f);
    ImVec4 accent = ImVec4(0.30f, 0.62f, 0.95f, 1.0f);
    c[ImGuiCol_WindowBg] = bg;
    c[ImGuiCol_ChildBg] = ImVec4(0.06f, 0.07f, 0.09f, 0.6f);
    c[ImGuiCol_PopupBg] = ImVec4(0.08f, 0.09f, 0.11f, 0.98f);
    c[ImGuiCol_TitleBg] = ImVec4(0.06f, 0.07f, 0.09f, 1.0f);
    c[ImGuiCol_TitleBgActive] = ImVec4(0.09f, 0.11f, 0.15f, 1.0f);
    c[ImGuiCol_FrameBg] = ImVec4(0.13f, 0.15f, 0.19f, 1.0f);
    c[ImGuiCol_FrameBgHovered] = ImVec4(0.18f, 0.21f, 0.27f, 1.0f);
    c[ImGuiCol_FrameBgActive] = ImVec4(0.22f, 0.27f, 0.35f, 1.0f);
    c[ImGuiCol_SliderGrab] = accent;
    c[ImGuiCol_SliderGrabActive] = ImVec4(0.45f, 0.72f, 1.0f, 1.0f);
    c[ImGuiCol_CheckMark] = accent;
    c[ImGuiCol_Button] = ImVec4(0.17f, 0.22f, 0.30f, 1.0f);
    c[ImGuiCol_ButtonHovered] = ImVec4(0.24f, 0.33f, 0.46f, 1.0f);
    c[ImGuiCol_ButtonActive] = ImVec4(0.30f, 0.45f, 0.66f, 1.0f);
    c[ImGuiCol_Header] = ImVec4(0.17f, 0.22f, 0.30f, 1.0f);
    c[ImGuiCol_HeaderHovered] = ImVec4(0.24f, 0.33f, 0.46f, 1.0f);
    c[ImGuiCol_HeaderActive] = ImVec4(0.30f, 0.45f, 0.66f, 1.0f);
    c[ImGuiCol_Tab] = ImVec4(0.11f, 0.13f, 0.17f, 1.0f);
    c[ImGuiCol_TabHovered] = ImVec4(0.24f, 0.33f, 0.46f, 1.0f);
    c[ImGuiCol_TabSelected] = ImVec4(0.17f, 0.24f, 0.34f, 1.0f);
    c[ImGuiCol_DockingPreview] = ImVec4(0.30f, 0.62f, 0.95f, 0.5f);
    c[ImGuiCol_DockingEmptyBg] = ImVec4(0, 0, 0, 0);
    c[ImGuiCol_Separator] = ImVec4(0.20f, 0.23f, 0.29f, 1.0f);
    ImPlotStyle &ps = ImPlot::GetStyle();
    ps.PlotPadding = ImVec2(6, 6);
    ps.Colors[ImPlotCol_PlotBg] = ImVec4(0.05f, 0.06f, 0.08f, 1.0f);
    ps.Colors[ImPlotCol_FrameBg] = ImVec4(0, 0, 0, 0);
}

// Creates the contexts if needed (first frame, or after a reload).
internal void ui_ensure_context(UiState *ui) {
    ImGui::SetAllocatorFunctions(ui_heap_alloc, ui_heap_free, &ui->heap);
    if (ui->imgui) {
        ImGui::SetCurrentContext(ui->imgui);
        ImPlot::SetCurrentContext(ui->implot);
        return;
    }
    ui->imgui = ImGui::CreateContext();
    ImGui::SetCurrentContext(ui->imgui);
    ui->implot = ImPlot::CreateContext();
    ImPlot::SetCurrentContext(ui->implot);
    ImGuiIO &io = ImGui::GetIO();
    io.IniFilename = 0;
    io.LogFilename = 0;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard | ImGuiConfigFlags_DockingEnable;
    io.BackendFlags |= ImGuiBackendFlags_RendererHasTextures | ImGuiBackendFlags_RendererHasVtxOffset;
    io.BackendRendererName = "simstack_gl";
    io.BackendPlatformName = "simstack";
    io.Fonts->TexDesiredFormat = ImTextureFormat_RGBA32;
    ImFontConfig cfg;
    cfg.FontDataOwnedByAtlas = false;   // the TTF bytes live in the UI font region
    if (ui->font_ui) io.Fonts->AddFontFromMemoryTTF(ui->font_ui, (int)ui->font_ui_size, 17.0f, &cfg);
    else io.Fonts->AddFontDefault();
    ui->mono = 0;
    if (ui->font_mono) ui->mono = io.Fonts->AddFontFromMemoryTTF(ui->font_mono, (int)ui->font_mono_size, 15.0f, &cfg);
    ui_style();
    ui->layout_built = 0;
    if (ui->ini_len) {
        ImGui::LoadIniSettingsFromMemory(ui->ini, ui->ini_len);
        ui->layout_built = 1;
    }
}

internal void ui_before_unload(UiState *ui) {
    if (!ui->imgui) return;
    ImGui::SetCurrentContext(ui->imgui);
    size_t len = 0;
    const char *ini = ImGui::SaveIniSettingsToMemory(&len);
    if (len < sizeof(ui->ini)) {
        memcpy(ui->ini, ini, len);
        ui->ini_len = (u32)len;
    }
}

// After a reload the contexts belong to code that is gone: forget them and
// take the whole heap back.
internal void ui_on_reload(UiState *ui) {
    ui->imgui = 0;
    ui->implot = 0;
    ui->mono = 0;
    ui_heap_reset(&ui->heap);
}

// --- input ----------------------------------------------------------------------

internal void ui_feed_input(SimInput *input) {
    ImGuiIO &io = ImGui::GetIO();
    io.DisplaySize = ImVec2((f32)input->window_width, (f32)input->window_height);
    io.DeltaTime = input->dt_frame > 1e-4f ? input->dt_frame : 1e-4f;
    if (input->mouse_inside) io.AddMousePosEvent(input->mouse_x, input->mouse_y);
    else io.AddMousePosEvent(-FLT_MAX, -FLT_MAX);
    for (u32 b = 0; b < 3; ++b) io.AddMouseButtonEvent((int)b, input->mouse_buttons[b].ended_down != 0);
    if (input->mouse_wheel != 0.0f) io.AddMouseWheelEvent(0.0f, input->mouse_wheel);
    struct { SimKey key; ImGuiKey im; } map[] = {
        { SimKey_Tab, ImGuiKey_Tab }, { SimKey_Left, ImGuiKey_LeftArrow }, { SimKey_Right, ImGuiKey_RightArrow },
        { SimKey_Up, ImGuiKey_UpArrow }, { SimKey_Down, ImGuiKey_DownArrow }, { SimKey_PageUp, ImGuiKey_PageUp },
        { SimKey_PageDown, ImGuiKey_PageDown }, { SimKey_Home, ImGuiKey_Home }, { SimKey_End, ImGuiKey_End },
        { SimKey_Delete, ImGuiKey_Delete }, { SimKey_Backspace, ImGuiKey_Backspace }, { SimKey_Enter, ImGuiKey_Enter },
        { SimKey_Escape, ImGuiKey_Escape }, { SimKey_Space, ImGuiKey_Space }, { SimKey_A, ImGuiKey_A },
        { SimKey_C, ImGuiKey_C }, { SimKey_V, ImGuiKey_V }, { SimKey_X, ImGuiKey_X }, { SimKey_Y, ImGuiKey_Y },
        { SimKey_Z, ImGuiKey_Z }, { SimKey_R, ImGuiKey_R }, { SimKey_P, ImGuiKey_P }, { SimKey_F1, ImGuiKey_F1 },
        { SimKey_F2, ImGuiKey_F2 },
    };
    for (u32 i = 0; i < ArrayCount(map); ++i) io.AddKeyEvent(map[i].im, input->keys[map[i].key].ended_down != 0);
    io.AddKeyEvent(ImGuiMod_Ctrl, input->keys[SimKey_Ctrl].ended_down != 0);
    io.AddKeyEvent(ImGuiMod_Shift, input->keys[SimKey_Shift].ended_down != 0);
    io.AddKeyEvent(ImGuiMod_Alt, input->keys[SimKey_Alt].ended_down != 0);
    if (input->text[0]) io.AddInputCharactersUTF8(input->text);
}

// --- output -----------------------------------------------------------------------

// Draw data and texture requests -> RenderCommands (arrays in the frame arena).
internal void ui_emit(RenderCommands *render, Arena *frame) {
    static_assert(sizeof(ImDrawVert) == sizeof(RenderUiVertex), "ImDrawVert must match RenderUiVertex");
    static_assert(sizeof(ImDrawIdx) == sizeof(u32), "ImDrawIdx must be 32-bit (imconfig_simstack.h)");
    ImDrawData *dd = ImGui::GetDrawData();
    render->tex_op_count = 0;
    render->ui_cmd_count = render->ui_vtx_count = render->ui_idx_count = 0;
    if (!dd) return;

    if (dd->Textures) {
        for (int i = 0; i < dd->Textures->Size; ++i) {
            ImTextureData *tex = (*dd->Textures)[i];
            if (render->tex_op_count >= RENDER_MAX_TEX_OPS) break;
            u32 id = (u32)tex->UniqueID + 1;
            if (tex->Status == ImTextureStatus_WantCreate || tex->Status == ImTextureStatus_WantUpdates) {
                Assert(tex->Format == ImTextureFormat_RGBA32);
                RenderTexOp *op = render->tex_ops + render->tex_op_count++;
                op->kind = RenderTexOp_Upload;
                op->id = id;
                op->width = (u32)tex->Width;
                op->height = (u32)tex->Height;
                op->rgba = (const u8 *)tex->GetPixels();
                tex->SetTexID((ImTextureID)id);
                tex->SetStatus(ImTextureStatus_OK);
            } else if (tex->Status == ImTextureStatus_WantDestroy && tex->UnusedFrames > 0) {
                RenderTexOp *op = render->tex_ops + render->tex_op_count++;
                op->kind = RenderTexOp_Destroy;
                op->id = id;
                tex->SetTexID(ImTextureID_Invalid);
                tex->SetStatus(ImTextureStatus_Destroyed);
            }
        }
    }

    u32 cmd_total = 0;
    for (int l = 0; l < dd->CmdLists.Size; ++l) cmd_total += (u32)dd->CmdLists[l]->CmdBuffer.Size;
    RenderUiVertex *vtx = push_array_nozero(frame, RenderUiVertex, (u64)dd->TotalVtxCount + 1);
    u32 *idx = push_array_nozero(frame, u32, (u64)dd->TotalIdxCount + 1);
    RenderUiCmd *cmds = push_array_nozero(frame, RenderUiCmd, (u64)cmd_total + 1);
    u32 vtx_base = 0, idx_base = 0, ncmd = 0;
    for (int l = 0; l < dd->CmdLists.Size; ++l) {
        const ImDrawList *list = dd->CmdLists[l];
        memcpy(vtx + vtx_base, list->VtxBuffer.Data, sizeof(ImDrawVert) * (u64)list->VtxBuffer.Size);
        memcpy(idx + idx_base, list->IdxBuffer.Data, sizeof(ImDrawIdx) * (u64)list->IdxBuffer.Size);
        for (int c = 0; c < list->CmdBuffer.Size; ++c) {
            const ImDrawCmd *cmd = &list->CmdBuffer[c];
            if (cmd->UserCallback || cmd->ElemCount == 0) continue;
            RenderUiCmd *out = cmds + ncmd++;
            out->idx_offset = idx_base + cmd->IdxOffset;
            out->elem_count = cmd->ElemCount;
            out->vtx_offset = vtx_base + cmd->VtxOffset;
            out->texture = (u32)cmd->GetTexID();
            out->clip_x0 = cmd->ClipRect.x - dd->DisplayPos.x;
            out->clip_y0 = cmd->ClipRect.y - dd->DisplayPos.y;
            out->clip_x1 = cmd->ClipRect.z - dd->DisplayPos.x;
            out->clip_y1 = cmd->ClipRect.w - dd->DisplayPos.y;
        }
        vtx_base += (u32)list->VtxBuffer.Size;
        idx_base += (u32)list->IdxBuffer.Size;
    }
    render->ui_vtx = vtx;
    render->ui_idx = idx;
    render->ui_cmds = cmds;
    render->ui_vtx_count = vtx_base;
    render->ui_idx_count = idx_base;
    render->ui_cmd_count = ncmd;
}

// --- helpers ----------------------------------------------------------------------

internal ImVec4 ui_v4(v4 c) { return ImVec4(c.r, c.g, c.b, c.a); }

internal b32 ui_slider_log(const char *label, f64 *value, f64 lo, f64 hi, const char *fmt) {
    f32 v = (f32)*value;
    b32 changed = ImGui::SliderFloat(label, &v, (f32)lo, (f32)hi, fmt, ImGuiSliderFlags_Logarithmic);
    if (changed) *value = (f64)v;
    return changed;
}

internal void ui_help(const char *text) {
    ImGui::SameLine();
    ImGui::TextDisabled("(?)");
    if (ImGui::BeginItemTooltip()) {
        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 30.0f);
        ImGui::TextUnformatted(text);
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
    }
}

internal void ui_bar(const char *label, u64 used, u64 size) {
    f32 frac = size ? (f32)((f64)used / (f64)size) : 0.0f;
    char overlay[96];
    snprintf(overlay, sizeof(overlay), "%s  %.1f / %.0f MB", label, (f64)used / 1048576.0, (f64)size / 1048576.0);
    ImGui::ProgressBar(frac, ImVec2(-1, 0), overlay);
}
