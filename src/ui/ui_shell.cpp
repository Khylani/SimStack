// App shell: main menu bar, transport bar (playback, speed, time), DPI-aware
// scaling, drawn icons (no icon font needed), shortcuts. Included by
// ui_panels.cpp after SimState is defined.

internal void ui_shell_defaults(UiState *ui) {
    if (ui->shell_ready) return;
    ui->show_controls = ui->show_plots = ui->show_performance = ui->show_inspector = 1;
    ui->show_grid = ui->show_labels = ui->show_legend = ui->show_hints = ui->show_axes = 1;
    ui->show_cells = ui->show_plants = 1;
    ui->ui_scale = 0.0f;
    snprintf(ui->screenshot_path, sizeof(ui->screenshot_path), "simstack_shot.ppm");
    ui->shell_ready = 1;
}

// 4K panels are ~2100 px tall; scale the UI with the window so text stays legible.
internal f32 ui_auto_scale(f32 window_h) { return window_h >= 2000.0f ? 1.5f : window_h >= 1400.0f ? 1.25f : 1.0f; }

internal void ui_apply_scale(UiState *ui) {
    f32 want = ui->ui_scale > 0.0f ? ui->ui_scale : ui_auto_scale(ImGui::GetIO().DisplaySize.y);
    if (want == ui->applied_scale) return;
    ui_style();
    ImGuiStyle &st = ImGui::GetStyle();
    st.ScaleAllSizes(want);
    st.FontScaleDpi = want;
    ImPlot::GetStyle().PlotPadding = ImVec2(6 * want, 6 * want);
    ui->applied_scale = want;
}

internal f32 ui_s(f32 v) { return v * ImGui::GetStyle().FontScaleDpi; }   // scale a pixel size

// A Controls section; `advanced` ones only show in Advanced mode.
internal b32 ui_section(UiState *ui, const char *name, ImGuiTreeNodeFlags flags, b32 advanced = 0) {
    if (advanced && !ui->advanced) return 0;
    return ImGui::CollapsingHeader(name, flags);
}

// Simple / Advanced switch and Defaults, under the scene picker.
internal b32 ui_controls_header(UiState *ui, const char *defaults_tip) {
    f32 w = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x * 2.0f) / 3.0f;
    for (u32 i = 0; i < 2; ++i) {
        b32 on = (ui->advanced != 0) == (i == 1);
        ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyleColorVec4(on ? ImGuiCol_ButtonActive : ImGuiCol_FrameBg));
        if (ImGui::Button(i ? "Advanced" : "Simple", ImVec2(w, 0))) ui->advanced = i;
        ImGui::PopStyleColor();
        if (ImGui::IsItemHovered()) ImGui::SetTooltip(i ? "Every parameter, solver and model detail" : "The parameters you usually change");
        ImGui::SameLine();
    }
    b32 defaults = ImGui::Button("Defaults", ImVec2(w, 0));
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", defaults_tip);
    return defaults;
}

// --- icons ------------------------------------------------------------------------------

enum UiIcon { UiIcon_Play, UiIcon_Pause, UiIcon_StepFwd, UiIcon_StepBack, UiIcon_Reset, UiIcon_Record, UiIcon_Frame, UiIcon_ToStart, UiIcon_ToEnd };

internal void ui_draw_icon(ImDrawList *dl, ImVec2 c, f32 r, u32 icon, ImU32 col) {
    switch (icon) {
        case UiIcon_Play:
            dl->AddTriangleFilled(ImVec2(c.x - 0.55f * r, c.y - 0.75f * r), ImVec2(c.x - 0.55f * r, c.y + 0.75f * r), ImVec2(c.x + 0.8f * r, c.y), col);
            break;
        case UiIcon_Pause:
            dl->AddRectFilled(ImVec2(c.x - 0.6f * r, c.y - 0.7f * r), ImVec2(c.x - 0.15f * r, c.y + 0.7f * r), col, 1.5f);
            dl->AddRectFilled(ImVec2(c.x + 0.15f * r, c.y - 0.7f * r), ImVec2(c.x + 0.6f * r, c.y + 0.7f * r), col, 1.5f);
            break;
        case UiIcon_StepFwd:
            dl->AddTriangleFilled(ImVec2(c.x - 0.65f * r, c.y - 0.65f * r), ImVec2(c.x - 0.65f * r, c.y + 0.65f * r), ImVec2(c.x + 0.35f * r, c.y), col);
            dl->AddRectFilled(ImVec2(c.x + 0.4f * r, c.y - 0.65f * r), ImVec2(c.x + 0.65f * r, c.y + 0.65f * r), col);
            break;
        case UiIcon_StepBack:
            dl->AddTriangleFilled(ImVec2(c.x + 0.65f * r, c.y - 0.65f * r), ImVec2(c.x + 0.65f * r, c.y + 0.65f * r), ImVec2(c.x - 0.35f * r, c.y), col);
            dl->AddRectFilled(ImVec2(c.x - 0.65f * r, c.y - 0.65f * r), ImVec2(c.x - 0.4f * r, c.y + 0.65f * r), col);
            break;
        case UiIcon_ToStart:
            dl->AddRectFilled(ImVec2(c.x - 0.75f * r, c.y - 0.65f * r), ImVec2(c.x - 0.5f * r, c.y + 0.65f * r), col);
            dl->AddTriangleFilled(ImVec2(c.x + 0.1f * r, c.y - 0.65f * r), ImVec2(c.x + 0.1f * r, c.y + 0.65f * r), ImVec2(c.x - 0.5f * r, c.y), col);
            dl->AddTriangleFilled(ImVec2(c.x + 0.75f * r, c.y - 0.65f * r), ImVec2(c.x + 0.75f * r, c.y + 0.65f * r), ImVec2(c.x + 0.1f * r, c.y), col);
            break;
        case UiIcon_ToEnd:
            dl->AddRectFilled(ImVec2(c.x + 0.5f * r, c.y - 0.65f * r), ImVec2(c.x + 0.75f * r, c.y + 0.65f * r), col);
            dl->AddTriangleFilled(ImVec2(c.x - 0.1f * r, c.y - 0.65f * r), ImVec2(c.x - 0.1f * r, c.y + 0.65f * r), ImVec2(c.x + 0.5f * r, c.y), col);
            dl->AddTriangleFilled(ImVec2(c.x - 0.75f * r, c.y - 0.65f * r), ImVec2(c.x - 0.75f * r, c.y + 0.65f * r), ImVec2(c.x - 0.1f * r, c.y), col);
            break;
        case UiIcon_Reset: {
            dl->PathArcTo(c, 0.62f * r, 0.6f, 5.6f, 20);
            dl->PathStroke(col, Maximum(1.5f, 0.2f * r));
            ImVec2 tip = ImVec2(c.x + 0.62f * r * cosf(0.6f), c.y + 0.62f * r * sinf(0.6f));
            dl->AddTriangleFilled(ImVec2(tip.x + 0.35f * r, tip.y - 0.05f * r), ImVec2(tip.x - 0.3f * r, tip.y - 0.1f * r), ImVec2(tip.x + 0.05f * r, tip.y + 0.4f * r), col);
        } break;
        case UiIcon_Record:
            dl->AddCircleFilled(c, 0.55f * r, col, 20);
            break;
        case UiIcon_Frame: {
            f32 a = 0.7f * r, b = 0.3f * r, t = Maximum(1.5f, 0.18f * r);
            for (s32 sx = -1; sx <= 1; sx += 2)
                for (s32 sy = -1; sy <= 1; sy += 2) {
                    ImVec2 k = ImVec2(c.x + sx * a, c.y + sy * a);
                    dl->AddLine(k, ImVec2(k.x - sx * b, k.y), col, t);
                    dl->AddLine(k, ImVec2(k.x, k.y - sy * b), col, t);
                }
        } break;
    }
}

// A square button with a drawn icon; `on` tints it with the accent.
internal b32 ui_icon_button(const char *id, u32 icon, const char *tip, b32 on = 0) {
    f32 h = ImGui::GetFrameHeight();
    ImVec2 p = ImGui::GetCursorScreenPos();
    b32 pressed = ImGui::InvisibleButton(id, ImVec2(h * 1.15f, h));
    ImDrawList *dl = ImGui::GetWindowDrawList();
    b32 hov = ImGui::IsItemHovered(), act = ImGui::IsItemActive();
    ImU32 bg = ImGui::GetColorU32(act ? ImGuiCol_ButtonActive : hov ? ImGuiCol_ButtonHovered : on ? ImGuiCol_ButtonActive : ImGuiCol_Button);
    dl->AddRectFilled(p, ImVec2(p.x + h * 1.15f, p.y + h), bg, ImGui::GetStyle().FrameRounding);
    ImU32 fg = on ? IM_COL32(255, 255, 255, 255) : IM_COL32(215, 222, 235, 255);
    ui_draw_icon(dl, ImVec2(p.x + h * 0.575f, p.y + h * 0.5f), h * 0.38f, icon, fg);
    if (tip && hov) ImGui::SetTooltip("%s", tip);
    return pressed;
}

// --- formatting ---------------------------------------------------------------------------

internal const char *ui_fmt_sim_time(SimState *state, char *buf, u32 n) {
    f64 t = state->sim_time;
    if (state->nft) {
        f64 day = floor(t / 86400.0), sec = t - day * 86400.0;
        snprintf(buf, n, "day %.0f  %02d:%02d", day, (int)(sec / 3600.0), (int)fmod(sec / 60.0, 60.0));
    } else if (t < 10.0) {
        snprintf(buf, n, "%.3f s", t);
    } else {
        snprintf(buf, n, "%.1f s", t);
    }
    return buf;
}

internal const char *ui_fmt_rate(f64 sim_s_per_s, char *buf, u32 n) {
    f64 r = sim_s_per_s;
    if (r >= 86400.0 * 2) snprintf(buf, n, "%.1f d/s", r / 86400.0);
    else if (r >= 3600.0 * 2) snprintf(buf, n, "%.1f h/s", r / 3600.0);
    else if (r >= 120.0) snprintf(buf, n, "%.1f min/s", r / 60.0);
    else snprintf(buf, n, "%.3gx", r);
    return buf;
}

// --- speed ------------------------------------------------------------------------------

struct UiSpeedPreset { const char *name; f32 value; };
global const UiSpeedPreset ui_nft_speeds[] = {
    { "real time", 1.0f }, { "1 min / s", 60.0f }, { "1 h / s", 3600.0f }, { "6 h / s", 21600.0f },
    { "1 day / s", 86400.0f }, { "1 week / s", 604800.0f },
};
global const UiSpeedPreset ui_chem_speeds[] = {
    { "0.01x", 0.01f }, { "0.1x", 0.1f }, { "real time", 1.0f }, { "10x", 10.0f }, { "100x", 100.0f },
};

internal void ui_speed_control(SimState *state) {
    UiState *ui = &state->ui;
    b32 nft = state->nft != 0;
    f32 *speed = nft ? &ui->nft_speed : &ui->speed;
    char label[48], rate[32];
    if (ui->max_speed) snprintf(label, sizeof(label), "max speed");
    else snprintf(label, sizeof(label), "%s", ui_fmt_rate(*speed, rate, sizeof(rate)));
    ImGui::SetNextItemWidth(ui_s(150));
    if (ImGui::BeginCombo("##speed", label, ImGuiComboFlags_HeightLarge)) {
        const UiSpeedPreset *p = nft ? ui_nft_speeds : ui_chem_speeds;
        u32 n = nft ? ArrayCount(ui_nft_speeds) : ArrayCount(ui_chem_speeds);
        for (u32 i = 0; i < n; ++i)
            if (ImGui::Selectable(p[i].name, !ui->max_speed && fabsf(*speed - p[i].value) < 1e-3f * p[i].value)) {
                *speed = p[i].value;
                ui->max_speed = 0;
            }
        ImGui::Separator();
        bool mx = ui->max_speed != 0;
        if (ImGui::Checkbox("max (whole frame budget)", &mx)) ui->max_speed = mx;
        ImGui::SetNextItemWidth(ui_s(220));
        ImGui::SliderFloat("##speedfine", speed, nft ? 1.0f : 0.001f, nft ? 2.6e6f : 1000.0f, "%.3g x", ImGuiSliderFlags_Logarithmic);
        ImGui::EndCombo();
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Simulated time per wall second");
}

// --- menu bar ---------------------------------------------------------------------------

internal void ui_switch_scene(SimState *state, SimMemory *memory, u32 scene, u32 zones);

internal void ui_export_csv(SimState *state, SimMemory *memory) {
    UiState *ui = &state->ui;
    if (state->dbd) {
        const char *saved = state->dbd->csv_path;
        state->dbd->csv_path = ui->export_path;
        dbd_scene_write_csv(state->dbd, &memory->platform, &state->frame);
        state->dbd->csv_path = saved;
    } else if (state->nft) {
        nft_scene_write_csv(state->nft, &memory->platform, &state->frame, ui->export_path);
    }
}

internal void ui_main_menu(SimState *state, SimMemory *memory, RenderCommands *render) {
    UiState *ui = &state->ui;
    if (!ImGui::BeginMainMenuBar()) return;
    if (ImGui::BeginMenu("File")) {
        ImGui::BeginDisabled(!state->dbd && !state->nft);
        if (ImGui::MenuItem("Export CSV", 0)) ui_export_csv(state, memory);
        ImGui::EndDisabled();
        ImGui::SetNextItemWidth(ui_s(220));
        ImGui::InputText("##csvpath", ui->export_path, sizeof(ui->export_path));
        ImGui::Separator();
        if (ImGui::MenuItem("Save screenshot")) {
            snprintf(ui->screenshot_path, sizeof(ui->screenshot_path), "simstack_shot_%03u.ppm", ++ui->screenshot_count);
            render->screenshot_path = ui->screenshot_path;
        }
        ImGui::Separator();
        if (ImGui::MenuItem("Quit", "Ctrl+Q")) render->request_quit = 1;
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("Scene")) {
        const char *names[] = { "Particles (demo)", "DBD reactor", "Operating-point sweep", "NFT system" };
        for (u32 s = 0; s < ArrayCount(names); ++s)
            if (ImGui::MenuItem(names[s], 0, state->scene == s) && state->scene != s) {
                u32 zones = s == SimScene_Sweep ? 1024 : s == SimScene_Nft ? 16 : 64;
                ui_switch_scene(state, memory, s, zones);
            }
        ImGui::Separator();
        if (ImGui::BeginMenu("NFT presets")) {
            for (u32 p = 0; p < NftPreset_Count; ++p)
                if (ImGui::MenuItem(nft_preset_names[p])) {
                    state->nft_params = nft_preset(p);
                    state->nft_params_set = 1;
                    memory->config.preset = p;
                    ui_switch_scene(state, memory, SimScene_Nft, state->nft_params.rails);
                }
            ImGui::EndMenu();
        }
        if (ImGui::MenuItem("DBD: one PAW skid module")) {
            memory->config.preset = 1;
            ui_switch_scene(state, memory, SimScene_Dbd, 64);
        }
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("Simulation")) {
        if (ImGui::MenuItem(state->paused ? "Play" : "Pause", "Space")) state->paused = !state->paused;
        if (ImGui::MenuItem("Step", ".")) { ui->step_once = 1; state->paused = 1; }
        if (ImGui::MenuItem("Reset", "R")) sim_reset_scene(state, memory);
        ImGui::Separator();
        bool mx = ui->max_speed != 0;
        if (ImGui::MenuItem("Max speed", 0, &mx)) ui->max_speed = mx;
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("View")) {
        bool c = ui->show_controls, p = ui->show_plots, f = ui->show_performance, i = ui->show_inspector;
        if (ImGui::MenuItem("Controls", 0, &c)) ui->show_controls = c;
        if (ImGui::MenuItem("Plots", 0, &p)) ui->show_plots = p;
        if (ImGui::MenuItem("Inspector", 0, &i)) ui->show_inspector = i;
        if (ImGui::MenuItem("Performance", 0, &f)) ui->show_performance = f;
        ImGui::Separator();
        if (ImGui::MenuItem("Frame all", "F")) { state->camera.frame_request = 1; state->camera.user_moved = 0; }
        if (ImGui::MenuItem("Reset camera", "Home")) { camera_reset(&state->camera, state->scene); state->camera.auto_orbit = 0; }
        bool orbit = state->camera.auto_orbit != 0;
        if (ImGui::MenuItem("Auto orbit", "F1", &orbit)) state->camera.auto_orbit = orbit;
        ImGui::Separator();
        bool g = ui->show_grid, l = ui->show_labels, lg = ui->show_legend, h = ui->show_hints, ax = ui->show_axes;
        if (ImGui::MenuItem("Floor grid", 0, &g)) ui->show_grid = g;
        if (ImGui::MenuItem("Labels", 0, &l)) ui->show_labels = l;
        if (ImGui::MenuItem("Colour legend", 0, &lg)) ui->show_legend = lg;
        if (ImGui::MenuItem("Axes", 0, &ax)) ui->show_axes = ax;
        if (ImGui::MenuItem("Mouse hints", 0, &h)) ui->show_hints = h;
        if (state->nft) {
            bool cl = ui->show_cells, pl = ui->show_plants;
            if (ImGui::MenuItem("Channel water", 0, &cl)) ui->show_cells = cl;
            if (ImGui::MenuItem("Plants", 0, &pl)) ui->show_plants = pl;
        }
        ImGui::Separator();
        if (ImGui::BeginMenu("UI scale")) {
            f32 opts[] = { 0.0f, 1.0f, 1.25f, 1.5f, 1.75f, 2.0f };
            const char *names[] = { "Automatic", "100 %", "125 %", "150 %", "175 %", "200 %" };
            for (u32 k = 0; k < ArrayCount(opts); ++k)
                if (ImGui::MenuItem(names[k], 0, ui->ui_scale == opts[k])) ui->ui_scale = opts[k];
            ImGui::EndMenu();
        }
        if (ImGui::MenuItem("Reset layout")) ui->relayout = 1;
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("Help")) {
        if (ImGui::MenuItem("Shortcuts", "F2")) ui->open_shortcuts = 1;
        if (ImGui::MenuItem("About SimStack")) ui->open_about = 1;
        ImGui::EndMenu();
    }
    ImGui::EndMainMenuBar();
}

internal void ui_shortcuts_popup(UiState *ui) {
    if (ui->open_shortcuts) { ImGui::OpenPopup("Shortcuts"); ui->open_shortcuts = 0; }
    if (ui->open_about) { ImGui::OpenPopup("About"); ui->open_about = 0; }
    if (ImGui::BeginPopupModal("Shortcuts", 0, ImGuiWindowFlags_AlwaysAutoResize)) {
        struct { const char *k, *v; } rows[] = {
            { "Space", "play / pause" }, { ". (period)", "step forward" }, { ", (comma)", "step back (keyframe)" },
            { "R", "reset the scene" }, { "F", "frame all" }, { "Home", "reset the camera" }, { "F1", "auto orbit" },
            { "Left drag", "orbit" }, { "Right / middle drag, Shift + left", "pan" }, { "Wheel", "zoom toward the cursor" },
            { "Hover", "inspect" }, { "Click", "select (pin in the Inspector)" }, { "Double-click", "focus the camera there" },
            { "Escape", "clear the selection / close menus" }, { "Ctrl+Q", "quit" },
        };
        if (ImGui::BeginTable("##keys", 2, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingFixedFit)) {
            for (u32 i = 0; i < ArrayCount(rows); ++i) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn(); ImGui::TextUnformatted(rows[i].k);
                ImGui::TableNextColumn(); ImGui::TextDisabled("%s", rows[i].v);
            }
            ImGui::EndTable();
        }
        if (ImGui::Button("Close", ImVec2(-1, 0)) || ImGui::IsKeyPressed(ImGuiKey_Escape)) ImGui::CloseCurrentPopup();
        ImGui::EndPopup();
    }
    if (ImGui::BeginPopupModal("About", 0, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::TextUnformatted("SimStack");
        ImGui::TextDisabled("DBD plasma chemistry, plasma-activated water and NFT hydroponics");
        ImGui::TextDisabled("C++ / AVX2 / CUDA, arena memory, hot-reloadable sim module");
        if (ImGui::Button("Close", ImVec2(-1, 0)) || ImGui::IsKeyPressed(ImGuiKey_Escape)) ImGui::CloseCurrentPopup();
        ImGui::EndPopup();
    }
}

// --- transport bar ----------------------------------------------------------------------

internal void ui_timeline_bar(SimState *state, SimMemory *memory, f32 width);   // ui_timeline.cpp

internal void ui_transport(SimState *state, SimMemory *memory, SimStats *stats) {
    UiState *ui = &state->ui;
    ImGuiViewport *vp = ImGui::GetMainViewport();
    f32 h = ImGui::GetFrameHeight() + 2.0f * ImGui::GetStyle().WindowPadding.y;
    ImGuiWindowFlags flags = ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoTitleBar;
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.055f, 0.062f, 0.080f, 1.0f));
    if (ImGui::BeginViewportSideBar("##transport", vp, ImGuiDir_Down, h, flags)) {
        if (ui_icon_button("##reset", UiIcon_Reset, "Reset the scene (R)")) sim_reset_scene(state, memory);
        ImGui::SameLine();
        if (ui_icon_button("##back", UiIcon_StepBack, "Step back to the previous keyframe (,)")) ui->step_back = 1;
        ImGui::SameLine();
        if (ui_icon_button("##play", state->paused ? UiIcon_Play : UiIcon_Pause, state->paused ? "Play (Space)" : "Pause (Space)", !state->paused))
            state->paused = !state->paused;
        ImGui::SameLine();
        if (ui_icon_button("##step", UiIcon_StepFwd, "Step forward (.)")) { ui->step_once = 1; state->paused = 1; }
        ImGui::SameLine();
        ui_speed_control(state);
        ImGui::SameLine();
        char tbuf[48];
        if (ui->mono) ImGui::PushFont(ui->mono, 0.0f);
        ImGui::AlignTextToFramePadding();
        ImGui::Text("%s", ui_fmt_sim_time(state, tbuf, sizeof(tbuf)));
        if (ui->mono) ImGui::PopFont();
        ImGui::SameLine();
        // status on the right, timeline in between
        f64 rate = ui->hist_count ? ui->hist_rate[(ui->hist_head + UI_HISTORY - 1) % UI_HISTORY] : 0.0;
        char status[96], rbuf[32];
        snprintf(status, sizeof(status), "%s  %.0f fps", ui_fmt_rate(rate, rbuf, sizeof(rbuf)),
                 ui->frame_ms_smooth > 0 ? 1000.0f / ui->frame_ms_smooth : 0.0f);
        f32 status_w = ImGui::CalcTextSize(status).x + ui_s(16);
        f32 avail = ImGui::GetContentRegionAvail().x - status_w;
        ui_timeline_bar(state, memory, Maximum(avail, ui_s(60)));
        ImGui::SameLine(ImGui::GetWindowWidth() - status_w);
        ImGui::AlignTextToFramePadding();
        ImGui::TextDisabled("%s", status);
        (void)stats;
    }
    ImGui::End();
    ImGui::PopStyleColor();
}
