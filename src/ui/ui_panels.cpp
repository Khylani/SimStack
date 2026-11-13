// Simstack UI panels. Included by simstack.cpp after SimState is defined.
//
//   ui_frame_begin  before the fixed steps: input, NewFrame, every window
//                   (edits apply to the scene before it steps this frame)
//   ui_frame_end    after the fixed steps: Render + emit render commands
//
// Plots show the state as of the previous frame's steps (one frame behind).

// Default layout: controls on the left; on the right the 3D view (the
// dockspace's central node) on top and plots / performance as tabs below.
// Everything can be re-docked by dragging; the layout survives hot reloads.
internal void ui_build_layout(ImGuiID dock, ImVec2 size) {
    ImGui::DockBuilderRemoveNode(dock);
    ImGui::DockBuilderAddNode(dock, (ImGuiDockNodeFlags)ImGuiDockNodeFlags_DockSpace | (ImGuiDockNodeFlags)ImGuiDockNodeFlags_PassthruCentralNode);
    ImGui::DockBuilderSetNodeSize(dock, size);
    f32 left_frac = Clamp(360.0f / Maximum(size.x, 1.0f), 0.2f, 0.45f);
    ImGuiID left, rest, bottom, center;
    ImGui::DockBuilderSplitNode(dock, ImGuiDir_Left, left_frac, &left, &rest);
    ImGui::DockBuilderSplitNode(rest, ImGuiDir_Down, 0.66f, &bottom, &center);
    ImGui::DockBuilderDockWindow("Controls", left);
    ImGui::DockBuilderDockWindow("Plots", bottom);
    ImGui::DockBuilderDockWindow("Performance", bottom);
    ImGui::DockBuilderFinish(dock);
}

#include "ui_nft.cpp"

internal void ui_switch_scene(SimState *state, SimMemory *memory, u32 scene, u32 zones) {
    memory->config.scene = scene;
    memory->config.zone_count = zones;
    state->scene = scene;
    sim_reset_scene(state, memory);
}

// --- Controls -----------------------------------------------------------------------

internal void ui_controls(SimState *state, SimMemory *memory, SimStats *stats) {
    UiState *ui = &state->ui;
    DbdScene *scene = state->dbd;
    if (!ImGui::Begin("Controls")) { ImGui::End(); return; }

    // Scene
    const char *scenes[] = { "Particles (M1 demo)", "DBD reactor (plug flow)", "Operating-point sweep", "NFT system" };
    s32 current = (s32)state->scene;
    ImGui::SetNextItemWidth(-1);
    if (ImGui::Combo("##scene", &current, scenes, ArrayCount(scenes)) && (u32)current != state->scene) {
        u32 zones = current == SimScene_Sweep ? 1024 : current == SimScene_Nft ? 16 : 64;
        ui_switch_scene(state, memory, (u32)current, zones);
        scene = state->dbd;
    }
    if (state->nft) {
        ui_nft_controls(state, memory);
        ImGui::End();
        return;
    }

    // Run control
    if (ImGui::Button(state->paused ? "  Run  " : "Pause")) state->paused = !state->paused;
    ImGui::SameLine();
    if (ImGui::Button("Step 1 ms")) { ui->step_once = 1; state->paused = 1; }
    ImGui::SameLine();
    if (ImGui::Button("Reset")) sim_reset_scene(state, memory);
    ImGui::SetNextItemWidth(-90);
    ImGui::BeginDisabled(ui->max_speed);
    ImGui::SliderFloat("##speed", &ui->speed, 0.01f, 1000.0f, "%.2fx real time", ImGuiSliderFlags_Logarithmic);
    ImGui::EndDisabled();
    ImGui::SameLine();
    bool max_speed = ui->max_speed != 0;
    if (ImGui::Checkbox("Max", &max_speed)) ui->max_speed = max_speed;
    ui_help("Speed asks for simulated seconds per wall second. The chemistry gets at most "
            "~12 ms of wall time per frame; if it cannot keep up, simulated time runs slower "
            "and the frame rate stays smooth. Max spends the whole budget every frame.");

    f64 rate = 0.0;
    if (ui->hist_count) rate = ui->hist_rate[(ui->hist_head + UI_HISTORY - 1) % UI_HISTORY];
    ImGui::Text("t = %.3f s   %.2f sim-s/s   %llu steps", state->sim_time, rate, (unsigned long long)state->step_count);

    if (!scene) {
        ImGui::Separator();
        ImGui::TextWrapped("The particle scene is the milestone-1 demo. Pick the reactor or the sweep above.");
        ImGui::End();
        return;
    }

    // Domain size
    if (ImGui::CollapsingHeader("Zones", ImGuiTreeNodeFlags_DefaultOpen)) {
        s32 zones = (s32)scene->zones.count;
        ImGui::SetNextItemWidth(-90);
        ImGui::InputInt("##zones", &zones, scene->chained ? 8 : 64, scene->chained ? 64 : 1024);
        ImGui::SameLine();
        zones = Clamp(zones, 8, 65536);
        if (ImGui::Button("Apply") && (u32)zones != scene->zones.count) {
            ui_switch_scene(state, memory, state->scene, (u32)zones);
            scene = state->dbd;
        }
        if (scene->chained) ImGui::TextDisabled("%u axial zones along the discharge tube", scene->zones.count);
        else ImGui::TextDisabled("%u x %u batch reactors (power x humidity)", scene->sweep_nx, scene->sweep_ny);
    }

    // Solver
    if (ImGui::CollapsingHeader("Solver", ImGuiTreeNodeFlags_DefaultOpen)) {
        s32 backend = (s32)scene->backend;
        ImGui::RadioButton("CPU", &backend, SimBackend_Cpu);
        ImGui::BeginDisabled(!memory->gpu.available);
        ImGui::SameLine();
        ImGui::RadioButton("CUDA", &backend, SimBackend_Gpu);
        ImGui::SameLine();
        ImGui::RadioButton("Hybrid", &backend, SimBackend_Hybrid);
        ImGui::EndDisabled();
        ui_help("Hybrid runs CUDA and the CPU work queue at the same time on disjoint zones, split by their "
                "measured throughput. Both perform identical arithmetic, so results are bit-identical to either "
                "device alone. Start with --gpu (built with ./build.sh --cuda) to enable CUDA.");
        if ((u32)backend != scene->backend) {
            scene->backend = (u32)backend;
            scene->gpu_ready = 0;
        }
        if (scene->backend == SimBackend_Hybrid)
            ImGui::TextDisabled("GPU %u / %u zones (step %.2f ms: CPU %.2f, GPU %.2f)", scene->hybrid.split, scene->zones.padded,
                                scene->hybrid.step_ms[scene->hybrid.split / 8], scene->hybrid.last_cpu_ms, scene->hybrid.last_gpu_ms);
        s32 precision = (s32)scene->precision;
        ImGui::RadioButton("f32 lanes", &precision, DbdPrecision_F32);
        ImGui::SameLine();
        ImGui::RadioButton("f64 lanes", &precision, DbdPrecision_F64);
        ui_help("f32: 8 zones per AVX2 register and the GPU's FP32 units; increments over the f64 "
                "state (products within ~1e-6 of f64). f64: the reference, exact atom bookkeeping "
                "for long runs.");
        scene->precision = (u32)precision;

        const char *methods[] = { "Ros2 (2 stages, order 2)", "Ros3 (3 stages, order 3)", "Rodas3 (4 stages, stiffly accurate)" };
        s32 method = (s32)scene->method_id;
        ImGui::SetNextItemWidth(-1);
        if (ImGui::Combo("##method", &method, methods, ArrayCount(methods))) {
            scene->method_id = (KinMethodId)method;
            scene->method = kin_method(scene->method_id);
            scene->gpu_ready = 0;
        }
        ImGui::SetNextItemWidth(-90);
        if (ui_slider_log("rtol", &scene->control.rtol, 1e-6, 1e-1, "%.0e")) scene->gpu_ready = 0;
        ImGui::SetNextItemWidth(-90);
        if (ui_slider_log("atol cm^-3", &scene->control.atol, 1.0, 1e8, "%.0e")) scene->gpu_ready = 0;
        ImGui::TextDisabled("%s kernels, %u threads", scene->generated ? "generated straight-line" : "table-driven",
                            memory->worker_count);
    }

    // Discharge / operating point
    if (ImGui::CollapsingHeader("Discharge", ImGuiTreeNodeFlags_DefaultOpen)) {
        s32 mode = (s32)scene->mode;
        ImGui::RadioButton("Pulsed", &mode, DbdDischarge_Pulsed);
        ImGui::SameLine();
        ImGui::RadioButton("Averaged", &mode, DbdDischarge_Averaged);
        ui_help("Pulsed: one radical burst per AC half-cycle, integrated exactly. Averaged: the same "
                "energy as a continuous source; ~100x cheaper, near-identical long-lived products at kHz.");
        scene->mode = (DbdDischargeMode)mode;

        DbdOperating *op = &scene->op;
        b32 changed = 0;
        ImGui::PushItemWidth(-110);
        if (scene->chained) {
            changed |= ui_slider_log("Power W/cm^3", &op->power_W_cm3, 0.001, 2.0, "%.3f");
            f32 rh = (f32)(op->x_H2O * 100.0);
            if (ImGui::SliderFloat("H2O %", &rh, 0.0f, 5.0f, "%.2f")) { op->x_H2O = rh / 100.0f; changed = 1; }
        }
        changed |= ui_slider_log("Frequency Hz", &op->freq_Hz, 50.0, 100000.0, "%.0f");
        f32 T = (f32)op->T_K;
        if (ImGui::SliderFloat("Gas T K", &T, 270.0f, 500.0f, "%.0f")) { op->T_K = T; changed = 1; }
        f32 p_kPa = (f32)(op->p_Pa / 1000.0);
        if (ImGui::SliderFloat("Pressure kPa", &p_kPa, 20.0f, 200.0f, "%.1f")) { op->p_Pa = p_kPa * 1000.0f; changed = 1; }
        s32 chem_ms = (s32)scene->chem_steps;
        ImGui::SetNextItemWidth(-110);
        if (ImGui::SliderInt("Chem step ms", &chem_ms, 1, 50)) scene->chem_steps = (u32)chem_ms;
        ui_help("Chemistry advances (and, in the reactor, plug-flow exchanges) happen every N master "
                "steps of 1 ms. Inside a step the solver chooses its own sub-steps; larger N lets it take "
                "long steps in quasi-steady chemistry and cuts per-step overhead (and GPU transfers).");
        f64 min_res = (f64)scene->zones.count * state->dt_fixed * (f64)scene->chem_steps * 1.01;
        if (scene->chained && scene->residence_s < min_res) scene->residence_s = min_res;
        if (ui_slider_log(scene->chained ? "Residence s" : "Duration s", &scene->residence_s, Maximum(0.05, min_res), 60.0, "%.2f")) {
            scene->residence_s = Maximum(scene->residence_s, min_res);
        }
        ImGui::PopItemWidth();
        if (changed) dbd_scene_apply_operating(scene);
        if (scene->chained) {
            f64 sei = op->power_W_cm3 * scene->residence_s * 1000.0;
            ImGui::TextDisabled("specific energy %.0f J/L, %.1f pulses per ms", sei, 2.0 * op->freq_Hz * 1e-3);
        } else if (scene->finished) {
            ImGui::TextColored(ImVec4(0.5f, 0.9f, 0.5f, 1), "sweep finished");
            ImGui::SameLine();
            if (ImGui::SmallButton("restart")) sim_reset_scene(state, memory);
        }
    }

    if (ImGui::CollapsingHeader("Electron-impact yields")) {
        ImGui::PushItemWidth(-110);
        f32 g[3] = { (f32)scene->source.g_O_per_100eV, (f32)scene->source.g_N_per_100eV, (f32)scene->source.g_OH_per_100eV };
        if (ImGui::SliderFloat("G(O) /100eV", &g[0], 0.0f, 12.0f, "%.2f")) scene->source.g_O_per_100eV = g[0];
        if (ImGui::SliderFloat("G(N) /100eV", &g[1], 0.0f, 4.0f, "%.2f")) scene->source.g_N_per_100eV = g[1];
        if (ImGui::SliderFloat("G(OH) /100eV", &g[2], 0.0f, 4.0f, "%.2f")) scene->source.g_OH_per_100eV = g[2];
        ImGui::PopItemWidth();
        ImGui::TextDisabled("radicals per 100 eV at reference air; surrogate until\nthe ChemPlasKin reduction tables exist");
    }

    // Readout
    if (ImGui::CollapsingHeader(scene->chained ? "Outlet" : "Corner zone (max power, wettest)", ImGuiTreeNodeFlags_DefaultOpen)) {
        DbdZones *z = &scene->zones;
        if (ImGui::BeginTable("##outlet", 3, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp)) {
            for (u32 p = 0; p < DBD_PLOT_SPECIES; ++p) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::ColorButton("##c", ui_v4(dbd_plot_species[p].color), ImGuiColorEditFlags_NoTooltip, ImVec2(10, 10));
                ImGui::SameLine();
                ImGui::TextUnformatted(dbd_plot_species[p].name);
                ImGui::TableNextColumn();
                if (ui->mono) ImGui::PushFont(ui->mono, 0.0f);
                ImGui::Text("%12.4g", dbd_ppm(z, scene->plot_index[p], z->count - 1));
                if (ui->mono) ImGui::PopFont();
                ImGui::TableNextColumn();
                ImGui::TextDisabled("ppm");
            }
            ImGui::EndTable();
        }
        ImGui::SetNextItemWidth(-80);
        ImGui::InputText("##export", ui->export_path, sizeof(ui->export_path));
        ImGui::SameLine();
        if (ImGui::Button("Export")) {
            const char *saved = scene->csv_path;
            scene->csv_path = ui->export_path;   // UiState is stable memory
            dbd_scene_write_csv(scene, &memory->platform, &state->frame);
            scene->csv_path = saved;
        }
    }
    (void)stats;
    ImGui::End();
}

// --- Plots ----------------------------------------------------------------------------

internal void ui_plots(SimState *state) {
    UiState *ui = &state->ui;
    DbdScene *scene = state->dbd;
    if (!ImGui::Begin("Plots")) { ImGui::End(); return; }
    if (state->nft) { ui_nft_plots(state); ImGui::End(); return; }
    if (!scene) { ImGui::TextDisabled("no chemistry in this scene"); ImGui::End(); return; }
    DbdZones *z = &scene->zones;

    if (ImGui::BeginTabBar("##plots")) {
        if (ImGui::BeginTabItem(scene->chained ? "Outlet history" : "Corner history")) {
            if (ImPlot::BeginPlot("##history", ImVec2(-1, -1))) {
                ImPlot::SetupAxes("t [s]", "mole fraction [ppm]", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
                ImPlot::SetupAxisScale(ImAxis_Y1, ImPlotScale_Log10);
                ImPlot::SetupLegend(ImPlotLocation_SouthEast);
                if (scene->series_count > 1) {
                    for (u32 p = 0; p < DBD_PLOT_SPECIES; ++p) {
                        ImPlotSpec spec;
                        spec.LineColor = ui_v4(dbd_plot_species[p].color);
                        spec.LineWeight = 2.0f;
                        spec.Offset = (int)scene->series_head;
                        spec.Stride = (int)(sizeof(f64) * (1 + DBD_PLOT_SPECIES));
                        ImPlot::PlotLine(dbd_plot_species[p].name, scene->series, scene->series + 1 + p,
                                         (int)scene->series_count, spec);
                    }
                }
                ImPlot::EndPlot();
            }
            ImGui::EndTabItem();
        }
        if (scene->chained && ImGui::BeginTabItem("Axial profile")) {
            ImGui::RadioButton("residence time", &ui->profile_x_time, 1);
            ImGui::SameLine();
            ImGui::RadioButton("zone index", &ui->profile_x_time, 0);
            TempMemory temp = begin_temp(&state->frame);
            f64 *xs = push_array_nozero(&state->frame, f64, z->count);
            f64 *ys = push_array_nozero(&state->frame, f64, z->count);
            for (u32 i = 0; i < z->count; ++i)
                xs[i] = ui->profile_x_time ? ((f64)i + 1.0) / (f64)z->count * scene->residence_s : (f64)i;
            if (ImPlot::BeginPlot("##profile", ImVec2(-1, -1))) {
                ImPlot::SetupAxes(ui->profile_x_time ? "residence time [s]" : "zone", "mole fraction [ppm]",
                                  ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
                ImPlot::SetupAxisScale(ImAxis_Y1, ImPlotScale_Log10);
                ImPlot::SetupLegend(ImPlotLocation_SouthEast);
                for (u32 p = 0; p < DBD_PLOT_SPECIES; ++p) {
                    for (u32 i = 0; i < z->count; ++i) ys[i] = dbd_ppm(z, scene->plot_index[p], i);
                    ImPlotSpec spec;
                    spec.LineColor = ui_v4(dbd_plot_species[p].color);
                    spec.LineWeight = 2.0f;
                    if (z->count <= 128) { spec.Marker = ImPlotMarker_Circle; spec.MarkerSize = 2.5f; }
                    ImPlot::PlotLine(dbd_plot_species[p].name, xs, ys, (int)z->count, spec);
                }
                ImPlot::EndPlot();
            }
            end_temp(temp);
            ImGui::EndTabItem();
        }
        if (!scene->chained && ImGui::BeginTabItem("Sweep map")) {
            const char *names[DBD_PLOT_SPECIES];
            for (u32 p = 0; p < DBD_PLOT_SPECIES; ++p) names[p] = dbd_plot_species[p].name;
            ImGui::SetNextItemWidth(120);
            ImGui::Combo("species", &ui->heat_species, names, DBD_PLOT_SPECIES);
            ImGui::SameLine();
            ImGui::TextDisabled("log10 ppm after %.2f s", scene->t);
            u32 nx = scene->sweep_nx, ny = scene->sweep_ny;
            TempMemory temp = begin_temp(&state->frame);
            f64 *vals = push_array_nozero(&state->frame, f64, (u64)nx * ny);
            f64 lo = 1e300, hi = -1e300;
            u32 s = scene->plot_index[ui->heat_species];
            for (u32 iy = 0; iy < ny; ++iy)
                for (u32 ix = 0; ix < nx; ++ix) {
                    f64 ppm = dbd_ppm(z, s, iy * nx + ix);
                    f64 v = log10(ppm > 1e-6 ? ppm : 1e-6);
                    vals[(u64)(ny - 1 - iy) * nx + ix] = v;   // heatmap row 0 is the top: wettest first
                    lo = Minimum(lo, v); hi = Maximum(hi, v);
                }
            if (hi <= lo) hi = lo + 1.0;
            ImPlot::PushColormap(ImPlotColormap_Viridis);
            f32 avail = ImGui::GetContentRegionAvail().x;
            if (ImPlot::BeginPlot("##heat", ImVec2(avail - 90, -1), ImPlotFlags_NoLegend)) {
                ImPlot::SetupAxes("log10 power [W/cm^3]", "H2O mole fraction", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
                f64 x0 = log10(0.01), x1 = log10(0.5);
                ImPlot::PlotHeatmap("##map", vals, (int)ny, (int)nx, lo, hi, nx * ny <= 256 ? "%.1f" : 0,
                                    ImPlotPoint(x0, 0.0), ImPlotPoint(x1, 0.04));
                ImPlot::EndPlot();
            }
            ImGui::SameLine();
            ImPlot::ColormapScale("##scale", lo, hi, ImVec2(80, -1), "%.1f");
            ImPlot::PopColormap();
            end_temp(temp);
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
    ImGui::End();
}

// --- Performance ------------------------------------------------------------------------

internal void ui_performance(SimState *state, SimMemory *memory, SimStats *stats) {
    UiState *ui = &state->ui;
    if (!ImGui::Begin("Performance")) { ImGui::End(); return; }
    if (ImGui::BeginTable("##perf", 2, ImGuiTableFlags_Resizable | ImGuiTableFlags_BordersInnerV)) {
        ImGui::TableNextColumn();
        if (ImPlot::BeginPlot("##frame", ImVec2(-1, -1))) {
            ImPlot::SetupAxes("frame", "ms", ImPlotAxisFlags_NoTickLabels, ImPlotAxisFlags_AutoFit);
            ImPlot::SetupAxisLimits(ImAxis_X1, 0, UI_HISTORY, ImPlotCond_Always);
            ImPlot::SetupAxis(ImAxis_Y2, "sim-s / wall-s", ImPlotAxisFlags_AuxDefault | ImPlotAxisFlags_AutoFit);
            ImPlot::SetupLegend(ImPlotLocation_NorthWest);
            ImPlotSpec a; a.Offset = (int)(ui->hist_count == UI_HISTORY ? ui->hist_head : 0);
            ImPlotSpec fill = a; fill.FillAlpha = 0.25f;
            ImPlot::PlotShaded("chemistry ms", ui->hist_chem_ms, (int)ui->hist_count, 0.0, 1.0, 0.0, fill);
            ImPlot::PlotLine("frame ms", ui->hist_frame_ms, (int)ui->hist_count, 1.0, 0.0, a);
            ImPlot::SetAxes(ImAxis_X1, ImAxis_Y2);
            ImPlot::PlotLine("speed", ui->hist_rate, (int)ui->hist_count, 1.0, 0.0, a);
            ImPlot::EndPlot();
        }
        ImGui::TableNextColumn();
        DbdScene *scene = state->dbd;
        if (state->nft) {
            NftSystem *ns = &state->nft->sys;
            ImGui::Text("NFT: %u groups of 8 channels x %u cells, %u substeps / window", ns->groups, ns->cells, ns->substeps);
            ImGui::Text("%.3f ms per 60 s window, %.0f simulated s per wall s", state->nft->last_window_wall_ms,
                        state->nft->last_window_wall_ms > 0 ? 60.0 / (state->nft->last_window_wall_ms * 1e-3) : 0.0);
            ImGui::Text("chemistry %.1f of %.1f ms budget, frame %.1f ms (%.0f fps)", ui->chem_ms_frame, ui->chem_budget_ms,
                        ui->frame_ms_smooth, ui->frame_ms_smooth > 0 ? 1000.0f / ui->frame_ms_smooth : 0.0f);
        }
        if (scene) {
            DbdZones *z = &scene->zones;
            f64 steps = 0.0, failed = 0.0;
            for (u32 i = 0; i < z->count; ++i) { steps += z->steps[i]; failed += z->failed[i]; }
            f64 ms = scene->t * 1000.0;
            ImGui::Text("%s, %s lanes, %s", scene->backend == SimBackend_Hybrid ? "CUDA + CPU" : scene->backend == SimBackend_Gpu ? "CUDA" : "CPU",
                        scene->precision == DbdPrecision_F32 ? "f32" : "f64", scene->method.name);
            ImGui::Text("%.0f steps / zone / ms   %.3f ms wall / step", ms > 0 ? steps / z->count / ms : 0.0,
                        scene->last_step_wall_ms);
            ImGui::Text("chemistry %.1f of %.1f ms budget, frame %.1f ms (%.0f fps)", ui->chem_ms_frame, ui->chem_budget_ms,
                        ui->frame_ms_smooth, ui->frame_ms_smooth > 0 ? 1000.0f / ui->frame_ms_smooth : 0.0f);
            bool auto_fps = !ui->manual_fps;
            if (ImGui::Checkbox("auto", &auto_fps)) ui->manual_fps = !auto_fps;
            ImGui::SameLine();
            ImGui::SetNextItemWidth(-90);
            ImGui::BeginDisabled(auto_fps);
            ImGui::SliderFloat("target fps", &ui->target_fps, 10.0f, 240.0f, "%.0f");
            ImGui::EndDisabled();
            ui_help("The chemistry runs as a background job while the frame is rendered and swapped, "
                    "and gets 90 % of the target frame time minus the UI work. Auto follows the display "
                    "rate (vsync), so the solver uses all idle time without dropping frames.");
            if (failed > 0) ImGui::TextColored(ImVec4(1, 0.4f, 0.3f, 1), "%.0f zones failed (max_steps)", failed);
        }
        ui_bar("permanent", stats->permanent_used, stats->permanent_size);
        ui_bar("frame", state->frame.used, state->frame.size);
        ui_bar("ui heap", ui->heap.arena.used, ui->heap.arena.size);
        if (memory->gpu.available) ui_bar("gpu arena", memory->gpu.gpu_arena_used(), memory->gpu.info.arena_bytes);
        DebugTable *table = memory->debug;
        if (table && table->cycles_per_ms > 0 &&
            ImGui::BeginTable("##prof", 3, ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY | ImGuiTableFlags_SizingStretchProp,
                              ImVec2(-1, -1))) {
            ImGui::TableSetupColumn("timed block");
            ImGui::TableSetupColumn("hits");
            ImGui::TableSetupColumn("ms");
            ImGui::TableHeadersRow();
            for (u32 i = 0; i < DEBUG_MAX_RECORDS; ++i) {
                DebugRecord *r = table->records + i;
                if (!r->name || !r->last_hits) continue;
                ImGui::TableNextRow();
                ImGui::TableNextColumn(); ImGui::TextUnformatted(r->name);
                ImGui::TableNextColumn(); ImGui::Text("%u", r->last_hits);
                ImGui::TableNextColumn(); ImGui::Text("%.3f", (f64)r->last_cycles / table->cycles_per_ms);
            }
            ImGui::EndTable();
        }
        ImGui::EndTable();
    }
    ImGui::End();
}

// --- frame ----------------------------------------------------------------------------

internal void ui_frame_begin(SimState *state, SimMemory *memory, SimInput *input, RenderCommands *render, SimStats *stats) {
    UiState *ui = &state->ui;
    ui_init_memory(ui, memory);
    ui_ensure_context(ui);
    ui_feed_input(input);
    ImGui::NewFrame();

    ImGuiViewport *vp = ImGui::GetMainViewport();
    ImGuiID dock = ImGui::DockSpaceOverViewport(0, vp, ImGuiDockNodeFlags_PassthruCentralNode);
    b32 fresh_layout = !ui->layout_built;
    if (!ui->layout_built) {
        ui_build_layout(dock, vp->Size);
        ui->layout_built = 1;
    }
    ImGuiDockNode *central = ImGui::DockBuilderGetCentralNode(dock);
    if (central) {
        render->scene_x = central->Pos.x - vp->Pos.x;
        render->scene_y = central->Pos.y - vp->Pos.y;
        render->scene_w = central->Size.x;
        render->scene_h = central->Size.y;
    }

    // Camera: drag to orbit, wheel to zoom, when the mouse is over the 3D view.
    ImGuiIO &io = ImGui::GetIO();
    b32 over_view = central && !io.WantCaptureMouse && input->mouse_inside &&
                    input->mouse_x >= render->scene_x && input->mouse_x < render->scene_x + render->scene_w &&
                    input->mouse_y >= render->scene_y && input->mouse_y < render->scene_y + render->scene_h;
    if (over_view && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) ui->dragging = 1;
    if (!ImGui::IsMouseDown(ImGuiMouseButton_Left)) ui->dragging = 0;
    if (ui->dragging) {
        state->camera.yaw += io.MouseDelta.x * 0.008f;
        state->camera.pitch = Clamp(state->camera.pitch + io.MouseDelta.y * 0.008f, -1.4f, 1.4f);
        state->camera.auto_orbit = 0;
    }
    if (over_view && input->mouse_wheel != 0.0f)
        state->camera.distance = Clamp(state->camera.distance * powf(0.9f, input->mouse_wheel), 1.0f, 30.0f);

    ui_controls(state, memory, stats);
    ui_performance(state, memory, stats);
    if (fresh_layout) ImGui::SetNextWindowFocus();   // Plots is the front tab of the shared node
    ui_plots(state);
}

// 3D view overlay: species picker, colour bar, inlet/outlet labels projected
// with this frame's camera.
internal void ui_view_overlay(SimState *state, RenderCommands *render) {
    UiState *ui = &state->ui;
    DbdScene *scene = state->dbd;
    if (render->scene_w <= 0) return;
    ImGuiViewport *vp = ImGui::GetMainViewport();
    ImVec2 origin = ImVec2(vp->Pos.x + render->scene_x, vp->Pos.y + render->scene_y);
    if (state->nft) { ui_nft_overlay(state, render, origin); return; }
    if (!scene) return;
    ImGui::SetNextWindowPos(ImVec2(origin.x + 10, origin.y + 10));
    ImGui::SetNextWindowBgAlpha(0.55f);
    ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings |
                             ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoMove;
    if (ImGui::Begin("##view_overlay", 0, flags)) {
        const char *names[DBD_PLOT_SPECIES];
        for (u32 p = 0; p < DBD_PLOT_SPECIES; ++p) names[p] = dbd_plot_species[p].name;
        ImGui::SetNextItemWidth(90);
        ImGui::Combo("##viewspecies", &ui->view_species, names, DBD_PLOT_SPECIES);
        ImGui::SameLine();
        ImGui::TextDisabled(scene->chained ? "along the tube, log10 ppm" : "over power x humidity, log10 ppm");
        ImDrawList *dl = ImGui::GetWindowDrawList();
        ImVec2 p0 = ImGui::GetCursorScreenPos();
        f32 w = 260.0f, h = 12.0f;
        const u32 segments = 32;
        for (u32 i = 0; i < segments; ++i) {
            ImU32 c0 = dbd_colormap((f32)i / segments), c1 = dbd_colormap((f32)(i + 1) / segments);
            dl->AddRectFilledMultiColor(ImVec2(p0.x + w * i / segments, p0.y), ImVec2(p0.x + w * (i + 1) / segments, p0.y + h),
                                        c0, c1, c1, c0);
        }
        ImGui::Dummy(ImVec2(w, h));
        ImGui::Text("%-10.3g", pow(10.0, scene->view_lo));
        ImGui::SameLine(w - 60);
        ImGui::Text("%10.3g ppm", pow(10.0, scene->view_hi));
    }
    ImGui::End();

    // labels at projected 3D positions
    if (scene->chained) {
        ImDrawList *fg = ImGui::GetForegroundDrawList();
        struct { v3 p; const char *text; } labels[] = { { V3(-1.6f, 0.55f, 0.0f), "inlet" }, { V3(1.6f, 0.55f, 0.0f), "outlet" } };
        for (u32 i = 0; i < ArrayCount(labels); ++i) {
            v4 clip = render->view_proj * V4(labels[i].p.x, labels[i].p.y, labels[i].p.z, 1.0f);
            if (clip.w <= 0.0f) continue;
            f32 nx = clip.x / clip.w, ny = clip.y / clip.w;
            ImVec2 pos = ImVec2(origin.x + (nx * 0.5f + 0.5f) * render->scene_w, origin.y + (0.5f - ny * 0.5f) * render->scene_h);
            fg->AddText(pos, IM_COL32(220, 225, 235, 220), labels[i].text);
        }
    }
}

internal void ui_frame_end(SimState *state, SimInput *input, RenderCommands *render) {
    UiState *ui = &state->ui;
    ui_view_overlay(state, render);
    // history ring
    f32 frame_ms = input->dt_frame * 1000.0f;
    f64 advanced = state->sim_time - ui->last_sim_time;
    ui->last_sim_time = state->sim_time;
    u32 slot = ui->hist_head;
    ui->hist_frame_ms[slot] = frame_ms;
    ui->hist_chem_ms[slot] = ui->chem_ms_frame;
    ui->hist_rate[slot] = input->dt_frame > 0 ? (f32)(advanced / input->dt_frame) : 0.0f;
    ui->hist_head = (ui->hist_head + 1) % UI_HISTORY;
    if (ui->hist_count < UI_HISTORY) ++ui->hist_count;

    ImGui::Render();
    ui_emit(render, &state->frame);
}
