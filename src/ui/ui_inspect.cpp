// Inspection: picking points in the 3D view (hover tooltip, click to pin,
// double-click to focus), the Inspector panel, and view decorations (axes
// gizmo, mouse hints). Included by ui_panels.cpp.

struct UiPointSet { const f32 *x, *y, *z; u32 count; u32 kind; };
enum UiPickKind { UiPick_None, UiPick_Particle, UiPick_DbdZone, UiPick_SweepZone, UiPick_NftCell, UiPick_NftPlant };

internal u32 ui_point_sets(SimState *state, UiPointSet *sets) {
    u32 n = 0;
    if (state->nft) {
        NftScene *sc = state->nft;
        if (state->ui.show_cells) sets[n++] = { sc->cx, sc->cy, sc->cz, sc->view_cells, UiPick_NftCell };
        if (state->ui.show_plants) sets[n++] = { sc->px, sc->py, sc->pz, sc->view_sites, UiPick_NftPlant };
    } else if (state->dbd) {
        DbdScene *d = state->dbd;
        sets[n++] = { d->view_x, d->view_y, d->view_z, d->view_count, d->chained ? (u32)UiPick_DbdZone : (u32)UiPick_SweepZone };
    } else {
        sets[n++] = { state->particles.x, state->particles.y, state->particles.z, state->particles.count, UiPick_Particle };
    }
    return n;
}

// Nearest point to the cursor in screen space (within `radius` px), ties to the nearer depth.
internal UiPick ui_pick(SimState *state, f32 mx, f32 my, f32 radius) {
    UiState *ui = &state->ui;
    UiPick best = {};
    if (ui->last_scene_w <= 0) return best;
    UiPointSet sets[4];
    u32 ns = ui_point_sets(state, sets);
    f32 best_d2 = radius * radius, best_w = 1e30f;
    m4 m = ui->last_view_proj;
    for (u32 s = 0; s < ns; ++s) {
        UiPointSet *ps = sets + s;
        u32 stride = ps->count > 300000 ? ps->count / 300000 + 1 : 1;   // big systems: sampled pick
        for (u32 i = 0; i < ps->count; i += stride) {
            v4 c = m * V4(ps->x[i], ps->y[i], ps->z[i], 1.0f);
            if (c.w <= 1e-4f) continue;
            f32 sx = ui->last_scene_x + (c.x / c.w * 0.5f + 0.5f) * ui->last_scene_w;
            f32 sy = ui->last_scene_y + (0.5f - c.y / c.w * 0.5f) * ui->last_scene_h;
            f32 dx = sx - mx, dy = sy - my, d2 = dx * dx + dy * dy;
            if (d2 < best_d2 * 0.999f || (d2 <= best_d2 && c.w < best_w)) {
                best_d2 = d2; best_w = c.w;
                best.valid = 1; best.kind = ps->kind; best.index = i;
                best.pos = V3(ps->x[i], ps->y[i], ps->z[i]);
            }
        }
    }
    return best;
}

// --- descriptions ---------------------------------------------------------------------------

internal void ui_kv(const char *k, const char *fmt, ...) {
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    ImGui::TextDisabled("%s", k);
    ImGui::TableNextColumn();
    char buf[128];
    va_list ap; va_start(ap, fmt); vsnprintf(buf, sizeof(buf), fmt, ap); va_end(ap);
    ImGui::TextUnformatted(buf);
}

internal const char *ui_pick_title(SimState *state, UiPick *p, char *buf, u32 n) {
    switch (p->kind) {
        case UiPick_NftCell: {
            NftSystem *s = &state->nft->sys;
            u32 r = p->index / s->cells, j = p->index % s->cells;
            snprintf(buf, n, "Channel %u (bank %c), %.2f m from the inlet", r + 1, nft_rail_bank(s, r) ? 'B' : 'A', (j + 0.5) * s->cell_m);
        } break;
        case UiPick_NftPlant: {
            NftSystem *s = &state->nft->sys;
            u32 r = p->index / s->sites, j = p->index % s->sites;
            const NftCrop *c = nft_crops + s->params.bank[nft_rail_bank(s, r) ? 1 : 0].crop;
            snprintf(buf, n, "%s, channel %u, plant %u", c->name, r + 1, j + 1);
        } break;
        case UiPick_DbdZone: {
            DbdScene *d = state->dbd;
            u32 i = p->index / DBD_TUBE_RING;
            snprintf(buf, n, "Zone %u of %u, residence %.3f s", i + 1, d->zones.count, (i + 1.0) / d->zones.count * d->residence_s);
        } break;
        case UiPick_SweepZone: {
            DbdScene *d = state->dbd;
            u32 i = p->index;
            snprintf(buf, n, "%.3g W/cm3, H2O %.2f %%", d->zones.power_W_cm3[i], 100.0 * d->zones.x_H2O[i]);
        } break;
        case UiPick_Particle: snprintf(buf, n, "Particle %u", p->index); break;
        default: snprintf(buf, n, "-"); break;
    }
    return buf;
}

// Key values (tooltip: `full` = 0) or everything (Inspector: `full` = 1).
internal void ui_pick_table(SimState *state, UiPick *p, b32 full) {
    if (!ImGui::BeginTable("##pick", 2, ImGuiTableFlags_SizingStretchProp | (full ? ImGuiTableFlags_RowBg : 0))) return;
    switch (p->kind) {
        case UiPick_NftCell: {
            NftSystem *s = &state->nft->sys;
            u32 r = p->index / s->cells, j = p->index % s->cells, g = r / NFT_LANES, l = r % NFT_LANES;
            f64 c[NftF_Count];
            for (u32 k = 0; k < NftF_Count; ++k) c[k] = nft_abs(s, g, j, k, l);
            ui_kv("film", "%.2f C, DO %.2f mg/L", c[NftF_T], c[NftF_DO]);
            ui_kv("rise vs inlet", "%+.3f K", (f64)s->group[g].rows[j].f[NftF_T][l]);
            ui_kv("EC / pH", "%.2f mS/cm, pH %.2f", nft_ec_mS(c), nft_cell_ph(s, g, j, l, s->pH));
            if (full) {
                for (u32 i = 0; i < NFT_IONS; ++i) ui_kv(nft_field_names[NFT_ION0 + i], "%.2f mg/L", c[NFT_ION0 + i]);
                NftCellCold *cc = s->group[g].cold + j;
                ui_kv("film depth", "%.2f mm (root fraction %.3f)", cc->depth[l] * 1e3, cc->phi[l]);
                ui_kv("velocity", "%.4f m/s", cc->vel[l]);
                ui_kv("channel flow", "%.3f L/min", s->group[g].lp.flow_L_s[l] * 60.0);
            }
        } break;
        case UiPick_NftPlant: {
            NftSystem *s = &state->nft->sys;
            u32 r = p->index / s->sites, j = p->index % s->sites, g = r / NFT_LANES, l = r % NFT_LANES;
            NftSiteRow *site = s->group[g].sites + j;
            ui_kv("dry mass", "%.3f g (%.0f %% of max)", site->W[l], 100.0 * site->W[l] / s->group[g].crop.wmax[l]);
            ui_kv("age", "%.1f days of %.0f", site->age[l] / 86400.0, s->group[g].crop.cycle_s[l] / 86400.0);
            ui_kv("limits", "fT %.2f  fN %.2f  fDO %.2f", site->fT[l], site->fN[l], site->fDO[l]);
            if (full) {
                ui_kv("water use", "%.1f mL/day", site->transp[l] * 86400.0);
                f64 upN = (site->uptake[NftF_NO3 - NFT_ION0][l] + site->uptake[NftF_NH4 - NFT_ION0][l]) * 86400.0;
                ui_kv("N uptake", "%.1f mg/day", upN);
                ui_kv("K / Ca uptake", "%.1f / %.1f mg/day", site->uptake[NftF_K - NFT_ION0][l] * 86400.0, site->uptake[NftF_Ca - NFT_ION0][l] * 86400.0);
                ui_kv("root O2", "%.2f mg/h", site->o2_rate[l] * 3600.0);
                ui_kv("harvests", "%.0f (%.1f g), %.0f bolting", site->harvests[l], site->harvested_g[l], site->bolted[l]);
                ui_kv("heat dose", "%.1f degree-days toward bolting", site->heat_dd[l]);
            }
        } break;
        case UiPick_DbdZone: case UiPick_SweepZone: {
            DbdScene *d = state->dbd;
            DbdZones *z = &d->zones;
            u32 i = p->kind == UiPick_DbdZone ? p->index / DBD_TUBE_RING : p->index;
            ui_kv("gas", "%.0f K, %.1f kPa, H2O %.2f %%", z->T_K[i], z->p_Pa[i] * 1e-3, 100.0 * z->x_H2O[i]);
            ui_kv("discharge", "%.3g W/cm3 at %.0f Hz", z->power_W_cm3[i], z->freq_Hz[i]);
            u32 n_show = full ? z->n_species : DBD_PLOT_SPECIES;
            for (u32 k = 0; k < n_show; ++k) {
                u32 s = full ? k : d->plot_index[k];
                f64 ppm = dbd_ppm(z, s, i);
                if (!full || ppm > 1e-9) ui_kv(d->model.species_name[s], "%.4g ppm", ppm);
            }
            if (full) ui_kv("solver", "%.0f steps, %.0f rejects%s", z->steps[i], z->rejects[i], z->failed[i] != 0.0 ? ", FAILED" : "");
        } break;
        case UiPick_Particle: {
            ui_kv("position", "%.3f %.3f %.3f", p->pos.x, p->pos.y, p->pos.z);
        } break;
    }
    ImGui::EndTable();
}

internal void ui_inspector(SimState *state) {
    UiState *ui = &state->ui;
    if (!ui_begin_panel("Inspector", &ui->show_inspector)) return;
    UiPick *p = &ui->selected;
    if (!p->valid) {
        ImGui::TextDisabled("Hover the 3D view to inspect, click to pin here,\ndouble-click to focus the camera.");
        ImGui::End();
        return;
    }
    char title[160];
    ImGui::TextUnformatted(ui_pick_title(state, p, title, sizeof(title)));
    ImGui::SameLine();
    if (ImGui::SmallButton("Focus")) { state->camera.target = p->pos; state->camera.auto_orbit = 0; }
    ImGui::SameLine();
    if (ImGui::SmallButton("Clear")) { p->valid = 0; ImGui::End(); return; }
    if (ImGui::BeginTable("##insp", 2, ImGuiTableFlags_Resizable | ImGuiTableFlags_BordersInnerV)) {
        ImGui::TableNextColumn();
        ui_pick_table(state, p, 1);
        ImGui::TableNextColumn();
        // profile through the selection, with a marker at it
        TempMemory temp = begin_temp(&state->frame);
        if (state->nft && (p->kind == UiPick_NftCell || p->kind == UiPick_NftPlant)) {
            NftScene *sc = state->nft;
            NftSystem *s = &sc->sys;
            u32 r = p->kind == UiPick_NftCell ? p->index / s->cells : p->index / s->sites;
            u32 jm = p->kind == UiPick_NftCell ? p->index % s->cells : s->site_offset + (p->index % s->sites) * s->site_stride;
            u32 layer = (u32)Clamp(ui->nft_layer, 0, NftLayer_Count - 1);
            if (nft_layers[layer].plants) layer = NftLayer_Temperature;
            f64 *xs = push_array_nozero(&state->frame, f64, s->cells), *ys = push_array_nozero(&state->frame, f64, s->cells);
            for (u32 j = 0; j < s->cells; ++j) { xs[j] = (j + 0.5) * s->cell_m; ys[j] = nft_layer_value(s, layer, r, j); }
            if (ImPlot::BeginPlot("##insp_profile", ImVec2(-1, -1))) {
                ImPlot::SetupAxes("m from inlet", nft_layers[layer].unit, ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
                ImPlot::PlotLine(nft_layers[layer].name, xs, ys, (int)s->cells);
                f64 mx = xs[jm], my = ys[jm];
                ImPlotSpec mk; mk.Marker = ImPlotMarker_Circle; mk.MarkerSize = 6.0f;
                ImPlot::PlotScatter("selected", &mx, &my, 1, mk);
                ImPlot::EndPlot();
            }
        } else if (state->dbd && p->kind == UiPick_DbdZone) {
            DbdScene *d = state->dbd;
            DbdZones *z = &d->zones;
            u32 i = p->index / DBD_TUBE_RING, sp = d->plot_index[ui->view_species % DBD_PLOT_SPECIES];
            f64 *xs = push_array_nozero(&state->frame, f64, z->count), *ys = push_array_nozero(&state->frame, f64, z->count);
            for (u32 k = 0; k < z->count; ++k) { xs[k] = (k + 1.0) / z->count * d->residence_s; ys[k] = dbd_ppm(z, sp, k); }
            if (ImPlot::BeginPlot("##insp_profile", ImVec2(-1, -1))) {
                ImPlot::SetupAxes("residence time [s]", "ppm", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
                ImPlot::SetupAxisScale(ImAxis_Y1, ImPlotScale_Log10);
                ImPlot::PlotLine(dbd_plot_species[ui->view_species % DBD_PLOT_SPECIES].name, xs, ys, (int)z->count);
                f64 mx = xs[i], my = ys[i];
                ImPlotSpec mk; mk.Marker = ImPlotMarker_Circle; mk.MarkerSize = 6.0f;
                ImPlot::PlotScatter("selected", &mx, &my, 1, mk);
                ImPlot::EndPlot();
            }
        } else {
            ImGui::TextDisabled("(no profile for this element)");
        }
        end_temp(temp);
        ImGui::EndTable();
    }
    ImGui::End();
}

// --- hover / click handling (in ui_frame_begin) ---------------------------------------------

internal void ui_view_interaction(SimState *state, SimMemory *memory, SimInput *input, b32 over_view) {
    UiState *ui = &state->ui;
    ImGuiIO &io = ImGui::GetIO();
    ui->hover.valid = 0;
    if (over_view && ui->dragging == 0) ui->hover = ui_pick(state, input->mouse_x, input->mouse_y, ui_s(14.0f));
    if (memory->config.has_pick && ui->last_scene_w > 0) {   // scripted inspection (--pick)
        f32 mx = ui->last_scene_x + memory->config.pick_x * ui->last_scene_w;
        f32 my = ui->last_scene_y + memory->config.pick_y * ui->last_scene_h;
        ui->hover = ui_pick(state, mx, my, ui_s(40.0f));
        ui->selected = ui->hover;
    }
    // click (press + release without dragging) selects; double-click focuses
    if (over_view && ImGui::IsMouseReleased(ImGuiMouseButton_Left) && ui->drag_moved < 4.0f) {
        UiPick p = ui_pick(state, input->mouse_x, input->mouse_y, ui_s(14.0f));
        ui->selected = p;
        if (p.valid) {
            ui->show_inspector = 1;
            ImGui::SetWindowFocus("Inspector");   // bring its tab to the front
        }
    }
    if (over_view && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left) && ui->selected.valid) {
        state->camera.target = ui->selected.pos;
        state->camera.distance = Maximum(state->camera.distance * 0.6f, 0.3f);
        state->camera.auto_orbit = 0;
    }
    if (ImGui::IsKeyPressed(ImGuiKey_Escape) && !io.WantCaptureKeyboard) ui->selected.valid = 0;
    if (ui->hover.valid && !ui->dragging) {
        char title[160];
        ImGui::BeginTooltip();
        ImGui::TextUnformatted(ui_pick_title(state, &ui->hover, title, sizeof(title)));
        ImGui::Separator();
        ui_pick_table(state, &ui->hover, 0);
        ImGui::TextDisabled("click to pin, double-click to focus");
        ImGui::EndTooltip();
    }
}

// Axes gizmo (bottom-left) and mouse hints (bottom-right) over the 3D view.
internal void ui_view_decorations(SimState *state, RenderCommands *render, ImVec2 origin) {
    UiState *ui = &state->ui;
    ImDrawList *fg = ImGui::GetForegroundDrawList();
    if (ui->show_axes) {
        f32 aspect = render->scene_h > 0 ? render->scene_w / render->scene_h : 1.0f;
        CameraBasis cb = camera_basis(&state->camera, aspect);
        ImVec2 c = ImVec2(origin.x + ui_s(44), origin.y + render->scene_h - ui_s(44));
        f32 len = ui_s(28);
        struct { v3 a; ImU32 col; const char *name; } ax[] = {
            { V3(1, 0, 0), IM_COL32(235, 95, 90, 255), "x" }, { V3(0, 1, 0), IM_COL32(120, 220, 110, 255), "y" },
            { V3(0, 0, 1), IM_COL32(95, 150, 245, 255), "z" },
        };
        fg->AddCircleFilled(c, ui_s(34), IM_COL32(10, 12, 18, 140), 32);
        for (u32 i = 0; i < 3; ++i) {
            f32 sx = dot(ax[i].a, cb.right), sy = dot(ax[i].a, cb.up);
            ImVec2 e = ImVec2(c.x + sx * len, c.y - sy * len);
            fg->AddLine(c, e, ax[i].col, ui_s(2.0f));
            fg->AddText(ImVec2(e.x + ui_s(2), e.y - ui_s(8)), ax[i].col, ax[i].name);
        }
    }
    if (ui->show_hints) {
        const char *hint = "drag orbit  |  right-drag pan  |  wheel zoom  |  F frame  |  click inspect";
        ImVec2 sz = ImGui::CalcTextSize(hint);
        if (sz.x > 0.75f * render->scene_w) hint = "drag orbit | right-drag pan | wheel zoom";
        sz = ImGui::CalcTextSize(hint);
        if (sz.x < 0.75f * render->scene_w) fg->AddText(ImVec2(origin.x + render->scene_w - sz.x - ui_s(12), origin.y + render->scene_h - sz.y - ui_s(10)),
                    IM_COL32(150, 160, 180, 150), hint);
    }
}
