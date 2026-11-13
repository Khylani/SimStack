// UI for the NFT scene: controls (structure, crops, hydraulics, climate,
// reservoir/chiller, nutrients), plots and the 3D layer overlay. Included by
// ui_panels.cpp after SimState is defined.

internal void ui_nft_rebuild(SimState *state, SimMemory *memory) {
    state->nft_params = state->nft->sys.params;   // keep every live edit
    sim_reset_scene(state, memory);
}

internal void ui_nft_apply_live(NftScene *sc) {
    nft_update_lane_params(&sc->sys);   // hydraulics, film, wraps
    nft_update_crop_lanes(&sc->sys);
}

internal const char *ui_duration(f64 seconds, char *buf, u32 size) {
    if (seconds < 120.0) snprintf(buf, size, "%.0f s", seconds);
    else if (seconds < 7200.0) snprintf(buf, size, "%.1f min", seconds / 60.0);
    else if (seconds < 172800.0) snprintf(buf, size, "%.1f h", seconds / 3600.0);
    else snprintf(buf, size, "%.1f days", seconds / 86400.0);
    return buf;
}

internal b32 ui_combo_u32(const char *label, u32 *value, const char *const *names, u32 count) {
    s32 v = (s32)*value;
    b32 changed = ImGui::Combo(label, &v, names, (s32)count);
    if (changed) *value = (u32)v;
    return changed;
}

internal b32 ui_slider_f32(const char *label, f32 *v, f32 lo, f32 hi, const char *fmt, b32 log_scale = 0) {
    return ImGui::SliderFloat(label, v, lo, hi, fmt, log_scale ? ImGuiSliderFlags_Logarithmic : 0);
}

internal void ui_nft_controls(SimState *state, SimMemory *memory) {
    UiState *ui = &state->ui;
    NftScene *sc = state->nft;
    NftSystem *s = &sc->sys;
    NftParams *p = &s->params;
    char buf[64];

    // run
    if (ImGui::Button(state->paused ? "  Run  " : "Pause")) state->paused = !state->paused;
    ImGui::SameLine();
    if (ImGui::Button("Step 1 min")) { ui->step_once = 1; state->paused = 1; }
    ImGui::SameLine();
    if (ImGui::Button("Reset")) ui_nft_rebuild(state, memory);
    ImGui::SetNextItemWidth(-90);
    ImGui::BeginDisabled(ui->max_speed);
    ImGui::SliderFloat("##nftspeed", &ui->nft_speed, 1.0f, 2.6e6f, "%.0fx real time", ImGuiSliderFlags_Logarithmic);
    ImGui::EndDisabled();
    ImGui::SameLine();
    bool max_speed = ui->max_speed != 0;
    if (ImGui::Checkbox("Max", &max_speed)) ui->max_speed = max_speed;
    f64 rate = ui->hist_count ? ui->hist_rate[(ui->hist_head + UI_HISTORY - 1) % UI_HISTORY] : 0.0;
    f64 hour = fmod(s->t / 3600.0, 24.0);
    ImGui::Text("day %.2f  %02d:%02d  %s  %s/s", s->t / 86400.0, (int)hour, (int)((hour - (int)hour) * 60.0),
                s->light > 0.5f ? "lights on" : "lights off", ui_duration(rate, buf, sizeof(buf)));

    // structure
    if (ImGui::CollapsingHeader("System", ImGuiTreeNodeFlags_DefaultOpen)) {
        static s32 channels = 0, plants = 0;
        static f32 length = 0.0f;
        if (!channels) { channels = (s32)s->rails; plants = (s32)s->sites; length = p->rail_length_m; }
        ImGui::PushItemWidth(-130);
        ImGui::InputInt("channels", &channels, 8, 64);
        ui_slider_f32("channel length m", &length, 0.5f, 60.0f, "%.2f", 1);
        ImGui::InputInt("plants / channel", &plants, 1, 10);
        ImGui::PopItemWidth();
        channels = Clamp(channels, 1, 8192); plants = Clamp(plants, 1, NFT_MAX_SITES);
        bool auto_size = p->auto_size != 0;
        if (ImGui::Checkbox("auto-size pump, pipes, tank, chiller", &auto_size)) p->auto_size = auto_size;
        if (ImGui::Button("Rebuild system")) {
            p->rails = (u32)channels; p->plants_per_rail = (u32)plants; p->rail_length_m = length;
            ui_nft_rebuild(state, memory);
            sc = state->nft; s = &sc->sys; p = &s->params;
            channels = 0;
        }
        ImGui::TextDisabled("%u x %.2f m, %u plants at %.0f mm, %u plants total", s->rails, p->rail_length_m, s->sites,
                            1000.0 * s->cell_m * s->site_stride, s->rails * s->sites);
        ImGui::TextDisabled("grid %u cells/channel (dx %.1f mm), %u substeps / min", s->cells, s->cell_m * 1000.0, s->substeps);
    }

    // crops and wraps per bank
    if (ImGui::CollapsingHeader("Crops & channel wrap", ImGuiTreeNodeFlags_DefaultOpen)) {
        const char *crops[NftCrop_Count], *wraps[NftWrap_Count];
        for (u32 i = 0; i < NftCrop_Count; ++i) crops[i] = nft_crops[i].name;
        for (u32 i = 0; i < NftWrap_Count; ++i) wraps[i] = nft_wraps[i].name;
        b32 live = 0;
        for (u32 b = 0; b < 2; ++b) {
            ImGui::PushID((int)b);
            ImGui::SeparatorText(b == 0 ? "Bank A (first half)" : "Bank B (second half)");
            ImGui::PushItemWidth(-130);
            live |= ui_combo_u32("crop", &p->bank[b].crop, crops, NftCrop_Count);
            live |= ui_combo_u32("wrap", &p->bank[b].wrap, wraps, NftWrap_Count);
            f32 grade = p->hyd.slope[b] * 100.0f;
            if (ui_slider_f32("grade %", &grade, 0.2f, 8.0f, "%.1f")) { p->hyd.slope[b] = grade / 100.0f; live = 1; }
            ui_slider_f32("stagger days", &p->bank[b].stagger_days, 0.0f, 120.0f, "%.0f");
            ImGui::PopItemWidth();
            const NftCrop *c = nft_crops + p->bank[b].crop;
            ImGui::TextDisabled("%.0f d cycle, root zone opt %.0f-%.0f C, bolts > %.0f C air", c->cycle_days, c->t_opt_lo,
                                c->t_opt_hi, c->t_bolt);
            ImGui::PopID();
        }
        if (live) ui_nft_apply_live(sc);
        ImGui::TextDisabled("stagger applies on rebuild");
    }

    // hydraulics
    if (ImGui::CollapsingHeader("Hydraulics (PVC)", ImGuiTreeNodeFlags_DefaultOpen)) {
        NftHydraulicParams *hp = &p->hyd;
        const char *pumps[NftPump_Count], *profiles[NftProfile_Count];
        for (u32 i = 0; i < NftPump_Count; ++i) pumps[i] = nft_pumps[i].name;
        for (u32 i = 0; i < NftProfile_Count; ++i) profiles[i] = nft_profiles[i].name;
        const char *schedules[] = { "Schedule 40", "Schedule 80" };
        b32 live = 0;
        ImGui::PushItemWidth(-130);
        live |= ui_combo_u32("pump", &hp->pump, pumps, NftPump_Count);
        bool sub = p->pump_submersible != 0;
        if (ImGui::Checkbox("submersible (all pump power heats the water)", &sub)) { p->pump_submersible = sub; live = 1; }
        live |= ui_combo_u32("PVC", &hp->schedule80, schedules, 2);
        live |= ui_combo_u32("supply pipe", &hp->supply_size, nft_pipe_names, NftPipe_Count);
        live |= ui_slider_f32("supply length m", &hp->supply_length_m, 0.2f, 50.0f, "%.1f", 1);
        live |= ui_slider_f32("fittings K", &hp->supply_k, 0.0f, 30.0f, "%.1f");
        live |= ui_slider_f32("lift m", &hp->lift_m, 0.0f, 5.0f, "%.2f");
        live |= ui_combo_u32("manifold", &hp->manifold_size, nft_pipe_names, NftPipe_Count);
        live |= ui_slider_f32("channel pitch m", &hp->channel_pitch_m, 0.05f, 1.0f, "%.2f");
        live |= ui_slider_f32("feed bore mm", &hp->tap_id_mm, 1.0f, 20.0f, "%.1f");
        live |= ui_slider_f32("feed tube m", &hp->tap_length_m, 0.0f, 3.0f, "%.2f");
        live |= ui_combo_u32("channel profile", &hp->profile, profiles, NftProfile_Count);
        bool bal = hp->balance_taps != 0;
        if (ImGui::Checkbox("balance taps (inline valves)", &bal)) { hp->balance_taps = bal; live = 1; }
        bool ed = hp->eductor_on != 0;
        if (ImGui::Checkbox("mixing eductor", &ed)) { hp->eductor_on = ed; live = 1; }
        if (hp->eductor_on) {
            live |= ui_slider_f32("nozzle mm", &hp->eductor_nozzle_mm, 2.0f, 20.0f, "%.1f");
            live |= ui_slider_f32("entrainment", &hp->eductor_ratio, 1.0f, 6.0f, "%.1f : 1");
        }
        live |= ui_combo_u32("return pipe", &hp->return_size, nft_pipe_names, NftPipe_Count);
        ImGui::PopItemWidth();
        if (live) { p->auto_size = 0; ui_nft_apply_live(sc); }

        NftHydraulics *h = &s->hyd;
        if (!h->ok) {
            ImGui::TextColored(ImVec4(1, 0.4f, 0.3f, 1), "pump cannot reach the manifold (shut-off head %.1f m)",
                               nft_pumps[hp->pump].h0_m);
        } else {
            ImGui::Text("pump %.1f L/min at %.2f m, %.0f W (eta %.2f)", h->q_pump * 60000.0, h->head_pump, h->pump_power_W, h->pump_eta);
            ImGui::Text("channels %.2f..%.2f L/min (spread %.1f %%)", h->q_min * 60000.0, h->q_max * 60000.0,
                        100.0 * (h->q_max / Maximum(h->q_min, 1e-12) - 1.0));
            if (hp->eductor_on)
                ImGui::Text("eductor %.1f L/min motive -> %.0f L/min, tank turnover %.1f /h", h->q_eductor_motive * 60000.0,
                            h->q_eductor_total * 60000.0, h->q_eductor_total * 3600.0 * 1000.0 / Maximum((f64)s->res.V_L, 1.0));
            f64 re = h->q_mean / nft_profiles[hp->profile].width_m / h->nu;
            ImGui::Text("root-free film %.2f mm, %.3f m/s, Re %.0f (%s)", s->group[0].lp.depth_m[0] * 1000.0, s->group[0].lp.velocity_m_s[0],
                        re, re < 400 ? "laminar" : re > 1600 ? "turbulent" : "transitional");
            ImVec4 rc = h->return_use > 0.8 ? ImVec4(1, 0.4f, 0.3f, 1) : ImVec4(0.6f, 0.65f, 0.7f, 1);
            ImGui::TextColored(rc, "return drain at %.0f %% of full-pipe capacity", 100.0 * h->return_use);
        }
    }

    // climate
    if (ImGui::CollapsingHeader("Climate")) {
        ImGui::PushItemWidth(-130);
        ui_slider_f32("ambient mean C", &p->ambient_mean_C, 5.0f, 40.0f, "%.1f");
        ui_slider_f32("day/night swing C", &p->ambient_amp_C, 0.0f, 15.0f, "%.1f");
        ui_slider_f32("photoperiod h", &p->photoperiod_h, 0.0f, 24.0f, "%.1f");
        ui_slider_f32("radiant on tops W/m2", &p->radiant_W_m2, 0.0f, 1000.0f, "%.0f");
        ui_slider_f32("relative humidity", &p->rh, 0.05f, 1.0f, "%.2f");
        ui_slider_f32("CO2 ppm", &p->co2_ppm, 200.0f, 2000.0f, "%.0f");
        b32 evap_changed = ui_slider_f32("air film h W/m2K", &p->h_conv_W_m2K, 1.0f, 30.0f, "%.1f");
        evap_changed |= ui_slider_f32("channel open fraction", &p->channel_open_frac, 0.0f, 1.0f, "%.2f");
        ui_slider_f32("tank open fraction", &p->tank_open_frac, 0.0f, 1.0f, "%.2f");
        if (evap_changed) s->cells_force = 1;
        ImGui::PopItemWidth();
        f64 es_a = nft_es_Pa(s->T_amb);
        ImGui::TextDisabled("ambient now %.1f C, VPD %.2f kPa, dew point %.1f C", s->T_amb, es_a * (1.0 - s->rh) * 1e-3,
                            243.04 * log(Maximum((f64)s->rh, 1e-3) * es_a / 610.94) / (17.625 - log(Maximum((f64)s->rh, 1e-3) * es_a / 610.94)));
        ImGui::TextDisabled("evaporation: channels %.2f L/day, tank %.2f L/day", s->evap_channel_L_s * 86400.0, s->evap_tank_L_s * 86400.0);
    }

    // water use, transpiration and root mats
    if (ImGui::CollapsingHeader("Water & roots", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::PushItemWidth(-130);
        ui_slider_f32("Baille A (radiation)", &p->baille_A, 0.0f, 1.0f, "%.2f");
        ui_slider_f32("Baille B W/m2/kPa", &p->baille_B, 0.0f, 60.0f, "%.1f");
        ui_slider_f32("night stomata", &p->night_stomata, 0.0f, 1.0f, "%.2f");
        ui_slider_f32("channel pitch m", &p->channel_pitch_m, 0.05f, 1.0f, "%.2f");
        b32 roots_changed = ui_slider_f32("root diameter mm", &p->root_diam_mm, 0.05f, 2.0f, "%.2f");
        roots_changed |= ui_slider_f32("root fresh/dry", &p->root_fresh_dry, 0.0f, 30.0f, "%.1f");
        roots_changed |= ui_slider_f32("packed mat fraction", &p->root_phi_max, 0.05f, 0.8f, "%.2f");
        if (roots_changed) s->cells_force = 1;
        ImGui::PopItemWidth();
        f64 plants = (f64)s->rails * s->sites;
        ImGui::Text("plants %.1f L/day (%.0f mL each), evaporation %.1f L/day", s->transp_L_s * 86400.0,
                    s->transp_L_s * 86400.0 / plants * 1000.0, (s->evap_channel_L_s + s->evap_tank_L_s) * 86400.0);
        ImGui::Text("channels hold %.1f L; deepest film %.1f mm (root-free %.1f mm)", s->acc.holdup_L, s->max_depth_m * 1e3,
                    s->group[0].lp.depth_m[0] * 1e3);
        if (s->overflow_cells)
            ImGui::TextColored(ImVec4(1, 0.4f, 0.3f, 1), "%u cells: root mats raise the film past 80 %% of the wall", s->overflow_cells);
        ImGui::TextDisabled("totals: plants %.0f L, evaporation %.0f L (%.1f kWh latent)", s->acc.transpired_L,
                            s->acc.evap_channel_L + s->acc.evap_tank_L, s->acc.evap_heat_J / 3.6e6);
    }

    // pH
    if (ImGui::CollapsingHeader("pH", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::PushItemWidth(-130);
        bool ctl = p->ph_control != 0;
        if (ImGui::Checkbox("pH control (HNO3 / KOH, 1 M)", &ctl)) p->ph_control = ctl;
        ui_slider_f32("pH setpoint", &p->ph_set, 4.5f, 7.5f, "%.2f");
        ui_slider_f32("pH band +-", &p->ph_band, 0.02f, 1.0f, "%.2f");
        ui_slider_f32("max dose mmol/h", &p->acid_mmol_per_h_max, 1.0f, 1000.0f, "%.0f", 1);
        ui_slider_f32("tap alkalinity meq/L", &p->tap_alk_meq, 0.0f, 6.0f, "%.2f");
        ImGui::PopItemWidth();
        f64 c[NftF_Count];
        for (u32 k = 0; k < NftF_Count; ++k) c[k] = nft_reservoir_conc(s, k);
        NftWaterChem w = nft_water_chem(c, nft_R_conc(s));
        ImGui::Text("tank pH %.2f, alkalinity %.3f meq/L, I %.3f M%s", s->pH, w.TA_meq, w.I,
                    s->acid_running ? ", dosing acid" : s->base_running ? ", dosing base" : "");
        ImGui::TextDisabled("acid %.0f mmol (%.2f L), base %.0f mmol; growth factor %.2f", s->acc.acid_mmol, s->acc.acid_L,
                            s->acc.base_mmol, nft_ph_factor(s->pH));
    }

    // reservoir + chiller
    if (ImGui::CollapsingHeader("Reservoir & chiller", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::PushItemWidth(-130);
        bool on = p->chiller_on != 0;
        if (ImGui::Checkbox("chiller", &on)) p->chiller_on = on;
        ui_slider_f32("setpoint C", &p->chiller_set_C, 10.0f, 30.0f, "%.1f");
        ui_slider_f32("hysteresis +-C", &p->chiller_band_C, 0.05f, 2.0f, "%.2f");
        ui_slider_f32("capacity W", &p->chiller_W, 50.0f, 20000.0f, "%.0f", 1);
        ui_slider_f32("COP", &p->chiller_cop, 1.0f, 6.0f, "%.1f");
        ImGui::PopItemWidth();
        f64 hours = Maximum(s->t / 3600.0, 1e-9);
        ImGui::Text("tank %.0f L at %.2f C, chiller %s", s->res.V_L, s->res.T_C, s->res.chiller_running ? "running" : "idle");
        ImGui::Text("chiller %.2f kWh (%.2f kWh/day), duty %.0f %%", s->acc.chiller_elec_J / 3.6e6,
                    s->acc.chiller_elec_J / 3.6e6 / Maximum(s->t / 86400.0, 1e-9), 100.0 * s->acc.chiller_on_s / (hours * 3600.0));
        ImGui::Text("pump heat %.0f W, peak load estimate %.0f W", nft_pump_heat_W(p, &s->hyd),
                    nft_peak_heat_W(s, p, nft_pump_heat_W(p, &s->hyd)));
    }

    // nutrients
    if (ImGui::CollapsingHeader("Nutrients", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::PushItemWidth(-130);
        bool dose = p->dosing_on != 0, matched = p->dose_matched != 0, topoff = p->topoff_on != 0;
        if (ImGui::Checkbox("EC dosing", &dose)) p->dosing_on = dose;
        ImGui::SameLine();
        if (ImGui::Checkbox("uptake-matched stock", &matched)) p->dose_matched = matched;
        if (ImGui::Checkbox("top-off", &topoff)) p->topoff_on = topoff;
        ui_slider_f32("EC setpoint mS/cm", &p->ec_set_mS, 0.4f, 3.5f, "%.2f");
        ui_slider_f32("solution change days", &p->dump_days, 0.0f, 60.0f, "%.0f");
        ImGui::PopItemWidth();
        f64 c[NftF_Count];
        for (u32 k = 0; k < NftF_Count; ++k) c[k] = nft_reservoir_conc(s, k);
        ImGui::Text("EC %.2f mS/cm, DO %.1f mg/L", nft_ec_mS(c), c[NftF_DO]);
        if (ImGui::BeginTable("##ions", 4, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp)) {
            for (u32 i = 0; i < NFT_IONS; ++i) {
                if (i % 2 == 0) ImGui::TableNextRow();
                ImGui::TableNextColumn(); ImGui::TextUnformatted(nft_field_names[NFT_ION0 + i]);
                ImGui::TableNextColumn(); ImGui::Text("%.1f", c[NFT_ION0 + i]);
            }
            ImGui::EndTable();
        }
        ImGui::TextDisabled("water used %.0f L, stock %.1f L, plants %.0f L, evaporated %.0f L", s->acc.topoff_L + s->acc.dosed_L,
                            s->acc.dosed_L, s->acc.transpired_L, s->acc.evap_channel_L + s->acc.evap_tank_L);
    }

    // harvest
    if (ImGui::CollapsingHeader("Harvest", ImGuiTreeNodeFlags_DefaultOpen)) {
        nft_collect_harvest(s);
        for (u32 b = 0; b < 2; ++b) {
            const NftCrop *c = nft_crops + p->bank[b].crop;
            f64 w, ft, fn, fd;
            nft_bank_plants(s, b, &w, &ft, &fn, &fd);
            ImGui::Text("%s: %s", b ? "B" : "A", c->name);
            ImGui::TextDisabled("  %.0f harvests, %.0f g dry (%.2f g each), %.0f bolting", s->acc.harvests[b], s->acc.harvest_g[b],
                                s->acc.harvests[b] > 0 ? s->acc.harvest_g[b] / s->acc.harvests[b] : 0.0, s->acc.bolted[b]);
            ImGui::TextDisabled("  now %.2f g mean, fT %.2f fN %.2f fDO %.2f", w, ft, fn, fd);
        }
        ImGui::SetNextItemWidth(-80);
        ImGui::InputText("##nftexport", ui->export_path, sizeof(ui->export_path));
        ImGui::SameLine();
        if (ImGui::Button("Export")) nft_scene_write_csv(sc, &memory->platform, &state->frame, ui->export_path);
    }
}

// --- plots ---------------------------------------------------------------------------

internal void ui_nft_line(NftScene *sc, const char *label, u32 col, ImVec4 color, ImAxis y_axis = ImAxis_Y1) {
    if (sc->hist_count < 2) return;
    ImPlot::SetAxes(ImAxis_X1, y_axis);
    ImPlotSpec spec;
    spec.LineColor = color;
    spec.LineWeight = 2.0f;
    spec.Offset = (int)sc->hist_head;
    spec.Stride = (int)(sizeof(f64) * NftCol_Count);
    ImPlot::PlotLine(label, sc->history + NftCol_Day, sc->history + col, (int)sc->hist_count, spec);
}

internal void ui_nft_plots(SimState *state) {
    NftScene *sc = state->nft;
    NftSystem *s = &sc->sys;
    UiState *ui = &state->ui;
    ImVec4 cA = ImVec4(0.35f, 0.75f, 1.0f, 1), cB = ImVec4(1.0f, 0.6f, 0.25f, 1);
    if (!ImGui::BeginTabBar("##nftplots")) return;
    if (ImGui::BeginTabItem("Temperatures")) {
        if (ImPlot::BeginPlot("##temps", ImVec2(-1, -1))) {
            ImPlot::SetupAxes("day", "C", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
            ImPlot::SetupAxis(ImAxis_Y2, "chiller duty", ImPlotAxisFlags_AuxDefault);
            ImPlot::SetupAxisLimits(ImAxis_Y2, 0, 4, ImPlotCond_Always);
            ImPlot::SetupLegend(ImPlotLocation_NorthEast);
            if (sc->hist_count > 1) {
                ImPlot::SetAxes(ImAxis_X1, ImAxis_Y2);
                ImPlotSpec sh; sh.FillColor = ImVec4(0.3f, 0.5f, 0.9f, 1); sh.FillAlpha = 0.25f;
                sh.Offset = (int)sc->hist_head; sh.Stride = (int)(sizeof(f64) * NftCol_Count);
                ImPlot::PlotShaded("chiller duty", sc->history + NftCol_Day, sc->history + NftCol_ChillerDuty, (int)sc->hist_count, 0.0, sh);
            }
            ui_nft_line(sc, "ambient", NftCol_TAmb, ImVec4(0.75f, 0.75f, 0.75f, 1));
            ui_nft_line(sc, "reservoir", NftCol_TRes, ImVec4(0.4f, 1.0f, 0.55f, 1));
            ui_nft_line(sc, "outlet A", NftCol_TOutA, cA);
            ui_nft_line(sc, "outlet B", NftCol_TOutB, cB);
            ImPlot::EndPlot();
        }
        ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("Nutrients")) {
        if (ImPlot::BeginPlot("##ions", ImVec2(-1, -1))) {
            ImPlot::SetupAxes("day", "mg/L", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
            ImPlot::SetupAxis(ImAxis_Y2, "EC mS/cm", ImPlotAxisFlags_AuxDefault | ImPlotAxisFlags_AutoFit);
            ImPlot::SetupLegend(ImPlotLocation_NorthEast);
            ImVec4 colors[NFT_IONS] = { {0.4f,0.8f,1,1}, {0.7f,0.7f,1,1}, {1,0.85f,0.3f,1}, {1,0.5f,0.3f,1}, {0.9f,0.9f,0.9f,1}, {0.5f,1,0.5f,1}, {0.9f,0.5f,1,1} };
            for (u32 i = 0; i < NFT_IONS; ++i) ui_nft_line(sc, nft_field_names[NFT_ION0 + i], NftCol_NO3 + i, colors[i]);
            ui_nft_line(sc, "DO", NftCol_DO, ImVec4(0.3f, 0.9f, 0.9f, 1));
            ui_nft_line(sc, "EC", NftCol_EC, ImVec4(1, 1, 1, 1), ImAxis_Y2);
            ImPlot::EndPlot();
        }
        ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("Plants")) {
        if (ImPlot::BeginPlot("##plants", ImVec2(-1, -1))) {
            ImPlot::SetupAxes("day", "g dry mass / plant", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
            ImPlot::SetupAxis(ImAxis_Y2, "harvested g", ImPlotAxisFlags_AuxDefault | ImPlotAxisFlags_AutoFit);
            ImPlot::SetupLegend(ImPlotLocation_NorthWest);
            ui_nft_line(sc, "mean A", NftCol_WA, cA);
            ui_nft_line(sc, "mean B", NftCol_WB, cB);
            ui_nft_line(sc, "harvest A", NftCol_HarvestA, ImVec4(0.2f, 0.5f, 0.8f, 1), ImAxis_Y2);
            ui_nft_line(sc, "harvest B", NftCol_HarvestB, ImVec4(0.8f, 0.45f, 0.2f, 1), ImAxis_Y2);
            ImPlot::EndPlot();
        }
        ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("pH")) {
        if (ImPlot::BeginPlot("##ph", ImVec2(-1, -1))) {
            ImPlot::SetupAxes("day", "pH", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
            ImPlot::SetupAxis(ImAxis_Y2, "mmol / meq/L", ImPlotAxisFlags_AuxDefault | ImPlotAxisFlags_AutoFit);
            ImPlot::SetupLegend(ImPlotLocation_NorthWest);
            ui_nft_line(sc, "tank pH", NftCol_pH, ImVec4(1, 0.85f, 0.3f, 1));
            ui_nft_line(sc, "acid mmol", NftCol_AcidMmol, ImVec4(1, 0.4f, 0.35f, 1), ImAxis_Y2);
            ui_nft_line(sc, "base mmol", NftCol_BaseMmol, ImVec4(0.4f, 0.6f, 1, 1), ImAxis_Y2);
            ImPlot::EndPlot();
        }
        ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("Water")) {
        if (ImPlot::BeginPlot("##water", ImVec2(-1, -1))) {
            ImPlot::SetupAxes("day", "L/day", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
            ImPlot::SetupAxis(ImAxis_Y2, "L / mm", ImPlotAxisFlags_AuxDefault | ImPlotAxisFlags_AutoFit);
            ImPlot::SetupLegend(ImPlotLocation_NorthWest);
            ui_nft_line(sc, "plants", NftCol_TranspLd, ImVec4(0.4f, 1, 0.5f, 1));
            ui_nft_line(sc, "evaporation", NftCol_EvapLd, ImVec4(0.4f, 0.8f, 1, 1));
            ui_nft_line(sc, "channel holdup L", NftCol_HoldupL, ImVec4(0.8f, 0.8f, 0.8f, 1), ImAxis_Y2);
            ui_nft_line(sc, "deepest film mm", NftCol_FilmMaxMm, ImVec4(0.9f, 0.6f, 0.3f, 1), ImAxis_Y2);
            ImPlot::EndPlot();
        }
        ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("Limits")) {
        if (ImPlot::BeginPlot("##limits", ImVec2(-1, -1))) {
            ImPlot::SetupAxes("day", "factor", ImPlotAxisFlags_AutoFit, 0);
            ImPlot::SetupAxisLimits(ImAxis_Y1, 0, 1.05, ImPlotCond_Always);
            ImPlot::SetupLegend(ImPlotLocation_SouthWest);
            ui_nft_line(sc, "fT A", NftCol_fTA, cA);
            ui_nft_line(sc, "fT B", NftCol_fTB, cB);
            ui_nft_line(sc, "fN A", NftCol_fNA, ImVec4(0.4f, 1, 0.5f, 1));
            ui_nft_line(sc, "fN B", NftCol_fNB, ImVec4(0.8f, 1, 0.3f, 1));
            ui_nft_line(sc, "fDO A", NftCol_fDOA, ImVec4(0.6f, 0.6f, 1, 1));
            ui_nft_line(sc, "fDO B", NftCol_fDOB, ImVec4(1, 0.6f, 1, 1));
            ImPlot::EndPlot();
        }
        ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("Along channel")) {
        const char *fields[NftF_Count];
        for (u32 k = 0; k < NftF_Count; ++k) fields[k] = nft_field_names[k];
        ImGui::SetNextItemWidth(120);
        ImGui::Combo("field", &ui->nft_profile_field, fields, NftF_Count);
        u32 k = (u32)Clamp(ui->nft_profile_field, 0, NftF_Count - 1);
        TempMemory temp = begin_temp(&state->frame);
        f64 *xs = push_array_nozero(&state->frame, f64, s->cells);
        f64 *ya = push_array_nozero(&state->frame, f64, s->cells);
        f64 *yb = push_array_nozero(&state->frame, f64, s->cells);
        for (u32 j = 0; j < s->cells; ++j) {
            xs[j] = ((f64)j + 0.5) * s->cell_m;
            ya[j] = nft_bank_cell_mean(s, 0, k, j);
            yb[j] = nft_bank_cell_mean(s, 1, k, j);
        }
        if (ImPlot::BeginPlot("##along", ImVec2(-1, -1))) {
            ImPlot::SetupAxes("distance from inlet [m]", k == NftF_T ? "C" : "mg/L", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
            ImPlotSpec a; a.LineColor = cA; a.LineWeight = 2.0f;
            ImPlotSpec b; b.LineColor = cB; b.LineWeight = 2.0f;
            ImPlot::PlotLine("bank A mean", xs, ya, (int)s->cells, a);
            ImPlot::PlotLine("bank B mean", xs, yb, (int)s->cells, b);
            ImPlot::EndPlot();
        }
        end_temp(temp);
        ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("Manifold")) {
        TempMemory temp = begin_temp(&state->frame);
        f64 *xs = push_array_nozero(&state->frame, f64, s->rails);
        f64 *q = push_array_nozero(&state->frame, f64, s->rails);
        f64 *hd = push_array_nozero(&state->frame, f64, s->rails);
        for (u32 r = 0; r < s->rails; ++r) { xs[r] = r; q[r] = s->hyd.q_channel[r] * 60000.0; hd[r] = s->hyd.head_tap[r]; }
        if (ImPlot::BeginPlot("##manifold", ImVec2(-1, -1))) {
            ImPlot::SetupAxes("channel (inlet end of the manifold = 0)", "L/min", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
            ImPlot::SetupAxis(ImAxis_Y2, "tap head m", ImPlotAxisFlags_AuxDefault | ImPlotAxisFlags_AutoFit);
            ImPlotSpec bars; bars.FillColor = ImVec4(0.35f, 0.7f, 1, 1);
            ImPlot::PlotBars("channel flow", xs, q, (int)s->rails, 0.7, bars);
            ImPlot::SetAxes(ImAxis_X1, ImAxis_Y2);
            ImPlotSpec ln; ln.LineColor = ImVec4(1, 0.8f, 0.3f, 1); ln.LineWeight = 2.0f;
            ImPlot::PlotLine("tap head", xs, hd, (int)s->rails, ln);
            ImPlot::EndPlot();
        }
        end_temp(temp);
        ImGui::EndTabItem();
    }
    ImGui::EndTabBar();
}

// --- 3D overlay --------------------------------------------------------------------------

internal void ui_nft_overlay(SimState *state, RenderCommands *render, ImVec2 origin) {
    UiState *ui = &state->ui;
    NftScene *sc = state->nft;
    ImGui::SetNextWindowPos(ImVec2(origin.x + 10, origin.y + 10));
    ImGui::SetNextWindowBgAlpha(0.55f);
    ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings |
                             ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoMove;
    if (ImGui::Begin("##nft_overlay", 0, flags)) {
        const char *names[NftLayer_Count];
        for (u32 i = 0; i < NftLayer_Count; ++i) names[i] = nft_layers[i].name;
        ImGui::SetNextItemWidth(240);
        ImGui::Combo("##layer", &ui->nft_layer, names, NftLayer_Count);
        const NftLayerInfo *info = nft_layers + Clamp(ui->nft_layer, 0, NftLayer_Count - 1);
        ImDrawList *dl = ImGui::GetWindowDrawList();
        ImVec2 p0 = ImGui::GetCursorScreenPos();
        f32 w = 240.0f, h = 12.0f;
        const u32 segments = 32;
        for (u32 i = 0; i < segments; ++i) {
            f32 t0 = (f32)i / segments, t1 = (f32)(i + 1) / segments;
            ImU32 c0 = info->diverging ? nft_diverging(2 * t0 - 1) : dbd_colormap(t0);
            ImU32 c1 = info->diverging ? nft_diverging(2 * t1 - 1) : dbd_colormap(t1);
            dl->AddRectFilledMultiColor(ImVec2(p0.x + w * t0, p0.y), ImVec2(p0.x + w * t1, p0.y + h), c0, c1, c1, c0);
        }
        ImGui::Dummy(ImVec2(w, h));
        // enough significant digits to tell the ends apart (a pH span can be 0.005)
        f32 mag = Maximum(fabsf(sc->view_lo), fabsf(sc->view_hi)), span = Maximum(sc->view_hi - sc->view_lo, 1e-30f);
        int digits = Clamp((int)ceilf(log10f(Maximum(mag, 1e-30f) / span)) + 2, 3, 7);
        ImGui::Text("%-9.*g", digits, sc->view_lo);
        ImGui::SameLine(w * 0.5f - 10); ImGui::TextDisabled("%s", info->unit);
        ImGui::SameLine(w - 70); ImGui::Text("%9.*g", digits, sc->view_hi);
        ImGui::TextDisabled("inlet at left, grade x10; plants rise with dry mass");
    }
    ImGui::End();
}
