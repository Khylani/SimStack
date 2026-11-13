// Timeline bar in the transport: recorded range, keyframe ticks, playhead,
// LIVE / PLAYBACK state, scrubbing, bake progress and the Bake popup.
// Requests go through UiState (scrub_t, bake_*); simstack.cpp applies them.

internal f64 ui_time_unit(SimState *state) { return state->nft ? 86400.0 : 1.0; }
internal const char *ui_time_unit_name(SimState *state) { return state->nft ? "days" : "s"; }

internal void ui_bake_popup(SimState *state) {
    UiState *ui = &state->ui;
    Timeline *tl = &state->timeline;
    if (ImGui::BeginPopup("Bake")) {
        ImGui::TextUnformatted("Bake a run");
        ImGui::TextDisabled("Simulate ahead as fast as the machine allows, recording keyframes;\n"
                            "then scrub and play it back freely. Edits during playback fork it.");
        ImGui::Separator();
        f64 unit = ui_time_unit(state);
        if (ui->bake_duration <= 0.0f) { ui->bake_duration = state->nft ? 30.0f * 86400.0f : 2.0f; ui->bake_keys = 1000; }
        f32 dur = (f32)(ui->bake_duration / unit);
        ImGui::SetNextItemWidth(ui_s(160));
        if (ImGui::DragFloat("duration", &dur, state->nft ? 0.5f : 0.05f, state->nft ? 0.1f : 0.01f, state->nft ? 3650.0f : 600.0f,
                             state->nft ? "%.1f days" : "%.2f s", ImGuiSliderFlags_Logarithmic))
            ui->bake_duration = dur * (f32)unit;
        ImGui::SetNextItemWidth(ui_s(160));
        ImGui::DragInt("keyframes", &ui->bake_keys, 10.0f, 10, TIMELINE_MAX_KEYS / 2);
        if (state->dbd) {
            bool p = ui->bake_precise != 0;
            if (ImGui::Checkbox("reference precision (f64 lanes, rtol 1e-5, 1 ms coupling)", &p)) ui->bake_precise = p;
        }
        f64 mb = (f64)tl->frame_bytes * (f64)ui->bake_keys / 1048576.0;
        f64 cap = (f64)tl->frame_bytes * (f64)tl->capacity / 1048576.0;
        ImGui::TextDisabled("%.1f MB per keyframe, %.0f MB for this bake (timeline holds %.0f MB)", (f64)tl->frame_bytes / 1048576.0, mb, cap);
        ImGui::TextDisabled("one keyframe every %.3g %s", ui->bake_duration / Maximum(ui->bake_keys, 1) / unit, ui_time_unit_name(state));
        if (ImGui::Button("Bake", ImVec2(ui_s(120), 0))) { ui->bake_go = 1; ImGui::CloseCurrentPopup(); }
        ImGui::SameLine();
        if (ImGui::Button("Cancel", ImVec2(ui_s(120), 0))) ImGui::CloseCurrentPopup();
        ImGui::EndPopup();
    }
}

internal void ui_timeline_bar(SimState *state, SimMemory *memory, f32 width) {
    UiState *ui = &state->ui;
    Timeline *tl = &state->timeline;
    (void)memory;
    if (memory->config.bake_s > 0.0 && !ui->bake_cli_done && tl->key_count) {   // --bake
        ui->bake_cli_done = 1;
        ui->bake_duration = (f32)memory->config.bake_s;
        ui->bake_keys = 1000;
        ui->bake_go = 1;
    }
    if (!timeline_available(tl)) {
        ImGui::AlignTextToFramePadding();
        ImGui::TextDisabled("(no timeline memory)");
        return;
    }
    // Bake button / progress
    if (tl->baking) {
        if (ImGui::Button("Stop bake")) ui->bake_cancel = 1;
    } else {
        if (ImGui::Button("Bake...")) ImGui::OpenPopup("Bake");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Simulate ahead and record keyframes for playback");
    }
    ui_bake_popup(state);
    ImGui::SameLine();
    f32 badge_w = ui_s(84);
    f32 w = Maximum(width - ImGui::GetItemRectSize().x - badge_w - 3.0f * ImGui::GetStyle().ItemSpacing.x, ui_s(40));
    f32 h = ImGui::GetFrameHeight();
    ImVec2 p0 = ImGui::GetCursorScreenPos();
    ImGui::InvisibleButton("##timeline", ImVec2(w, h));
    b32 hovered = ImGui::IsItemHovered(), active = ImGui::IsItemActive();
    ImDrawList *dl = ImGui::GetWindowDrawList();
    f64 t_now = tl->playback ? tl->playhead : state->sim_time;
    f64 t0 = timeline_start(tl);
    f64 t1 = Maximum(Maximum(timeline_end(tl), state->sim_time), tl->baking ? tl->bake_end : 0.0);
    if (t1 <= t0) t1 = t0 + 1e-6;
    auto tx = [&](f64 t) { return p0.x + (f32)((t - t0) / (t1 - t0)) * w; };
    f32 cy = p0.y + 0.5f * h, th = 0.36f * h;
    // track, recorded range, bake progress
    dl->AddRectFilled(ImVec2(p0.x, cy - th), ImVec2(p0.x + w, cy + th), IM_COL32(28, 33, 44, 255), th);
    if (tl->key_count > 1) dl->AddRectFilled(ImVec2(tx(timeline_start(tl)), cy - th), ImVec2(tx(timeline_end(tl)), cy + th),
                                             IM_COL32(52, 92, 150, 200), th);
    if (tl->baking) dl->AddRectFilled(ImVec2(tx(tl->bake_start), cy - th), ImVec2(tx(state->sim_time), cy + th), IM_COL32(220, 150, 60, 200), th);
    // keyframe ticks (when they fit)
    if (tl->key_count > 1 && (f32)tl->key_count < w / 4.0f)
        for (u32 i = 0; i < tl->key_count; ++i) {
            f32 x = tx(tl->keys[i].sim_time);
            dl->AddLine(ImVec2(x, cy - 0.55f * th), ImVec2(x, cy + 0.55f * th), IM_COL32(150, 190, 240, 160), 1.0f);
        }
    // playhead
    f32 px = tx(t_now);
    ImU32 head = tl->playback ? IM_COL32(120, 190, 255, 255) : IM_COL32(110, 230, 140, 255);
    dl->AddLine(ImVec2(px, p0.y + 1), ImVec2(px, p0.y + h - 1), head, Maximum(2.0f, ui_s(2.0f)));
    dl->AddTriangleFilled(ImVec2(px - ui_s(5), p0.y), ImVec2(px + ui_s(5), p0.y), ImVec2(px, p0.y + ui_s(6)), head);
    // scrub
    if (active && tl->key_count && !tl->baking) {
        f32 mx = Clamp(ImGui::GetIO().MousePos.x, p0.x, p0.x + w);
        ui->scrub_t = t0 + (f64)((mx - p0.x) / w) * (t1 - t0);
    }
    if (hovered) {
        f32 mx = Clamp(ImGui::GetIO().MousePos.x, p0.x, p0.x + w);
        f64 t = t0 + (f64)((mx - p0.x) / w) * (t1 - t0);
        char buf[48];
        f64 saved = state->sim_time;
        state->sim_time = t;
        ImGui::SetTooltip("%s\n%u keyframes, every %.3g %s%s", ui_fmt_sim_time(state, buf, sizeof(buf)), tl->key_count,
                          tl->interval / ui_time_unit(state), ui_time_unit_name(state), tl->key_count ? "\ndrag to scrub" : "");
        state->sim_time = saved;
    }
    // state badge
    ImGui::SameLine();
    const char *badge = tl->baking ? "BAKING" : tl->playback ? "PLAYBACK" : "LIVE";
    ImVec4 bc = tl->baking ? ImVec4(0.86f, 0.59f, 0.24f, 1) : tl->playback ? ImVec4(0.47f, 0.74f, 1.0f, 1) : ImVec4(0.43f, 0.90f, 0.55f, 1);
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(bc.x * 0.25f, bc.y * 0.25f, bc.z * 0.25f, 1));
    ImGui::PushStyleColor(ImGuiCol_Text, bc);
    if (ImGui::Button(badge, ImVec2(badge_w, 0)) && tl->playback) ui->scrub_t = timeline_end(tl) + 1.0;   // jump to the live end
    ImGui::PopStyleColor(2);
    if (ImGui::IsItemHovered()) {
        if (tl->baking) ImGui::SetTooltip("Baking to %.3g %s (%.0f %%)", (tl->bake_end - tl->bake_start) / ui_time_unit(state), ui_time_unit_name(state),
                                          100.0 * (state->sim_time - tl->bake_start) / Maximum(tl->bake_end - tl->bake_start, 1e-12));
        else if (tl->playback) ImGui::SetTooltip("Showing recorded keyframes. Click to jump to the end; edit anything to fork from here.");
        else ImGui::SetTooltip("Simulating and recording keyframes");
    }
}
