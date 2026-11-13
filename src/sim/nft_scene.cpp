// NFT scene: the NftSystem plus an hourly history ring, CSV log, 3D view and
// the headless summary. Included by simstack.cpp.

enum NftCol {           // history columns (one row per simulated hour)
    NftCol_Day,
    NftCol_TAmb, NftCol_TRes, NftCol_TOutA, NftCol_TOutB,
    NftCol_ChillerDuty, NftCol_ChillerKWh,
    NftCol_EC, NftCol_NO3, NftCol_NH4, NftCol_P, NftCol_K, NftCol_Ca, NftCol_Mg, NftCol_S, NftCol_DO,
    NftCol_VRes,
    NftCol_WA, NftCol_WB, NftCol_HarvestA, NftCol_HarvestB,
    NftCol_fTA, NftCol_fTB, NftCol_fNA, NftCol_fNB, NftCol_fDOA, NftCol_fDOB,
    NftCol_WaterL,
    NftCol_pH, NftCol_Alk,                         // tank pH, total alkalinity (meq/L)
    NftCol_TranspLd, NftCol_EvapLd,                // plant water use / evaporation (channels + tank), L/day over the hour
    NftCol_AcidMmol, NftCol_BaseMmol,              // cumulative pH-control doses
    NftCol_HoldupL, NftCol_FilmMaxMm,              // water held in the channels; deepest film (root mats)
    NftCol_Count,
};
global const char *const nft_col_names[NftCol_Count] = {
    "day", "T_ambient_C", "T_reservoir_C", "T_outlet_A_C", "T_outlet_B_C", "chiller_duty", "chiller_kWh",
    "EC_mS_cm", "NO3N_mg_L", "NH4N_mg_L", "P_mg_L", "K_mg_L", "Ca_mg_L", "Mg_mg_L", "S_mg_L", "DO_mg_L",
    "reservoir_L", "plant_dm_A_g", "plant_dm_B_g", "harvest_dm_A_g", "harvest_dm_B_g",
    "fT_A", "fT_B", "fN_A", "fN_B", "fDO_A", "fDO_B", "water_used_L",
    "pH", "alkalinity_meq_L", "plant_water_L_day", "evaporation_L_day", "acid_mmol", "base_mmol",
    "channel_holdup_L", "film_max_mm",
};

#define NFT_HISTORY (24 * 730)    // two years of hourly rows

struct NftScene {
    NftSystem sys;
    f64 *history;                // ring [row][NftCol_Count]
    u32 hist_count, hist_head;
    f64 next_sample_t;
    f64 chiller_on_at_sample;
    f64 transp_at_sample, evap_at_sample, t_at_sample;
    const char *csv_path;
    f64 last_window_wall_ms, wall_ms_total;
    u64 windows;
    // 3D view (allocated once): rails (cells) and plants (sites)
    u32 view_cells, view_sites;
    f32 *cx, *cy, *cz; u32 *ccol;
    f32 *px, *py, *pz; u32 *pcol;
    f32 view_lo, view_hi;
};

internal void nft_scene_sample(NftScene *sc) {
    NftSystem *s = &sc->sys;
    u32 slot;
    if (sc->hist_count < NFT_HISTORY) slot = (sc->hist_head + sc->hist_count++) % NFT_HISTORY;
    else { slot = sc->hist_head; sc->hist_head = (sc->hist_head + 1) % NFT_HISTORY; }
    f64 *row = sc->history + (u64)slot * NftCol_Count;
    f64 c[NftF_Count];
    for (u32 k = 0; k < NftF_Count; ++k) c[k] = nft_reservoir_conc(s, k);
    row[NftCol_Day] = s->t / 86400.0;
    row[NftCol_TAmb] = s->T_amb;
    row[NftCol_TRes] = s->res.T_C;
    row[NftCol_TOutA] = nft_bank_cell_mean(s, 0, NftF_T, s->cells - 1);
    row[NftCol_TOutB] = nft_bank_cell_mean(s, 1, NftF_T, s->cells - 1);
    row[NftCol_ChillerDuty] = (s->acc.chiller_on_s - sc->chiller_on_at_sample) / 3600.0;
    sc->chiller_on_at_sample = s->acc.chiller_on_s;
    row[NftCol_ChillerKWh] = s->acc.chiller_elec_J / 3.6e6;
    row[NftCol_EC] = nft_ec_mS(c);
    for (u32 i = 0; i < NFT_IONS; ++i) row[NftCol_NO3 + i] = c[NFT_ION0 + i];
    row[NftCol_DO] = c[NftF_DO];
    row[NftCol_VRes] = s->res.V_L;
    f64 w, ft, fn, fd;
    nft_bank_plants(s, 0, &w, &ft, &fn, &fd);
    row[NftCol_WA] = w; row[NftCol_fTA] = ft; row[NftCol_fNA] = fn; row[NftCol_fDOA] = fd;
    nft_bank_plants(s, 1, &w, &ft, &fn, &fd);
    row[NftCol_WB] = w; row[NftCol_fTB] = ft; row[NftCol_fNB] = fn; row[NftCol_fDOB] = fd;
    nft_collect_harvest(s);
    row[NftCol_HarvestA] = s->acc.harvest_g[0];
    row[NftCol_HarvestB] = s->acc.harvest_g[1];
    row[NftCol_WaterL] = s->acc.topoff_L + s->acc.dosed_L;
    row[NftCol_pH] = s->pH;
    row[NftCol_Alk] = nft_water_chem(c, nft_R_conc(s)).TA_meq;
    f64 dt_days = (s->t - sc->t_at_sample) / 86400.0;
    f64 evap = s->acc.evap_channel_L + s->acc.evap_tank_L;
    row[NftCol_TranspLd] = dt_days > 0.0 ? (s->acc.transpired_L - sc->transp_at_sample) / dt_days : 0.0;
    row[NftCol_EvapLd] = dt_days > 0.0 ? (evap - sc->evap_at_sample) / dt_days : 0.0;
    sc->transp_at_sample = s->acc.transpired_L;
    sc->evap_at_sample = evap;
    sc->t_at_sample = s->t;
    row[NftCol_AcidMmol] = s->acc.acid_mmol;
    row[NftCol_BaseMmol] = s->acc.base_mmol;
    row[NftCol_HoldupL] = s->acc.holdup_L;
    row[NftCol_FilmMaxMm] = s->max_depth_m * 1e3;
}

internal void nft_scene_init(NftScene *sc, Arena *arena, SimConfig *config, NftParams *params) {
    nft_init(&sc->sys, arena, params ? *params : nft_default_params());
    sc->history = push_array(arena, f64, (u64)NFT_HISTORY * NftCol_Count);
    sc->csv_path = config->csv_path;
    sc->view_cells = sc->sys.rails * sc->sys.cells;
    sc->view_sites = sc->sys.rails * sc->sys.sites;
    sc->cx = SoaPush(arena, f32, sc->view_cells); sc->cy = SoaPush(arena, f32, sc->view_cells);
    sc->cz = SoaPush(arena, f32, sc->view_cells); sc->ccol = push_array_aligned(arena, u32, SoaPadCount(sc->view_cells), SIMD_ALIGN);
    sc->px = SoaPush(arena, f32, sc->view_sites); sc->py = SoaPush(arena, f32, sc->view_sites);
    sc->pz = SoaPush(arena, f32, sc->view_sites); sc->pcol = push_array_aligned(arena, u32, SoaPadCount(sc->view_sites), SIMD_ALIGN);
    nft_environment(&sc->sys, 0.0);
    nft_scene_sample(sc);
    sc->next_sample_t = 3600.0;
}

internal void nft_scene_write_csv(NftScene *sc, PlatformAPI *api, Arena *scratch, const char *path) {
    if (!path) return;
    TempMemory temp = begin_temp(scratch);
    u64 cap = Megabytes(32);
    char *buf = (char *)push_size_(scratch, cap, 16, 0);
    u64 len = 0;
#define CSV(...) len += (u64)snprintf(buf + len, len < cap ? cap - len : 0, __VA_ARGS__)
    NftSystem *s = &sc->sys;
    CSV("# simstack NFT: %u channels x %.2f m x %u plants, bank A %s / %s, bank B %s / %s, reservoir %.0f L, chiller %s %.1f C %.0f W\n",
        s->rails, s->params.rail_length_m, s->sites, nft_crops[s->params.bank[0].crop].name, nft_wraps[s->params.bank[0].wrap].name,
        nft_crops[s->params.bank[1].crop].name, nft_wraps[s->params.bank[1].wrap].name, s->params.reservoir_L,
        s->params.chiller_on ? "on" : "off", s->params.chiller_set_C, s->params.chiller_W);
    for (u32 c = 0; c < NftCol_Count; ++c) CSV("%s%s", c ? "," : "", nft_col_names[c]);
    CSV("\n");
    for (u32 r = 0; r < sc->hist_count && len < cap - 1024; ++r) {
        f64 *row = sc->history + (u64)((sc->hist_head + r) % NFT_HISTORY) * NftCol_Count;
        for (u32 c = 0; c < NftCol_Count; ++c) CSV("%s%.6g", c ? "," : "", row[c]);
        CSV("\n");
    }
#undef CSV
    if (len >= cap) len = cap - 1;
    if (!api->write_entire_file(path, buf, len)) api->log("[nft] cannot write %s\n", path);
    end_temp(temp);
}

// Advances one coupling window and samples the history every simulated hour.
internal void nft_scene_window(NftScene *sc, SimMemory *memory, Arena *frame) {
    u64 w0 = memory->platform.now_ns();
    nft_window(&sc->sys, &memory->platform, memory->work_queue, memory->worker_count, frame);
    ++sc->windows;
    if (sc->sys.t >= sc->next_sample_t - 1e-6) {
        nft_scene_sample(sc);
        sc->next_sample_t += 3600.0;
        // CSV once per simulated day
        if (sc->csv_path && fmod(sc->sys.t, 86400.0) < sc->sys.window_s * 0.5)
            nft_scene_write_csv(sc, &memory->platform, frame, sc->csv_path);
    }
    sc->last_window_wall_ms = (f64)(memory->platform.now_ns() - w0) * 1e-6;
    sc->wall_ms_total += sc->last_window_wall_ms;
}

// --- 3D view ------------------------------------------------------------------------

enum NftLayer {
    NftLayer_Temperature, NftLayer_DeltaInlet, NftLayer_DeltaAmbient, NftLayer_DO,
    NftLayer_NO3, NftLayer_K, NftLayer_Ca, NftLayer_EC,
    NftLayer_Velocity, NftLayer_Depth, NftLayer_Flow,
    NftLayer_PlantMass, NftLayer_fT, NftLayer_fN, NftLayer_fDO,
    NftLayer_pH, NftLayer_RootFrac, NftLayer_Evap,
    NftLayer_Count,
};
struct NftLayerInfo { const char *name, *unit; b32 diverging; b32 plants; };
global const NftLayerInfo nft_layers[NftLayer_Count] = {
    { "Film temperature",            "C",       0, 0 },
    { "Temperature rise vs inlet",   "K",       1, 0 },
    { "Temperature vs ambient",      "K",       1, 0 },
    { "Dissolved O2",                "mg/L",    0, 0 },
    { "Nitrate-N",                   "mg/L",    0, 0 },
    { "Potassium",                   "mg/L",    0, 0 },
    { "Calcium",                     "mg/L",    0, 0 },
    { "EC (local)",                  "mS/cm",   0, 0 },
    { "Film velocity",               "m/s",     0, 0 },
    { "Film depth",                  "mm",      0, 0 },
    { "Channel flow",                "L/min",   0, 0 },
    { "Plant dry mass / max",        "",        0, 1 },
    { "Plant temperature factor fT", "",        0, 1 },
    { "Plant nutrient factor fN",    "",        0, 1 },
    { "Plant O2 factor fDO",         "",        0, 1 },
    { "pH (local)",                  "",        0, 0 },
    { "Root mat volume fraction",    "",        0, 0 },
    { "Evaporative cooling",         "W/m",     0, 0 },
};

// Diverging map for signed quantities: blue (-) - near-white (0) - red (+).
internal u32 nft_diverging(f32 t) {             // t in [-1, 1]
    t = Clamp(t, -1.0f, 1.0f);
    f32 r, g, b;
    if (t < 0.0f) { f32 a = -t; r = 0.93f - 0.80f * a; g = 0.93f - 0.55f * a; b = 0.95f - 0.20f * a; }
    else          { f32 a = t;  r = 0.95f - 0.20f * a; g = 0.93f - 0.75f * a; b = 0.93f - 0.80f * a; }
    return 0xFF000000u | ((u32)(b * 255.0f) << 16) | ((u32)(g * 255.0f) << 8) | (u32)(r * 255.0f);
}

// Local pH. Exact (speciation solve warm-started from the tank) for views up to
// NFT_PH_EXACT_CELLS; above that, linearized: pH_tank + dTA / beta with the
// tank's buffer capacity beta = dTA/dpH (cheap enough for a million cells a frame).
#define NFT_PH_EXACT_CELLS 32768
global f64 nft_ph_beta;          // meq/L per pH unit, refreshed per frame by nft_scene_fill_render
internal f32 nft_cell_ph_view(NftSystem *s, u32 g, u32 j, u32 l) {
    if ((u64)s->rails * s->cells <= NFT_PH_EXACT_CELLS) return (f32)nft_cell_ph(s, g, j, l, s->pH);
    f64 dTA = 0.0;
    for (u32 i = 0; i < NFT_IONS; ++i) dTA += nft_charge_sign[i] * (f64)s->group[g].rows[j].f[NFT_ION0 + i][l] / nft_meq_mg[i];
    return (f32)(s->pH + dTA / Maximum(nft_ph_beta, 1e-6));
}

internal f64 nft_tank_buffer_meq(NftSystem *s) {
    f64 c[NftF_Count];
    for (u32 k = 0; k < NftF_Count; ++k) c[k] = nft_reservoir_conc(s, k);
    NftWaterChem w = nft_water_chem(c, nft_R_conc(s));
    NftAcidBase ab = nft_acid_base(s->res.T_C, w.I);
    f64 pc = nft_pco2_atm(&s->params);
    return (nft_alk_species(&ab, s->pH + 0.01, pc, w.PT, w.NT) - nft_alk_species(&ab, s->pH - 0.01, pc, w.PT, w.NT)) / 0.02 * 1e3;
}

// Scalar of a layer at channel r, cell j.
internal f32 nft_layer_value(NftSystem *s, u32 layer, u32 r, u32 j) {
    u32 g = r / NFT_LANES, l = r % NFT_LANES;
    NftGroup *grp = s->group + g;
    switch (layer) {
        case NftLayer_Temperature:  return nft_abs(s, g, j, NftF_T, l);
        case NftLayer_DeltaInlet:   return grp->rows[j].f[NftF_T][l];                    // delta from the inlet (ref)
        case NftLayer_DeltaAmbient: return nft_abs(s, g, j, NftF_T, l) - s->T_amb;
        case NftLayer_DO:           return nft_abs(s, g, j, NftF_DO, l);
        case NftLayer_NO3:          return nft_abs(s, g, j, NftF_NO3, l);
        case NftLayer_K:            return nft_abs(s, g, j, NftF_K, l);
        case NftLayer_Ca:           return nft_abs(s, g, j, NftF_Ca, l);
        case NftLayer_EC: {
            f64 c[NftF_Count];
            for (u32 k = 0; k < NftF_Count; ++k) c[k] = nft_abs(s, g, j, k, l);
            return (f32)nft_ec_mS(c);
        }
        case NftLayer_pH:           return nft_cell_ph_view(s, g, j, l);
        case NftLayer_Velocity:     return grp->cold[j].vel[l];
        case NftLayer_Depth:        return grp->cold[j].depth[l] * 1000.0f;
        case NftLayer_Flow:         return grp->lp.flow_L_s[l] * 60.0f;
        case NftLayer_RootFrac:     return grp->cold[j].phi[l];
        case NftLayer_Evap: {       // latent heat leaving the open film, per metre of channel
            f64 Tw = nft_abs(s, g, j, NftF_T, l);
            f64 kg_s = (f64)grp->cold[j].ev[l] / s->dt_sub * (nft_es_Pa(Tw) - s->rh * nft_es_Pa(s->T_amb));
            return (f32)(nft_latent(Tw) * kg_s / s->cell_m);
        }
        default:                    return 0.0f;
    }
}

internal f32 nft_site_value(NftSystem *s, u32 layer, u32 r, u32 j) {
    NftGroup *grp = s->group + r / NFT_LANES;
    u32 l = r % NFT_LANES;
    NftSiteRow *site = grp->sites + j;
    switch (layer) {
        case NftLayer_fT:  return site->fT[l];
        case NftLayer_fN:  return site->fN[l];
        case NftLayer_fDO: return site->fDO[l];
        default:           return site->W[l] / grp->crop.wmax[l];
    }
}

// Channels side by side along x (inlet at -x, drawn with the bank's grade
// exaggerated 10x), the manifold across the inlet ends, plants above their
// sites. Channel cells are coloured by the chosen layer; plant layers colour
// the plants and grey the film.
internal void nft_scene_fill_render(NftScene *sc, RenderCommands *render, u32 layer) {
    TIMED_FUNCTION();
    NftSystem *s = &sc->sys;
    layer %= NftLayer_Count;
    const NftLayerInfo *info = nft_layers + layer;
    f32 lo = 1e30f, hi = -1e30f;
    if (layer == NftLayer_pH) nft_ph_beta = nft_tank_buffer_meq(s);
    if (info->plants) {
        lo = 0.0f; hi = 1.0f;
    } else {
        for (u32 r = 0; r < s->rails; ++r)
            for (u32 j = 0; j < s->cells; ++j) {
                f32 v = nft_layer_value(s, layer, r, j);
                lo = Minimum(lo, v); hi = Maximum(hi, v);
            }
        if (info->diverging) {
            f32 m = Maximum(Maximum(fabsf(lo), fabsf(hi)), 0.01f);
            lo = -m; hi = m;
        } else if (hi - lo < 1e-6f * Maximum(fabsf(hi), 1.0f)) {
            f32 m = 0.5f * (hi + lo), d = Maximum(fabsf(m) * 0.01f, 0.01f);
            lo = m - d; hi = m + d;
        }
    }
    sc->view_lo = lo; sc->view_hi = hi;
    f32 span_z = 2.4f, dz = s->rails > 1 ? span_z / (f32)(s->rails - 1) : 0.0f;
    f32 len_x = 3.6f;
    u32 n = 0, m = 0;
    for (u32 r = 0; r < s->rails; ++r) {
        NftGroup *g = s->group + r / NFT_LANES;
        u32 l = r % NFT_LANES;
        u32 bank = nft_rail_bank(s, r) ? 1 : 0;
        f32 z = -0.5f * span_z + dz * (f32)r + (bank ? 0.12f : -0.12f);
        f32 drop = 10.0f * s->params.hyd.slope[bank] * len_x;   // grade, exaggerated 10x
        for (u32 j = 0; j < s->cells; ++j) {
            f32 fx = (f32)j / (f32)(s->cells - 1);
            sc->cx[n] = -0.5f * len_x + len_x * fx;
            sc->cy[n] = 0.5f * drop - drop * fx;
            sc->cz[n] = z;
            if (info->plants) {
                sc->ccol[n] = 0xFF4A4440u;
            } else {
                f32 v = nft_layer_value(s, layer, r, j);
                sc->ccol[n] = info->diverging ? nft_diverging(v / hi) : dbd_colormap((v - lo) / (hi - lo));
            }
            ++n;
        }
        for (u32 j = 0; j < s->sites; ++j) {
            f32 fx = (f32)(s->site_offset + j * s->site_stride) / (f32)(s->cells - 1);
            f32 frac = g->sites[j].W[l] / g->crop.wmax[l];
            sc->px[m] = -0.5f * len_x + len_x * fx;
            sc->py[m] = 0.5f * drop - drop * fx + 0.04f + 0.30f * Clamp(frac, 0.0f, 1.2f);
            sc->pz[m] = z;
            if (info->plants) {
                sc->pcol[m] = dbd_colormap(nft_site_value(s, layer, r, j));
            } else {
                u32 gch = (u32)(90.0f + 150.0f * Clamp(frac, 0.0f, 1.0f));
                sc->pcol[m] = 0xFF000000u | (40u << 16) | (gch << 8) | 50u;
            }
            ++m;
        }
    }
    render->batch_count = 2;
    RenderPointBatch *b = render->batches;
    b->count = n; b->px = sc->cx; b->py = sc->cy; b->pz = sc->cz; b->rgba = sc->ccol;
    b->color = V4(1, 1, 1, 1); b->point_size = n <= 4096 ? 5.0f : 3.0f;
    b = render->batches + 1;
    b->count = m; b->px = sc->px; b->py = sc->py; b->pz = sc->pz; b->rgba = sc->pcol;
    b->color = V4(1, 1, 1, 1); b->point_size = m <= 1024 ? 8.0f : 4.0f;
}

internal void nft_scene_summary(NftScene *sc, char *out, u32 size) {
    NftSystem *s = &sc->sys;
    f64 c[NftF_Count];
    for (u32 k = 0; k < NftF_Count; ++k) c[k] = nft_reservoir_conc(s, k);
    f64 wa, wb, ft, fn, fd;
    nft_bank_plants(s, 0, &wa, &ft, &fn, &fd);
    nft_bank_plants(s, 1, &wb, &ft, &fn, &fd);
    nft_collect_harvest(s);
    snprintf(out, size,
             "nft %u ch x %u plants day %.2f | T amb %.1f res %.2f out A %.2f B %.2f C | chiller %.2f kWh | EC %.2f NO3-N %.0f K %.0f Ca %.0f DO %.1f | "
             "pH %.2f, acid %.0f base %.0f mmol | water: plants %.1f L, evaporation %.1f L, film max %.1f mm, holdup %.1f L | "
             "plant dm A %.2f B %.2f g, harvests A %.0f (%.0f g, %.0f bolted) B %.0f (%.0f g, %.0f bolted) | %.3f ms/window",
             s->rails, s->sites, s->t / 86400.0, s->T_amb, s->res.T_C, nft_bank_cell_mean(s, 0, NftF_T, s->cells - 1),
             nft_bank_cell_mean(s, 1, NftF_T, s->cells - 1), s->acc.chiller_elec_J / 3.6e6, nft_ec_mS(c), c[NftF_NO3],
             c[NftF_K], c[NftF_Ca], c[NftF_DO], s->pH, s->acc.acid_mmol, s->acc.base_mmol, s->acc.transpired_L,
             s->acc.evap_channel_L + s->acc.evap_tank_L, s->max_depth_m * 1e3, s->acc.holdup_L, wa, wb, s->acc.harvests[0], s->acc.harvest_g[0], s->acc.bolted[0],
             s->acc.harvests[1], s->acc.harvest_g[1], s->acc.bolted[1],
             sc->windows ? sc->wall_ms_total / (f64)sc->windows : 0.0);
}

internal u64 nft_scene_hash(NftScene *sc) {
    NftSystem *s = &sc->sys;
    u64 h = hash_fnv1a_64(&s->res, sizeof(s->res));
    h = hash_fnv1a_64(s->ref, sizeof(s->ref), h);
    h = hash_fnv1a_64(&s->R_meq, sizeof(s->R_meq), h);
    h = hash_fnv1a_64(&s->acc, sizeof(s->acc), h);
    for (u32 g = 0; g < s->groups; ++g) {
        h = hash_fnv1a_64(s->group[g].rows, sizeof(NftRow) * s->cells, h);
        h = hash_fnv1a_64(s->group[g].sites, sizeof(NftSiteRow) * s->sites, h);
    }
    return h;
}
