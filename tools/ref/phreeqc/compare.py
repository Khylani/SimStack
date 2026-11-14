#!/usr/bin/env python3
"""Sim water chemistry vs PHREEQC 3 (USGS), on the farm's source water and the NFT solution.

    tools/ref/phreeqc/build.sh && python3 -I tools/ref/phreeqc/compare.py [--phreeqc PATH] [--db PATH]

Cases (the sim side is build/ref_sim_water):
  source    speciation of the source water: DIC, CO2, log pCO2, ionic strength, saturation
  batch     closed-system titration with nitric + nitrous acid (15 % nitrite, the model's
            batch share): pH against acid N, and the N that reaches the batch pH
  nft       the working solution made up in the source water, open to 420 ppm CO2: pH,
            and the nitric acid that brings it to the pH set point

Nitrogen gets its own non-redox elements (Nta nitrate, Ntb nitrite, Amm ammonium) so PHREEQC
does not reduce nitrate at the default pe. HNO2 uses llnl.dat's log K and enthalpy;
everything else is phreeqc.dat. PHREEQC works in molality (mol/kgw); at these strengths the
difference from mol/L is ~0.2 % and is ignored.
"""
import argparse, os, re, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
REF = os.path.expanduser('~/Play/ref/phreeqc')

DEFS = """
SOLUTION_MASTER_SPECIES
    Nta     NtaO3-   0  Nta   14.0067
    Ntb     NtbO2-   0  Ntb   14.0067
    Amm     AmmH+    0  Amm   14.0067
SOLUTION_SPECIES
    NtaO3- = NtaO3-
        -log_k 0
        -gamma 3 0
    NtbO2- = NtbO2-
        -log_k 0
        -gamma 3 0
    NtbO2- + H+ = HNtbO2
        -log_k 3.2206
        -delta_h -14.782 kJ
    AmmH+ = AmmH+
        -log_k 0
        -gamma 2.5 0
    AmmH+ = Amm + H+
        -log_k -9.252
        -delta_h 12.48 kcal
    AmmH+ + SO4-2 = AmmHSO4-
        -log_k 1.211
        -delta_h 8.56 kJ
        -gamma 2.08 -0.0416
"""

def read_sim(exe):
    out = subprocess.run([exe], capture_output=True, text=True, check=True).stdout
    d, titr, ions = {}, {}, {}
    for line in out.splitlines():
        parts = line.split()
        key = parts[0]
        case, field = key.split('.', 1)
        if field == 'titr':
            titr.setdefault(case, []).append((float(parts[1]), float(parts[2])))
        elif field == 'ion':
            ions.setdefault(case, {})[parts[1]] = float(parts[2])
        else:
            d[key] = float(parts[1])
    return d, titr, ions

def run_phreeqc(phreeqc, db, text):
    with tempfile.TemporaryDirectory() as tmp:
        inp, out, sel = (os.path.join(tmp, n) for n in ('in.pqi', 'out.pqo', 'sel.tsv'))
        open(inp, 'w').write(text.replace('@SEL@', sel))
        r = subprocess.run([phreeqc, inp, out, db], capture_output=True, text=True)
        log = open(out).read() if os.path.exists(out) else ''
        if r.returncode != 0 or 'ERROR' in log:
            errs = '\n'.join(l for l in log.splitlines() if 'ERROR' in l)
            sys.exit('phreeqc failed:\n' + (errs or r.stderr or r.stdout))
        rows = [l.split('\t') for l in open(sel).read().splitlines() if l.strip()]
    head = [h.strip() for h in rows[0]]
    return [dict(zip(head, (v.strip() for v in row))) for row in rows[1:]]

SEL = """SELECTED_OUTPUT 1
    -file @SEL@
    -reset false
    -state true
    -solution true
    -step true
    -temperature true
    -pH true
    -ionic_strength true
    -alkalinity true
    -charge_balance true
    -percent_error true
    -totals C(4) Ca Nta Ntb
    -molalities CO2 HCO3- CO3-2 Ca+2 OH-
    -saturation_indices CO2(g) Calcite SiO2(a) Hydroxyapatite Gypsum
"""

def source_block(n, sim, case, alk_meq):
    T, ph = sim[case + '.T_C'], sim[case + '.source_pH']
    if case == 'w78':
        ions = "Ca 11\n    Mg 11\n    Na 14\n    K 0.7\n    Cl 20\n    S(6) 3.4 as SO4\n    Si 36 as SiO2\n    F 0.2"
        bal = ''
    else:   # design figure: only pH and KH were given; Na balances the charge
        ions = "Na 1 charge"
        bal = ''
    return f"""SOLUTION {n} {case}
    temp {T}
    pH {ph}
    units mg/kgw
    Alkalinity {alk_meq * 61.0168} as HCO3
    {ions}{bal}
"""

def fmt(x, d=3):
    return f"{x:.{d}f}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phreeqc', default=os.path.join(REF, 'bin', 'phreeqc'))
    ap.add_argument('--db', default=os.path.join(REF, 'share', 'doc', 'phreeqc', 'database', 'phreeqc.dat'))
    ap.add_argument('--sim', default=os.path.join(ROOT, 'build', 'ref_sim_water'))
    ap.add_argument('--out', default=os.path.join(HERE, 'results.md'))
    a = ap.parse_args()
    sim, titr, ions = read_sim(a.sim)
    lines = []
    P = lambda s='': (print(s), lines.append(s))

    P('# Sim water chemistry vs PHREEQC 3.8.6 (phreeqc.dat)')
    P()
    # ---------------------------------------------------------------- source water
    P('## Source water speciation (closed, at the skid source temperature)')
    P()
    P('| | design: sim | design: PHREEQC | W78: sim | W78: PHREEQC |')
    P('| --- | --- | --- | --- | --- |')
    res = {}
    for case in ('design', 'w78'):
        kh = sim[case + '.kh_meq']
        # the sim's KH is carbonate alkalinity; PHREEQC's input is total alkalinity
        # (carbonate + OH - H). Two passes: the first gives OH- and H+.
        alk = kh
        for _ in range(3):
            r = run_phreeqc(a.phreeqc, a.db, DEFS + SEL + source_block(1, sim, case, alk) + "END\n")[0]
            oh = float(r['m_OH-']) * 1e3
            h = 10 ** -float(r['pH']) * 1e3  # activity ~ concentration at this strength (fine for the correction)
            alk = kh + oh - h
        res[case] = r
    def row(label, s_design, p_design, s_w78, p_w78):
        P(f'| {label} | {s_design} | {p_design} | {s_w78} | {p_w78} |')
    rd, rw = res['design'], res['w78']
    row('DIC, mmol/L', fmt(sim['design.dic_mmol']), fmt(float(rd['C(4)']) * 1e3), fmt(sim['w78.dic_mmol']), fmt(float(rw['C(4)']) * 1e3))
    row('CO₂*, mmol/L', fmt(sim['design.co2_mmol'], 5), fmt(float(rd['m_CO2']) * 1e3, 5), fmt(sim['w78.co2_mmol']), fmt(float(rw['m_CO2']) * 1e3))
    row('log pCO₂', fmt(sim['design.log_pco2'], 2), fmt(float(rd['si_CO2(g)']), 2), fmt(sim['w78.log_pco2'], 2), fmt(float(rw['si_CO2(g)']), 2))
    row('ionic strength, mmol/L', fmt(sim['design.I'] * 1e3, 2), fmt(float(rd['mu']) * 1e3, 2), fmt(sim['w78.I'] * 1e3, 2), fmt(float(rw['mu']) * 1e3, 2))
    row('calcite SI', '—', fmt(float(rd['si_Calcite']), 2) if rd['si_Calcite'] not in ('-999.999', '') else '—', '—', fmt(float(rw['si_Calcite']), 2))
    row('amorphous silica SI', '—', '—', '—', fmt(float(rw['si_SiO2(a)']), 2))
    row('charge balance error, %', '—', fmt(float(rd['pct_err']), 2), '—', fmt(float(rw['pct_err']), 2))
    P()
    # ---------------------------------------------------------------- batch titration
    P('## Batch: closed-system acid N against pH (85 % nitric, 15 % nitrous)')
    P()
    P('| acid N, mmol/L | design: sim | design: PHREEQC | W78: sim | W78: PHREEQC |')
    P('| --- | --- | --- | --- | --- |')
    curves = {}
    for case in ('design', 'w78'):
        kh = sim[case + '.kh_meq']
        alk = kh + float(res[case]['m_OH-']) * 1e3 - 10 ** -float(res[case]['pH']) * 1e3
        steps = [n for n, _ in titr[case] if n > 0]
        react = "REACTION 1\n    HNtaO3 0.85\n    HNtbO2 0.15\n    " + ' '.join(f'{n * 1e-3:.6e}' for n in steps) + "\nINCREMENTAL_REACTIONS false\n"
        rr = run_phreeqc(a.phreeqc, a.db, DEFS + SEL + source_block(1, sim, case, alk) + "USE solution 1\n" + react + "END\n")
        ph = [float(r['pH']) for r in rr if r['state'] == 'react']
        curves[case] = [(0.0, float(res[case]['pH']))] + list(zip(steps, ph))
    worst = {}
    for i, (n, _) in enumerate(titr['design']):
        sd, sw = titr['design'][i][1], titr['w78'][i][1]
        pd, pw = curves['design'][i][1], curves['w78'][i][1]
        worst['design'] = max(worst.get('design', 0), abs(sd - pd))
        worst['w78'] = max(worst.get('w78', 0), abs(sw - pw))
        if i % 3 == 0 or abs(n - 1.3) < 1e-9 or abs(n - 1.4) < 1e-9:
            P(f'| {n:.1f} | {sd:.3f} | {pd:.3f} | {sw:.3f} | {pw:.3f} |')
    def n_at(curve, target):
        for (n0, p0), (n1, p1) in zip(curve, curve[1:]):
            if p0 >= target >= p1:
                return n0 + (n1 - n0) * (p0 - target) / (p0 - p1)
        return float('nan')
    tgt = sim['w78.batch_pH']
    P()
    P(f"Acid N to reach pH {tgt:.1f}: design sim {sim['design.n_to_target_mmol']:.3f} vs PHREEQC "
      f"{n_at(curves['design'], tgt):.3f} mmol/L; W78 sim {sim['w78.n_to_target_mmol']:.3f} vs PHREEQC {n_at(curves['w78'], tgt):.3f} mmol/L. "
      f"Worst pH difference along the curves: design {worst['design']:.3f}, W78 {worst['w78']:.3f}.")
    P()
    # ---------------------------------------------------------------- NFT solution
    P('## NFT working solution, open to 420 ppm CO₂')
    P()
    P('| case | pH: sim | pH: PHREEQC | HNO₃ to pH 5.8 (mmol/L): sim | PHREEQC | ionic strength: sim | PHREEQC | free Ca²⁺ / total | hydroxyapatite SI | gypsum SI |')
    P('| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |')
    for case in ('nft_w78_20', 'nft_w78_30', 'nft_design_20'):
        T = sim[case + '.T_C']; ion = ions[case]
        extra = ''
        if sim[case + '.na'] > 0: extra += f"\n    Na {sim[case + '.na']}"
        if sim[case + '.sio2'] > 0: extra += f"\n    Si {sim[case + '.sio2']} as SiO2"
        sol = f"""SOLUTION 1 {case}
    temp {T}
    pH 6.5
    units mg/kgw
    Alkalinity {sim[case + '.TA_meq'] * 61.0168} as HCO3
    Nta {ion['NO3-N']}
    Amm {ion['NH4-N']}
    P {ion['P']}
    K {ion['K']}
    Ca {ion['Ca']}
    Mg {ion['Mg']}
    S(6) {ion['S']}{extra}
    Cl 10 charge
EQUILIBRIUM_PHASES 1
    CO2(g) {sim[case + '.log_pco2']} 10
"""
        rr = run_phreeqc(a.phreeqc, a.db, DEFS + SEL + sol + "END\n")
        eq = [r for r in rr if r['state'] == 'react'][-1]
        # nitric acid titration, open system: find the acid that reaches the set point
        set_ph = sim[case + '.ph_set']
        steps = [0.1 * k for k in range(1, 31)]
        react = "USE solution 1\nEQUILIBRIUM_PHASES 1\n    CO2(g) " + str(sim[case + '.log_pco2']) + " 10\nREACTION 1\n    HNtaO3 1\n    " + \
                ' '.join(f'{n * 1e-3:.6e}' for n in steps) + "\nINCREMENTAL_REACTIONS false\n"
        rt = run_phreeqc(a.phreeqc, a.db, DEFS + SEL + sol.split('EQUILIBRIUM_PHASES')[0] + "SAVE solution 1\nEND\n" + react + "END\n")
        tcurve = [(0.0, float(eq['pH']))] + [(n, float(r['pH'])) for n, r in zip(steps, [r for r in rt if r['state'] == 'react'][-len(steps):])]
        p_acid = n_at(tcurve, set_ph)
        free_ca = float(eq['m_Ca+2']) / float(eq['Ca'])
        P(f"| {case} | {sim[case + '.pH']:.3f} | {float(eq['pH']):.3f} | {sim[case + '.acid_to_set_mmol']:.3f} | {p_acid:.3f} | "
          f"{sim[case + '.I'] * 1e3:.1f} | {float(eq['mu']) * 1e3:.1f} | {free_ca:.2f} | {float(eq['si_Hydroxyapatite']):.2f} | {float(eq['si_Gypsum']):.2f} |")
    P()
    open(a.out, 'w').write('\n'.join(lines) + '\n')
    print(f'\nwrote {a.out}')

if __name__ == '__main__':
    main()
