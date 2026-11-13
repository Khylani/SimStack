#!/usr/bin/env python3
"""Extends ChemPlasKin's air_plasma.yaml with N2O5 (NO2 + NO3 <=> N2O5), the NOx
reservoir of ozone-rich air that the combustion-oriented mechanism lacks. Without
it, NO2 + O -> NO + O2 / NO + O3 run as a catalytic ozone sink and the reference
wrongly falls into NOx mode at 330 K.

Rates as src/sim/kinetics/air_mech.cpp (JPL, Fc = 0.6; reverse = forward / Keq,
Keq = 2.7e-27 exp(11000/T) cm^3); thermo NIST (constant cp); transport only
because the phase declares it (unused in 0D).

usage: add_n2o5.py <ChemPlasKin>/data/PAC_kinetics/air_plasma.yaml out.yaml
"""
import sys
NA = 6.02214076e23
src = open(sys.argv[1]).read()
src = src.replace("NO2-, N2O-, NO3-, OH-, H-, O4,", "NO2-, N2O-, NO3-, OH-, H-, O4, N2O5,", 1)
assert "O4, N2O5," in src
species = '''- name: N2O5
  composition: {N: 2, O: 5}
  thermo:
    model: constant-cp
    T0: 298.15
    h0: 13.3 kJ/mol
    s0: 355.7 J/mol/K
    cp0: 95.3 J/mol/K
  transport:
    model: gas
    geometry: nonlinear
    diameter: 4.8
    well-depth: 400.0
'''
i = src.index('\nreactions:')
src = src[:i] + '\n' + species + src[i:]
Ea = 11000 * 1.98720425864
k0f, kif = 2.0e-30 * NA**2 * 300**4.4, 1.4e-12 * NA * 300**0.7
k0r, kir = 2.0e-30 / 2.7e-27 * NA * 300**4.4, 1.4e-12 / 2.7e-27 * 300**0.7
src = src.rstrip('\n') + f'''
- equation: NO2 + NO3 (+M) => N2O5 (+M)
  type: falloff
  low-P-rate-constant: {{A: {k0f:.6e}, b: -4.4, Ea: 0}}
  high-P-rate-constant: {{A: {kif:.6e}, b: -0.7, Ea: 0}}
  Troe: {{A: 0.6, T3: 1.0e-30, T1: 1.0e+30}}
- equation: N2O5 (+M) => NO2 + NO3 (+M)
  type: falloff
  low-P-rate-constant: {{A: {k0r:.6e}, b: -4.4, Ea: {Ea:.1f}}}
  high-P-rate-constant: {{A: {kir:.6e}, b: -0.7, Ea: {Ea:.1f}}}
  Troe: {{A: 0.6, T3: 1.0e-30, T1: 1.0e+30}}
'''
open(sys.argv[2], 'w').write(src)
