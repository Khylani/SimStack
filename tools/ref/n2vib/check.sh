#!/bin/bash
# Regression for the N2(v) reference: detailed balance, the vibrational temperature against
# the CARS measurement in a 1 atm air DBD (Baeva et al. 1999: ~800 K), and the vibrational
# (Zeldovich) share of fixed nitrogen at the PAW skid's conditions under all three
# Zeldovich models. Exit status 1 on any failure.
cd "$(dirname "$0")/../../.."
B=./build/n2vib_ref
[ -x $B ] || tools/ref/n2vib/build.sh >/dev/null
fail=0
db=$($B 330 0 30 1.64 0.05 180 check 2>/dev/null | sed -n 's/.*worst relative change of any species \([0-9.e+-]*\).*/\1/p')
python3 -c "import sys; sys.exit(0 if float('$db') < 1e-12 else 1)" && echo "detailed balance: $db" || { echo "FAIL detailed balance $db"; fail=1; }
for T in 330 413; do for X in 0 0.001; do
  base=$($B $T $X 30 1.64 1.32 180 novib 2>/dev/null | sed -n 's/.*fixed N \([0-9.]*\) ppm.*/\1/p')
  for M in "" step13 step11; do
    out=$($B $T $X 30 1.64 1.32 180 $M 2>/dev/null)
    fn=$(echo "$out" | sed -n 's/.*fixed N \([0-9.]*\) ppm.*/\1/p'); tv=$(echo "$out" | sed -n 's/.*T_v(0,1) \([0-9]*\) K.*/\1/p')
    r=$(python3 -c "print('%.4f' % ($fn/$base - 1))")
    printf "T %s K, H2O %4.0f ppm, %-7s: T_v %s K, vibrational share of fixed N %+.2f %%\n" $T $(python3 -c "print($X*1e6)") "${M:-FM}" $tv $(python3 -c "print(100*$r)")
    python3 -c "import sys; sys.exit(0 if abs($r) < 0.01 else 1)" || { echo "  FAIL: vibrational share above 1 %"; fail=1; }
    if [ "$X" = 0 ]; then python3 -c "import sys; sys.exit(0 if 650 <= $tv <= 900 else 1)" || { echo "  FAIL: dry-air T_v outside 650-900 K (CARS ~800 K)"; fail=1; }; fi
  done
done; done
exit $fail
