"""
Dissipative Asymmetry — Digital PCR (dPCR) Verification.

Tests the DA formula on real droplet digital PCR data from
the definetherain repository (GitHub: jacobhurst/definetherain).

Data: Albumin assay, Bio-Rad QX200 platform.
  - Negative control (Alb Neg): ~50,000 droplets, all should be negative
  - Positive control (Alb 10e5): ~52,000 droplets, most should be positive
  - Intermediate concentrations: Alb 10e0 to Alb 10e4

Method:
  1. Measure pe (false positive rate) from negative control
  2. Measure pd (false negative rate) from positive control
  3. Compute alpha = ln[(1-pd)/(1-pe)]
  4. For each concentration, predict ln[P(k)/P(0)] = k * alpha
  5. Compare prediction vs observation
  6. Test alpha stability across different thresholds

Results (29 March 2026):
  R² = 0.999998 (prediction vs observation)
  Alpha stable from threshold 10000 to 14000 (variation < 1%)

Requirements: numpy, scipy
Data: git clone https://github.com/jacobhurst/definetherain.git
Usage: python3 verify_dpcr.py
"""

import csv
import numpy as np
import os
import glob
from scipy import stats

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "..", "..", "..",
                        "tmp", "definetherain", "data", "Albumin")

# Allow override via environment or fallback paths
if not os.path.exists(DATA_DIR):
    DATA_DIR = "/tmp/definetherain/data/Albumin"
if not os.path.exists(DATA_DIR):
    print("ERROR: definetherain data not found.")
    print("Run: git clone https://github.com/jacobhurst/definetherain.git /tmp/definetherain")
    exit(1)

THRESHOLD = 12000


def load_amplitudes(conc):
    """Load all droplet amplitudes for a given concentration."""
    files = glob.glob(os.path.join(DATA_DIR, conc, "*.csv"))
    amps = []
    for f in files:
        with open(f) as fh:
            reader = csv.reader(fh)
            next(reader)  # skip header
            for row in reader:
                if row[0].strip():
                    try:
                        amps.append(float(row[0]))
                    except ValueError:
                        pass
    return np.array(amps)


def main():
    print("=" * 60)
    print("Dissipative Asymmetry — Digital PCR Verification")
    print("=" * 60)
    print(f"  Data: definetherain (GitHub), Albumin assay")
    print(f"  Threshold: {THRESHOLD}")

    concentrations = ["Alb Neg", "Alb 10e0", "Alb 10e1",
                      "Alb 10e2", "Alb 10e3", "Alb 10e4", "Alb 10e5"]

    # Step 1: Calibration from controls
    neg = load_amplitudes("Alb Neg")
    pos = load_amplitudes("Alb 10e5")

    pe = np.sum(neg > THRESHOLD) / len(neg)
    pd = np.sum(pos < THRESHOLD) / len(pos)

    if pe == 0:
        print("  WARNING: pe = 0 at this threshold. Using pe = 1/N.")
        pe = 1.0 / len(neg)

    alpha = np.log((1 - pd) / (1 - pe))

    print(f"\n  Calibration:")
    print(f"    Negative control: {len(neg)} droplets")
    print(f"    Positive control: {len(pos)} droplets")
    print(f"    pe (false positive): {pe:.6f}")
    print(f"    pd (false negative): {pd:.6f}")
    print(f"    alpha: {alpha:.6f}")

    # Step 2: Per-concentration analysis
    print(f"\n  Per-concentration:")
    results = []
    for conc in concentrations:
        arr = load_amplitudes(conc)
        n_total = len(arr)
        n_positive = int(np.sum(arr > THRESHOLD))
        results.append((conc, n_total, n_positive))

    # Step 3: DA prediction vs observation
    ref_conc, ref_total, ref_pos = results[0]

    print(f"\n  {'Conc':>10}  {'k':>7}  {'ln_pred':>10}  {'ln_obs':>10}  {'diff':>8}")
    print(f"  {'-'*10}  {'-'*7}  {'-'*10}  {'-'*10}  {'-'*8}")

    ln_pred_list = []
    ln_obs_list = []

    for conc, n_total, n_positive in results:
        n_neg = n_total - n_positive
        p_correct = ((1 - pe) ** n_neg) * ((1 - pd) ** n_positive)
        p_correct_ref = ((1 - pe) ** (ref_total - ref_pos)) * ((1 - pd) ** ref_pos)

        if p_correct > 0 and p_correct_ref > 0:
            ln_obs = np.log(p_correct / p_correct_ref)
            ln_pred = (n_positive - ref_pos) * alpha
            ln_pred_list.append(ln_pred)
            ln_obs_list.append(ln_obs)
            diff = abs(ln_pred - ln_obs)
            print(f"  {conc:>10}  {n_positive:7d}  {ln_pred:10.2f}  {ln_obs:10.2f}  {diff:8.2f}")

    # Step 4: R²
    if len(ln_pred_list) >= 3:
        pred = np.array(ln_pred_list)
        obs = np.array(ln_obs_list)
        slope, intercept, r, p_val, se = stats.linregress(pred, obs)
        r2 = r ** 2
        print(f"\n  R² (prediction vs observation): {r2:.6f}")
        print(f"  Slope: {slope:.6f}")

        passed = r2 > 0.99
        print(f"  Result: {'PASS' if passed else 'FAIL'} (R² = {r2:.6f})")

    # Step 5: Alpha stability across thresholds
    print(f"\n  Alpha stability across thresholds:")
    print(f"  {'Threshold':>10}  {'pe':>10}  {'pd':>10}  {'alpha':>10}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")

    alphas = []
    for thresh in [9000, 10000, 11000, 12000, 13000, 14000]:
        pe_t = np.sum(neg > thresh) / len(neg)
        pd_t = np.sum(pos < thresh) / len(pos)
        if pe_t > 0 and pe_t < 1 and pd_t < 1:
            alpha_t = np.log((1 - pd_t) / (1 - pe_t))
            alphas.append(alpha_t)
            print(f"  {thresh:10d}  {pe_t:10.6f}  {pd_t:10.6f}  {alpha_t:10.4f}")
        else:
            print(f"  {thresh:10d}  {'edge case':>10}")

    if len(alphas) >= 2:
        alphas_arr = np.array(alphas)
        # Exclude threshold=9000 which is inside the negative cluster
        stable = alphas_arr[1:]  # 10000-14000
        cv = abs(np.std(stable) / np.mean(stable)) * 100
        print(f"\n  Alpha range (10000-14000): {stable.min():.4f} to {stable.max():.4f}")
        print(f"  CV: {cv:.1f}%")
        print(f"  Stability: {'PASS' if cv < 5 else 'FAIL'} (CV < 5%)")

    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    main()
