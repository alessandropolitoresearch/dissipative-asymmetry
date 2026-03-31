"""
Dissipative Asymmetry — Monte Carlo verification across 5 domains.

What this script does:
  Verifies that ln[P(k)/P(0)] = k * alpha holds for independent
  discrete elements with asymmetric transition rates, using Monte
  Carlo simulation across 5 physically distinct domains.

What this script does NOT do:
  - It does NOT use real hardware data (see verify_ibm_quantum.py for quantum)
  - It does NOT prove the formula (the formula is an algebraic identity)
  - It confirms that the Monte Carlo simulation matches the theory

Where the parameters come from:
  - Qubit: pd=0.03, pe=0.005 — typical IBM superconducting qubit
    readout error rates. Source: Tannu & Qureshi, MICRO 2019 [1]
  - DRAM: pd=0.02, pe=0.01 — typical retention error rates.
    Source: CMU-SAFARI ReadDisturbanceVTS25 [5]
  - DNA: pd=0.05, pe=0.026 — scaled to maintain the measured
    GC->AT / AT->GC ratio of 1.93. Source: Lynch, PNAS 2010 [4]
  - SPAD: pd=0.01, pe=0.05 — inverted asymmetry (pe > pd),
    dark count exceeds missed detection.
    Source: Bronzi et al., IEEE Sensors J. 2016 [8]
  - CCD: pd=0.04, pe=0.001 — extreme asymmetry typical of
    radiation-damaged CCD pixels.
    Source: STScI MAST Archive, HST ACS/WFC dark frames [10]

  Reference numbers [n] refer to the paper:
  "Dissipative Asymmetry" (Polito, 2026), DOI: 10.5281/zenodo.19164107

How it works:
  For each domain, the script:
  1. Sets pd (P(1->0 error)) and pe (P(0->1 error))
  2. Computes alpha_theory = ln[(1-pd)/(1-pe)]
  3. Simulates N_SIM independent trials for each composition k
     (k = number of elements in excited state)
  4. Measures P(all elements read correctly | k excited)
  5. Computes ln[P(k)/P(0)] and fits a line vs k
  6. Compares the measured slope to alpha_theory
  7. Reports R^2 of the linear fit

Expected output:
  R^2 > 0.99 for all 5 domains (the formula is exact;
  deviations are sampling noise from finite simulations)

Requirements: numpy, scipy
Usage: python3 verify_qubit.py
"""

import numpy as np
from scipy import stats

# Number of Monte Carlo trials per composition k
N_SIM = 100000


def verify_domain(name, pd, pe, N_elements, n_sim=N_SIM):
    """
    Verify DA formula for one domain via Monte Carlo.

    Args:
        name: domain name (for display)
        pd: probability of 1->0 error (dominant in passive systems)
        pe: probability of 0->1 error (rare in passive systems)
        N_elements: number of binary elements in the system
        n_sim: number of Monte Carlo trials per k

    Returns:
        R^2 of linear fit ln[P(k)/P(0)] vs k, or None if insufficient data
    """
    # Theoretical alpha from pd and pe
    alpha_theory = np.log((1 - pd) / (1 - pe))

    k_values = range(0, N_elements + 1)
    p_correct = []

    # For each composition k (number of excited elements):
    # simulate n_sim trials and count how many have all elements read correctly
    for k in k_values:
        n_correct = 0
        for _ in range(n_sim):
            ok = True
            for i in range(N_elements):
                if i < k:
                    # Element is excited (1): error with probability pd
                    if np.random.random() < pd:
                        ok = False
                        break
                else:
                    # Element is ground (0): error with probability pe
                    if np.random.random() < pe:
                        ok = False
                        break
            if ok:
                n_correct += 1
        p_correct.append(n_correct / n_sim)

    # P(0) = probability of correct readout when all elements are ground
    p0 = p_correct[0]
    if p0 == 0:
        return None

    # Compute log-ratios: ln[P(k)/P(0)]
    log_ratios = []
    valid_k = []
    for k, p in zip(k_values, p_correct):
        if p > 0:
            log_ratios.append(np.log(p / p0))
            valid_k.append(k)

    if len(valid_k) < 3:
        return None

    # Linear fit: ln[P(k)/P(0)] = slope * k + intercept
    # DA predicts: slope = alpha, intercept = 0
    slope, intercept, r_value, p_value, std_err = stats.linregress(valid_k, log_ratios)
    r_squared = r_value ** 2

    print(f"\n  {name}")
    print(f"  pd={pd}, pe={pe}, N={N_elements}")
    print(f"  alpha_theory = {alpha_theory:.6f}")
    print(f"  alpha_MC     = {slope:.6f}")
    print(f"  R^2          = {r_squared:.6f}")
    print(f"  Match: {'PASS' if abs(slope - alpha_theory) < 0.01 else 'FAIL'}")

    return r_squared


if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 60)
    print("Dissipative Asymmetry — Monte Carlo Verification")
    print("5 domains, independent discrete elements")
    print("=" * 60)

    # Each tuple: (name, pd, pe, N_elements)
    # pd = P(excited element misread as ground)
    # pe = P(ground element misread as excited)
    # N_elements = number of independent binary elements
    domains = [
        ("Qubit (T1 relaxation)",       0.03,  0.005, 4),   # [1]
        ("DRAM (charge leakage)",       0.02,  0.01,  8),   # [5]
        ("DNA (cytosine deamination)",  0.05,  0.026, 4),   # [4]
        ("SPAD (inverted asymmetry)",   0.01,  0.05,  4),   # [8]
        ("CCD (dark current)",          0.04,  0.001, 8),   # [10]
    ]

    results = []
    for name, pd, pe, N in domains:
        r2 = verify_domain(name, pd, pe, N)
        if r2 is not None:
            results.append((name, r2))

    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    all_pass = True
    for name, r2 in results:
        status = "PASS" if r2 > 0.99 else "FAIL"
        if r2 <= 0.99:
            all_pass = False
        print(f"  {name}: R^2 = {r2:.6f} [{status}]")

    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print(f"  Domains tested: {len(results)}/5")
