"""
Dissipative Asymmetry — Property 8: Additivity verification.

What this script does:
  Verifies that when multiple independent error mechanisms act on
  the same element, the asymmetry coefficients add exactly:
  alpha_total = alpha_1 + alpha_2 + ... + alpha_N

  Three levels of verification:
  1. Algebraic: direct computation, error must be < 1e-14
  2. Monte Carlo: simulate combined mechanisms, compare measured
     alpha to sum of individual alphas
  3. Stress test: 1000 random mechanism combinations, all must
     match to machine precision

What this script does NOT do:
  - It does NOT test composition on real hardware data
  - It does NOT test series (cascaded channels) — only parallel
    (independent mechanisms on the same element)

Where the parameters come from:
  The algebraic test uses 4 representative error mechanisms:
  - Radiation: pd=0.03, pe=0.005 (cosmic ray damage)
  - Crosstalk: pd=0.02, pe=0.008 (electromagnetic coupling)
  - Thermal:   pd=0.01, pe=0.003 (thermal noise)
  - Aging:     pd=0.008, pe=0.004 (device degradation)
  These are typical values; exact sources vary by system.

  The Monte Carlo test uses two mechanisms (A and B) with
  parameters from Qubit and DRAM domains.

  The stress test uses randomly generated pd/pe values.

Why additivity holds:
  For independent mechanisms, survival probabilities multiply:
    (1 - pd_total) = product_i (1 - pd_i)
    (1 - pe_total) = product_i (1 - pe_i)
  Therefore:
    alpha_total = ln[product(1-pd_i) / product(1-pe_i)]
                = sum_i ln[(1-pd_i) / (1-pe_i)]
                = sum_i alpha_i
  This is a property of logarithms, not a new result.

Expected output:
  - Algebraic: difference < 1e-14 (machine precision)
  - Monte Carlo: mean error < 5% (sampling noise)
  - Stress test: max error < 1e-12 across 1000 tests

Requirements: numpy, scipy
Usage: python3 verify_additivity.py
"""

import numpy as np
from scipy import stats


def verify_algebraic():
    """
    Algebraic verification: compute alpha_total two ways and compare.
    Way 1: sum individual alpha_i
    Way 2: compute alpha from combined pd_total, pe_total
    They must be identical to machine precision.
    """
    print("=" * 60)
    print("1. ALGEBRAIC VERIFICATION")
    print("=" * 60)

    # Four independent error mechanisms with different rates
    mechanisms = [
        ("Radiation",  0.03,  0.005),
        ("Crosstalk",  0.02,  0.008),
        ("Thermal",    0.01,  0.003),
        ("Aging",      0.008, 0.004),
    ]

    alphas = []
    prod_1_minus_pd = 1.0
    prod_1_minus_pe = 1.0

    print(f"\n  {'Mechanism':<15} {'pd':<8} {'pe':<8} {'alpha_i':<12}")
    print(f"  {'-' * 43}")

    for name, pd, pe in mechanisms:
        alpha_i = np.log((1 - pd) / (1 - pe))
        alphas.append(alpha_i)
        # Accumulate products for combined calculation
        prod_1_minus_pd *= (1 - pd)
        prod_1_minus_pe *= (1 - pe)
        print(f"  {name:<15} {pd:<8} {pe:<8} {alpha_i:<12.6f}")

    # Way 1: sum of individual alphas
    alpha_sum = sum(alphas)

    # Way 2: alpha from combined pd_total, pe_total
    pd_total = 1 - prod_1_minus_pd
    pe_total = 1 - prod_1_minus_pe
    alpha_combined = np.log((1 - pd_total) / (1 - pe_total))

    print(f"\n  Sum of alpha_i:     {alpha_sum:.10f}")
    print(f"  alpha_combined:     {alpha_combined:.10f}")
    print(f"  Difference:         {abs(alpha_sum - alpha_combined):.2e}")
    print(f"  Exact:              {'PASS' if abs(alpha_sum - alpha_combined) < 1e-14 else 'FAIL'}")

    return abs(alpha_sum - alpha_combined) < 1e-14


def verify_monte_carlo(n_trials=50, N=4, n_sim=200000):
    """
    Monte Carlo verification: apply two mechanisms to the same elements,
    measure alpha of the combined system, compare to alpha_A + alpha_B.

    Args:
        n_trials: number of independent trials
        N: number of elements per pixel (bits)
        n_sim: pixels per trial
    """
    print(f"\n{'=' * 60}")
    print(f"2. MONTE CARLO VERIFICATION ({n_trials} trials)")
    print(f"{'=' * 60}")

    # Two mechanisms with different asymmetry
    pd_a, pe_a = 0.03, 0.005  # mechanism A (e.g., radiation)
    pd_b, pe_b = 0.02, 0.008  # mechanism B (e.g., crosstalk)

    alpha_a_true = np.log((1 - pd_a) / (1 - pe_a))
    alpha_b_true = np.log((1 - pd_b) / (1 - pe_b))

    print(f"\n  Mechanism A: pd={pd_a}, pe={pe_a}, alpha={alpha_a_true:.6f}")
    print(f"  Mechanism B: pd={pd_b}, pe={pe_b}, alpha={alpha_b_true:.6f}")
    print(f"  Theoretical sum: {alpha_a_true + alpha_b_true:.6f}")

    errors = []
    for trial in range(n_trials):
        np.random.seed(trial)
        pixels = np.random.randint(0, 256, size=N * 50000).astype(np.uint16)

        # Apply mechanism A: each bit independently
        correct_a = np.ones(len(pixels), dtype=bool)
        for bit in range(8):
            b = (pixels >> bit) & 1
            correct_a &= ~((b == 1) & (np.random.random(len(pixels)) < pd_a))
            correct_a &= ~((b == 0) & (np.random.random(len(pixels)) < pe_a))

        # Apply mechanism B: each bit independently
        correct_b = np.ones(len(pixels), dtype=bool)
        for bit in range(8):
            b = (pixels >> bit) & 1
            correct_b &= ~((b == 1) & (np.random.random(len(pixels)) < pd_b))
            correct_b &= ~((b == 0) & (np.random.random(len(pixels)) < pe_b))

        # Combined: correct only if BOTH mechanisms don't flip
        correct_combined = correct_a & correct_b

        # Compute Hamming weight of each pixel
        weights = np.zeros(len(pixels), dtype=int)
        t = pixels.copy()
        while np.any(t > 0):
            weights += (t & 1).astype(int)
            t >>= 1

        # Measure alpha from combined system
        ks, lrs = [], []
        p0 = None
        for k in range(9):
            mask = weights == k
            n = np.sum(mask)
            if n >= 30:
                pc = np.sum(correct_combined[mask]) / n
                if k == 0:
                    p0 = pc
                if p0 and p0 > 0 and pc > 0:
                    ks.append(k)
                    lrs.append(np.log(pc / p0))

        if len(ks) >= 3:
            sl, _, _, _, _ = stats.linregress(ks, lrs)
            # Compare measured alpha to theoretical sum
            pred = alpha_a_true + alpha_b_true
            err = abs(sl - pred) / abs(pred) * 100
            errors.append(err)

    errors = np.array(errors)
    print(f"\n  Trials completed: {len(errors)}/{n_trials}")
    print(f"  Mean error: {errors.mean():.2f}%")
    print(f"  Max error:  {errors.max():.2f}%")
    print(f"  Trials < 5%: {np.sum(errors < 5)}/{len(errors)}")
    print(f"  Result: {'PASS' if errors.mean() < 5 else 'FAIL'}")

    return errors.mean() < 5


def verify_stress_test():
    """
    Stress test: 1000 random combinations of 2-9 mechanisms.
    Verifies algebraic identity holds for arbitrary pd/pe values.
    All must match to machine precision (< 1e-12).
    """
    print(f"\n{'=' * 60}")
    print("3. STRESS TEST (1000 random combinations)")
    print(f"{'=' * 60}")

    np.random.seed(42)
    max_error = 0

    for _ in range(1000):
        # Random number of mechanisms (2 to 9)
        n_mech = np.random.randint(2, 10)
        # Random pd and pe values
        pds = np.random.uniform(0.001, 0.1, n_mech)
        pes = np.random.uniform(0.0001, 0.05, n_mech)

        # Way 1: sum of individual alphas
        alphas = np.log((1 - pds) / (1 - pes))
        alpha_sum = np.sum(alphas)

        # Way 2: alpha from combined rates
        pd_tot = 1 - np.prod(1 - pds)
        pe_tot = 1 - np.prod(1 - pes)
        alpha_combined = np.log((1 - pd_tot) / (1 - pe_tot))

        error = abs(alpha_combined - alpha_sum)
        max_error = max(max_error, error)

    print(f"\n  Max error across 1000 tests: {max_error:.2e}")
    print(f"  Machine precision: {'PASS' if max_error < 1e-12 else 'FAIL'}")

    return max_error < 1e-12


if __name__ == "__main__":
    print("=" * 60)
    print("Dissipative Asymmetry — Additivity (Property 8)")
    print("alpha_total = alpha_1 + alpha_2 + ... + alpha_N")
    print("=" * 60)

    r1 = verify_algebraic()
    r2 = verify_monte_carlo()
    r3 = verify_stress_test()

    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    print(f"  Algebraic:    {'PASS' if r1 else 'FAIL'}")
    print(f"  Monte Carlo:  {'PASS' if r2 else 'FAIL'}")
    print(f"  Stress test:  {'PASS' if r3 else 'FAIL'}")
    print(f"  Overall:      {'ALL PASS' if all([r1, r2, r3]) else 'SOME FAILED'}")
