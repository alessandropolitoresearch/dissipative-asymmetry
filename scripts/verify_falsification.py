"""
Dissipative Asymmetry — Falsification tests.

What this script does:
  Systematically attempts to break the invariant ln[P(k)/P(0)] = k * alpha
  by violating each assumption one at a time. The invariant should hold
  under valid conditions and break under invalid ones.

What this script does NOT do:
  - It does NOT use real hardware data
  - It does NOT test all possible violation modes
  - The global correlation test uses a simplified model;
    the R^2 = 0.94 value cited in the paper comes from a
    more detailed simulation in earlier sessions

Tests:
  1. F-invariance: vary operating conditions F from 0.80 to 1.00.
     Alpha must not change. (Tests Property 1)
  2. N-invariance: vary number of elements N from 2 to 128.
     Alpha must not change. (Tests Property 3)
  3. Non-uniform pd: spread pd by 50% across elements.
     Invariant must hold within 5%. (Tests robustness)
  4. Adjacent-element correlation: introduce c=0.05 between
     adjacent elements. Independent form should still hold
     approximately. (Tests condition 3 boundary)
  5. Global correlation: all elements correlated by shared
     factor. Both forms should break. (Tests condition 3 violation)

Where the parameters come from:
  - pd=0.03, pe=0.005: typical qubit readout error rates [1]
  - c=0.05: weak adjacent-element correlation (typical crosstalk)
  - 50% spread: simulates manufacturing variation in pd

Expected output:
  Tests 1-2: PASS (alpha identical, variation < 1e-14)
  Test 3: PASS (difference < 5%)
  Test 4: PASS (R^2 > 0.9, structure preserved with weak correlation)
  Test 5: reported from paper (R^2 = 0.94 with global correlation)

Requirements: numpy, scipy
Usage: python3 verify_falsification.py
"""

import numpy as np
from scipy import stats


def test_f_invariance():
    """
    Test 1: alpha unchanged when operating conditions F vary.

    Computes P(k)/P(0) analytically for 5 values of F.
    F^N cancels in the ratio, so alpha must be identical
    regardless of F. Variation must be < 1e-14 (machine precision).
    """
    print("\n  Test 1: F varied from 0.80 to 1.00")
    pd, pe, N, k = 0.03, 0.005, 8, 4

    alphas = []
    for F in [0.80, 0.85, 0.90, 0.95, 1.00]:
        # P(k) = (1-pe)^(N-k) * (1-pd)^k * F^N
        Pk = ((1 - pe) ** (N - k)) * ((1 - pd) ** k) * (F ** N)
        # P(0) = (1-pe)^N * F^N
        P0 = ((1 - pe) ** N) * (F ** N)
        # Ratio: F^N cancels
        ratio = Pk / P0
        alpha_measured = np.log(ratio) / k
        alphas.append(alpha_measured)

    variation = max(alphas) - min(alphas)
    print(f"    alpha range: {min(alphas):.10f} to {max(alphas):.10f}")
    print(f"    Variation: {variation:.4e}")
    passed = variation < 1e-14
    print(f"    Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_n_invariance():
    """
    Test 2: alpha unchanged when number of elements N varies.

    Alpha = ln[(1-pd)/(1-pe)] depends only on pd and pe,
    not on N. Verifies this for N = 2, 4, 8, 16, 32, 64, 128.
    """
    print("\n  Test 2: N varied from 2 to 128")
    pd, pe = 0.03, 0.005

    alphas = []
    for N in [2, 4, 8, 16, 32, 64, 128]:
        k = N // 2
        Pk = ((1 - pe) ** (N - k)) * ((1 - pd) ** k)
        P0 = ((1 - pe) ** N)
        ratio = Pk / P0
        alpha_measured = np.log(ratio) / k
        alphas.append(alpha_measured)

    variation = max(alphas) - min(alphas)
    print(f"    alpha range: {min(alphas):.10f} to {max(alphas):.10f}")
    print(f"    Variation: {variation:.4e}")
    passed = variation < 1e-14
    print(f"    Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_nonuniform_pd():
    """
    Test 3: invariant holds with non-uniform pd across elements.

    In real systems, not all elements have identical pd.
    This test gives each element a different pd drawn from
    [pd_mean * 0.5, pd_mean * 1.5] (50% spread).
    The measured alpha should still match the uniform case
    within 5%, because the average pd determines alpha.
    """
    print("\n  Test 3: Non-uniform pd (50% spread across elements)")
    np.random.seed(42)

    pd_mean, pe = 0.03, 0.005
    N = 100
    n_sim = 50000

    # Case 1: uniform pd (all elements have pd_mean)
    p_correct_uniform = []
    for k in range(0, N + 1, 10):
        correct = 0
        for _ in range(n_sim):
            ok = True
            for i in range(N):
                if i < k:
                    if np.random.random() < pd_mean:
                        ok = False
                        break
                else:
                    if np.random.random() < pe:
                        ok = False
                        break
            if ok:
                correct += 1
        p_correct_uniform.append((k, correct / n_sim))

    # Case 2: non-uniform pd (each element has different pd)
    pd_spread = np.random.uniform(pd_mean * 0.5, pd_mean * 1.5, N)
    p_correct_nonuniform = []
    for k in range(0, N + 1, 10):
        correct = 0
        for _ in range(n_sim):
            ok = True
            for i in range(N):
                if i < k:
                    if np.random.random() < pd_spread[i]:
                        ok = False
                        break
                else:
                    if np.random.random() < pe:
                        ok = False
                        break
            if ok:
                correct += 1
        p_correct_nonuniform.append((k, correct / n_sim))

    # Fit alpha for both cases
    def fit_alpha(data):
        ks = [d[0] for d in data if d[1] > 0]
        p0 = data[0][1]
        if p0 == 0:
            return None, None
        lrs = [np.log(d[1] / p0) for d in data if d[1] > 0]
        if len(ks) < 3:
            return None, None
        sl, _, rv, _, _ = stats.linregress(ks, lrs)
        return sl, rv ** 2

    alpha_u, r2_u = fit_alpha(p_correct_uniform)
    alpha_nu, r2_nu = fit_alpha(p_correct_nonuniform)

    if alpha_u and alpha_nu:
        diff_pct = abs(alpha_u - alpha_nu) / abs(alpha_u) * 100
        print(f"    Uniform alpha:     {alpha_u:.6f} (R^2={r2_u:.4f})")
        print(f"    Non-uniform alpha: {alpha_nu:.6f} (R^2={r2_nu:.4f})")
        print(f"    Difference: {diff_pct:.1f}%")
        passed = diff_pct < 5
        print(f"    Result: {'PASS (< 5%)' if passed else 'FAIL'}")
        return passed
    return False


def test_correlation():
    """
    Test 4 & 5: effect of correlation between elements.

    Test 4 (adjacent-element, c=0.05): weak correlation between
    adjacent elements. The log-linear structure should still hold
    approximately (R^2 > 0.9).

    Note: the paper also reports that global correlation degrades
    R^2 to 0.94. That test requires a specialized correlation model
    not included in this script.
    """
    print("\n  Test 4: Adjacent-element correlation c=0.05")

    np.random.seed(42)
    pd, pe = 0.03, 0.005
    N = 20
    n_sim = 50000

    def simulate(correlation_type, n_sim=n_sim):
        """Simulate readout with optional correlation."""
        results = []
        for k in range(0, N + 1, 2):
            correct = 0
            for _ in range(n_sim):
                # Generate initial state: k excited, N-k ground
                state = np.zeros(N, dtype=int)
                state[:k] = 1
                np.random.shuffle(state)

                # Apply correlation (modifies state before readout)
                if correlation_type == "nearest":
                    # Adjacent-element: if neighbor is excited,
                    # 5% chance this element also becomes excited
                    for i in range(1, N):
                        if state[i - 1] == 1 and np.random.random() < 0.05:
                            state[i] = 1
                # Readout with asymmetric errors
                ok = True
                for i in range(N):
                    if state[i] == 1:
                        if np.random.random() < pd:
                            ok = False
                            break
                    else:
                        if np.random.random() < pe:
                            ok = False
                            break
                if ok:
                    correct += 1
            results.append((k, correct / n_sim))
        return results

    # Run two cases
    ind_results = simulate("none")        # independent (baseline)
    nn_results = simulate("nearest")      # adjacent-element

    def fit_r2(data):
        """Fit ln[P(k)/P(0)] vs k and return R^2."""
        ks = [d[0] for d in data if d[1] > 0]
        p0 = data[0][1]
        if p0 == 0:
            return 0
        lrs = [np.log(d[1] / p0) for d in data if d[1] > 0]
        if len(ks) < 3:
            return 0
        _, _, rv, _, _ = stats.linregress(ks, lrs)
        return rv ** 2

    r2_ind = fit_r2(ind_results)
    r2_nn = fit_r2(nn_results)

    print(f"    Independent:       R^2 = {r2_ind:.4f}")
    print(f"    Adjacent-element:  R^2 = {r2_nn:.4f}")

    passed_nn = r2_nn > 0.9
    print(f"    NN preserves structure: {'PASS' if passed_nn else 'FAIL'}")

    return passed_nn


def test_global_correlation():
    """
    Test 5: global (mean-field) correlation breaks the invariant.

    Model: each element's error rate depends on the total fraction
    of excited elements (mean-field coupling):
      pd_eff = pd0 * exp(J * k / N)

    With J=1.0, the linear fit degrades to R^2 ~ 0.94,
    confirming that global correlation breaks the log-linear structure.

    Parameters: N=8, pd0=0.05, pe0=0.005, J=1.0
    Source: correlated_bda.py (mean-field model)
    """
    print("\n  Test 5: Global (mean-field) correlation J=1.0")

    np.random.seed(42)
    N = 8
    pd0, pe0 = 0.05, 0.005
    J = 1.0
    n_sim = 200000

    from collections import defaultdict
    surv_by_k = defaultdict(lambda: [0, 0])

    for _ in range(n_sim):
        # Random codeword
        codeword = np.random.randint(0, 2, size=N)
        k = int(np.sum(codeword))

        # Mean-field modulated error rates
        pd_eff = min(pd0 * np.exp(J * k / N), 0.999)
        pe_eff = min(pe0 * np.exp(J * k / N), 0.999)

        # Apply errors
        survived = True
        for i in range(N):
            p_flip = pd_eff if codeword[i] == 1 else pe_eff
            if np.random.random() < p_flip:
                survived = False
                break

        surv_by_k[k][1] += 1
        if survived:
            surv_by_k[k][0] += 1

    # Compute log-ratios
    k_vals, ln_ratio_vals = [], []
    p0 = None
    for k in sorted(surv_by_k.keys()):
        s, t = surv_by_k[k]
        if t >= 50 and s >= 5:
            p = s / t
            if k == 0:
                p0 = p
            if p0 and p0 > 0:
                k_vals.append(k)
                ln_ratio_vals.append(np.log(p / p0))

    if len(k_vals) >= 3:
        k_arr = np.array(k_vals, dtype=float)
        ln_arr = np.array(ln_ratio_vals)
        slope, intercept, r_value, p_value, se = stats.linregress(k_arr, ln_arr)
        r2 = r_value ** 2
        print(f"    R^2 (linear fit) = {r2:.4f}")
        print(f"    Expected: ~0.94 (degrades from ~1.00 under independence)")
        passed = r2 < 0.96  # must break below 0.96
        print(f"    Invariant breaks: {'PASS' if passed else 'FAIL'}")
        return passed

    print(f"    Insufficient data")
    return False


if __name__ == "__main__":
    print("=" * 60)
    print("Dissipative Asymmetry — Falsification Tests")
    print("=" * 60)

    r1 = test_f_invariance()
    r2 = test_n_invariance()
    r3 = test_nonuniform_pd()
    r4 = test_correlation()
    r5 = test_global_correlation()

    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    print(f"  F-invariance:        {'PASS' if r1 else 'FAIL'}")
    print(f"  N-invariance:        {'PASS' if r2 else 'FAIL'}")
    print(f"  Non-uniform pd:      {'PASS' if r3 else 'FAIL'}")
    print(f"  NN correlation:      {'PASS' if r4 else 'FAIL'}")
    print(f"  Global correlation:  {'PASS' if r5 else 'FAIL'} (invariant breaks as expected)")
    print(f"  Overall:             {'ALL PASS' if all([r1, r2, r3, r4, r5]) else 'SOME FAILED'}")
