"""
Dissipative Asymmetry Calculator
DOI: 10.5281/zenodo.19308787
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(page_title="DA Calculator", page_icon="📐", layout="wide")

st.title("📐 Dissipative Asymmetry Calculator")
st.markdown("Enter error rates and explore the algebraic properties of asymmetric discrete systems. [Paper DOI: 10.5281/zenodo.19308787](https://doi.org/10.5281/zenodo.19308787)")

tab0, tab1, tab2, tab3, tab4 = st.tabs(["📖 Tutorial", "Binary (α)", "Correlation (β)", "M-state", "Field-dependent"])

# ============================================================
# TAB 0: Tutorial
# ============================================================
with tab0:
    st.header("What is Dissipative Asymmetry?")

    st.markdown("""
    Many physical systems are made of **independent pieces** that can be in two states: **on** or **off**
    (excited or ground, 1 or 0). When you read these pieces, errors happen — but not equally in both directions.

    **Example:** imagine a sensor with 100 pixels.
    - Some pixels are "on" (excited)
    - Some pixels are "off" (ground)
    - When you read them, two types of error can occur:
    """)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        **p_d — dominant error**

        An "on" pixel is misread as "off."
        This is the common error — things
        naturally turn off (charge leaks,
        qubits decay, DNA degrades).

        *Example: p_d = 0.04 (4%)*
        """)
    with c2:
        st.markdown("""
        **p_e — excitation error**

        An "off" pixel is misread as "on."
        This is the rare error — things
        don't spontaneously turn on
        without energy.

        *Example: p_e = 0.001 (0.1%)*
        """)

    st.divider()
    st.subheader("The coefficient α")
    st.markdown("""
    From these two error rates, you compute a single number:

    **α = ln[(1 − p_d) / (1 − p_e)]**

    With the example above: α = ln[0.96 / 0.999] = **−0.0398**

    This number tells you how different the two error directions are.
    """)

    st.divider()
    st.subheader("What is k?")
    st.markdown("""
    **k** is how many elements are "on" (excited) in your system.

    - k = 0 → all pixels are off (easiest to read)
    - k = 50 → half are on (harder to read)
    - k = 100 → all are on (hardest to read)

    The formula says:

    **ln[P(k) / P(0)] = k × α**

    Meaning: every extra "on" element makes the system harder to read by the same amount α.
    """)

    st.divider()
    st.subheader("What is F?")
    st.markdown("""
    **F** represents the operating conditions: temperature, voltage, humidity, etc.

    The key property: **F cancels exactly** in the ratio P(k)/P(0).

    This means:
    - If you change the temperature → total errors change, but **α stays the same**
    - If α changes → the system itself has physically changed, not the environment

    **α is a property of the system, not of the conditions.**
    """)

    st.divider()
    st.subheader("Three things you can do with α")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        **1. Measure once**

        Compute α from p_d and p_e.
        This single number characterizes
        your system. No need to
        measure again unless the
        system physically changes.
        """)
    with c2:
        st.markdown("""
        **2. Estimate without measuring**

        For any k, the readout accuracy
        relative to k=0 is e^(k×α).
        Just multiply k by α.
        No extra measurements needed.
        """)
    with c3:
        st.markdown("""
        **3. Detect degradation**

        If α changes over time →
        the system has degraded.
        If α is stable but errors
        increase → only the
        environment changed.
        """)

    st.divider()
    st.subheader("The four tabs")
    st.markdown("""
    | Tab | What it does |
    |-----|-------------|
    | **Binary (α)** | The basic case: two states, independent elements. Enter p_d, p_e, N → get α and plots. |
    | **Correlation (β)** | When adjacent elements influence each other. Adds a second coefficient β. |
    | **M-state** | When elements have more than 2 states (e.g., multi-level memory cells). |
    | **Field-dependent** | When an external field changes α itself. Shows the critical field F* where the asymmetry inverts. |
    """)

# ============================================================
# TAB 1: Binary
# ============================================================
with tab1:
    col_in, col_res, col_plot = st.columns([1, 1, 2])

    with col_in:
        st.subheader("Input")
        pd1 = st.number_input("p_d (dominant error)", min_value=0.0001, max_value=0.9999, value=0.04, step=0.001, format="%.4f",
                              help="Excited element misread as ground", key="pd1")
        pe1 = st.number_input("p_e (excitation error)", min_value=0.0001, max_value=0.9999, value=0.001, step=0.001, format="%.4f",
                              help="Ground element misread as excited", key="pe1")
        N1 = st.number_input("N (elements)", min_value=2, max_value=100000, value=100, step=1, key="N1")
        alpha1 = np.log((1 - pd1) / (1 - pe1))

    with col_res:
        st.subheader("Result")
        st.metric("α", f"{alpha1:.6f}")
        st.metric("p_d / p_e", f"{pd1/pe1:.1f}×")
        st.metric("Direction", "Passive (α<0)" if alpha1 < 0 else "Active (α>0)")
        ks = sorted(set([0, 1, int(N1*0.1), int(N1*0.3), int(N1*0.5), N1]))
        ks = [k for k in ks if 0 <= k <= N1]
        df = pd.DataFrame({
            "k": ks,
            "k×α": [f"{k*alpha1:.3f}" for k in ks],
            "Accuracy": [f"{np.exp(k*alpha1)*100:.1f}%" for k in ks]
        })
        st.dataframe(df, use_container_width=True, hide_index=True, height=250)

    with col_plot:
        st.subheader("Visualization")
        k_plot = np.arange(0, N1 + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
        ax1.plot(k_plot, k_plot * alpha1, color='#1f77b4', linewidth=1.5)
        ax1.set_xlabel('k'); ax1.set_ylabel('ln[P(k)/P(0)]'); ax1.set_title('Log-ratio')
        ax1.axhline(y=0, color='gray', linewidth=0.5, linestyle='--'); ax1.grid(True, alpha=0.3)
        ax2.plot(k_plot, np.exp(k_plot * alpha1) * 100, color='#d62728', linewidth=1.5)
        ax2.set_xlabel('k'); ax2.set_ylabel('Accuracy (%)'); ax2.set_title('Accuracy')
        ax2.axhline(y=100, color='gray', linewidth=0.5, linestyle='--'); ax2.set_ylim(0, 105); ax2.grid(True, alpha=0.3)
        plt.tight_layout(); st.pyplot(fig)

# ============================================================
# TAB 2: Correlation (β)
# ============================================================
with tab2:
    col_in2, col_res2, col_plot2 = st.columns([1, 1, 2])

    with col_in2:
        st.subheader("Input")
        st.caption("For systems where adjacent elements influence each other.")
        pd2 = st.number_input("p_d", min_value=0.0001, max_value=0.9999, value=0.04, step=0.001, format="%.4f", key="pd2")
        pe2 = st.number_input("p_e", min_value=0.0001, max_value=0.9999, value=0.001, step=0.001, format="%.4f", key="pe2")
        c2 = st.number_input("c (correlation strength)", min_value=0.0, max_value=1.0, value=0.05, step=0.01, format="%.3f",
                             help="c=0 means independent. c>0 means adjacent excited elements increase each other's error.")
        N2 = st.number_input("N (elements)", min_value=2, max_value=1000, value=50, step=1, key="N2")

        alpha2 = np.log((1 - pd2) / (1 - pe2))
        beta2 = np.log((1 - pd2 * (1 + c2)) / (1 - pd2)) if pd2 * (1 + c2) < 1 else 0.0

    with col_res2:
        st.subheader("Result")
        st.metric("α", f"{alpha2:.6f}")
        st.metric("β", f"{beta2:.6f}")
        st.caption("ln[P(k,n)/P(0,0)] = k×α + n×β")
        st.caption("k = excited elements, n = adjacent excited pairs")
        st.markdown(f"""
        - **α** measures the asymmetry per element
        - **β** measures the extra effect when two excited elements are neighbors
        - When c = 0: β = 0 and the formula reduces to the binary case
        """)

    with col_plot2:
        st.subheader("Visualization: α vs β contribution")
        k_range = np.arange(0, N2 + 1)
        # n approximation: for random placement, n ≈ k*(k-1)/(N-1)
        n_approx = k_range * (k_range - 1) / max(N2 - 1, 1)
        ln_indep = k_range * alpha2
        ln_corr = k_range * alpha2 + n_approx * beta2

        fig2, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(k_range, ln_indep, color='#1f77b4', linewidth=1.5, label=f'Independent (α only)')
        ax.plot(k_range, ln_corr, color='#d62728', linewidth=1.5, linestyle='--', label=f'With correlation (α + β)')
        ax.set_xlabel('k (excited elements)'); ax.set_ylabel('ln[P/P(0)]')
        ax.set_title(f'Effect of correlation c={c2:.3f}')
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout(); st.pyplot(fig2)

# ============================================================
# TAB 3: M-state
# ============================================================
with tab3:
    col_in3, col_res3, col_plot3 = st.columns([1, 1, 2])

    with col_in3:
        st.subheader("Input")
        st.caption("For systems with more than 2 states (e.g., MLC Flash, multi-level cells).")
        M3 = st.number_input("M (number of states)", min_value=2, max_value=8, value=3, step=1, key="M3")
        N3 = st.number_input("N (elements)", min_value=2, max_value=1000, value=100, step=1, key="N3")

        st.markdown(f"Enter exit rates for each state (probability of misreading):")
        q_vals = []
        for s in range(M3):
            default = 0.999 if s == 0 else max(0.99 - s * 0.05, 0.5)
            q = st.number_input(f"q_{s} (correct readout prob for state {s})", min_value=0.01, max_value=0.999,
                                value=default, step=0.01, format="%.3f", key=f"q_{s}")
            q_vals.append(q)

    with col_res3:
        st.subheader("Result")
        st.caption("α_s = ln(q_s / q_0) for each state s")
        alphas_m = []
        for s in range(M3):
            a_s = np.log(q_vals[s] / q_vals[0]) if q_vals[0] > 0 else 0
            alphas_m.append(a_s)
            st.metric(f"α_{s}", f"{a_s:.6f}")

        st.markdown(f"""
        **Formula:** ln[P(k₁,...,k_{{M-1}})/P(0,...,0)] = Σ k_s × α_s

        Each state contributes independently. The total log-ratio
        is the sum of contributions from all states.
        """)

    with col_plot3:
        st.subheader("Visualization: α per state")
        fig3, ax3 = plt.subplots(figsize=(10, 3.5))
        states = list(range(M3))
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, M3))
        ax3.bar(states, alphas_m, color=colors)
        ax3.set_xlabel('State s'); ax3.set_ylabel('α_s')
        ax3.set_title(f'Asymmetry coefficient per state (M={M3})')
        ax3.set_xticks(states)
        ax3.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
        ax3.grid(True, alpha=0.3, axis='y')
        plt.tight_layout(); st.pyplot(fig3)

# ============================================================
# TAB 4: Field-dependent (Type 2)
# ============================================================
with tab4:
    col_in4, col_res4, col_plot4 = st.columns([1, 1, 2])

    with col_in4:
        st.subheader("Input")
        st.caption("When an external field enters the transition rates asymmetrically, α becomes field-dependent.")
        pd4 = st.number_input("p_d", min_value=0.0001, max_value=0.9999, value=0.04, step=0.001, format="%.4f", key="pd4")
        pe_base4 = st.number_input("p_e,base", min_value=0.0001, max_value=0.9999, value=0.001, step=0.001, format="%.4f", key="pe_base4",
                                   help="Base excitation error rate without field")
        delta4 = st.number_input("δ (field coupling)", min_value=0.0001, max_value=0.1, value=0.005, step=0.001, format="%.4f", key="delta4",
                                 help="How strongly the field affects p_e")
        F_range = st.slider("F range", min_value=0.0, max_value=10.0, value=(0.0, 5.0), step=0.1, key="F_range4")

    with col_res4:
        st.subheader("Result")
        F_star = (pd4 - pe_base4) / delta4 if delta4 > 0 else float('inf')
        st.metric("F* (critical field)", f"{F_star:.3f}")
        st.caption("At F*, the bias vanishes. Beyond F*, it inverts.")

        alpha_at_0 = np.log((1 - pd4) / (1 - pe_base4))
        st.metric("α(F=0)", f"{alpha_at_0:.6f}")

        if F_star < F_range[1]:
            pe_at_star = pe_base4 + delta4 * F_star
            if pe_at_star < 1:
                alpha_at_star = np.log((1 - pd4) / (1 - pe_at_star))
                st.metric("α(F*)", f"{alpha_at_star:.6f}")

        st.markdown("""
        **Formula:** α(F) = ln[(1−p_d)/(1−p_e,base−δF)]

        As F increases, p_e grows. At F*, p_e = p_d
        and α = 0. Beyond F*, the asymmetry inverts.
        """)

    with col_plot4:
        st.subheader("Visualization: α(F)")
        F_vals = np.linspace(F_range[0], F_range[1], 200)
        alpha_F = []
        for F in F_vals:
            pe_F = pe_base4 + delta4 * F
            if pe_F < 1 and pe_F > 0:
                alpha_F.append(np.log((1 - pd4) / (1 - pe_F)))
            else:
                alpha_F.append(np.nan)

        fig4, ax4 = plt.subplots(figsize=(10, 3.5))
        ax4.plot(F_vals, alpha_F, color='#2ca02c', linewidth=2)
        ax4.axhline(y=0, color='gray', linewidth=1, linestyle='--')
        if F_star < F_range[1]:
            ax4.axvline(x=F_star, color='red', linewidth=1, linestyle=':', label=f'F* = {F_star:.2f}')
            ax4.legend()
        ax4.set_xlabel('External field F'); ax4.set_ylabel('α(F)')
        ax4.set_title('Field-dependent asymmetry coefficient')
        ax4.grid(True, alpha=0.3)
        plt.tight_layout(); st.pyplot(fig4)

# ============================================================
# Bottom: What this is / is not / Examples
# ============================================================
st.divider()
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("""
    🔬 **Characterization**

    Measure p_d and p_e once. Get α —
    one number that describes your system's
    intrinsic asymmetry, independent of
    temperature, voltage, or any other
    operating condition.
    """)
with c2:
    st.markdown("""
    🔧 **Monitoring**

    Measure α today: −0.04.
    Six months later: −0.06.
    Nothing else changed. Your system
    has physically degraded.
    If α is still −0.04 but errors
    increased — only the environment changed.
    """)
with c3:
    st.markdown("""
    📊 **Comparison**

    Two devices, different manufacturers,
    different conditions. You cannot compare
    raw error rates. But you can compare
    α — because α does not depend on
    conditions. Same α = same intrinsic quality.
    """)

st.caption("[DOI: 10.5281/zenodo.19308787](https://doi.org/10.5281/zenodo.19308787) · © 2026 A. Polito · CC BY-NC-ND 4.0")
