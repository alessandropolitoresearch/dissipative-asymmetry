# Dissipative Asymmetry

A study in statistics and mathematics, based on well-known principles of Bernoulli probability theory (1713).

**Paper:** [DOI: 10.5281/zenodo.19308787](https://doi.org/10.5281/zenodo.19308787)

## What is this?

In systems made of independent discrete elements with asymmetric transition rates, a single coefficient

**α = ln[(1 − p_d) / (1 − p_e)]**

characterizes the intrinsic asymmetry of the system. This coefficient does not change when operating conditions (temperature, voltage, humidity) change — they cancel exactly in the ratio of readout probabilities.

The same algebraic structure has been verified across 11 physically distinct domains, including quantum processors, semiconductor memories, DNA replication, and space telescope detectors.

## Quick start

```bash
# Monte Carlo verification across 5 domains
python3 scripts/verify_qubit.py

# Property 8: additivity
python3 scripts/verify_additivity.py

# Falsification tests (5 tests)
python3 scripts/verify_falsification.py

# Digital PCR verification (requires definetherain data)
git clone https://github.com/jacobhurst/definetherain.git /tmp/definetherain
python3 scripts/verify_dpcr.py

# IBM Quantum (requires account at quantum.ibm.com)
python3 scripts/verify_ibm_quantum.py
```

## Calculator

Interactive web calculator with 5 tabs (Tutorial, Binary, Correlation, M-state, Field-dependent):

```bash
pip install streamlit numpy matplotlib pandas
streamlit run calculator/app.py
```

## Scripts

| Script | What it does | Data needed |
|--------|-------------|-------------|
| `verify_qubit.py` | Monte Carlo across 5 domains | None (simulation) |
| `verify_additivity.py` | Property 8: α₁ + α₂ = α_total | None (simulation) |
| `verify_falsification.py` | 5 tests to break the invariant | None (simulation) |
| `verify_dpcr.py` | Real dPCR data, R² = 0.999998 | definetherain repo |
| `verify_ibm_quantum.py` | IBM Quantum hardware test | IBM Quantum account |

## Requirements

```
Python 3.8+
numpy
scipy
```

For the calculator: `streamlit matplotlib pandas`

For IBM Quantum: `qiskit qiskit-ibm-runtime`

## Results

| Domain | Measurements | Result |
|--------|-------------|--------|
| IBM Fez (156 qubits) | 160,000 | R² = 0.88–0.97 |
| IBM Kingston (156 qubits) | 160,000 | R² = 0.88–0.97 |
| DRAM retention | 304,024 | 1.74× asymmetry |
| DRAM read disturb | 147,456 | 1.90× asymmetry |
| HST ACS/WFC CCD | 8,570,000 | **R² = 0.9996** |
| JWST NIRCam | 4,194,304 | R² = 0.97–0.99 |
| dPCR (Albumin) | ~350,000 | R² = 0.999998 |

## License

CC BY-NC-ND 4.0

## Author

Alessandro Polito — Independent Researcher

Contact: ap.writer@proton.me
