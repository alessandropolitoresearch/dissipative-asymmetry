# Dissipative Asymmetry

A coefficient α characterizes asymmetric transition rates in independent discrete systems. Operating conditions cancel exactly. The same structure is observed across quantum processors, semiconductor memories, DNA replication, optical sensors, and digital PCR. No new physics, no new mathematics — Bernoulli probability theory (1713).

**Paper:** [DOI: 10.5281/zenodo.19308787](https://doi.org/10.5281/zenodo.19308787)

**Calculator:** [asymmetry-calculator.netlify.app](https://asymmetry-calculator.netlify.app/)

## The coefficient

In systems made of independent discrete elements with asymmetric transition rates, a single coefficient

**α = ln[(1 − p_d) / (1 − p_e)]**

characterizes the intrinsic asymmetry of the system. Operating conditions (temperature, voltage, humidity) cancel exactly in the ratio of readout probabilities.

## Properties

| # | Property | Description |
|---|----------|-------------|
| 1 | Definition | α = ln[(1 − p_d) / (1 − p_e)] |
| 2 | Linearity | ln[P(k)/P(0)] = k · α |
| 3 | F-independence | Operating conditions cancel in the ratio |
| 4 | Symmetry | α = 0 when p_d = p_e |
| 5 | Sign | α > 0 when p_d > p_e, α < 0 otherwise |
| 6 | Monotonicity | \|α\| increases with asymmetry |
| 7 | Composition | Ratio of two compositions depends only on their difference |
| 8 | Additivity | α_total = α₁ + α₂ for independent mechanisms |

## Cross-domain observations

The same algebraic structure has been observed across multiple physically distinct domains:

| Domain | Measurements | Fit |
|--------|-------------|-----|
| DRAM retention | 304,024 | 1.74× asymmetry |
| DRAM read disturb | 147,456 | 1.90× asymmetry |
| HST ACS/WFC CCD | 8,570,000 | R² = 0.9996 |
| JWST NIRCam | 4,194,304 | R² = 0.97–0.99 |
| dPCR (Albumin) | ~350,000 | R² = 0.999998 |

## Scripts

| Script | What it does | Data needed |
|--------|-------------|-------------|
| `verify_qubit.py` | Monte Carlo across 5 domains | None (simulation) |
| `verify_additivity.py` | Property 8: α₁ + α₂ = α_total | None (simulation) |
| `verify_falsification.py` | 5 tests to break the invariant | None (simulation) |
| `verify_dpcr.py` | Real dPCR data, R² = 0.999998 | [definetherain](https://github.com/jacobhurst/definetherain) repo |

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
```

## Requirements

```
Python 3.8+
numpy
scipy
```

## License

CC BY-NC-ND 4.0

## Author

Alessandro Polito — Independent Researcher

Contact: ap.writer@proton.me
