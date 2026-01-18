# Traceability Matrix: Longer Form Paper

This document maps every key statistic in the paper to its source code and results file.

**Last Updated:** 2026-01-18 12:42 UTC
**Verification Rate:** 100.0% (43/43 claims verified)

## Status Summary
- ✓ = Verified (result matches tex)
- ⚠ = Minor discrepancy (documented)
- ✗ = Not verified

---

## Fixes Applied (2026-01-18)

| Issue | Status | Details |
|-------|--------|---------|
| Data symlink missing | ✓ FIXED | Copied actual `wallet_fills_data.parquet` (120MB) to `data/` folder |
| AWS credentials exposed | ✓ FIXED | Removed hardcoded credentials from 10 files, now use env vars |
| statsmodels API compatibility | ✓ FIXED | Fixed `cov_params()` handling in 2 scripts |
| Missing figure generation | ✓ FIXED | Created `generate_all_figures.py` with 24 figure functions |
| Verification scripts | ✓ FIXED | Created 7 verification scripts with JSON outputs |
| Rolling OOS discrepancy | ✓ FIXED | Updated tex from 3.05 to 2.94 bps (matches computed mean) |
| Hardcoded absolute paths | ✓ FIXED | Converted 36 files to use relative paths |

---

## 1. Main Results (Section 3: Toxicity Differential)

| Tex Claim | Value | Results File | Code File | Status |
|-----------|-------|--------------|-----------|--------|
| Trade-weighted toxicity | 3.05 bps | midmove_toxicity_results.json | midmove_toxicity_analysis.py | ✓ |
| Trade-weighted t-stat | 5.84 | midmove_toxicity_results.json | midmove_toxicity_analysis.py | ✓ |
| Taker-level toxicity | 19.18 bps | midmove_toxicity_results.json | midmove_toxicity_analysis.py | ✓ |
| Taker-level t-stat | 24.38 | clustered_inference_main_results.json | clustered_inference_main_results.py | ✓ |
| N trades | 693K | midmove_toxicity_results.json | midmove_toxicity_analysis.py | ✓ |

## 2. Classification Stability (Table 4)

| Tex Claim | Value | Results File | Code File | Status |
|-----------|-------|--------------|-----------|--------|
| Day 1 vs Day 2 Spearman ρ | 0.71 | wallet_markout_persistence_results.json | wallet_markout_persistence.py | ✓ |
| Week 1 vs Week 2 Spearman ρ | 0.78 | wallet_markout_persistence_results.json | wallet_markout_persistence.py | ✓ |
| Month 1 vs Month 2 Spearman ρ | 0.82 | wallet_markout_persistence_results.json | wallet_markout_persistence.py | ✓ |
| First half vs Second half ρ | 0.84 | wallet_markout_persistence_results.json | wallet_markout_persistence.py | ✓ |
| Top quintile overlap (monthly) | 78.1% | wallet_markout_persistence_results.json | wallet_markout_persistence.py | ✓ |
| Sample size | 5,396 wallets | wallet_markout_persistence_results.json | wallet_markout_persistence.py | ✓ |

## 3. Rolling Out-of-Sample (Table 5)

| Tex Claim | Value | Results File | Code File | Status |
|-----------|-------|--------------|-----------|--------|
| 8/8 windows significant | p < 0.01 | rolling_oos_toxicity_results.json | rolling_oos_toxicity.py | ✓ |
| Mean toxicity differential | 2.94 bps | rolling_oos_toxicity_results.json | rolling_oos_toxicity.py | ✓ |
| Std dev | 0.19 bps | rolling_oos_toxicity_results.json | rolling_oos_toxicity.py | ✓ |
| CV | 6.5% | rolling_oos_toxicity_results.json | rolling_oos_toxicity.py | ✓ |
| t-stat range | 7.31–8.64 | rolling_oos_toxicity_results.json | rolling_oos_toxicity.py | ✓ |

## 4. Multi-Event Analysis (Section 4)

| Tex Claim | Value | Results File | Code File | Status |
|-----------|-------|--------------|-----------|--------|
| Jan 20 Congestion 1 | +0.70 bps | sign_test_results.json | sign_test_multi_event.py | ✓ |
| Jan 20 Congestion 2 | +1.10 bps | sign_test_results.json | sign_test_multi_event.py | ✓ |
| Jul 29 API Outage | +2.60 bps | sign_test_results.json | sign_test_multi_event.py | ✓ |
| Jul 30 Stress Event | +2.25 bps | sign_test_results.json | sign_test_multi_event.py | ✓ |
| Events positive | 4/4 | sign_test_results.json | sign_test_multi_event.py | ✓ |
| Sign test p-value | 0.0625 | sign_test_results.json | sign_test_multi_event.py | ✓ |

## 5. Robust Inference (Table 9)

| Tex Claim | Value | Results File | Code File | Status |
|-----------|-------|--------------|-----------|--------|
| Asset-clustered t-stat | 2.25 | randomization_inference_results.json | randomization_inference_outage.py | ✓ |
| Asset clusters N | 10 | randomization_inference_results.json | randomization_inference_outage.py | ✓ |
| Randomization inference p | 0.14 | randomization_inference_results.json | randomization_inference_outage.py | ✓ |
| Block bootstrap t-stat | 10.75 | randomization_inference_results.json | randomization_inference_outage.py | ✓ |
| Headline coefficient | 2.60 bps | randomization_inference_results.json | randomization_inference_outage.py | ✓ |

## 6. Market Structure During Outage

| Tex Claim | Value | Results File | Code File | Status |
|-----------|--------|--------------|-----------|--------|
| Maker participation change | +84% | market_structure_outage_results.json | market_structure_outage.py | ✓ |
| HHI change | -13% | market_structure_outage_results.json | market_structure_outage.py | ✓ |
| Quote updates change | -68% | market_structure_outage_results.json | market_structure_outage.py | ✓ |
| Spread widening | +87% | market_structure_outage_results.json | market_structure_outage.py | ✓ |
| Top-5 best price share | 78%→52% | market_structure_outage_results.json | market_structure_outage.py | ✓ |
| Top-MPSC collapse | -84% | market_structure_outage_results.json | market_structure_outage.py | ✓ |

## 7. Economic Magnitude

| Tex Claim | Value | Results File | Code File | Status |
|-----------|-------|--------------|-----------|--------|
| Volume during outage | $847M | economic_magnitude_results.json | economic_magnitude_calculations.py | ✓ |
| Excess spread cost | $0.24M | economic_magnitude_results.json | economic_magnitude_calculations.py | ✓ |
| Forced liquidation losses | $1.5M | economic_magnitude_results.json | economic_magnitude_calculations.py | ✓ |
| Total excess cost | $1.7M | economic_magnitude_results.json | economic_magnitude_calculations.py | ✓ |
| Welfare transfer | $0.9M | economic_magnitude_results.json | economic_magnitude_calculations.py | ✓ |

## 8. Composition Shift

| Tex Claim | Value | Results File | Code File | Status |
|-----------|-------|--------------|-----------|--------|
| Informed ratio | 1.19 → 1.25 | oos_classification_results.json | out_of_sample_classification.py | ✓ |
| Informed ratio change | +5.1% | oos_classification_results.json | out_of_sample_classification.py | ✓ |
| Informed share increase | +4.83 pp | oos_classification_results.json | out_of_sample_classification.py | ✓ |
| DiD coefficient | +2.8 pp | wallet_did_results.json | wallet_did_regression.py | ✓ |
| DiD t-stat | 1.33 | wallet_did_results.json | wallet_did_regression.py | ✓ |

---

## Figure Generation Status

| Figure | Generation Script | Status |
|--------|-------------------|--------|
| figure1_cross_sectional | generate_all_figures.py | ✓ Generated |
| figure2_time_evolution | generate_all_figures.py | ✓ Generated |
| figure3_horizon_analysis | generate_all_figures.py | ✓ Generated |
| figure4_classification_validation | generate_all_figures.py | ✓ Generated |
| figure5_rolling_oos | generate_all_figures.py | ✓ Generated |
| figure6_event_study | generate_all_figures.py | ✓ Generated |
| figure7_mpsc_timeseries | generate_all_figures.py | ✓ Generated |
| figure8_mpsc_withdrawal | generate_all_figures.py | ✓ Generated |
| figure9_evidence_summary | generate_all_figures.py | ✓ Generated |
| figure10_event_design | generate_all_figures.py | ✓ Generated |
| figure_maker_concentration | generate_all_figures.py | ✓ Generated |
| figure_informed_outage | generate_all_figures.py | ✓ Generated |
| figure_concentration_timeseries | generate_all_figures.py | ✓ Generated |
| figure_outage_event_study | generate_all_figures.py | ✓ Generated |
| figure_rd_* | generate_all_figures.py | ✓ Generated |
| figure_jan2025_congestion | generate_all_figures.py | ✓ Generated |
| Additional figures | generate_all_figures.py | ✓ Generated |

---

## Code Execution Order

To reproduce all results from scratch:
```bash
cd code/

# 1. Run verification scripts (generates key result JSONs)
python run_all_verification.py

# 2. Run main analysis scripts
python midmove_toxicity_analysis.py
python clustered_inference_main_results.py
python out_of_sample_classification.py
python wallet_did_regression.py
python within_tercile_clustered_analysis.py

# 3. Generate all figures
python generate_all_figures.py
```

---

## Portability Note

All scripts now use **relative paths** via:
```python
_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CODE_DIR)
_DATA_DIR = os.path.join(_PROJECT_ROOT, 'data')
_RESULTS_DIR = os.path.join(_PROJECT_ROOT, 'results')
_FIGURES_DIR = os.path.join(_PROJECT_ROOT, 'figures')
```

This makes the entire folder self-contained and portable.

---

## Security Note

AWS credentials have been removed from all code files. Scripts now use environment variables:
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`

To run scripts requiring S3 access, set these environment variables or use a `.env` file.

---

## Verification Summary

| Category | Claims | Verified | Rate |
|----------|--------|----------|------|
| Main toxicity results | 5 | 5 | 100% |
| Classification stability | 6 | 6 | 100% |
| Rolling OOS | 5 | 5 | 100% |
| Multi-event | 6 | 6 | 100% |
| Robust inference | 5 | 5 | 100% |
| Market structure | 6 | 6 | 100% |
| Economic magnitude | 5 | 5 | 100% |
| Composition shift | 5 | 5 | 100% |
| **Total** | **43** | **43** | **100%** |

---

*Updated: Sat 18 Jan 2026 12:42:00 UTC*
