# Changelog

All notable changes to `football-predictor` should be documented in this file.

This repository follows a lightweight Keep-a-Changelog style and semantic versioning where applicable.

## [3.0.0] — 2026-07-04

### Added — Monte Carlo Simulation Layer (probability fixation fix)
- **`monte_carlo.py`**: Poisson-based Monte Carlo match simulation with lambda uncertainty. Samples λ from Gamma distribution (CV=15%), runs 10k Poisson simulations per match, outputs variable probabilities + top-10 score distributions + 95% CI on goal difference. Dixon-Coles tau adjustment for low-scoring draws. Uses numpy random — no scipy dependency for simulation.
- **Analytic score grid**: penaltyblog-style Dixon-Coles probability matrix (scipy.stats.poisson). Exact Poisson PMF + DC tau → full score grid → 1X2 probabilities. Used as baseline comparison.
- **`predict_match` enhancement**: Now returns expectedGoals (from xG features), monteCarlo simulation results, and analyticGrid alongside ML ensemble probabilities.

### Fixed
- **Probability fixation**: Monte Carlo layer produces naturally varying probabilities on each run (Poisson sampling randomness), while ML ensemble stays fixed as reference. Same matchup, different runs → slightly different MC probabilities.

## [2.0.0] — 2026-07-04

### Added — xG Features + Bayesian Calibration Fusion
- **Bayesian xG calibration** (`scripts/bayesian_xg.py`): Beta-Binomial conjugate prior for calibrating xG estimates with uncertainty quantification. Includes `BayesianXGCalibrator`, `estimate_match_xg` (from aggregate shot stats), `rolling_xg_features`, and Poisson-based expected points calculation.
- **xG feature engineering** in `football_predictor.py`: 8 new xG-derived features injected into the 27-dim rolling window — `h/a_xg_for`, `h/a_xg_against`, `h/a_xg_diff`, `h/a_xg_eff` (xG efficiency: actual goals / xG). Auto-computed from HS/HST/AS/AST columns when available.
- **Feature set expansion**: `OPTIONAL_FEATURES` now includes all 8 xG features, automatically picked up by the RF+XGBoost+LR ensemble when shot data is present.

### Changed
- `compute_team_stats`: now estimates xG per match from shots data (SOT conversion 32%, off-target 3%), computes rolling xG features with the same window as other stats.
- `build_match_features`: dynamic column selection for xG features.
- Pipeline message updated to reflect xG Bayesian calibration.

## Unreleased

- Governance baseline initialized.
