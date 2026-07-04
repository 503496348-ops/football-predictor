# Changelog

All notable changes to `football-predictor` should be documented in this file.

This repository follows a lightweight Keep-a-Changelog style and semantic versioning where applicable.

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
