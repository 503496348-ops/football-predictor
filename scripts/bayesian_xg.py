# -*- coding: utf-8 -*-
"""
Bayesian xG Calibration — 贝叶斯期望进球校准模块
================================================
Uses Beta-Binomial conjugate prior to calibrate xG estimates from
shot-level data into match-level expected goals with uncertainty bounds.

Reference: An & Martin (2025) — Score more by xG or by tactics?
           Dixon & Coles (1997) — Modelling association football scores.

No external deps beyond numpy.
"""
from __future__ import annotations
import math
from typing import Dict, Optional, Tuple
import numpy as np


# ── Beta-Binomial conjugate prior ───────────────────────────────────────────
# Prior: Beta(alpha=2, beta=3) → mean ≈ 0.4 (typical shot conversion rate)
# Updated with observed shot outcomes → posterior Beta(alpha + goals, beta + misses)

PRIOR_ALPHA = 2.0  # prior goals
PRIOR_BETA = 3.0   # prior misses


class BayesianXGCalibrator:
    """
    Bayesian xG calibration using Beta-Binomial conjugate prior.

    Takes raw xG estimates (sum of per-shot xG values) and actual goals,
    then calibrates via posterior inference.
    """

    def __init__(self, prior_alpha: float = PRIOR_ALPHA, prior_beta: float = PRIOR_BETA):
        self.alpha = prior_alpha
        self.beta = prior_beta
        self.n_matches = 0
        self.total_xg = 0.0
        self.total_goals = 0

    def update(self, xg_sum: float, actual_goals: int) -> Tuple[float, float]:
        """
        Update posterior with one match observation.

        Args:
            xg_sum: Sum of per-shot xG values for the team in this match.
            actual_goals: Actual goals scored.

        Returns:
            (posterior_mean, posterior_std) — calibrated xG with uncertainty.
        """
        # Likelihood: goals ~ Binomial(n_shots, p) approximated by
        # Poisson(xg) → update alpha/beta with goal outcome
        self.alpha += actual_goals
        self.beta += max(0, xg_sum - actual_goals)
        self.n_matches += 1
        self.total_xg += xg_sum
        self.total_goals += actual_goals

        return self.posterior_mean, self.posterior_std

    @property
    def posterior_mean(self) -> float:
        """Calibrated xG expectation (posterior mean of conversion rate)."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def posterior_std(self) -> float:
        """Posterior standard deviation (uncertainty)."""
        a, b = self.alpha, self.beta
        return math.sqrt((a * b) / ((a + b) ** 2 * (a + b + 1)))

    @property
    def credible_interval_90(self) -> Tuple[float, float]:
        """90% credible interval using Beta quantiles (approximation)."""
        mean = self.posterior_mean
        std = self.posterior_std
        # Normal approximation for large alpha+beta
        lo = max(0, mean - 1.645 * std)
        hi = min(1, mean + 1.645 * std)
        return (round(lo, 4), round(hi, 4))

    def calibrate_xg(self, raw_xg: float) -> dict:
        """
        Calibrate a raw xG estimate using the posterior.

        Args:
            raw_xg: Raw xG sum from shot-level model.

        Returns:
            Dict[str, Any] with calibrated_xg, raw_xg, adjustment_factor, uncertainty, ci_90.
        """
        # Adjustment factor: posterior conversion rate / league average
        league_avg = self.total_goals / max(1, self.n_matches)
        if league_avg > 0:
            adj = self.posterior_mean / (league_avg / max(1, self.n_matches))
        else:
            adj = 1.0

        calibrated = raw_xg * adj
        return {
            "calibrated_xg": round(calibrated, 3),
            "raw_xg": round(raw_xg, 3),
            "adjustment_factor": round(adj, 4),
            "uncertainty": round(self.posterior_std, 4),
            "ci_90": list(self.credible_interval_90),
        }

    def to_dict(self) -> dict:
        """Export state for persistence."""
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "n_matches": self.n_matches,
            "total_xg": self.total_xg,
            "total_goals": self.total_goals,
        }


def estimate_match_xg(shots_home: int, sot_home: int, goals_home: int,
                       shots_away: int, sot_away: int, goals_away: int) -> Dict[str, float]:
    """
    Estimate match-level xG from aggregate shot stats (no shot-level data needed).

    Uses a simple logistic model calibrated on historical data:
      xG ≈ shots_on_target × conversion_rate + shots_off_target × 0.03

    Args:
        shots_home/away: Total shots.
        sot_home/away: Shots on target.
        goals_home/away: Actual goals (for calibration).

    Returns:
        Dict with xG_home, xG_away, xG_diff, expected_points_home, expected_points_away.
    """
    # Base conversion rates (from historical European league averages)
    SOT_CONV = 0.32   # shots on target → goal probability
    SHOT_CONV = 0.03  # shots off target → goal probability (deflections, etc.)

    xg_home = sot_home * SOT_CONV + max(0, shots_home - sot_home) * SHOT_CONV
    xg_away = sot_away * SOT_CONV + max(0, shots_away - sot_away) * SHOT_CONV

    # Expected points from xG (Poisson model)
    ep_home, ep_away = _expected_points_from_xg(xg_home, xg_away)

    return {
        "xg_home": round(xg_home, 3),
        "xg_away": round(xg_away, 3),
        "xg_diff": round(xg_home - xg_away, 3),
        "expected_points_home": round(ep_home, 3),
        "expected_points_away": round(ep_away, 3),
        "actual_goals_home": goals_home,
        "actual_goals_away": goals_away,
    }


def _expected_points_from_xg(xg_h: float, xg_a: float) -> Tuple[float, float]:
    """
    Compute expected points from xG using Poisson scoreline matrix.

    Returns (ep_home, ep_away) where 3=win, 1=draw, 0=loss.
    """
    max_goals = 8
    p_home_win = 0.0
    p_draw = 0.0
    p_away_win = 0.0

    for hg in range(max_goals + 1):
        p_hg = _poisson_pmf(hg, max(xg_h, 0.01))
        for ag in range(max_goals + 1):
            p_ag = _poisson_pmf(ag, max(xg_a, 0.01))
            prob = p_hg * p_ag
            if hg > ag:
                p_home_win += prob
            elif hg < ag:
                p_away_win += prob
            else:
                p_draw += prob

    ep_home = 3 * p_home_win + 1 * p_draw
    ep_away = 3 * p_away_win + 1 * p_draw
    return ep_home, ep_away


def _poisson_pmf(k: int, lam: float) -> float:
    """Poisson probability mass function."""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def rolling_xg_features(team_history: list, window: int = 10) -> Dict[str, float]:
    """
    Compute rolling xG features from a team's match history.

    Args:
        team_history: List of dicts with keys: xg_for, xg_against, goals_for, goals_against.
        window: Rolling window size.

    Returns:
        Dict with rolling xG averages, xG difference, xG efficiency, etc.
    """
    if not team_history:
        return {
            "xg_for_avg": 0.0, "xg_against_avg": 0.0,
            "xg_diff_avg": 0.0, "xg_efficiency": 0.0,
            "xg_overperformance": 0.0,
        }

    recent = team_history[-window:]
    n = len(recent)

    xg_for = [m.get("xg_for", 0) for m in recent]
    xg_against = [m.get("xg_against", 0) for m in recent]
    goals_for = [m.get("goals_for", 0) for m in recent]
    goals_against = [m.get("goals_against", 0) for m in recent]

    avg_xgf = sum(xg_for) / n
    avg_xga = sum(xg_against) / n
    avg_gf = sum(goals_for) / n
    avg_ga = sum(goals_against) / n

    # xG efficiency: actual goals / xG (>1 = overperforming)
    total_xgf = sum(xg_for)
    efficiency = sum(goals_for) / max(total_xgf, 0.01)

    # xG overperformance: actual - expected
    overperf = avg_gf - avg_xgf

    return {
        "xg_for_avg": round(avg_xgf, 3),
        "xg_against_avg": round(avg_xga, 3),
        "xg_diff_avg": round(avg_xgf - avg_xga, 3),
        "xg_efficiency": round(efficiency, 3),
        "xg_overperformance": round(overperf, 3),
    }
