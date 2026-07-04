#!/usr/bin/env python3
"""
Monte Carlo match simulation for football_predictor.py.
Fixes probability fixation by injecting Poisson lambda uncertainty.

Reference: penaltyblog FootballProbabilityGrid + Dixon-Coles tau adjustment.
Uses numpy for Poisson sampling — no scipy dependency.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


def poisson_sample(lambda_val: float, rng: np.random.Generator = None) -> int:
    """Sample from Poisson distribution."""
    if rng is None:
        rng = np.random.default_rng()
    if lambda_val <= 0:
        return 0
    return int(rng.poisson(lambda_val))


def gamma_sample(shape: float, scale: float, rng: np.random.Generator = None) -> float:
    """Sample from Gamma distribution."""
    if rng is None:
        rng = np.random.default_rng()
    return float(rng.gamma(shape, scale))


def dixon_coles_tau(h: int, a: int, lh: float, la: float, rho: float) -> float:
    """Dixon-Coles tau adjustment for low-scoring draws."""
    if h == 0 and a == 0:
        return 1.0 - rho * lh * la
    if h == 1 and a == 0:
        return 1.0 + rho * lh
    if h == 0 and a == 1:
        return 1.0 + rho * la
    if h == 1 and a == 1:
        return 1.0 - rho
    return 1.0


def analytic_score_grid(
    lambda_home: float, lambda_away: float,
    rho: float = -0.13, max_goals: int = 8
) -> Dict:
    """
    Compute analytic score probability grid (penaltyblog-style Dixon-Coles).

    Args:
        lambda_home: Home expected goals
        lambda_away: Away expected goals
        rho: Dixon-Coles correlation (default -0.13)
        max_goals: Grid size

    Returns:
        Dict with homeWin, draw, awayWin, mostLikelyScore, grid
    """
    from scipy.stats import poisson as poisson_dist

    h_goals = np.arange(max_goals + 1)
    a_goals = np.arange(max_goals + 1)

    home_probs = poisson_dist.pmf(h_goals, lambda_home)
    away_probs = poisson_dist.pmf(a_goals, lambda_away)

    grid = np.outer(home_probs, away_probs)

    # Apply DC tau adjustment
    if rho != 0.0:
        grid[0, 0] *= 1.0 - rho * lambda_home * lambda_away
        grid[1, 0] *= 1.0 + rho * lambda_home
        grid[0, 1] *= 1.0 + rho * lambda_away
        grid[1, 1] *= 1.0 - rho

    # Normalize
    grid /= grid.sum()

    # Compute 1X2
    I, J = np.indices(grid.shape)
    home_win = float(grid[I > J].sum())
    draw = float(grid[I == J].sum())
    away_win = float(grid[I < J].sum())

    # Most likely score
    idx = np.unravel_index(np.argmax(grid), grid.shape)
    most_likely = f"{idx[0]}-{idx[1]}"

    return {
        "homeWin": round(home_win, 4),
        "draw": round(draw, 4),
        "awayWin": round(away_win, 4),
        "mostLikelyScore": most_likely,
    }


def monte_carlo_simulate(
    lambda_home: float, lambda_away: float,
    rho: float = -0.13,
    n_sim: int = 10000,
    lambda_uncertainty: float = 0.15,
    seed: Optional[int] = None,
) -> Dict:
    """
    Monte Carlo match simulation with lambda uncertainty.

    Instead of fixed (λH, λA) → fixed probability, we:
    1. Sample λH' ~ Gamma(k, λH/k) and λA' ~ Gamma(k, λA/k)
    2. Sample goals from Poisson(λH') and Poisson(λA')
    3. Repeat N times, aggregate outcomes

    Each call produces slightly different probabilities because
    the Poisson sampling introduces genuine randomness.

    Args:
        lambda_home: Base home expected goals
        lambda_away: Base away expected goals
        rho: Dixon-Coles rho (default -0.13)
        n_sim: Number of simulations (default 10000)
        lambda_uncertainty: CV of lambda uncertainty (default 0.15)
        seed: Optional random seed for reproducibility

    Returns:
        Dict with homeWin, draw, awayWin, topScores, confidence95_goalDiff
    """
    rng = np.random.default_rng(seed)

    # Gamma shape parameter — higher = less lambda uncertainty
    shape_k = 1.0 / (lambda_uncertainty ** 2)

    home_wins = 0
    draws = 0
    away_wins = 0
    score_counts = {}
    goal_diffs = []

    for _ in range(n_sim):
        # Sample uncertain lambdas from Gamma
        lh = max(0.01, gamma_sample(shape_k, lambda_home / shape_k, rng))
        la = max(0.01, gamma_sample(shape_k, lambda_away / shape_k, rng))

        # Sample goals from Poisson
        hg = poisson_sample(lh, rng)
        ag = poisson_sample(la, rng)

        # Apply DC tau adjustment probabilistically for low scores
        if hg <= 1 and ag <= 1 and rho != 0:
            tau = dixon_coles_tau(hg, ag, lh, la, rho)
            if rng.random() > abs(tau):
                continue  # reject sample

        if hg > ag:
            home_wins += 1
        elif hg == ag:
            draws += 1
        else:
            away_wins += 1

        key = f"{hg}-{ag}"
        score_counts[key] = score_counts.get(key, 0) + 1
        goal_diffs.append(hg - ag)

    n = home_wins + draws + away_wins
    goal_diffs_sorted = sorted(goal_diffs)

    # Top 10 most likely scores
    top_scores = sorted(
        [{"score": s, "prob": round(c / n, 4)} for s, c in score_counts.items()],
        key=lambda x: -x["prob"]
    )[:10]

    # 95% confidence interval on goal difference
    ci_lo = goal_diffs_sorted[int(n * 0.025)]
    ci_hi = goal_diffs_sorted[min(int(n * 0.975), n - 1)]

    return {
        "homeWin": round(home_wins / n, 4),
        "draw": round(draws / n, 4),
        "awayWin": round(away_wins / n, 4),
        "nSimulations": n,
        "topScores": top_scores,
        "expectedGoalDiff": round(sum(goal_diffs) / n, 2),
        "confidence95_goalDiff": [ci_lo, ci_hi],
    }


def full_prediction(
    lambda_home: float, lambda_away: float,
    rho: float = -0.13,
    n_sim: int = 10000,
    lambda_uncertainty: float = 0.15,
    seed: Optional[int] = None,
) -> Dict:
    """
    Full prediction: analytic grid + Monte Carlo with uncertainty.

    Args:
        lambda_home: Home expected goals
        lambda_away: Away expected goals
        rho: Dixon-Coles rho
        n_sim: Number of MC simulations
        lambda_uncertainty: CV of lambda uncertainty
        seed: Optional random seed

    Returns:
        Dict with analytic, monteCarlo, and live probabilities
    """
    analytic = analytic_score_grid(lambda_home, lambda_away, rho)
    mc = monte_carlo_simulate(lambda_home, lambda_away, rho, n_sim, lambda_uncertainty, seed)

    return {
        "analytic": analytic,
        "monteCarlo": mc,
        "live": {
            "homeWin": mc["homeWin"],
            "draw": mc["draw"],
            "awayWin": mc["awayWin"],
        },
    }


if __name__ == "__main__":
    import sys

    # Quick test: Brazil vs Argentina (neutral)
    lh = float(sys.argv[1]) if len(sys.argv) > 1 else 1.3
    la = float(sys.argv[2]) if len(sys.argv) > 2 else 1.4

    print(f"\n  λ_home={lh:.2f}  λ_away={la:.2f}\n")

    result = full_prediction(lh, la)

    print(f"  Analytic (Dixon-Coles):")
    print(f"    Home win:  {result['analytic']['homeWin']*100:.1f}%")
    print(f"    Draw:      {result['analytic']['draw']*100:.1f}%")
    print(f"    Away win:  {result['analytic']['awayWin']*100:.1f}%")
    print(f"    Most likely: {result['analytic']['mostLikelyScore']}")

    print(f"\n  Monte Carlo ({result['monteCarlo']['nSimulations']} sims):")
    print(f"    Home win:  {result['monteCarlo']['homeWin']*100:.1f}%")
    print(f"    Draw:      {result['monteCarlo']['draw']*100:.1f}%")
    print(f"    Away win:  {result['monteCarlo']['awayWin']*100:.1f}%")
    scores_str = ", ".join(f"{s['score']}({s['prob']*100:.1f}%)" for s in result['monteCarlo']['topScores'][:5])
    print(f"    Top scores: {scores_str}")
    print(f"    95% CI goal diff: {result['monteCarlo']['confidence95_goalDiff']}")
