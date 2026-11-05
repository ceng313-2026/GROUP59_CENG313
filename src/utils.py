"""Evaluation utilities for the Hull Tactical Market Prediction competition."""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_ALLOCATION = 0.0
MAX_ALLOCATION = 2.0
TRADING_DAYS_PER_YEAR = 252
VOL_LIMIT = 1.2


def clip_allocation(allocation) -> np.ndarray:
    """Clip allocations to the valid competition range [0, 2]."""
    return np.clip(np.asarray(allocation, dtype=float), MIN_ALLOCATION, MAX_ALLOCATION)


def adjusted_sharpe(
    allocation,
    forward_returns,
    risk_free_rate,
    return_details: bool = False,
):
    """Volatility-adjusted Sharpe ratio used by the competition.

    Parameters
    ----------
    allocation : array-like
        Daily position in the S&P 500, in [0, 2].
    forward_returns : array-like
        Realised next-day S&P 500 returns.
    risk_free_rate : array-like
        Daily risk-free rate for the same days.
    return_details : bool
        If True, also return a dict with the intermediate quantities.
    """
    pos = np.asarray(allocation, dtype=float)
    fwd = np.asarray(forward_returns, dtype=float)
    rf = np.asarray(risk_free_rate, dtype=float)

    if not (len(pos) == len(fwd) == len(rf)):
        raise ValueError("allocation, forward_returns and risk_free_rate must have the same length")
    if len(pos) < 2:
        raise ValueError("need at least two days to compute a Sharpe ratio")
    if pos.min() < MIN_ALLOCATION or pos.max() > MAX_ALLOCATION:
        raise ValueError(f"allocations must be in [{MIN_ALLOCATION}, {MAX_ALLOCATION}]")

    n = len(pos)

    strat_ret = rf * (1 - pos) + pos * fwd
    strat_excess = strat_ret - rf
    strat_mean_excess = np.prod(1 + strat_excess) ** (1 / n) - 1
    strat_std = strat_ret.std(ddof=1)

    if strat_std == 0:
        sharpe = 0.0
    else:
        sharpe = strat_mean_excess / strat_std * np.sqrt(TRADING_DAYS_PER_YEAR)
    strat_vol = strat_std * np.sqrt(TRADING_DAYS_PER_YEAR) * 100

    mkt_excess = fwd - rf
    mkt_mean_excess = np.prod(1 + mkt_excess) ** (1 / n) - 1
    mkt_std = fwd.std(ddof=1)
    mkt_vol = mkt_std * np.sqrt(TRADING_DAYS_PER_YEAR) * 100

    excess_vol = max(0.0, strat_vol / mkt_vol - VOL_LIMIT) if mkt_vol > 0 else 0.0
    vol_penalty = 1 + excess_vol

    return_gap = max(0.0, (mkt_mean_excess - strat_mean_excess) * 100 * TRADING_DAYS_PER_YEAR)
    return_penalty = 1 + (return_gap ** 2) / 100

    score = float(min(sharpe / (vol_penalty * return_penalty), 1_000_000))

    if return_details:
        return score, {
            "sharpe": float(sharpe),
            "strategy_vol": float(strat_vol),
            "market_vol": float(mkt_vol),
            "vol_penalty": float(vol_penalty),
            "return_gap": float(return_gap),
            "return_penalty": float(return_penalty),
        }
    return score


def evaluate_folds(fold_scores: dict) -> pd.DataFrame:
    """Turn {model_name: [score_fold1, score_fold2, ...]} into a summary table."""
    rows = []
    for name, scores in fold_scores.items():
        scores = np.asarray(scores, dtype=float)
        rows.append({
            "model": name,
            "mean": scores.mean(),
            "std": scores.std(ddof=1) if len(scores) > 1 else 0.0,
            "min": scores.min(),
            "max": scores.max(),
            "n_folds": len(scores),
        })
    return pd.DataFrame(rows).sort_values("mean", ascending=False).reset_index(drop=True)