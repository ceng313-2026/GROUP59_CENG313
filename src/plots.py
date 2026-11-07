"""Plotting helpers used in the project notebooks."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS_PER_YEAR = 252


def _band(n: int) -> float:
    """Approximate 95% band for a correlation estimated from n observations."""
    return 1.96 / np.sqrt(n)


def plot_feature_availability(df: pd.DataFrame, features: list[str], id_col: str = "date_id"):
    """Heatmap of which features are available (black) on which day."""
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.imshow(
        df[features].notna().T,
        aspect="auto",
        cmap="Greys",
        interpolation="nearest",
        extent=(df[id_col].min(), df[id_col].max(), len(features) - 0.5, -0.5),
    )
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=5)
    ax.set_xlabel(id_col)
    ax.set_title("Feature availability over time (black = available)")
    plt.show()


def plot_available_feature_count(df: pd.DataFrame, features: list[str], id_col: str = "date_id"):
    """Number of non-missing features per day."""
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.plot(df[id_col], df[features].notna().sum(axis=1))
    ax.set_xlabel(id_col)
    ax.set_ylabel("available features")
    ax.set_title("Number of available features per day")
    plt.show()


def plot_cumulative_and_daily_returns(returns: pd.Series):
    """Cumulative growth of buy-and-hold (log scale) and the daily returns below it."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    (1 + returns).cumprod().plot(ax=axes[0], logy=True)
    axes[0].set_title("Cumulative growth of 1 unit invested (buy-and-hold, log scale)")
    returns.plot(ax=axes[1], lw=0.5)
    axes[1].set_title("Daily forward returns")
    axes[1].set_xlabel("date_id")
    plt.tight_layout()
    plt.show()


def plot_return_distribution(returns: pd.Series, bins: int = 100) -> pd.Series:
    """Histogram of returns against a normal distribution with the same mean and std.

    Returns skewness, excess kurtosis and the share of positive days.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.hist(returns, bins=bins, density=True, alpha=0.7, label=returns.name)
    x = np.linspace(returns.min(), returns.max(), 300)
    ax.plot(x, stats.norm.pdf(x, returns.mean(), returns.std()), "r",
            label="normal distribution (same mean/std)")
    ax.set_title("Distribution of daily forward returns")
    ax.legend()
    plt.show()

    return pd.Series({
        "skewness": returns.skew(),
        "excess_kurtosis": returns.kurt(),
        "share_positive_days": (returns > 0).mean(),
    }).round(3)


def plot_rolling_volatility(returns: pd.Series, window: int = TRADING_DAYS_PER_YEAR):
    """Rolling annualised volatility."""
    rolling_vol = returns.rolling(window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    fig, ax = plt.subplots(figsize=(14, 3))
    rolling_vol.plot(ax=ax)
    ax.set_title("Rolling 1-year annualised volatility of forward returns")
    ax.set_xlabel("date_id")
    ax.set_ylabel("volatility")
    plt.show()


def plot_autocorrelation(returns: pd.Series, max_lag: int = 20) -> pd.DataFrame:
    """Autocorrelation of returns and of absolute returns, with an approximate 95% band."""
    lags = range(1, max_lag + 1)
    acf = pd.DataFrame({
        "returns": [returns.autocorr(lag) for lag in lags],
        "absolute returns": [returns.abs().autocorr(lag) for lag in lags],
    }, index=lags)

    ax = acf.plot.bar(figsize=(14, 4))
    band = _band(len(returns))
    ax.axhline(band, color="grey", ls="--", lw=1)
    ax.axhline(-band, color="grey", ls="--", lw=1)
    ax.set_xlabel("lag (days)")
    ax.set_title("Autocorrelation of daily returns vs. absolute returns (dashed = 95% band)")
    plt.show()
    return acf


def plot_top_correlations(corr: pd.DataFrame, n_obs: int, top_n: int = 20):
    """Bar chart of the features with the largest |Spearman| correlation, coloured by group.

    `corr` needs the columns `group`, `spearman` and `abs_spearman`.
    """
    top = corr.sort_values("abs_spearman", ascending=False).head(top_n)
    groups = list(dict.fromkeys(corr["group"]))
    group_colors = {g: f"C{i}" for i, g in enumerate(groups)}

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(top.index, top["spearman"], color=top["group"].map(group_colors))
    band = _band(n_obs)
    ax.axhline(band, color="grey", ls="--", lw=1)
    ax.axhline(-band, color="grey", ls="--", lw=1)
    ax.set_title(f"Top {top_n} features by |Spearman correlation| with the target (dashed = approx. 95% band)")
    ax.set_ylabel("Spearman correlation")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in group_colors.values()]
    ax.legend(handles, group_colors.keys(), title="group", ncol=len(groups))
    plt.show()


def plot_decile_means(df: pd.DataFrame, features: list[str], target: str, n_bins: int = 10):
    """Mean target in each decile of each feature."""
    fig, axes = plt.subplots(1, len(features), figsize=(4 * len(features), 3.5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, f in zip(axes, features):
        bins = pd.qcut(df[f].rank(method="first"), n_bins, labels=range(1, n_bins + 1))
        df.groupby(bins, observed=True)[target].mean().plot.bar(ax=ax)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(f)
        ax.set_xlabel("feature decile (low → high)")
    axes[0].set_ylabel("mean target")
    plt.suptitle(f"Mean target by feature decile for the {len(features)} strongest features", y=1.03)
    plt.show()


def plot_feature_correlation_heatmap(df: pd.DataFrame, features: list[str], threshold: float = 0.9) -> int:
    """Spearman correlation heatmap between features.

    Returns the number of feature pairs whose absolute correlation exceeds `threshold`.
    """
    feat_corr = df[features].corr(method="spearman")

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(feat_corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(features, fontsize=5, rotation=90)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=5)
    ax.set_title("Spearman correlation between features")
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.show()

    return int(((feat_corr.abs() > threshold).sum().sum() - len(features)) // 2)


def plot_period_correlations(block_corr: pd.DataFrame, labels: list[str]):
    """Annotated heatmap of feature–target correlations per period (rows = features)."""
    n_rows, n_cols = block_corr.shape
    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(block_corr, cmap="RdBu_r", vmin=-0.15, vmax=0.15, aspect="auto")
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(labels, rotation=30)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(block_corr.index)
    for i in range(n_rows):
        for j in range(n_cols):
            v = block_corr.iloc[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
    ax.set_xlabel("period (date_id range)")
    ax.set_title(f"Spearman correlation with the target per period (top {n_rows} features, blank = not available)")
    plt.colorbar(im, ax=ax)
    plt.show()


def plot_rolling_correlation(df: pd.DataFrame, features: list[str], target: str,
                             window: int = 504, min_periods: int = 252, id_col: str = "date_id"):
    """Rolling correlation of each feature with the target."""
    s = df.set_index(id_col)
    fig, ax = plt.subplots(figsize=(14, 4))
    for f in features:
        s[f].rolling(window, min_periods=min_periods).corr(s[target]).plot(ax=ax, label=f)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel(id_col)
    ax.set_ylabel("correlation")
    ax.set_title(f"Rolling {window // TRADING_DAYS_PER_YEAR}-year correlation with the target (top {len(features)} features)")
    ax.legend()
    plt.show()


def plot_folds(splits, date_ids: pd.Series):
    """Training (blue) and validation (orange) range of each walk-forward fold."""
    fig, ax = plt.subplots(figsize=(14, 3))
    for k, (tr, va) in enumerate(splits):
        ax.barh(k + 1, len(tr), left=date_ids.iloc[tr[0]], color="C0")
        ax.barh(k + 1, len(va), left=date_ids.iloc[va[0]], color="C1")
    ax.invert_yaxis()
    ax.set_xlabel("date_id")
    ax.set_ylabel("fold")
    ax.set_title("Walk-forward folds (blue = training, orange = validation)")
    plt.show()


def plot_fold_scores(fold_table: pd.DataFrame):
    """Grouped bar chart of the competition score per fold (rows = folds, columns = models)."""
    ax = fold_table.plot.bar(figsize=(14, 4))
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("adjusted Sharpe")
    ax.set_title("Competition score per validation fold")
    plt.show()


def plot_strategies(allocations: dict, val: pd.DataFrame, show_allocation=("Ridge", "LightGBM")):
    """Cumulative growth of each strategy and the daily allocation of selected models.

    `allocations` maps a model name to its allocations on the rows of `val`.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    rf = val["risk_free_rate"].to_numpy()
    fwd = val["forward_returns"].to_numpy()
    for name, alloc in allocations.items():
        alloc = np.asarray(alloc, dtype=float)
        strat = rf * (1 - alloc) + alloc * fwd
        axes[0].plot(val["date_id"], np.cumprod(1 + strat), label=name)
    axes[0].set_yscale("log")
    axes[0].set_title("Cumulative growth over the validation period (log scale)")
    axes[0].legend()

    for name in show_allocation:
        axes[1].plot(val["date_id"], np.asarray(allocations[name]), lw=0.6, label=name)
    axes[1].axhline(1, color="black", lw=0.8, ls="--")
    axes[1].set_title("Daily allocation chosen by the models")
    axes[1].set_xlabel("date_id")
    axes[1].legend()
    plt.tight_layout()
    plt.show()