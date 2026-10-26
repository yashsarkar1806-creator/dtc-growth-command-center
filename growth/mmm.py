"""Lightweight marketing-mix model with diminishing returns + budget optimizer.

Model: revenue_t = base * season_t + sum_c beta_c * log1p(spend_ct / k_c) * season_t
k_c (saturation half-scale) is chosen per channel by grid search, betas by non-negative
least squares. This is intentionally simple and explainable, not a production MMM.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize, nnls


@dataclass
class ChannelModel:
    channels: list[str]
    beta: dict[str, float]
    k: dict[str, float]
    base: float
    r2: float
    mape: float
    extra: dict = field(default_factory=dict)

    def channel_revenue(self, channel: str, spend: float | np.ndarray, season: float = 1.0):
        return self.beta[channel] * np.log1p(np.asarray(spend) / self.k[channel]) * season

    def predict(self, spend: dict[str, float], season: float = 1.0) -> float:
        return float(self.base * season + sum(self.channel_revenue(c, spend.get(c, 0.0), season)
                                              for c in self.channels))

    def marginal_roas(self, channel: str, spend: float) -> float:
        """d revenue / d spend at a given spend level (next-dollar return)."""
        return float(self.beta[channel] / (self.k[channel] + spend))


def _design(df: pd.DataFrame, channels: list[str], k: dict[str, float]) -> np.ndarray:
    season = df["season_index"].to_numpy() if "season_index" in df else np.ones(len(df))
    cols = [season] + [np.log1p(df[c].to_numpy() / k[c]) * season for c in channels]
    return np.column_stack(cols)


def fit(df: pd.DataFrame, channels: list[str], k_grid=None, n_rounds: int = 2) -> ChannelModel:
    """Fit by coordinate-wise grid search on k and NNLS on betas."""
    y = df["revenue"].to_numpy(dtype=float)
    k_grid = k_grid if k_grid is not None else None
    k = {c: max(float(df[c].mean()), 1.0) for c in channels}

    def score(kk):
        X = _design(df, channels, kk)
        coef, _ = nnls(X, y)
        resid = y - X @ coef
        return float(resid @ resid), coef

    for _ in range(n_rounds):
        for c in channels:
            mean = max(float(df[c].mean()), 1.0)
            grid = k_grid if k_grid is not None else mean * np.array([0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1, 1.5, 2, 3, 5])
            best = min(grid, key=lambda g: score({**k, c: float(g)})[0])
            k[c] = float(best)
    sse, coef = score(k)
    X = _design(df, channels, k)
    pred = X @ coef
    r2 = 1 - sse / float(((y - y.mean()) ** 2).sum())
    mape = float(np.mean(np.abs((y - pred) / y)))
    return ChannelModel(channels, {c: float(b) for c, b in zip(channels, coef[1:])}, k,
                        float(coef[0]), float(r2), mape)


def contributions(model: ChannelModel, df: pd.DataFrame) -> pd.DataFrame:
    season = df["season_index"].to_numpy() if "season_index" in df else np.ones(len(df))
    rows = []
    for c in model.channels:
        rev = model.channel_revenue(c, df[c].to_numpy(), season).sum()
        spend = df[c].sum()
        rows.append({"channel": c, "spend": spend, "modelled_revenue": rev,
                     "roas": rev / spend if spend else np.nan,
                     "marginal_roas_at_avg_spend": model.marginal_roas(c, float(df[c].mean()))})
    return pd.DataFrame(rows).sort_values("modelled_revenue", ascending=False).reset_index(drop=True)


def optimise_budget(model: ChannelModel, total_budget: float, current: dict[str, float],
                    max_change: float = 0.5, min_spend: dict[str, float] | None = None) -> pd.DataFrame:
    """Maximise predicted weekly revenue subject to sum(spend)=budget and +/- max_change per channel."""
    chans = model.channels
    min_spend = min_spend or {}
    bounds = []
    for c in chans:
        cur = current.get(c, 0.0)
        lo = max(cur * (1 - max_change), min_spend.get(c, 0.0))
        hi = cur * (1 + max_change) if cur > 0 else total_budget * 0.2
        bounds.append((lo, max(hi, lo)))
    lo_sum, hi_sum = sum(b[0] for b in bounds), sum(b[1] for b in bounds)
    if not lo_sum <= total_budget <= hi_sum:
        raise ValueError(f"Budget {total_budget:,.0f} infeasible with bounds [{lo_sum:,.0f}, {hi_sum:,.0f}]; "
                         "widen max_change.")
    x0 = np.array([current.get(c, 0.0) for c in chans], dtype=float)
    x0 = np.clip(x0 * total_budget / max(x0.sum(), 1e-9), [b[0] for b in bounds], [b[1] for b in bounds])

    def neg(x):
        return -model.predict(dict(zip(chans, x)))

    def grad(x):
        return -np.array([model.marginal_roas(c, xi) for c, xi in zip(chans, x)])

    res = minimize(neg, x0, jac=grad, bounds=bounds, method="SLSQP",
                   constraints=[{"type": "eq", "fun": lambda x: x.sum() - total_budget,
                                 "jac": lambda x: np.ones_like(x)}],
                   options={"maxiter": 500, "ftol": 1e-9})
    if not res.success:
        raise RuntimeError(f"Optimiser failed: {res.message}")
    opt = res.x
    out = pd.DataFrame({"channel": chans,
                        "current_spend": [current.get(c, 0.0) for c in chans],
                        "recommended_spend": opt.round(0)})
    out["change"] = out["recommended_spend"] - out["current_spend"]
    out["current_revenue"] = [float(model.channel_revenue(c, current.get(c, 0.0))) for c in chans]
    out["recommended_revenue"] = [float(model.channel_revenue(c, s)) for c, s in zip(chans, opt)]
    out["marginal_roas_after"] = [model.marginal_roas(c, s) for c, s in zip(chans, opt)]
    return out
