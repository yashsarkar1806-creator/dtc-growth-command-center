import numpy as np
import pandas as pd
import pytest

from growth import data, mmm


@pytest.fixture(scope="module")
def weekly():
    return data.make_channel_weekly(seed=11)


@pytest.fixture(scope="module")
def model(weekly):
    return mmm.fit(weekly, data.CHANNELS)


def test_fit_quality(model):
    assert model.r2 > 0.85
    assert model.mape < 0.08
    assert all(b >= 0 for b in model.beta.values())


def test_diminishing_returns(model):
    for ch in model.channels:
        assert model.marginal_roas(ch, 1_000) > model.marginal_roas(ch, 20_000)


def test_recovers_known_curve_on_clean_data():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"A": rng.uniform(1_000, 20_000, 200), "B": rng.uniform(500, 5_000, 200)})
    df["revenue"] = 5_000 + 30_000 * np.log1p(df["A"] / 5_000) + 8_000 * np.log1p(df["B"] / 1_000)
    m = mmm.fit(df, ["A", "B"])
    assert m.r2 > 0.999
    pred = m.predict({"A": 10_000, "B": 2_000})
    truth = 5_000 + 30_000 * np.log1p(2) + 8_000 * np.log1p(2)
    assert pred == pytest.approx(truth, rel=0.02)


def test_optimiser_respects_budget_and_bounds_and_does_not_lose(model, weekly):
    current = {c: float(weekly[c].tail(4).mean()) for c in model.channels}
    budget = sum(current.values())
    plan = mmm.optimise_budget(model, budget, current, max_change=0.3)
    assert plan["recommended_spend"].sum() == pytest.approx(budget, rel=1e-3)
    for r in plan.itertuples():
        assert r.recommended_spend >= r.current_spend * 0.7 - 1
        assert r.recommended_spend <= r.current_spend * 1.3 + 1
    new = model.predict(dict(zip(plan["channel"], plan["recommended_spend"])))
    assert new >= model.predict(current) - 1


def test_optimiser_equalises_marginal_roas_for_interior_channels(model, weekly):
    current = {c: float(weekly[c].mean()) for c in model.channels}
    plan = mmm.optimise_budget(model, sum(current.values()), current, max_change=1.0)
    interior = plan[(plan["recommended_spend"] > 1) & (plan["recommended_spend"] < plan["current_spend"] * 2 - 1)]
    if len(interior) >= 2:
        assert interior["marginal_roas_after"].std() < 0.05


def test_infeasible_budget_raises(model, weekly):
    current = {c: float(weekly[c].mean()) for c in model.channels}
    with pytest.raises(ValueError):
        mmm.optimise_budget(model, sum(current.values()) * 3, current, max_change=0.1)
