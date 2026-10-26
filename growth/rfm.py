"""RFM segmentation and cohort retention."""
from __future__ import annotations

import pandas as pd

SEGMENT_ACTIONS = {
    "Champions": "Reward & referral program; early access to launches.",
    "Loyal": "Upsell bundles and subscriptions; ask for reviews/UGC.",
    "Potential Loyalists": "Second-purchase nudge flow with a cross-sell.",
    "New Customers": "Onboarding email/SMS series; how-to content.",
    "Needs Attention": "Personalised replenishment reminder.",
    "At Risk": "Win-back offer before they lapse; survey for friction.",
    "Hibernating": "Low-cost reactivation (email only); suppress from paid.",
}


def compute_rfm(orders: pd.DataFrame, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """Return one row per customer with recency (days), frequency, monetary and 1-5 scores."""
    if orders.empty:
        raise ValueError("orders is empty")
    as_of = as_of or (orders["order_date"].max() + pd.Timedelta(days=1))
    g = orders.groupby("customer_id").agg(
        last_order=("order_date", "max"), frequency=("order_id", "count"), monetary=("revenue", "sum"))
    g["recency"] = (as_of - g["last_order"]).dt.days
    g["R"] = _score(g["recency"], reverse=True)
    g["F"] = _score(g["frequency"])
    g["M"] = _score(g["monetary"])
    g["segment"] = [segment_for(r, f) for r, f in zip(g["R"], g["F"])]
    return g.drop(columns="last_order").reset_index()


def _score(s: pd.Series, reverse: bool = False) -> pd.Series:
    """Quintile score 1-5 using ranks (robust to ties). Higher is better."""
    ranks = s.rank(method="first", ascending=not reverse)
    return pd.qcut(ranks, 5, labels=[1, 2, 3, 4, 5]).astype(int)


def segment_for(r: int, f: int) -> str:
    """Simplified RFM grid on Recency x Frequency scores."""
    if r >= 4 and f >= 4:
        return "Champions"
    if r >= 3 and f >= 3:
        return "Loyal"
    if r >= 4 and f <= 2:
        return "New Customers" if f == 1 else "Potential Loyalists"
    if r == 3:
        return "Needs Attention"
    if r <= 2 and f >= 3:
        return "At Risk"
    return "Hibernating"


def segment_summary(rfm: pd.DataFrame) -> pd.DataFrame:
    s = rfm.groupby("segment").agg(customers=("customer_id", "count"), revenue=("monetary", "sum"),
                                   avg_orders=("frequency", "mean"), avg_recency_days=("recency", "mean"))
    s["share_of_revenue"] = s["revenue"] / s["revenue"].sum()
    s["recommended_action"] = s.index.map(SEGMENT_ACTIONS)
    return s.sort_values("revenue", ascending=False).round(3)


def cohort_retention(orders: pd.DataFrame, max_months: int = 12) -> pd.DataFrame:
    """Monthly acquisition cohorts x months-since-first-order, values = share of cohort active."""
    o = orders[["customer_id", "order_date"]].copy()
    o["order_month"] = o["order_date"].dt.to_period("M")
    o["cohort"] = o.groupby("customer_id")["order_month"].transform("min")
    o["age"] = (o["order_month"] - o["cohort"]).apply(lambda x: x.n)
    o = o[o["age"] <= max_months]
    active = o.groupby(["cohort", "age"])["customer_id"].nunique().unstack(fill_value=0)
    sizes = active[0]
    ret = active.divide(sizes, axis=0)
    ret.index = ret.index.astype(str)
    return ret.round(4)
