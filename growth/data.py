"""Synthetic data generator for a FICTIONAL DTC brand.

Everything produced here is made-up sample data for demonstration only.
"Fernway Botanics" is not a real company and no figures relate to any real business.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

BRAND_NAME = "Fernway Botanics (fictional DTC skincare brand)"
CHANNELS = ["Paid Social", "Paid Search", "Email/SMS", "Influencer", "Affiliate"]
PRODUCTS = ["Cleanser", "Serum", "Moisturiser", "SPF", "Bundle"]

# "True" response curves used to simulate weekly revenue: revenue_c = beta * log1p(spend / k)
_TRUE_BETA = {"Paid Social": 52_000, "Paid Search": 38_000, "Email/SMS": 14_000,
              "Influencer": 30_000, "Affiliate": 12_000}
_TRUE_K = {"Paid Social": 9_000, "Paid Search": 6_000, "Email/SMS": 800,
           "Influencer": 12_000, "Affiliate": 3_000}
_SPEND_RANGE = {"Paid Social": (6_000, 30_000), "Paid Search": (4_000, 18_000),
                "Email/SMS": (300, 2_500), "Influencer": (0, 25_000), "Affiliate": (1_000, 7_000)}

_POS = [
    "Love the {p}, my skin feels so soft and hydrated.",
    "The {p} absorbed quickly and smells amazing. Will reorder!",
    "Fast shipping and the packaging was beautiful.",
    "Best {p} I've tried, gentle and effective.",
    "Customer support swapped my order in a day, great service.",
    "Great value in the bundle, the {p} is my new favourite.",
]
_NEG = [
    "The {p} broke me out, really disappointed.",
    "Delivery took two weeks and nobody answered my emails.",
    "Too expensive for such a small bottle of {p}.",
    "The pump on the {p} stopped working after a week.",
    "Package arrived damaged and the {p} had leaked.",
    "Subscription was hard to cancel, frustrating experience.",
]
_NEU = [
    "The {p} is okay, nothing special.",
    "Arrived on time. Haven't noticed a difference yet with the {p}.",
    "Decent {p} but the scent is a bit strong for me.",
]


@dataclass(frozen=True)
class DemoData:
    customers: pd.DataFrame
    orders: pd.DataFrame
    channel_weekly: pd.DataFrame
    reviews: pd.DataFrame


def make_orders(n_customers: int = 2_000, start: str = "2025-01-01", months: int = 18,
                seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate customers (with acquisition channel) and their order history."""
    rng = np.random.default_rng(seed)
    start_ts = pd.Timestamp(start)
    end_ts = start_ts + pd.DateOffset(months=months)
    span_days = (end_ts - start_ts).days

    acq_days = np.sort(rng.integers(0, span_days - 1, n_customers))
    channel_p = np.array([0.32, 0.24, 0.10, 0.20, 0.14])
    customers = pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(n_customers)],
        "acquired_at": start_ts + pd.to_timedelta(acq_days, unit="D"),
        "acq_channel": rng.choice(CHANNELS, n_customers, p=channel_p),
    })
    # Channel-specific loyalty: email/SMS-acquired customers repeat more, affiliate less.
    loyalty = {"Paid Social": 0.9, "Paid Search": 1.0, "Email/SMS": 1.4,
               "Influencer": 0.8, "Affiliate": 0.7}

    rows = []
    oid = 0
    for cust, acq, ch in customers.itertuples(index=False):
        t = acq
        repeat_rate = 0.045 * loyalty[ch]  # expected orders per week-ish
        while t < end_ts:
            product = rng.choice(PRODUCTS, p=[0.22, 0.26, 0.2, 0.17, 0.15])
            base = {"Cleanser": 28, "Serum": 54, "Moisturiser": 42, "SPF": 34, "Bundle": 109}[product]
            value = round(float(base * rng.uniform(0.85, 1.25)), 2)
            rows.append((f"O{oid:06d}", cust, t, product, value))
            oid += 1
            # gap until next order: exponential, increasing churn over time
            if rng.random() < 0.38 / loyalty[ch]:
                break  # churned
            gap = rng.exponential(1 / repeat_rate) + 14
            t = t + pd.Timedelta(days=int(gap))
    orders = pd.DataFrame(rows, columns=["order_id", "customer_id", "order_date", "product", "revenue"])
    return customers, orders


def make_channel_weekly(weeks: int = 78, seed: int = 11) -> pd.DataFrame:
    """Simulate weekly spend per channel and total revenue with diminishing returns + noise."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-06", periods=weeks, freq="W-MON")
    df = pd.DataFrame({"week": dates})
    season = 1 + 0.12 * np.sin(np.arange(weeks) / 52 * 2 * np.pi) + np.where(
        (dates.month == 11) | (dates.month == 12), 0.18, 0.0)
    revenue = 18_000 * season  # baseline (organic/brand) revenue
    for ch in CHANNELS:
        lo, hi = _SPEND_RANGE[ch]
        spend = rng.uniform(lo, hi, weeks).round(0)
        if ch == "Influencer":  # bursty
            spend = np.where(rng.random(weeks) < 0.35, 0, spend)
        df[ch] = spend
        revenue = revenue + _TRUE_BETA[ch] * np.log1p(spend / _TRUE_K[ch]) * season
    df["revenue"] = (revenue * rng.normal(1, 0.05, weeks)).round(0)
    df["season_index"] = season.round(4)
    return df


def make_reviews(n: int = 240, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        r = rng.random()
        p = rng.choice(PRODUCTS)
        if r < 0.58:
            text, stars = rng.choice(_POS), int(rng.choice([4, 5], p=[0.3, 0.7]))
        elif r < 0.85:
            text, stars = rng.choice(_NEG), int(rng.choice([1, 2], p=[0.5, 0.5]))
        else:
            text, stars = rng.choice(_NEU), 3
        out.append({"review_id": f"R{i:04d}", "product": p, "stars": stars,
                    "text": text.format(p=p.lower())})
    return pd.DataFrame(out)


def load_demo(seed: int = 7) -> DemoData:
    customers, orders = make_orders(seed=seed)
    return DemoData(customers, orders, make_channel_weekly(seed=seed + 4), make_reviews(seed=seed - 4))
