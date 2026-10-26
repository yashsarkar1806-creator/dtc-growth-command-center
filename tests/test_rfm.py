import pandas as pd
import pytest

from growth import data, rfm


@pytest.fixture(scope="module")
def orders():
    return data.make_orders(n_customers=300, months=12, seed=1)[1]


def test_rfm_one_row_per_customer_and_scores_in_range(orders):
    r = rfm.compute_rfm(orders)
    assert r["customer_id"].is_unique
    assert len(r) == orders["customer_id"].nunique()
    for col in "RFM":
        assert r[col].between(1, 5).all()
    assert set(r["segment"]) <= set(rfm.SEGMENT_ACTIONS)


def test_rfm_recency_direction():
    o = pd.DataFrame({
        "order_id": [f"o{i}" for i in range(10)],
        "customer_id": [f"c{i}" for i in range(10)],
        "order_date": pd.to_datetime("2025-01-01") + pd.to_timedelta(range(0, 100, 10), unit="D"),
        "revenue": [10.0] * 10,
    })
    r = rfm.compute_rfm(o).set_index("customer_id")
    assert r.loc["c9", "R"] == 5 and r.loc["c0", "R"] == 1  # most recent buyer scores highest


def test_segment_grid():
    assert rfm.segment_for(5, 5) == "Champions"
    assert rfm.segment_for(5, 1) == "New Customers"
    assert rfm.segment_for(1, 5) == "At Risk"
    assert rfm.segment_for(1, 1) == "Hibernating"


def test_segment_summary_shares_sum_to_one(orders):
    s = rfm.segment_summary(rfm.compute_rfm(orders))
    assert s["share_of_revenue"].sum() == pytest.approx(1, abs=0.01)
    assert s["recommended_action"].notna().all()


def test_cohort_retention_month0_is_100pct_and_bounded(orders):
    ret = rfm.cohort_retention(orders)
    assert (ret[0] == 1).all()
    assert ((ret.values >= 0) & (ret.values <= 1)).all()


def test_empty_orders_raise():
    with pytest.raises(ValueError):
        rfm.compute_rfm(pd.DataFrame(columns=["order_id", "customer_id", "order_date", "revenue"]))
