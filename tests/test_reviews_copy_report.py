import pandas as pd

from growth import copygen, data, mmm, reviews, rfm
from growth.report import build_report


def test_lexicon_sentiment():
    assert reviews.lexicon_sentiment("Love it, amazing and gentle")[0] == "POSITIVE"
    assert reviews.lexicon_sentiment("Package arrived damaged and leaked")[0] == "NEGATIVE"
    assert reviews.lexicon_sentiment("It is a bottle")[0] == "NEUTRAL"


def test_theme_tagging():
    assert "Shipping & delivery" in reviews.tag_themes("Delivery took two weeks")
    assert reviews.tag_themes("hello") == ["Other"]


def test_analyse_with_fallback_engine():
    df, summary, engine = reviews.analyse(data.make_reviews(n=60), use_model=False)
    assert engine.startswith("Lexicon")
    assert len(df) == 60 and set(df["label"]) <= {"POSITIVE", "NEGATIVE", "NEUTRAL"}
    assert summary["priority"].is_monotonic_decreasing


def test_model_failure_falls_back(monkeypatch):
    def boom(*a, **k):
        raise OSError("offline")
    monkeypatch.setattr(reviews, "_load_pipeline", boom)
    out, engine = reviews.score_reviews(["great product"], use_model=True)
    assert "fallback" in engine.lower() and out.loc[0, "label"] == "POSITIVE"


def test_template_copy_respects_char_limits():
    b = copygen.Brief("Barrier Repair Serum", "busy parents", "Acquisition", "Paid Search",
                      "calm, hydrated skin with a very long benefit statement", "20% off first order")
    out = copygen.template_copy(b)
    assert all(len(h) <= 30 for h in out["headlines"])
    assert all(len(p) <= 90 for p in out["primary_text"])
    assert out["brief"]["Primary KPI"].startswith("CAC")


def test_ai_headlines_falls_back(monkeypatch):
    monkeypatch.setattr(copygen, "_load_generator", lambda *a, **k: (_ for _ in ()).throw(OSError("offline")))
    b = copygen.Brief("SPF", "runners", "Launch", "Paid Social", "no white cast", "")
    hl, engine = copygen.ai_headlines(b)
    assert len(hl) == 3 and engine.startswith("Template")


def test_clean_generated_lines():
    raw = "1. Calm skin in 7 days\n2. Calm skin in 7 days\n\nNote: I adjusted the tone\n- Hydration that lasts"
    assert copygen.clean_generated_lines(raw) == ["Calm skin in 7 days", "Hydration that lasts"]


def test_report_contains_all_sections():
    d = data.load_demo(seed=5)
    seg = rfm.segment_summary(rfm.compute_rfm(d.orders))
    ret = rfm.cohort_retention(d.orders)
    m = mmm.fit(d.channel_weekly, data.CHANNELS)
    cur = {c: float(d.channel_weekly[c].mean()) for c in data.CHANNELS}
    plan = mmm.optimise_budget(m, sum(cur.values()), cur)
    _, themes, engine = reviews.analyse(d.reviews, use_model=False)
    brief = copygen.template_copy(copygen.Brief("Serum", "x", "Win-back", "Email/SMS", "y", "z"))["brief"]
    md = build_report(seg, ret, mmm.contributions(m, d.channel_weekly), plan, themes, engine, brief)
    for h in ["## 1. Customers", "## 2. Channel mix", "## 3. Voice of customer", "## 4. Next campaign brief",
              "SYNTHETIC"]:
        assert h in md


def test_demo_data_is_deterministic():
    a, b = data.load_demo(seed=9), data.load_demo(seed=9)
    pd.testing.assert_frame_equal(a.orders, b.orders)
