"""DTC Growth Command Center — Streamlit app.

Run:  streamlit run app.py
All data is SYNTHETIC for a fictional brand unless you upload your own CSVs.
"""
from __future__ import annotations

import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from growth import copygen, data, mmm, reviews, rfm
from growth.report import build_report

st.set_page_config(page_title="DTC Growth Command Center", page_icon="📈", layout="wide")

USE_HF_DEFAULT = os.environ.get("USE_HF_MODELS", "1") == "1"


@st.cache_data(show_spinner=False)
def get_demo(seed: int):
    return data.load_demo(seed)


@st.cache_data(show_spinner="Scoring reviews…")
def get_review_analysis(df: pd.DataFrame, use_model: bool):
    return reviews.analyse(df, use_model=use_model)


@st.cache_resource(show_spinner=False)
def get_model(df: pd.DataFrame):
    return mmm.fit(df, [c for c in data.CHANNELS if c in df.columns])


# ---------------- Sidebar ----------------
st.sidebar.title("📈 Growth Command Center")
st.sidebar.caption(f"Demo brand: **{data.BRAND_NAME}**  \nAll numbers are synthetic sample data.")
seed = st.sidebar.number_input("Synthetic data seed", 1, 999, 7, help="Change to regenerate a different fictional dataset.")
use_hf = st.sidebar.toggle("Use Hugging Face models", value=USE_HF_DEFAULT,
                           help="Falls back to rule-based engines automatically if models can't be downloaded.")
with st.sidebar.expander("Bring your own data (CSV)"):
    st.caption("orders: order_id, customer_id, order_date, revenue  \n"
               "channel weekly: week, revenue, + one column per channel  \nreviews: review_id, text, stars, product")
    up_orders = st.file_uploader("Orders CSV", type="csv")
    up_weekly = st.file_uploader("Channel weekly CSV", type="csv")
    up_reviews = st.file_uploader("Reviews CSV", type="csv")

demo = get_demo(int(seed))
orders = demo.orders
weekly = demo.channel_weekly
review_df = demo.reviews
if up_orders:
    orders = pd.read_csv(up_orders, parse_dates=["order_date"])
if up_weekly:
    weekly = pd.read_csv(up_weekly, parse_dates=["week"])
if up_reviews:
    review_df = pd.read_csv(up_reviews)
channels = [c for c in data.CHANNELS if c in weekly.columns]

st.title("DTC Growth Command Center")
st.caption("One workspace for the questions a DTC growth team asks every Monday: who are our best customers, "
           "where should the next dollar go, what are customers complaining about, and what do we launch next?")

# KPIs
rfm_df = rfm.compute_rfm(orders)
seg = rfm.segment_summary(rfm_df)
ret = rfm.cohort_retention(orders)
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Customers", f"{orders['customer_id'].nunique():,}")
k2.metric("Revenue", f"${orders['revenue'].sum():,.0f}")
k3.metric("AOV", f"${orders['revenue'].mean():,.2f}")
k4.metric("Repeat-customer rate", f"{(rfm_df['frequency'] > 1).mean():.1%}")
k5.metric("Month-1 retention", f"{ret[1].dropna().mean():.1%}" if 1 in ret.columns else "–")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["👥 Customers & retention", "💸 Channel mix & budget",
                                        "💬 Review intelligence", "✍️ Campaign brief & copy", "📄 Insights export"])

# ---------------- Tab 1 ----------------
with tab1:
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("RFM segments")
        fig = px.treemap(seg.reset_index(), path=["segment"], values="revenue", color="avg_orders",
                         color_continuous_scale="Teal", hover_data={"customers": True})
        fig.update_layout(margin=dict(t=10, l=0, r=0, b=0), height=360)
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.subheader("Recency vs frequency")
        fig = px.scatter(rfm_df, x="recency", y="frequency", color="segment", size="monetary", opacity=0.6,
                         labels={"recency": "Days since last order", "frequency": "Orders"})
        fig.update_layout(margin=dict(t=10, l=0, r=0, b=0), height=360)
        st.plotly_chart(fig, width="stretch")
    st.dataframe(seg.style.format({"revenue": "${:,.0f}", "share_of_revenue": "{:.0%}", "avg_orders": "{:.2f}",
                                   "avg_recency_days": "{:.0f}"}), width="stretch")
    st.subheader("Monthly cohort retention")
    show = ret.iloc[:, 1:] if ret.shape[1] > 1 else ret
    fig = go.Figure(go.Heatmap(z=show.values, x=[f"M{c}" for c in show.columns], y=show.index,
                               colorscale="Teal", text=(show.values * 100).round(0), texttemplate="%{text}%",
                               hovertemplate="Cohort %{y}, %{x}: %{z:.1%}<extra></extra>"))
    fig.update_layout(height=520, margin=dict(t=10, l=0, r=0, b=0), yaxis_autorange="reversed")
    st.plotly_chart(fig, width="stretch")
    st.download_button("Download customer segments (CSV)", rfm_df.to_csv(index=False), "rfm_segments.csv")

# ---------------- Tab 2 ----------------
with tab2:
    model = get_model(weekly)
    contrib = mmm.contributions(model, weekly)
    st.caption(f"Log-saturation regression on {len(weekly)} weeks · R² {model.r2:.2f} · MAPE {model.mape:.1%}. "
               "Directional, not causal — validate with holdout tests.")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Response curves")
        curves = []
        for ch in channels:
            mx = float(weekly[ch].max()) * 1.6
            for s in [mx * i / 40 for i in range(41)]:
                curves.append({"channel": ch, "weekly_spend": s, "modelled_revenue": float(model.channel_revenue(ch, s))})
        fig = px.line(pd.DataFrame(curves), x="weekly_spend", y="modelled_revenue", color="channel")
        fig.update_layout(height=380, margin=dict(t=10, l=0, r=0, b=0))
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.subheader("ROAS: average vs next dollar")
        m = contrib.melt(id_vars="channel", value_vars=["roas", "marginal_roas_at_avg_spend"],
                         var_name="metric", value_name="value")
        m["metric"] = m["metric"].map({"roas": "Average ROAS", "marginal_roas_at_avg_spend": "Marginal ROAS"})
        fig = px.bar(m, x="channel", y="value", color="metric", barmode="group")
        fig.add_hline(y=1, line_dash="dot", annotation_text="break-even")
        fig.update_layout(height=380, margin=dict(t=10, l=0, r=0, b=0))
        st.plotly_chart(fig, width="stretch")

    st.subheader("Budget reallocation")
    current = {ch: float(weekly[ch].tail(4).mean()) for ch in channels}
    cur_total = sum(current.values())
    a, b = st.columns(2)
    budget = a.slider("Weekly budget ($)", int(cur_total * 0.5), int(cur_total * 1.5), int(cur_total), step=500)
    max_change = b.slider("Max change per channel", 0.1, 1.0, 0.5, 0.05, format="%.2f",
                          help="0.5 = each channel can move ±50% from its current level")
    try:
        plan = mmm.optimise_budget(model, budget, current, max_change=max_change)
        cur_rev = model.predict(current)
        new_rev = model.predict(dict(zip(plan["channel"], plan["recommended_spend"])))
        x1, x2, x3 = st.columns(3)
        x1.metric("Current plan revenue / wk", f"${cur_rev:,.0f}")
        x2.metric("Optimised revenue / wk", f"${new_rev:,.0f}", f"{new_rev / cur_rev - 1:+.1%}")
        x3.metric("Blended ROAS (optimised)", f"{(new_rev - model.base) / budget:.2f}")
        fig = go.Figure()
        fig.add_bar(x=plan["channel"], y=plan["current_spend"], name="Current")
        fig.add_bar(x=plan["channel"], y=plan["recommended_spend"], name="Recommended")
        fig.update_layout(barmode="group", height=340, margin=dict(t=10, l=0, r=0, b=0))
        st.plotly_chart(fig, width="stretch")
        st.dataframe(plan.style.format({c: "${:,.0f}" for c in ["current_spend", "recommended_spend", "change",
                                                                 "current_revenue", "recommended_revenue"]}
                                       | {"marginal_roas_after": "{:.2f}"}), width="stretch")
        st.session_state["plan"] = plan
    except ValueError as e:
        st.warning(str(e))
        st.session_state["plan"] = None
    st.session_state["contrib"] = contrib

# ---------------- Tab 3 ----------------
with tab3:
    rev_scored, themes, engine = get_review_analysis(review_df, use_hf)
    st.caption(f"Sentiment engine: **{engine}**")
    c1, c2, c3 = st.columns(3)
    vc = rev_scored["label"].value_counts(normalize=True)
    c1.metric("Positive", f"{vc.get('POSITIVE', 0):.0%}")
    c2.metric("Neutral / low-confidence", f"{vc.get('NEUTRAL', 0):.0%}")
    c3.metric("Negative", f"{vc.get('NEGATIVE', 0):.0%}")
    l, r = st.columns([1, 1])
    with l:
        st.subheader("CX fix priority (mentions × negativity)")
        fig = px.bar(themes.reset_index(), x="priority", y="themes", orientation="h", color="negative_share",
                     color_continuous_scale="OrRd")
        fig.update_layout(height=360, margin=dict(t=10, l=0, r=0, b=0), yaxis_autorange="reversed")
        st.plotly_chart(fig, width="stretch")
    with r:
        st.subheader("Sentiment vs star rating")
        fig = px.box(rev_scored, x="stars", y="score", points="all")
        fig.update_layout(height=360, margin=dict(t=10, l=0, r=0, b=0))
        st.plotly_chart(fig, width="stretch")
    mismatch = rev_scored[((rev_scored["stars"] >= 4) & (rev_scored["label"] == "NEGATIVE")) |
                          ((rev_scored["stars"] <= 2) & (rev_scored["label"] == "POSITIVE"))]
    st.markdown(f"**Star/sentiment mismatches** (worth a human read): {len(mismatch)}")
    st.dataframe(rev_scored[["product", "stars", "label", "score", "themes", "text"]], width="stretch", height=260)
    st.markdown("**Try it:** paste a review")
    txt = st.text_input("Review text", "The serum is lovely but shipping took forever.")
    if txt:
        one, eng1 = reviews.score_reviews([txt], use_model=use_hf)
        st.write(f"→ **{one.loc[0, 'label']}** ({one.loc[0, 'score']:+.2f}) · themes: {', '.join(reviews.tag_themes(txt))} · _{eng1}_")
    st.session_state["themes"], st.session_state["engine"] = themes, engine

# ---------------- Tab 4 ----------------
with tab4:
    top_theme = themes.index[0] if len(themes) else "Other"
    st.caption(f"Pre-filled from the data: top CX issue is **{top_theme}**; largest at-risk group drives a win-back idea.")
    c1, c2 = st.columns(2)
    product = c1.text_input("Product", "Barrier Repair Serum")
    audience = c1.text_input("Audience", "lapsed customers aged 25-40 with sensitive skin")
    benefit = c1.text_input("Key benefit", "calm, hydrated skin in 7 days")
    objective = c2.selectbox("Objective", list(copygen.OBJECTIVE_ANGLES), index=2)
    channel = c2.selectbox("Channel", list(copygen.CHANNEL_SPECS))
    offer = c2.text_input("Offer", "Free shipping this week")
    brief = copygen.Brief(product, audience, objective, channel, benefit, offer)
    out = copygen.template_copy(brief)
    l, r = st.columns([1, 1])
    with l:
        st.subheader("Campaign brief")
        for k_, v_ in out["brief"].items():
            st.markdown(f"**{k_}:** {v_}")
    with r:
        st.subheader("Ad copy (template engine)")
        for h in out["headlines"]:
            st.markdown(f"- {h}")
        for ptxt in out["primary_text"]:
            st.info(ptxt)
        st.caption(out["note"])
        if st.button("✨ Generate AI headline variants", disabled=not use_hf):
            with st.spinner("Generating with a small Hugging Face model…"):
                hl, eng = copygen.ai_headlines(brief)
            st.write(f"_{eng}_")
            for h in hl:
                st.markdown(f"- {h}")
    st.session_state["brief"] = out["brief"]

# ---------------- Tab 5 ----------------
with tab5:
    md = build_report(seg, ret, st.session_state.get("contrib", mmm.contributions(get_model(weekly), weekly)),
                      st.session_state.get("plan"), st.session_state.get("themes", themes),
                      st.session_state.get("engine", engine), st.session_state.get("brief"))
    st.markdown(md.replace("$", "\\$"))  # escape $ so Streamlit does not render LaTeX
    st.download_button("Download insights (Markdown)", md, "growth_insights.md")
