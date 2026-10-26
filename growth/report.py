"""Exportable insights summary (Markdown)."""
from __future__ import annotations

from datetime import date

import pandas as pd

from .data import BRAND_NAME


def _fmt_money(x: float, signed: bool = False) -> str:
    sign = "-" if x < 0 else ("+" if signed else "")
    return f"{sign}${abs(x):,.0f}"


def build_report(seg: pd.DataFrame, retention: pd.DataFrame, contrib: pd.DataFrame,
                 plan: pd.DataFrame | None, themes: pd.DataFrame, engine: str,
                 brief: dict | None = None) -> str:
    lines = [f"# Growth insights — {BRAND_NAME}", "",
             f"_Generated {date.today().isoformat()} by DTC Growth Command Center. "
             "All data is SYNTHETIC sample data for a fictional brand._", ""]

    top = seg.iloc[0]
    lines += ["## 1. Customers (RFM)",
              f"- Largest revenue segment: **{seg.index[0]}** — {top['customers']:.0f} customers, "
              f"{top['share_of_revenue']:.0%} of revenue. Action: {top['recommended_action']}"]
    if "At Risk" in seg.index:
        ar = seg.loc["At Risk"]
        lines.append(f"- **At Risk**: {ar['customers']:.0f} customers holding {_fmt_money(ar['revenue'])} "
                     f"of historic revenue. Action: {ar['recommended_action']}")
    m1 = retention[1].dropna().mean() if 1 in retention.columns else float("nan")
    m3 = retention[3].dropna().mean() if 3 in retention.columns else float("nan")
    lines += [f"- Average month-1 retention {m1:.1%}; month-3 {m3:.1%}.", ""]

    lines += ["## 2. Channel mix"]
    for r in contrib.itertuples():
        lines.append(f"- {r.channel}: modelled ROAS {r.roas:.2f}, next-dollar ROAS {r.marginal_roas_at_avg_spend:.2f}")
    if plan is not None:
        up = plan.sort_values("change", ascending=False).iloc[0]
        down = plan.sort_values("change").iloc[0]
        uplift = plan["recommended_revenue"].sum() - plan["current_revenue"].sum()
        lines += [f"- Reallocation: move budget toward **{up.channel}** ({_fmt_money(up.change, signed=True)}/wk) and away from "
                  f"**{down.channel}** ({_fmt_money(down.change, signed=True)}/wk); modelled uplift ≈ {_fmt_money(uplift)}/wk "
                  "(directional — validate with a holdout/geo test)."]
    lines.append("")

    lines += ["## 3. Voice of customer", f"_Sentiment engine: {engine}_"]
    for name, r in themes.head(3).iterrows():
        lines.append(f"- **{name}**: {r['mentions']:.0f} mentions, {r['negative_share']:.0%} negative")
    lines.append("")

    if brief:
        lines += ["## 4. Next campaign brief"] + [f"- **{k}**: {v}" for k, v in brief.items()] + [""]

    lines += ["## Caveats", "- Synthetic data; model is a simple log-saturation regression, not causal.",
              "- Treat recommendations as hypotheses to test, not conclusions."]
    return "\n".join(lines)
