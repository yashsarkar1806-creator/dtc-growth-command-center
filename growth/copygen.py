"""Campaign brief + ad copy generator.

Default: deterministic template engine (always works, no downloads).
Optional: a small Hugging Face instruct model rewrites/extends headline ideas.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

HF_COPY_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"

CHANNEL_SPECS = {
    "Paid Social": {"headline_chars": 40, "primary_chars": 125, "format": "Reels/Stories 9:16, UGC-style"},
    "Paid Search": {"headline_chars": 30, "primary_chars": 90, "format": "Responsive search ad"},
    "Email/SMS": {"headline_chars": 50, "primary_chars": 160, "format": "Subject line + SMS body"},
    "Influencer": {"headline_chars": 60, "primary_chars": 200, "format": "Creator talking points"},
}

OBJECTIVE_ANGLES = {
    "Acquisition": ["first-order offer", "social proof", "problem/solution"],
    "Retention / repeat purchase": ["replenishment reminder", "routine building", "loyalty perk"],
    "Win-back": ["we miss you", "what's new", "time-limited incentive"],
    "Launch": ["newness", "founder story", "early access"],
}


@dataclass
class Brief:
    product: str
    audience: str
    objective: str
    channel: str
    benefit: str
    offer: str
    tone: str = "warm, confident, no hype"


def _trim(s: str, n: int) -> str:
    """Trim to n chars at a word boundary (adds an ellipsis only if it had to cut)."""
    if len(s) <= n:
        return s
    cut = s[: n - 1].rsplit(" ", 1)[0].rstrip(",;:-— ")
    return cut + "…"


def _fit(options: list[str], n: int) -> str:
    """Return the first option that fits the channel limit, else a word-boundary trim of the shortest."""
    for o in options:
        if o and len(o) <= n:
            return o
    return _trim(min((o for o in options if o), key=len), n)


def template_copy(b: Brief) -> dict:
    spec = CHANNEL_SPECS.get(b.channel, CHANNEL_SPECS["Paid Social"])
    angles = OBJECTIVE_ANGLES.get(b.objective, OBJECTIVE_ANGLES["Acquisition"])
    hl = spec["headline_chars"]
    pc = spec["primary_chars"]
    benefit = b.benefit[:1].upper() + b.benefit[1:]
    headlines = [
        _fit([f"{benefit} with {b.product}", benefit, f"Meet {b.product}"], hl),
        _fit([f"Your routine, upgraded: {b.product}", f"Meet {b.product}", "Your routine, upgraded"], hl),
        _fit([f"{b.offer} on {b.product}", b.offer, f"Try {b.product}"] if b.offer else [f"Meet {b.product}"], hl),
    ]
    primary = [
        _fit([f"Made for {b.audience}. {b.product} helps you get {b.benefit}. {b.offer}".strip(),
              f"{b.product} helps you get {b.benefit}. {b.offer}".strip(),
              f"{b.product}: {b.benefit}."], pc),
        _fit([f"{b.product}: {b.benefit}, without the 10-step routine. Try it today.",
              f"{b.product}: {b.benefit}. Try it today."], pc),
    ]
    return {
        "brief": {
            "Objective": b.objective,
            "Audience": b.audience,
            "Single-minded proposition": f"{b.product} = {b.benefit}",
            "Angles to test": ", ".join(angles),
            "Channel / format": f"{b.channel} — {spec['format']}",
            "Offer": b.offer or "None (brand-led)",
            "Tone": b.tone,
            "Primary KPI": {"Acquisition": "CAC / first-order ROAS", "Retention / repeat purchase": "Repeat rate, 90-day LTV",
                            "Win-back": "Reactivation rate", "Launch": "Launch-week revenue, waitlist conversion"}.get(b.objective, "ROAS"),
            "Test plan": "3 headlines x 2 bodies, equal split, 7 days or 1,000 clicks per cell before calling a winner.",
        },
        "headlines": headlines,
        "primary_text": primary,
        "cta": "Shop now" if b.objective != "Retention / repeat purchase" else "Restock now",
        "note": "Only use product claims you can substantiate (e.g. clinical or review data).",
    }


@lru_cache(maxsize=1)
def _load_generator(model_name: str = HF_COPY_MODEL):
    from transformers import pipeline

    return pipeline("text-generation", model=model_name)


def ai_headlines(b: Brief, n: int = 3, max_new_tokens: int = 80) -> tuple[list[str], str]:
    """Try the HF model; fall back to template headlines. Returns (headlines, engine)."""
    try:
        gen = _load_generator()
        msgs = [{"role": "user", "content": (
            f"Write {n} short ad headlines (under 8 words each) for {b.product}, a skincare product, "
            f"for {b.audience}. Benefit: {b.benefit}. Tone: {b.tone}. One per line, no numbering.")}]
        out = gen(msgs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.8, top_p=0.9)
        text = out[0]["generated_text"][-1]["content"]
        lines = clean_generated_lines(text, n)
        if lines:
            # top up with template headlines if the small model returned fewer than n
            for h in template_copy(b)["headlines"]:
                if len(lines) >= n:
                    break
                if h not in lines:
                    lines.append(h)
            return lines, f"Hugging Face: {HF_COPY_MODEL}"
        raise ValueError("empty generation")
    except Exception as exc:
        return template_copy(b)["headlines"][:n], f"Template fallback ({type(exc).__name__})"


def clean_generated_lines(text: str, n: int = 3) -> list[str]:
    """Strip numbering/bullets, drop commentary and duplicates, keep plausible headlines."""
    out: list[str] = []
    for raw in text.splitlines():
        line = raw.strip().lstrip("-*•").strip()
        line = line.lstrip("0123456789").lstrip(".)").strip().strip('"')
        if not line or line.lower().startswith(("note", "here", "sure", "headline")):
            continue
        if len(line.split()) < 2 or len(line) > 100:
            continue
        if line.lower() not in {o.lower() for o in out}:
            out.append(line)
    return out[:n]
