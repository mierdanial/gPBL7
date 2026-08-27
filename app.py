"""
app.py
------
Streamlit dashboard for the Subway Surfers Fitness AI system.

Wraps the existing pipeline (mock_data -> analytics -> llm_analyzer ->
report_generator) in a visual UI: run a session, see the heart-rate /
movement / game-event story, the AI-generated coaching report, and
cross-session progress.

Run with:

    streamlit run app.py

Works out of the box in mock mode (no API key needed). Toggle "Use real
LLM" in the sidebar to call OpenRouter instead (requires LLM_API_KEY set,
either in a .env file or pasted into the sidebar field for this session).

NOTE ON THIS FILE: the pipeline calls below (all backend logic lives in
core.py) are unchanged from the original implementation. Everything
visual — CSS, layout, cards, the Plotly theme — lives in the "Design
system" section directly below; the rest of this file wires data to
those components.
"""

import os
import json
from datetime import datetime
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import core



# ========================================================================
# Design system (was theme.py)
# ========================================================================
#
# Owns every visual decision so the rest of this file can stay focused on
# data flow: CSS design tokens, a shared Plotly theme, and small HTML
# component builders (KPI cards, badges, the hero rating gauge,
# DATA/INSIGHT/ACTION cards). Nothing here touches the pipeline in
# core.py — nothing about the actual analysis, only how it's presented.
#
# Palette (dark instrumentation-panel aesthetic — AI + IoT + biometrics,
# restrained rather than "gaming neon"):
#     background   #0A0E16   surface   #121826   surface-2  #1A2233
#     border       #232C40   text hi   #E8ECF4   text lo    #8992A9
#     measured     #5B8DEF  (raw sensor data: HR line, KPIs)
#     ai           #34D0C3  (AI-generated content: insights, recommendations)
#     positive     #3ECF8E   caution   #F0B429   negative   #F0596B
#     action       #A78BFA  (things the player should do next)
#
# Type: "Space Grotesk" for headings/labels (technical, geometric),
# "Inter" for body copy, "IBM Plex Mono" for every numeric readout — the
# mono face is what makes KPIs read as *instrument data* rather than
# marketing stats.

from typing import Dict, List, Optional

FONT_IMPORT = (
    "https://fonts.googleapis.com/css2?"
    "family=Space+Grotesk:wght@500;600;700&"
    "family=Inter:wght@400;500;600&"
    "family=IBM+Plex+Mono:wght@500;600&display=swap"
)

COLORS = {
    "bg": "#0A0E16",
    "surface": "#121826",
    "surface2": "#1A2233",
    "border": "#232C40",
    "text_hi": "#E8ECF4",
    "text_lo": "#8992A9",
    "text_muted": "#57607A",
    "measured": "#5B8DEF",
    "ai": "#34D0C3",
    "positive": "#3ECF8E",
    "caution": "#F0B429",
    "negative": "#F0596B",
    "action": "#A78BFA",
}


def inject_css() -> str:
    c = COLORS
    return f"""
<style>
@import url('{FONT_IMPORT}');

:root {{
    --bg: {c['bg']};
    --surface: {c['surface']};
    --surface-2: {c['surface2']};
    --border: {c['border']};
    --text-hi: {c['text_hi']};
    --text-lo: {c['text_lo']};
    --text-muted: {c['text_muted']};
    --measured: {c['measured']};
    --ai: {c['ai']};
    --positive: {c['positive']};
    --caution: {c['caution']};
    --negative: {c['negative']};
    --action: {c['action']};
}}

.stApp {{
    background: radial-gradient(ellipse 1200px 600px at 50% -10%, #131b2c 0%, var(--bg) 55%);
    color: var(--text-hi);
    font-family: 'Inter', sans-serif;
}}

section[data-testid="stSidebar"] {{
    background: var(--surface);
    border-right: 1px solid var(--border);
}}
section[data-testid="stSidebar"] * {{ font-family: 'Inter', sans-serif; }}

h1, h2, h3 {{ font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.01em; }}

#MainMenu, footer {{ visibility: hidden; }}
.block-container {{ padding-top: 1.6rem; max-width: 1180px; }}

hr {{ border-color: var(--border) !important; margin: 1.6rem 0 !important; }}

/* ---- shared card shell ---- */
.panel {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px 22px;
}}

/* ---- status strip ---- */
.status-strip {{
    display: flex; align-items: center; justify-content: space-between;
    flex-wrap: wrap; gap: 10px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 12px 20px;
    margin-bottom: 22px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12.5px;
    color: var(--text-lo);
}}
.status-strip .brand {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 15px; font-weight: 600; color: var(--text-hi);
    letter-spacing: 0.01em;
}}
.status-strip .sep {{ color: var(--border); }}

/* ---- badges / pills ---- */
.badge {{
    display: inline-flex; align-items: center; gap: 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11.5px; font-weight: 600; letter-spacing: 0.03em;
    padding: 3px 10px; border-radius: 20px; text-transform: uppercase;
    white-space: nowrap;
}}
.badge-dot {{ width: 6px; height: 6px; border-radius: 50%; display: inline-block; }}
.badge-positive {{ background: rgba(62,207,142,0.12); color: var(--positive); }}
.badge-caution  {{ background: rgba(240,180,41,0.14); color: var(--caution); }}
.badge-negative {{ background: rgba(240,89,107,0.14); color: var(--negative); }}
.badge-ai       {{ background: rgba(52,208,195,0.14); color: var(--ai); }}
.badge-neutral  {{ background: rgba(137,146,169,0.14); color: var(--text-lo); }}

/* ---- KPI cards ---- */
.kpi-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px 16px 12px 16px;
    height: 100%;
}}
.kpi-label {{
    font-size: 11px; color: var(--text-lo); text-transform: uppercase;
    letter-spacing: 0.06em; font-weight: 600; margin-bottom: 6px;
}}
.kpi-value {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 26px; font-weight: 600; color: var(--text-hi); line-height: 1.1;
}}
.kpi-unit {{ font-size: 13px; color: var(--text-lo); font-weight: 500; margin-left: 3px; }}
.kpi-delta {{ margin-top: 6px; font-size: 12px; font-family: 'IBM Plex Mono', monospace; }}
.delta-up {{ color: var(--positive); }}
.delta-down {{ color: var(--negative); }}
.delta-flat {{ color: var(--text-muted); }}

/* ---- section header ---- */
.section-head {{ display: flex; align-items: baseline; gap: 10px; margin: 6px 0 14px 0; }}
.section-head .title {{
    font-family: 'Space Grotesk', sans-serif; font-size: 18px; font-weight: 600; color: var(--text-hi);
}}
.section-head .subtitle {{ font-size: 12.5px; color: var(--text-lo); }}

/* ---- hero rating ---- */
.hero-wrap {{
    display: flex; align-items: center; gap: 30px; flex-wrap: wrap;
}}
.hero-sub {{ flex: 1; min-width: 220px; }}
.hero-sub-row {{ margin-bottom: 12px; }}
.hero-sub-label {{
    display: flex; justify-content: space-between; font-size: 12.5px;
    color: var(--text-lo); margin-bottom: 4px; font-family: 'IBM Plex Mono', monospace;
}}
.hero-sub-track {{ background: var(--surface-2); border-radius: 6px; height: 6px; overflow: hidden; }}
.hero-sub-fill {{ height: 100%; border-radius: 6px; }}

/* ---- data / insight / action triad ---- */
.triad {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; }}
.triad-card {{
    border-radius: 12px; padding: 14px 16px; border: 1px solid var(--border);
    background: var(--surface-2);
}}
.triad-label {{
    font-family: 'IBM Plex Mono', monospace; font-size: 10.5px; font-weight: 600;
    letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 8px;
    display: flex; align-items: center; gap: 6px;
}}
.triad-text {{ font-size: 13.5px; color: var(--text-hi); line-height: 1.5; }}

/* ---- list cards (strengths / improvements / etc) ---- */
.list-card {{
    border: 1px solid var(--border); border-radius: 12px; padding: 16px 18px;
    background: var(--surface); height: 100%;
}}
.list-card-title {{
    font-family: 'Space Grotesk', sans-serif; font-size: 14.5px; font-weight: 600;
    margin-bottom: 10px; display: flex; align-items: center; gap: 8px;
}}
.list-item {{
    display: flex; gap: 9px; font-size: 13.5px; color: var(--text-hi);
    line-height: 1.5; margin-bottom: 9px;
}}
.list-item:last-child {{ margin-bottom: 0; }}
.list-marker {{ flex-shrink: 0; margin-top: 3px; }}

/* ---- summary callout ---- */
.summary-callout {{
    border-left: 3px solid var(--ai);
    background: linear-gradient(90deg, rgba(52,208,195,0.08), transparent 70%);
    border-radius: 8px;
    padding: 14px 18px;
    font-size: 14.5px; line-height: 1.6; color: var(--text-hi);
}}

/* Streamlit widget restyling */
div[data-testid="stMetric"] {{ display: none; }}
.stButton>button, .stDownloadButton>button {{
    background: var(--surface-2); border: 1px solid var(--border); color: var(--text-hi);
    font-family: 'Inter', sans-serif; font-weight: 500;
}}
.stButton>button[kind="primary"] {{ background: var(--measured); border: none; color: #06121F; font-weight: 600; }}
.stTabs [data-baseweb="tab"] {{ font-family: 'Inter', sans-serif; }}
</style>
"""


def status_strip(
    player_id: str,
    session_number: int,
    timestamp_str: str,
    hardware_mode_label: str,
    hardware_ok: bool,
    llm_label: str,
) -> str:
    hw_class = "badge-caution" if not hardware_ok else "badge-positive"
    return f"""
<div class="status-strip">
    <div style="display:flex;align-items:center;gap:16px;">
        <span class="brand">🏃 SUBWAY FITNESS AI</span>
        <span class="sep">|</span>
        <span>PLAYER <b style="color:var(--text-hi)">{player_id}</b></span>
        <span class="sep">·</span>
        <span>SESSION <b style="color:var(--text-hi)">#{session_number}</b></span>
        <span class="sep">·</span>
        <span>{timestamp_str}</span>
    </div>
    <div style="display:flex;align-items:center;gap:8px;">
        <span class="badge {hw_class}"><span class="badge-dot" style="background:currentColor;"></span>{hardware_mode_label}</span>
        <span class="badge badge-ai">🤖 {llm_label}</span>
    </div>
</div>
"""


def badge(text: str, kind: str = "neutral", icon: str = "") -> str:
    return f'<span class="badge badge-{kind}">{icon} {text}</span>'


def section_head(title: str, subtitle: str = "") -> str:
    sub = f'<span class="subtitle">{subtitle}</span>' if subtitle else ""
    return f'<div class="section-head"><span class="title">{title}</span>{sub}</div>'


def _delta_html(delta_pct: Optional[float], good_when: str = "up") -> str:
    """good_when: 'up' if higher is better (score), 'down' if lower is better (collisions)."""
    if delta_pct is None:
        return '<div class="kpi-delta delta-flat">— no prior session</div>'
    if abs(delta_pct) < 0.5:
        return '<div class="kpi-delta delta-flat">≈ flat vs last session</div>'
    direction = "up" if delta_pct > 0 else "down"
    is_good = (direction == good_when)
    cls = "delta-up" if is_good else "delta-down"
    arrow = "↑" if direction == "up" else "↓"
    return f'<div class="kpi-delta {cls}">{arrow} {abs(delta_pct):.1f}% vs last session</div>'


def kpi_card(label: str, value: str, unit: str = "", delta_pct: Optional[float] = None,
             good_when: str = "up") -> str:
    unit_html = f'<span class="kpi-unit">{unit}</span>' if unit else ""
    return f"""
<div class="kpi-card">
    <div class="kpi-label">{label}</div>
    <div class="kpi-value">{value}{unit_html}</div>
    {_delta_html(delta_pct, good_when)}
</div>
"""


def hero_gauge_svg(value: float, max_value: float = 10, color: str = "#34D0C3", size: int = 148) -> str:
    r = 60
    circumference = 2 * 3.14159265 * r
    frac = max(0.0, min(1.0, value / max_value))
    dash = circumference * frac
    return f"""
<svg width="{size}" height="{size}" viewBox="0 0 140 140">
    <circle cx="70" cy="70" r="{r}" fill="none" stroke="#1A2233" stroke-width="12" />
    <circle cx="70" cy="70" r="{r}" fill="none" stroke="{color}" stroke-width="12"
        stroke-linecap="round" stroke-dasharray="{dash:.1f} {circumference:.1f}"
        transform="rotate(-90 70 70)" />
    <text x="70" y="66" text-anchor="middle" font-family="IBM Plex Mono, monospace"
        font-size="30" font-weight="600" fill="#E8ECF4">{value:g}</text>
    <text x="70" y="86" text-anchor="middle" font-family="Inter, sans-serif"
        font-size="12" fill="#8992A9">out of {max_value:g}</text>
</svg>
"""


def hero_subscore_row(label: str, value: float, max_value: float = 10, color: str = "#5B8DEF") -> str:
    pct = max(0.0, min(1.0, value / max_value)) * 100
    return f"""
<div class="hero-sub-row">
    <div class="hero-sub-label"><span>{label}</span><span style="color:var(--text-hi);font-weight:600;">{value:g}/{max_value:g}</span></div>
    <div class="hero-sub-track"><div class="hero-sub-fill" style="width:{pct:.0f}%;background:{color};"></div></div>
</div>
"""


def triad(data_text: str, insight_text: str, action_text: str) -> str:
    c = COLORS
    return f"""
<div class="triad">
    <div class="triad-card">
        <div class="triad-label" style="color:{c['measured']};">📊 DATA — measured</div>
        <div class="triad-text">{data_text}</div>
    </div>
    <div class="triad-card">
        <div class="triad-label" style="color:{c['ai']};">🧠 INSIGHT — AI interpretation</div>
        <div class="triad-text">{insight_text}</div>
    </div>
    <div class="triad-card">
        <div class="triad-label" style="color:{c['action']};">🎯 ACTION — next step</div>
        <div class="triad-text">{action_text}</div>
    </div>
</div>
"""


def list_card(title: str, items: List[str], icon: str = "•", color: str = "#5B8DEF") -> str:
    rows = "".join(
        f'<div class="list-item"><span class="list-marker" style="color:{color};">{icon}</span><span>{it}</span></div>'
        for it in items
    )
    return f"""
<div class="list-card">
    <div class="list-card-title">{title}</div>
    {rows}
</div>
"""


def summary_callout(text: str) -> str:
    return f'<div class="summary-callout">{text}</div>'


PLOTLY_LAYOUT: Dict = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color=COLORS["text_lo"], size=12),
    margin=dict(l=10, r=10, t=30, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                font=dict(size=11)),
    xaxis=dict(gridcolor="#1E2740", zerolinecolor="#1E2740"),
    yaxis=dict(gridcolor="#1E2740", zerolinecolor="#1E2740"),
)


def style_fig(fig, height: int = 320, **layout_overrides):
    layout = dict(PLOTLY_LAYOUT)
    layout["height"] = height
    layout.update(layout_overrides)
    fig.update_layout(**layout)
    return fig


# ----------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------

st.set_page_config(
    page_title="Subway Surfers Fitness AI",
    page_icon="🏃",
    layout="wide",
)
st.markdown(inject_css(), unsafe_allow_html=True)

if "result" not in st.session_state:
    st.session_state.result = None  # will hold (session, analytics, report, profile, history)


# ----------------------------------------------------------------------
# Sidebar: session controls (unchanged logic, lightly restyled)
# ----------------------------------------------------------------------

st.sidebar.markdown(
    "<div style='font-family:Space Grotesk,sans-serif;font-size:16px;font-weight:600;"
    "color:var(--text-hi);margin-bottom:2px;'>🎮 Session Setup</div>"
    "<div style='font-size:12px;color:var(--text-lo);margin-bottom:14px;'>"
    "Configure and run a simulated session</div>",
    unsafe_allow_html=True,
)

player_id = st.sidebar.text_input("Player ID", value=core.DEFAULT_PLAYER_ID)
duration = st.sidebar.slider("Session duration (seconds)", 30, 600, core.SESSION_DURATION_SECONDS, step=30)
interval = st.sidebar.slider("Sample interval (seconds)", 1, 15, core.SAMPLE_INTERVAL_SECONDS)

st.sidebar.divider()
st.sidebar.subheader("🤖 LLM Settings")

use_real_llm = st.sidebar.toggle("Use real LLM (OpenRouter)", value=not core.MOCK_LLM_MODE)
api_key_input = ""
if use_real_llm:
    api_key_input = st.sidebar.text_input(
        "OpenRouter API key",
        value=core.LLM_API_KEY,
        type="password",
        help="Get one at https://openrouter.ai/keys. Leave blank to use LLM_API_KEY from .env.",
    )
    st.sidebar.caption(f"Model: `{core.LLM_MODEL}`")
else:
    st.sidebar.caption("Using the built-in mock analyzer — no API key needed.")

st.sidebar.divider()
run_clicked = st.sidebar.button("▶️ Run Session", type="primary", width='stretch')

st.sidebar.divider()
if st.sidebar.button("🗑️ Clear this player's history", width='stretch'):
    history_data = {}
    if os.path.exists(core.PLAYER_HISTORY_FILE):
        with open(core.PLAYER_HISTORY_FILE, "r", encoding="utf-8") as f:
            history_data = json.load(f)
    history_data.pop(player_id, None)
    os.makedirs(core.DATA_DIR, exist_ok=True)
    with open(core.PLAYER_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history_data, f, indent=2)
    st.sidebar.success(f"Cleared history for {player_id}")


# ----------------------------------------------------------------------
# Run the pipeline when the button is clicked (unchanged)
# ----------------------------------------------------------------------

if run_clicked:
    with st.spinner("Simulating session and running analysis..."):
        session = core.generate_mock_session(
            duration_seconds=duration, interval_seconds=interval, player_id=player_id
        )
        session_analytics = core.compute_session_analytics(session)

        history = core.get_player_history(player_id)
        profile = core.build_player_profile(player_id, include_current=None)

        try:
            analyzer = core.LLMAnalyzer(
                mock_mode=not use_real_llm,
                api_key=api_key_input or None,
            )
            report = analyzer.analyze_session(session_analytics, history=history, profile=profile)
            error = None
        except Exception as exc:
            report = None
            error = str(exc)

        if report is not None:
            core.save_session_result(player_id, session_analytics)

        st.session_state.result = {
            "session": session,
            "analytics": session_analytics,
            "history": history,  # sessions BEFORE this one (used for deltas)
            "profile": profile,
            "report": report,
            "error": error,
            "session_number": profile["sessions_completed"] + 1,
        }


# ----------------------------------------------------------------------
# Small local helpers (data shaping only — no visual decisions here)
# ----------------------------------------------------------------------

def pct_delta(curr: float, prev: Optional[float]) -> Optional[float]:
    if prev is None or prev == 0:
        return None
    return (curr - prev) / prev * 100


ACTION_COLORS = {
    "JUMP": COLORS["action"],
    "LEFT": COLORS["measured"],
    "RIGHT": COLORS["ai"],
    "CENTER": COLORS["text_lo"],
}


# ----------------------------------------------------------------------
# Main content
# ----------------------------------------------------------------------

result = st.session_state.result

hw_mock = core.MOCK_HARDWARE_MODE
hw_label = "DEMO / SIMULATION MODE" if hw_mock else "LIVE HARDWARE"

if result is None:
    st.markdown(
        status_strip(player_id, "—", datetime.now().strftime("%Y-%m-%d %H:%M"),
                            hw_label, not hw_mock, "Mock Analyzer" if not use_real_llm else "OpenRouter"),
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="panel" style="text-align:center;padding:48px 20px;">'
        '<div style="font-size:32px;margin-bottom:10px;">🏃‍♂️</div>'
        '<div style="font-family:Space Grotesk,sans-serif;font-size:18px;font-weight:600;'
        'color:var(--text-hi);margin-bottom:6px;">No session yet</div>'
        '<div style="color:var(--text-lo);font-size:13.5px;">Set your session options in the sidebar '
        'and click <b>Run Session</b> to generate a personalized fitness report.</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.stop()

if result["error"]:
    st.markdown(
        status_strip(player_id, result["session_number"], datetime.now().strftime("%Y-%m-%d %H:%M"),
                            hw_label, not hw_mock, "OpenRouter (failed)"),
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="panel" style="border-color:var(--negative);">'
        f'<div style="color:var(--negative);font-weight:600;margin-bottom:6px;">⚠️ LLM analysis failed</div>'
        f'<div style="color:var(--text-lo);font-size:13px;">{result["error"]}</div>'
        f'<div style="color:var(--text-lo);font-size:13px;margin-top:8px;">Tip: toggle off '
        f'"Use real LLM" in the sidebar to fall back to the mock analyzer.</div></div>',
        unsafe_allow_html=True,
    )
    st.stop()

a = result["analytics"]
report = result["report"]
session = result["session"]
history = result["history"]  # sessions BEFORE this one, oldest first
prev = history[-1] if history else None

# --- Status strip ---
session_dt = datetime.fromisoformat(session.start_time) if isinstance(session.start_time, str) else datetime.now()
st.markdown(
    status_strip(
        player_id, result["session_number"], session_dt.strftime("%Y-%m-%d %H:%M"),
        hw_label, not hw_mock, "OpenRouter (live)" if use_real_llm else "Mock Analyzer",
    ),
    unsafe_allow_html=True,
)

# --- HERO: overall session rating + subscores + AI summary ---
hero_l, hero_r = st.columns([1, 2], gap="large")
with hero_l:
    st.markdown(
        f'<div class="panel" style="text-align:center;">'
        f'<div class="kpi-label" style="margin-bottom:10px;">SESSION RATING</div>'
        f'{hero_gauge_svg(report.overall_rating, 10, color=COLORS["ai"])}'
        f'</div>',
        unsafe_allow_html=True,
    )
with hero_r:
    subrows = (
        hero_subscore_row("Gameplay", report.gameplay_rating, 10, COLORS["measured"])
        + hero_subscore_row("Movement", report.movement_rating, 10, COLORS["positive"])
        + hero_subscore_row("Engagement", report.engagement_rating, 10, COLORS["action"])
    )
    st.markdown(
        f'<div class="panel">'
        f'<div class="kpi-label" style="margin-bottom:10px;">BREAKDOWN</div>'
        f'{subrows}'
        f'<div style="margin-top:4px;">{summary_callout(report.session_summary)}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

st.markdown("<div style='height:22px;'></div>", unsafe_allow_html=True)

# --- KPI grid ---
st.markdown(section_head("Key Metrics", "This session vs. your last one"), unsafe_allow_html=True)
k1, k2, k3, k4, k5, k6 = st.columns(6)
mm, ss = divmod(a["duration_seconds"], 60)
with k1:
    st.markdown(kpi_card("Score", f"{a['score']:,}", delta_pct=pct_delta(a["score"], prev.get("score") if prev else None), good_when="up"), unsafe_allow_html=True)
with k2:
    st.markdown(kpi_card("Coins", f"{a['coins']:,}", delta_pct=pct_delta(a["coins"], prev.get("coins") if prev else None), good_when="up"), unsafe_allow_html=True)
with k3:
    st.markdown(kpi_card("Collisions", f"{a['collisions']}", delta_pct=pct_delta(a["collisions"], prev.get("collisions") if prev else None), good_when="down"), unsafe_allow_html=True)
with k4:
    st.markdown(kpi_card("Avg HR", f"{a['average_heart_rate']:g}", unit="bpm", delta_pct=pct_delta(a["average_heart_rate"], prev.get("average_heart_rate") if prev else None), good_when="neutral"), unsafe_allow_html=True)
with k5:
    st.markdown(kpi_card("Peak HR", f"{a['max_heart_rate']}", unit="bpm", delta_pct=pct_delta(a["max_heart_rate"], prev.get("max_heart_rate") if prev else None), good_when="neutral"), unsafe_allow_html=True)
with k6:
    st.markdown(kpi_card("Duration", f"{mm:02d}:{ss:02d}", delta_pct=None), unsafe_allow_html=True)

if a.get("unusually_high_heart_rate"):
    st.markdown(
        f'<div style="margin-top:10px;">{badge("Peak heart rate was unusually high for this session — not a medical diagnosis", "caution", "⚠️")}</div>',
        unsafe_allow_html=True,
    )

st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
st.divider()

# --- Heart Rate + Movement + Game Events (signature combined chart) ---
st.markdown(
    section_head("Heart Rate ↔ Movement ↔ Game Events",
                        "What was happening in the game when heart rate changed"),
    unsafe_allow_html=True,
)

df = pd.DataFrame([{
    "seconds": s.seconds,
    "timestamp": s.timestamp,
    "heart_rate": s.heart_rate,
    "movement_intensity": s.movement_intensity,
    "action": s.action,
} for s in session.samples])
ts_to_sec = dict(zip(df["timestamp"], df["seconds"]))

fig = go.Figure()

# Shaded high-intensity windows (measured, not an AI claim)
for i, period in enumerate(a.get("high_intensity_periods", [])):
    x0 = ts_to_sec.get(period["start"], None)
    x1 = ts_to_sec.get(period["end"], None)
    if x0 is None or x1 is None:
        continue
    fig.add_vrect(x0=x0, x1=x1, fillcolor=COLORS["caution"], opacity=0.08, line_width=0)
    if i == 0:
        fig.add_annotation(x=(x0 + x1) / 2, y=1.06, yref="paper", showarrow=False,
                            text="High-intensity window", font=dict(size=10, color=COLORS["caution"]))

fig.add_trace(go.Scatter(
    x=df["seconds"], y=df["heart_rate"], mode="lines", name="Heart rate (BPM)",
    line=dict(color=COLORS["negative"], width=2.2), yaxis="y",
))
fig.add_trace(go.Scatter(
    x=df["seconds"], y=df["movement_intensity"], mode="lines", name="Movement intensity",
    line=dict(color=COLORS["measured"], width=1.6), fill="tozeroy",
    fillcolor="rgba(91,141,239,0.10)", yaxis="y2",
))
jumps = df[df["action"] == "JUMP"]
if not jumps.empty:
    fig.add_trace(go.Scatter(
        x=jumps["seconds"], y=jumps["heart_rate"], mode="markers", name="Jump",
        marker=dict(symbol="triangle-up", size=9, color=COLORS["action"]), yaxis="y",
    ))

fig.update_layout(
    yaxis=dict(title="Heart rate (BPM)", gridcolor="#1E2740"),
    yaxis2=dict(title="Movement intensity", overlaying="y", side="right", showgrid=False, range=[0, 1]),
    xaxis=dict(title="Session time (seconds)", gridcolor="#1E2740"),
)
style_fig(fig, height=380)
st.plotly_chart(fig, width='stretch')
st.caption(
    "Measured sensor data only — shaded bands mark periods of high movement intensity. "
    "AI interpretation of this relationship is in the AI Coach section below."
)

st.divider()

# --- Movement analysis ---
st.markdown(section_head("Player Movement", "Distribution, consistency, and correlation with heart rate"), unsafe_allow_html=True)
mv1, mv2 = st.columns([1, 1], gap="large")

with mv1:
    counts = {
        "JUMP": a["jump_count"], "LEFT": a["left_count"],
        "RIGHT": a["right_count"], "CENTER": a["center_count"],
    }
    order = sorted(counts, key=counts.get)
    fig_actions = go.Figure(data=[go.Bar(
        x=[counts[k] for k in order], y=order, orientation="h",
        marker_color=[ACTION_COLORS[k] for k in order],
        text=[counts[k] for k in order], textposition="outside",
    )])
    fig_actions.update_layout(xaxis=dict(title="Count"), showlegend=False)
    style_fig(fig_actions, height=260)
    st.plotly_chart(fig_actions, width='stretch')
    chips = "".join([
        badge(f"Consistency {a['movement_consistency']*100:.0f}%", "neutral"),
        " ",
        badge(f"{a['jump_frequency_per_minute']}/min jumps", "neutral"),
        " ",
        badge(f"{a['lane_change_frequency_per_minute']}/min lane changes", "neutral"),
        " ",
        badge(f"Intensity: {a['movement_intensity_label']}", "neutral"),
    ])
    st.markdown(f"<div>{chips}</div>", unsafe_allow_html=True)

with mv2:
    corr = a["heart_rate_movement_correlation"]
    corr_label = a["heart_rate_movement_correlation_label"]
    st.markdown(
        badge(f"r = {corr}  ·  {corr_label} relationship", "ai", "🔗"),
        unsafe_allow_html=True,
    )
    fig_scatter = go.Figure(data=[go.Scatter(
        x=df["movement_intensity"], y=df["heart_rate"], mode="markers",
        marker=dict(color=COLORS["ai"], size=7, opacity=0.75),
    )])
    fig_scatter.update_layout(
        xaxis=dict(title="Movement intensity"), yaxis=dict(title="Heart rate (BPM)"),
    )
    style_fig(fig_scatter, height=260)
    st.plotly_chart(fig_scatter, width='stretch')

st.divider()

# --- Game performance rates ---
st.markdown(section_head("Game Performance Rates", "Per-minute pace, compared with your last session"), unsafe_allow_html=True)
g1, g2, g3, g4, g5 = st.columns(5)
with g1:
    st.markdown(kpi_card("Score / min", f"{a['score_per_minute']:g}", delta_pct=pct_delta(a["score_per_minute"], prev.get("score_per_minute") if prev else None), good_when="up"), unsafe_allow_html=True)
with g2:
    st.markdown(kpi_card("Coins / min", f"{a['coins_per_minute']:g}", delta_pct=pct_delta(a["coins_per_minute"], prev.get("coins_per_minute") if prev else None), good_when="up"), unsafe_allow_html=True)
with g3:
    st.markdown(kpi_card("Collisions / min", f"{a['collision_rate_per_minute']:g}", delta_pct=pct_delta(a["collision_rate_per_minute"], prev.get("collision_rate_per_minute") if prev else None), good_when="down"), unsafe_allow_html=True)
with g4:
    st.markdown(kpi_card("Jumps / min", f"{a['jump_frequency_per_minute']:g}", delta_pct=pct_delta(a["jump_frequency_per_minute"], prev.get("jump_frequency_per_minute") if prev else None), good_when="up"), unsafe_allow_html=True)
with g5:
    st.markdown(kpi_card("Lane changes / min", f"{a['lane_change_frequency_per_minute']:g}", delta_pct=pct_delta(a["lane_change_frequency_per_minute"], prev.get("lane_change_frequency_per_minute") if prev else None), good_when="neutral"), unsafe_allow_html=True)

st.divider()

# --- AI Coach Analysis ---
st.markdown(section_head("🧠 AI Coach Analysis", "AI-generated interpretation of the data above — not medical advice"), unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Physical Activity", "Gameplay", "Heart Rate + Movement"])
with tab1:
    st.markdown(f'<div class="panel">{report.physical_activity_analysis}</div>', unsafe_allow_html=True)
with tab2:
    st.markdown(f'<div class="panel">{report.gameplay_analysis}</div>', unsafe_allow_html=True)
with tab3:
    st.markdown(f'<div class="panel">{report.heart_rate_movement_analysis}</div>', unsafe_allow_html=True)

st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

action_text = report.next_session_goals[0] if report.next_session_goals else (
    report.recommendations[0] if report.recommendations else "Keep monitoring this pattern next session."
)
st.markdown(
    triad(
        data_text=(
            f"Heart rate moved from {a['start_heart_rate']} to {a['max_heart_rate']} BPM "
            f"(peak +{a['heart_rate_increase']} BPM) alongside a {a['heart_rate_movement_correlation_label']} "
            f"correlation (r = {a['heart_rate_movement_correlation']}) with movement intensity."
        ),
        insight_text=report.heart_rate_movement_analysis,
        action_text=action_text,
    ),
    unsafe_allow_html=True,
)

st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")
with col1:
    st.markdown(list_card("✅ Strengths", report.strengths, "✓", COLORS["positive"]), unsafe_allow_html=True)
with col2:
    st.markdown(list_card("⚠️ Areas to Improve", report.areas_for_improvement, "!", COLORS["caution"]), unsafe_allow_html=True)

st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

col3, col4 = st.columns(2, gap="large")
with col3:
    st.markdown(list_card("💡 Recommendations", report.recommendations, "→", COLORS["ai"]), unsafe_allow_html=True)
with col4:
    st.markdown(list_card("🎯 Next Session Goals", report.next_session_goals, "◎", COLORS["action"]), unsafe_allow_html=True)

st.markdown(
    f'<div style="margin-top:16px;color:var(--text-lo);font-size:12px;font-style:italic;">'
    f'⚕️ {report.safety_note}</div>',
    unsafe_allow_html=True,
)

st.divider()

# --- Progress across sessions ---
history_full = core.get_player_history(player_id)  # includes this session now
if len(history_full) > 1:
    st.markdown(section_head("Progress Across Sessions", f"{len(history_full)} sessions recorded for {player_id}"), unsafe_allow_html=True)
    hist_df = pd.DataFrame(history_full)
    hist_df["session_number"] = range(1, len(hist_df) + 1)

    trend_specs = [
        ("score", "Score Trend", COLORS["positive"]),
        ("collisions", "Collision Trend", COLORS["negative"]),
        ("average_heart_rate", "Average HR Trend", COLORS["measured"]),
        ("average_movement_intensity", "Movement Intensity Trend", COLORS["ai"]),
    ]
    ht1, ht2 = st.columns(2)
    ht3, ht4 = st.columns(2)
    slots = [ht1, ht2, ht3, ht4]
    for (col_key, title, color), slot in zip(trend_specs, slots):
        with slot:
            fig_t = go.Figure(data=[go.Scatter(
                x=hist_df["session_number"], y=hist_df[col_key], mode="lines+markers",
                line=dict(color=color, width=2.5), marker=dict(size=6),
            )])
            fig_t.update_layout(title=dict(text=title, font=dict(size=13)),
                                 xaxis=dict(title="Session #"), yaxis=dict(title=""))
            style_fig(fig_t, height=250)
            st.plotly_chart(fig_t, width='stretch')

    st.divider()

# --- Export ---
st.markdown(section_head("Export"), unsafe_allow_html=True)
col_d1, col_d2 = st.columns(2)

text_report = core.generate_text_report(a, report, session_number=result["session_number"])
json_report = core.generate_json_report(a, report)

with col_d1:
    st.download_button(
        "📄 Download text report",
        data=text_report,
        file_name=f"{player_id}_session_{result['session_number']}.txt",
        mime="text/plain",
        width='stretch',
    )
with col_d2:
    st.download_button(
        "🧾 Download JSON report",
        data=json.dumps(json_report, indent=2),
        file_name=f"{player_id}_session_{result['session_number']}.json",
        mime="application/json",
        width='stretch',
    )

with st.expander("🔍 Raw analytics (what Python calculated for the LLM)"):
    st.json(a)
