import streamlit.components.v1 as components
import plotly.graph_objects as go
import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pandas as pd
from numba import njit
from pathlib import Path
from urllib.parse import quote

# ── 페이지 설정 ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="기후 모델링 연구 대시보드",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── 전역 CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* ── Google Font: DM Sans (웹안전 fallback 포함) ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700;800&display=swap');

/* ── Reset & Base ─────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    -webkit-font-smoothing: antialiased;
}
[data-testid="stAppViewContainer"] { background: #f0f4f8; }
[data-testid="stHeader"] { background: transparent; }
section[data-testid="stSidebar"] { display: none; }
.block-container {
    padding: 1.6rem !important;
    max-width: 1400px;
}

/* ── Hero Card ────────────────────────────────────────────── */
.hero {
    background: linear-gradient(135deg, #1a56a0 0%, #1e6fb8 55%, #2185d0 100%);
    border-radius: 20px;
    padding: 1.6rem;
    color: #fff;
    margin-bottom: 1.6rem;
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: '';
    position: absolute;
    right: -60px; top: -60px;
    width: 320px; height: 320px;
    border-radius: 50%;
    background: rgba(255,255,255,0.05);
    pointer-events: none;
}
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.18);
    border: 1px solid rgba(255,255,255,0.28);
    color: #e0f0ff;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    padding: 0.3rem 0.75rem;
    border-radius: 999px;
    margin-bottom: 0.9rem;
}
.hero-title {
    font-size: 2.3rem;
    font-weight: 800;
    line-height: 1.2;
    margin-bottom: 0.8rem;
    letter-spacing: -0.03em;
}
.hero-desc {
    font-size: 0.97rem;
    line-height: 1.75;
    color: #d5e5f7;
    max-width: 980px;
}
.hero-chips {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin-top: 1rem;
}
.hero-chip {
    background: rgba(255,255,255,0.13);
    border: 1px solid rgba(255,255,255,0.22);
    color: #e8f3ff;
    font-size: 0.8rem;
    font-weight: 600;
    padding: 0.35rem 0.75rem;
    border-radius: 999px;
}

/* ── Section Title ────────────────────────────────────────── */
.sec-title {
    font-size: 1.05rem;
    font-weight: 800;
    color: #0f2744;
    margin: 1.6rem 0 0.9rem 0;
    letter-spacing: -0.02em;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.sec-title::before {
    content: '';
    display: inline-block;
    width: 4px;
    height: 1.1em;
    background: #1a56a0;
    border-radius: 2px;
}

/* ── Metric Cards ─────────────────────────────────────────── */
.mcard {
    background: #ffffff;
    border: 1px solid #d6e2f0;
    border-radius: 14px;
    padding: 1.2rem;
    flex: 1;
    min-width: 140px;
    box-shadow: 0 2px 8px rgba(26,86,160,0.06);
    position: relative;
    overflow: hidden;
}
.mcard::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #1a56a0, #2185d0);
    border-radius: 14px 14px 0 0;
}
.mcard-label {
    font-size: 0.78rem;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.55rem;
}
.mcard-val {
    font-size: 1.75rem;
    font-weight: 800;
    color: #0f2744;
    line-height: 1.1;
    letter-spacing: -0.02em;
}
.mcard-unit {
    font-size: 0.88rem;
    font-weight: 500;
    color: #64748b;
    margin-left: 2px;
}
.mcard-note {
    font-size: 0.77rem;
    color: #94a3b8;
    margin-top: 0.35rem;
    line-height: 1.5;
}

/* ── Module Cards (Home) ──────────────────────────────────── */
.modcard {
    background: #ffffff;
    border: 1px solid #d6e2f0;
    border-radius: 16px;
    padding: 1.5rem;
    min-height: 170px;
    height: 100%;
    box-shadow: 0 2px 10px rgba(26,86,160,0.06);
    transition: transform 0.18s, box-shadow 0.18s, border-color 0.18s;
    cursor: pointer;
    position: relative;
    overflow: hidden;
    text-decoration: none !important;
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
    color: inherit !important;
    box-sizing: border-box;
    margin: 0 !important;
}
.modcard:hover {
    transform: translateY(-3px);
    border-color: #1a56a0;
    box-shadow: 0 8px 24px rgba(26,86,160,0.13);
}
.modcard-num {
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    color: #1a56a0;
    margin: 0 0 1rem 0;
    display: flex;
    align-items: center;
    gap: 0.4rem;
    line-height: 1;
}
.modcard-num::before {
    content: '';
    display: inline-block;
    width: 18px;
    height: 2px;
    background: #1a56a0;
    border-radius: 1px;
}
.modcard-title {
    font-size: 1rem;
    font-weight: 800;
    color: #0f2744;
    margin: 0 0 1rem 0;
    line-height: 1.35;
    letter-spacing: -0.01em;
}
.modcard-desc {
    font-size: 0.88rem;
    line-height: 1.75;
    color: #475569;
    margin: 0;
}

/* ── Info / Note Box ──────────────────────────────────────── */
.infobox {
    background: #f0f6ff;
    border: 1px solid #bdd6f5;
    border-left: 4px solid #1a56a0;
    border-radius: 10px;
    padding: 1.1rem 1.2rem;
    margin-bottom: 1.2rem;
}
.infobox-title {
    font-size: 0.82rem;
    font-weight: 800;
    color: #1a56a0;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.35rem;
}
.infobox-body {
    font-size: 0.93rem;
    line-height: 1.8;
    color: #334155;
}

/* ── Left Nav Panel ───────────────────────────────────────── */
.nav-panel {
    background: #f7f7f8;
    border: 2px solid #c9d8ea;
    border-radius: 26px;
    padding: 1.2rem;
    box-shadow: none;
    margin-bottom: 1rem;
    position: sticky;
    top: 1rem;
}
.nav-panel-title {
    font-size: 1.25rem;
    font-weight: 800;
    color: #2b5ea7;
    letter-spacing: -0.03em;
    line-height: 1.15;
    margin-bottom: 1.2rem;
    padding: 0 0.25rem 0.85rem 0.25rem;
    border-bottom: 3px solid #dde7f2;
}

/* ── nav-link 전용 스타일 ── */
.nav-links {
    display: flex;
    flex-direction: column;
    gap: 0.85rem;
}

.nav-link {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 100%;
    min-height: 4.2rem;
    border-radius: 20px;
    border: 2px solid #b9cfea;
    background: #ffffff;
    color: #2b5ea7 !important;
    font-weight: 700;
    font-size: 0.95rem;
    text-decoration: none !important;
    transition: transform 0.18s, box-shadow 0.18s, border-color 0.18s;
}

.nav-link:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(26,86,160,0.13);
}

.nav-link.active {
    border-color: #1a56a0;
    background: #eef5ff;
}
.nav-hint { display: none; }

/* ── Settings Shell ───────────────────────────────────────── */
.settings-shell {
    background: #f7f7f8;
    border: 2px solid #c9d8ea;
    border-radius: 26px;
    padding: 1.2rem;
    box-shadow: none;
    margin-top: 0.2rem;
    margin-bottom: 0.9rem;
}
.settings-title {
    font-size: 1.25rem;
    font-weight: 800;
    color: #2b5ea7;
    letter-spacing: -0.03em;
    line-height: 1.15;
    margin-bottom: 1.15rem;
    padding-bottom: 0.9rem;
    border-bottom: 3px solid #dde7f2;
}

/* ── Source Expander ──────────────────────────────────────── */
.src-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 0.7rem 0.8rem;
    margin-top: 0.6rem;
}

/* ── Paper Card ───────────────────────────────────────────── */
.pcard {
    background: #ffffff;
    border: 1px solid #d6e2f0;
    border-radius: 16px;
    padding: 1.2rem;
    box-shadow: 0 2px 10px rgba(26,86,160,0.06);
    min-height: 185px;
    height: 100%;
}
.pcard-tag {
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #1a56a0;
    margin-bottom: 0.8rem;
}
.pcard-title {
    font-size: 1rem;
    font-weight: 800;
    color: #0f2744;
    margin-bottom: 0.8rem;
    line-height: 1.4;
}
.pcard-body {
    font-size: 0.91rem;
    line-height: 1.75;
    color: #475569;
}

/* ── Abstract Box ─────────────────────────────────────────── */
.abstract-box {
    background: linear-gradient(180deg, #f0f6ff 0%, #e8f2ff 100%);
    border: 1px solid #bdd6f5;
    border-radius: 12px;
    padding: 1.2rem;
    margin: 0.4rem 0 1.2rem 0;
}
.abstract-label {
    font-size: 0.78rem;
    font-weight: 800;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #1a56a0;
    margin-bottom: 0.5rem;
}
.abstract-text {
    font-size: 0.95rem;
    line-height: 1.85;
    color: #1e3a5f;
}

/* ── Buttons ──────────────────────────────────────────────── */
div[data-testid="stButton"] > button {
    width: 100%;
    border-radius: 20px;
    border: 2px solid #b9cfea;
    background: #ffffff;
    color: #2b5ea7;
    font-weight: 700;
    font-size: 0.95rem;
    min-height: 4.2rem;
    box-shadow: none;
    transition: all 0.15s ease;
    letter-spacing: -0.02em;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}
div[data-testid="stButton"] > button:hover {
    border-color: #8fb3df;
    background: #f7fbff;
    color: #2b5ea7;
}
div[data-testid="stButton"] > button:focus {
    box-shadow: none;
    border-color: #8fb3df;
    outline: none;
}
div[data-testid="stButton"][key="main_reset_experiment"] > button,
div[data-testid="stButton"] > button[kind="secondary"] {
    background: #ffffff !important;
    color: #2b5ea7 !important;
    border: 2px solid #b9cfea !important;
    font-size: 1.05rem !important;
    font-weight: 800 !important;
    border-radius: 18px !important;
    min-height: 3.4rem !important;
}

/* ── Download button ──────────────────────────────────────── */
div[data-testid="stDownloadButton"] > button {
    border-radius: 12px;
    border: none;
    background: linear-gradient(135deg, #1560bd 0%, #2196f3 100%);
    color: #ffffff !important;
    font-weight: 800;
    font-size: 0.9rem;
    min-height: 3rem;
    box-shadow: 0 3px 12px rgba(21,96,189,0.35);
    letter-spacing: -0.01em;
    transition: background 0.15s, box-shadow 0.15s;
}
div[data-testid="stDownloadButton"] > button:hover {
    background: linear-gradient(135deg, #1050a8 0%, #1a80e0 100%);
    box-shadow: 0 5px 18px rgba(21,96,189,0.45);
}

/* ══ Slider — 안정적인 깔끔 버전 ════════════════════════ */
div[data-testid="stSlider"] {
    padding: 0.15rem 0 0.9rem 0;
    margin-bottom: 0.25rem;
}
div[data-testid="stSlider"] label {
    display: block !important;
    margin-bottom: 0.5rem !important;
}
div[data-testid="stSlider"] label p {
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    color: #18365c !important;
    margin: 0 !important;
    line-height: 1.3 !important;
}
div[data-testid="stSlider"] [data-baseweb="slider"] {
    padding-top: 0.2rem !important;
    padding-bottom: 0.2rem !important;
}
div[data-testid="stSlider"] [data-baseweb="slider"] > div:first-child {
    height: 8px !important;
    border-radius: 999px !important;
    background: #d9e6f5 !important;
    box-shadow: none !important;
    overflow: visible !important;
}
div[data-testid="stSlider"] [data-baseweb="slider"] > div:first-child > div {
    height: 8px !important;
    border-radius: 999px !important;
}
div[data-testid="stSlider"] [data-baseweb="slider"] > div:first-child > div:first-child {
    background: #2b6cb0 !important;
}
div[data-testid="stSlider"] [data-baseweb="slider"] > div:first-child > div:last-child {
    background: #d9e6f5 !important;
}
div[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    width: 18px !important;
    height: 18px !important;
    border-radius: 50% !important;
    background: #ffffff !important;
    border: 2px solid #2b6cb0 !important;
    box-shadow: 0 2px 6px rgba(43, 108, 176, 0.18) !important;
    cursor: grab !important;
}
div[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"]:hover {
    box-shadow: 0 3px 8px rgba(43, 108, 176, 0.24) !important;
}
div[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"]:active {
    cursor: grabbing !important;
}
div[data-testid="stSlider"] [data-baseweb="slider"] [data-testid="stSliderThumbValue"] {
    font-size: 0.82rem !important;
    font-weight: 700 !important;
    color: #2b6cb0 !important;
    background: transparent !important;
    top: -24px !important;
    white-space: nowrap !important;
}
div[data-testid="stSlider"] [data-testid="stTickBarMin"],
div[data-testid="stSlider"] [data-testid="stTickBarMax"] {
    display: none !important;
}

/* ── Select & Input ───────────────────────────────────────── */
div[data-baseweb="select"] > div {
    min-height: 64px !important;
    border: 2px solid #c9d8ea !important;
    border-radius: 18px !important;
    background: #ffffff !important;
    box-shadow: none !important;
    color: #18365c !important;
    font-weight: 700 !important;
}
div[data-baseweb="select"] > div:hover {
    border-color: #9dbce2 !important;
}

/* ── Expander ─────────────────────────────────────────────── */
details[data-testid="stExpander"] {
    border: 1px solid #d6e2f0;
    border-radius: 10px;
    background: #ffffff;
}
details[data-testid="stExpander"] summary {
    font-weight: 700;
    color: #0f2744;
    font-size: 0.93rem;
    padding: 0.75rem 1rem;
}
details[data-testid="stExpander"] summary:hover { background: #f0f6ff; }

/* ── Misc ─────────────────────────────────────────────────── */
div[data-testid="stSpinner"] { color: #1a56a0 !important; }
div[data-testid="stInfo"] {
    background: #f0f6ff;
    border: 1px solid #bdd6f5;
    border-radius: 10px;
    color: #1a56a0;
}
div[data-testid="stCaptionContainer"] { color: #94a3b8; font-size: 0.8rem; }
div[data-testid="stImage"] img,
div[data-testid="stPyplotRootElement"] { border-radius: 12px; }

/* ── Page Title ───────────────────────────────────────────── */
.page-title {
    font-size: 1.7rem;
    font-weight: 800;
    color: #0f2744;
    letter-spacing: -0.03em;
    line-height: 1.2;
    margin-bottom: 0.25rem;
}
.page-subtitle {
    font-size: 0.9rem;
    color: #64748b;
    font-weight: 400;
    margin-bottom: 1.1rem;
}

/* ── Condition Bar ────────────────────────────────────────── */
.cond-bar {
    background: #ffffff;
    border: 1px solid #d6e2f0;
    border-radius: 14px;
    padding: 1rem 1.2rem;
    display: flex;
    gap: 2rem;
    flex-wrap: wrap;
    align-items: center;
    margin-bottom: 1.4rem;
    box-shadow: 0 2px 8px rgba(26,86,160,0.05);
}
.cond-item { display: flex; flex-direction: column; gap: 0.1rem; }
.cond-label {
    font-size: 0.72rem;
    font-weight: 700;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}
.cond-val {
    font-size: 1.05rem;
    font-weight: 800;
    color: #0f2744;
    letter-spacing: -0.01em;
}
.cond-base { font-size: 0.75rem; color: #94a3b8; }

/* ── Responsive tweak ─────────────────────────────────────── */
@media (max-width: 900px) {
    .hero-title { font-size: 1.6rem; }
    .block-container { padding: 1rem !important; }
}

/* ── Left Nav Links ───────────────────────────────────────── */
.nav-links {
    display: flex;
    flex-direction: column;
    gap: 0.85rem;
}

.nav-link {
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 4.2rem;
    border-radius: 20px;
    border: 2px solid #b9cfea;
    background: #ffffff;
    color: #2b5ea7 !important;
    font-weight: 700;
    font-size: 0.95rem;
    letter-spacing: -0.02em;
    text-decoration: none !important;
    box-shadow: none;
    transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease, background 0.18s ease;
}

.nav-link:hover {
    transform: translateY(-3px);
    border-color: #1a56a0;
    background: #f7fbff;
    box-shadow: 0 8px 24px rgba(26,86,160,0.13);
}

.nav-link:active {
    transform: translateY(-1px);
    box-shadow: 0 3px 10px rgba(26,86,160,0.10);
}

.nav-link.active {
    border-color: #1a56a0;
    background: #eef5ff;
    color: #1a56a0 !important;
    box-shadow: 0 4px 14px rgba(26,86,160,0.12);
}

.scenario-scale-title {
    font-size: 0.92rem;
    font-weight: 800;
    color: #18365c;
    margin: 0.85rem 0 0.7rem 0;
    letter-spacing: -0.02em;
}

.scenario-scale {
    position: relative;
    margin: 0.35rem 0 0.8rem 0;
    padding: 0.2rem 0 1.7rem 0;
}

.scenario-scale-line {
    position: absolute;
    top: 8px;
    left: 12px;
    right: 12px;
    height: 6px;
    border-radius: 999px;
    background: linear-gradient(90deg, #55c58f 0%, #2b6cb0 50%, #ef4444 100%);
    opacity: 0.22;
}

.scenario-scale-fill {
    position: absolute;
    top: 8px;
    left: 12px;
    height: 6px;
    border-radius: 999px;
    background: linear-gradient(90deg, #55c58f 0%, #2b6cb0 50%, #ef4444 100%);
}

.scenario-scale-steps {
    position: relative;
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
}

.scenario-step {
    width: 18%;
    text-align: center;
    position: relative;
    text-decoration: none !important;
}

.scenario-dot {
    width: 20px;
    height: 20px;
    border-radius: 50%;
    background: #ffffff;
    border: 3px solid #bfd0e5;
    margin: 0 auto 0.3rem auto;
    transition: all 0.2s ease;
}

/* hover */
.scenario-step:hover .scenario-dot {
    transform: scale(1.15);
    border-color: #5b8fd6;
}

/* active */
.scenario-step.active .scenario-dot {
    border: 7px solid #2b5ea7;
    transform: scale(1.2);
}

.scenario-step-label {
    font-size: 0.7rem;
    font-weight: 800;
    color: #334155;
    line-height: 1.2;
    letter-spacing: -0.02em;
    white-space: nowrap;

    opacity: 0;                 
    transform: translateY(6px); 
    transition: all 0.2s ease;
}

.scenario-step:hover .scenario-step-label {
    opacity: 1;
    transform: translateY(0);
    color: #2b5ea7;
}

.scenario-step.active .scenario-step-label {
    opacity: 1;
    transform: translateY(0);
    font-size: 0.8rem;
    color: #1a56a0;
}
.scenario-scale-foot {
    display: flex;
    justify-content: space-between;
    margin-top: 0.55rem;
    font-size: 0.76rem;
    font-weight: 700;
}

.scenario-scale-foot .low {
    color: #34a853;
}

.scenario-scale-foot .mid {
    color: #64748b;
}

.scenario-scale-foot .high {
    color: #ef4444;
}

.scenario-current {
    margin-top: 0.9rem;
    font-size: 0.93rem;
    color: #64748b;
    font-weight: 600;
    text-align: center;
}

.scenario-current strong {
    color: #1a56a0;
    font-weight: 900;
}
.scenario-hero {
    background: linear-gradient(135deg, #1f5fbf 0%, #256fcb 55%, #3a86e8 100%);
    border-radius: 20px;
    padding: 1.3rem;
    color: #ffffff;
    margin-bottom: 1rem;
    box-shadow: 0 10px 28px rgba(31, 95, 191, 0.2);
}

.scenario-hero-top {
    font-size: 0.75rem;
    font-weight: 700;
    opacity: 0.9;
    margin-bottom: 0.4rem;
}

.scenario-hero-name {
    font-size: 1.8rem;
    font-weight: 900;
    margin-bottom: 0.4rem;
}

.scenario-hero-desc {
    font-size: 0.9rem;
    opacity: 0.95;
    line-height: 1.6;
}
/* 카드 */
.source-card {
    border-radius: 16px;
    padding: 0.9rem 1.1rem;
    background: #ffffff;
    box-shadow: 0 6px 18px rgba(30, 64, 175, 0.08);
    border: 1px solid #e2e8f0;
    cursor: pointer;
    transition: all 0.2s ease;
}

/* hover */
.source-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 24px rgba(30, 64, 175, 0.15);
}

/* 헤더 */
.source-header {
    display: flex;
    align-items: center;
    gap: 10px;
}

/* 화살표 */
.source-arrow {
    font-size: 0.9rem;
    transition: transform 0.2s ease;
    color: #1e40af;
}

/* 제목 */
.source-title {
    font-size: 0.95rem;
    font-weight: 800;
    color: #1e293b;
}

.source-content {
    overflow: hidden;
    background: #f8fafc;
    border-radius: 12px;
    margin-top: 0.5rem;
    padding: 0.9rem 1rem;
}

/* 링크 카드 스타일 */
.source-item {
    display: block;
    padding: 0.75rem 0.8rem;
    margin-bottom: 0.55rem;
    border-radius: 13px;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    text-decoration: none !important;
    transition: all 0.16s ease;
}

.source-item:hover {
    background: #eef5ff;
    border-color: #b9cfea;
    transform: translateX(3px);
}

/* 링크 제목 */
.source-name {
    display: block;
    font-size: 0.83rem;
    font-weight: 900;
    color: #1a56a0;
    margin-bottom: 0.25rem;
}

/* 설명 */
.source-desc {
    display: block;
    font-size: 0.73rem;
    font-weight: 600;
    color: #64748b;
}

.source-details[open] .source-content {
    max-height: 800px;
    opacity: 1;
    transform: translateY(0);
    padding: 0.9rem 1rem;
}

.source-details[open] .source-arrow {
    transform: rotate(90deg);
}

.source-content {
    max-height: 0;
    opacity: 0;
    transform: translateY(-8px);
    overflow: hidden;
    background: #f8fafc;
    border-radius: 12px;
    margin-top: 0.5rem;
    padding: 0 1rem;
    transition: max-height 0.35s ease, opacity 0.25s ease, transform 0.25s ease, padding 0.25s ease;
}

.source-details[open] .source-content {
    max-height: 900px;
    opacity: 1;
    transform: translateY(0);
    padding: 0.9rem 1rem;
}

.source-details[open] .source-item:nth-of-type(1) { animation-delay: 0.03s; }
.source-details[open] .source-item:nth-of-type(2) { animation-delay: 0.07s; }
.source-details[open] .source-item:nth-of-type(3) { animation-delay: 0.11s; }
.source-details[open] .source-item:nth-of-type(4) { animation-delay: 0.15s; }
.source-details[open] .source-item:nth-of-type(5) { animation-delay: 0.19s; }

@keyframes sourceFadeUp {
    from {
        opacity: 0;
        transform: translateY(-6px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

    
.param-shell {
    background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
    border: 2px solid #c9d8ea;
    border-radius: 28px;
    padding: 1.15rem;
    box-shadow: 0 8px 24px rgba(26, 86, 160, 0.08);
    margin-top: 0.2rem;
    margin-bottom: 0.9rem;
}

.param-head {
    padding: 0.25rem 0.2rem 0.95rem 0.2rem;
    border-bottom: 3px solid #e0ebf7;
    margin-bottom: 1rem;
}

.param-title {
    font-size: 1.28rem;
    font-weight: 900;
    color: #2b5ea7;
    letter-spacing: -0.04em;
    line-height: 1.15;
}

.param-subtitle {
    margin-top: 0.35rem;
    font-size: 0.78rem;
    font-weight: 650;
    color: #7a8da8;
    line-height: 1.5;
}

.param-reset-wrap {
    margin-bottom: 1rem;
}

.param-card {
    background: #ffffff;
    border: 1px solid #dbe7f5;
    border-radius: 22px;
    padding: 1rem 1rem 0.8rem 1rem;
    margin-bottom: 1rem;
    box-shadow: 0 5px 16px rgba(26, 86, 160, 0.07);
}

.param-card-top {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 0.8rem;
    margin-bottom: 0.75rem;
}

.param-label {
    font-size: 0.92rem;
    font-weight: 900;
    color: #14365f;
    line-height: 1.3;
    letter-spacing: -0.03em;
}

.param-desc {
    margin-top: 0.25rem;
    font-size: 0.72rem;
    font-weight: 650;
    color: #7a8da8;
    line-height: 1.45;
}

.param-value {
    min-width: 4.2rem;
    text-align: right;
}

.param-value-main {
    font-size: 1.15rem;
    font-weight: 950;
    color: #2768bf;
    line-height: 1.05;
}

.param-value-unit {
    font-size: 0.68rem;
    font-weight: 800;
    color: #8da0b8;
    margin-top: 0.15rem;
}

.param-range {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: -0.25rem;
    padding: 0 0.05rem;
    font-size: 0.68rem;
    font-weight: 800;
    color: #8da0b8;
}

.param-pill {
    background: #eef5ff;
    color: #2b6cb0;
    padding: 0.2rem 0.5rem;
    border-radius: 999px;
    font-size: 0.65rem;
    font-weight: 900;
}

/* Streamlit slider inside parameter cards */
.param-card div[data-testid="stSlider"] {
    padding: 0 !important;
    margin: 0 !important;
}

.param-card div[data-testid="stSlider"] label {
    display: none !important;
}

.param-card div[data-testid="stSlider"] [data-baseweb="slider"] {
    padding-top: 0.15rem !important;
    padding-bottom: 0.15rem !important;
}

.param-card div[data-testid="stSlider"] [data-baseweb="slider"] > div:first-child {
    height: 9px !important;
    border-radius: 999px !important;
    background: #dce8f6 !important;
}

.param-card div[data-testid="stSlider"] [data-baseweb="slider"] > div:first-child > div {
    height: 9px !important;
    border-radius: 999px !important;
}

.param-card div[data-testid="stSlider"] [data-baseweb="slider"] > div:first-child > div:first-child {
    background: linear-gradient(90deg, #ff4b4b 0%, #2b6cb0 100%) !important;
}

.param-card div[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    width: 26px !important;
    height: 26px !important;
    background: #ffffff !important;
    border: 5px solid #2b6cb0 !important;
    box-shadow: 0 5px 14px rgba(43, 108, 176, 0.26) !important;
}

.param-card div[data-testid="stSlider"] [data-testid="stSliderThumbValue"] {
    display: none !important;
}

.param-card div[data-testid="stTickBarMin"],
.param-card div[data-testid="stTickBarMax"] {
    display: none !important;
}

.param-reset-wrap div[data-testid="stButton"] > button {
    min-height: 3.5rem !important;
    border-radius: 18px !important;
    background: #ffffff !important;
    border: 2px solid #b9cfea !important;
    color: #2b5ea7 !important;
    font-size: 1rem !important;
    font-weight: 900 !important;
}
/* Parameter panel final */
.st-key-param_panel {
    background: #ffffff;
    border: 1px solid #d6e2f0;
    border-radius: 24px;
    padding: 1.35rem 1.45rem 1.1rem 1.45rem;
    box-shadow: 0 8px 22px rgba(26,86,160,0.08);
    margin: 1.2rem 0 1.5rem 0;
}

.st-key-param_panel div[data-testid="stVerticalBlockBorderWrapper"] {
    border: none !important;
    padding: 0 !important;
}

.st-key-param_reset_btn button {
    min-height: 2.35rem !important;
    height: 2.35rem !important;
    border-radius: 999px !important;
    font-size: 0.82rem !important;
    font-weight: 800 !important;
    padding: 0 0.9rem !important;
    border: 1.5px solid #b9cfea !important;
    color: #2b5ea7 !important;
    background: #f8fbff !important;
}

.st-key-param_panel div[data-testid="stSlider"] label p {
    font-size: 0.95rem !important;
    font-weight: 900 !important;
    color: #0f2744 !important;
}

.st-key-param_panel div[data-testid="stSlider"] [data-baseweb="slider"] > div:first-child {
    height: 12px !important;
    border-radius: 999px !important;
    background: #dce8f6 !important;
}

.st-key-param_panel div[data-testid="stSlider"] [data-baseweb="slider"] > div:first-child > div:first-child {
    background: linear-gradient(90deg, #ff4b4b 0%, #2b6cb0 100%) !important;
}

.st-key-param_panel div[data-testid="stSlider"] [role="slider"] {
    width: 28px !important;
    height: 28px !important;
    background: #ffffff !important;
    border: 5px solid #2b6cb0 !important;
    box-shadow: 0 5px 14px rgba(43,108,176,0.28) !important;
}

.st-key-param_panel div[data-testid="stSlider"] [data-testid="stSliderThumbValue"] {
    font-size: 0.88rem !important;
    font-weight: 900 !important;
    color: #2b6cb0 !important;
    top: -30px !important;
}

</style>
""",
    unsafe_allow_html=True,
)


# ── 관측 데이터 ───────────────────────────────────────────────────────────────
START_YEAR, END_YEAR = 1925, 2025
YEARS = END_YEAR - START_YEAR + 1
years_axis = np.arange(START_YEAR, END_YEAR + 1)
dt = 24 * 3600
land_frac, ocean_frac = 0.29, 0.71
C_land, C_mixed, C_deep = 2.0e7, 1.2e8, 2.0e9

obs_datasets = {
    "NASA GISS (GISTEMP v4)": {
        1925: -0.22, 1940: 0.10, 1950: -0.18, 1965: -0.10,
        1980: 0.26, 1990: 0.48, 2000: 0.62, 2010: 0.88, 2020: 1.18, 2025: 1.38,
    },
    "Berkeley Earth (Land + Ocean)": {
        1925: -0.18, 1940: 0.12, 1950: -0.15, 1965: -0.08,
        1980: 0.28, 1990: 0.51, 2000: 0.65, 2010: 0.92, 2020: 1.24, 2025: 1.42,
    },
    "Hadley Centre (HadCRUT5)": {
        1925: -0.25, 1940: 0.08, 1950: -0.20, 1965: -0.12,
        1980: 0.22, 1990: 0.45, 2000: 0.60, 2010: 0.85, 2020: 1.15, 2025: 1.35,
    },
}

# ── 물리 함수 (핵심 계산 – 변경 금지) ─────────────────────────────────────────
@st.cache_data
def co2_forcing(C, T):
    return 5.35 * np.log(C / 280.0) * (1 + 0.01 * max(0, T))


@st.cache_data
def aerosol_effect(y, mult):
    base = np.interp(y, [1925, 1960, 1985, 2025], [0, -0.4, -0.9, -0.15])
    return mult * (base - 0.12 * np.abs(base) ** 0.5)


@njit
def fast_core(
    total_steps, START_YEAR, dt, land_frac, ocean_frac,
    C_land, C_mixed, C_deep, INITIAL_TEMP,
    lambda_base, aer_mult, k_lo, enso_amp, co2_path,
):
    Tl = np.zeros(total_steps)
    Tm = np.zeros(total_steps)
    Td = np.zeros(total_steps)
    Tl[0] = Tm[0] = Td[0] = INITIAL_TEMP
    y_arr = START_YEAR + np.arange(total_steps) / 365.0
    xp_aer = np.array([1925.0, 1960.0, 1985.0, 2025.0])
    fp_aer = np.array([0.0, -0.4, -0.9, -0.15])
    base_aer = np.interp(y_arr, xp_aer, fp_aer)
    aer_arr = aer_mult * (base_aer - 0.12 * np.abs(base_aer) ** 0.5)
    f_non_co2 = 0.75 * ((y_arr - 1925.0) / 100.0) ** 2.2
    f_osc_arr = enso_amp * (
        np.sin(2 * np.pi * y_arr / 3.8) + np.sin(2 * np.pi * y_arr / 5.5)
    )
    f_volc_arr = np.zeros(total_steps)
    volc_data = np.array([
        [1963.2, -0.8, 1.2],
        [1982.3, -1.3, 1.5],
        [1991.4, -1.8, 1.8],
    ])
    for v in range(3):
        ys, s, d = volc_data[v, 0], volc_data[v, 1], volc_data[v, 2]
        for i in range(total_steps):
            if y_arr[i] >= ys:
                f_volc_arr[i] += s * np.exp(-(y_arr[i] - ys) / d)
    co2_init = 5.35 * np.log(max(1.0, 306.0) / 280.0) * (
        1.0 + 0.01 * max(0.0, INITIAL_TEMP)
    )
    aer_init = aer_mult * (
        np.interp(float(START_YEAR), xp_aer, fp_aer)
        - 0.12 * np.abs(np.interp(float(START_YEAR), xp_aer, fp_aer)) ** 0.5
    )
    F_offset = (lambda_base * INITIAL_TEMP) - (co2_init + aer_init)
    base_forcing = f_non_co2 + aer_arr + f_volc_arr + f_osc_arr + F_offset
    for i in range(total_steps - 1):
        curr_T = land_frac * Tl[i] + ocean_frac * Tm[i]
        dynamic_lambda = max(0.1, lambda_base - 0.15 * max(0.0, curr_T))
        f_co2 = 5.35 * np.log(max(1.0, co2_path[i]) / 280.0) * (
            1.0 + 0.01 * max(0.0, curr_T)
        )
        total_f = f_co2 + base_forcing[i]
        h_lo = k_lo * (Tl[i] - Tm[i])
        h_md = 0.45 * (1.0 - 0.05 * max(0.0, Tm[i])) * (Tm[i] - Td[i])
        Tl[i + 1] = Tl[i] + (
            (total_f - dynamic_lambda * Tl[i] - h_lo) / C_land
        ) * dt
        Tm[i + 1] = Tm[i] + (
            (total_f - dynamic_lambda * Tm[i] + h_lo - h_md) / C_mixed
        ) * dt
        Td[i + 1] = Td[i] + (h_md / C_deep) * dt
        if Tl[i + 1] > 100.0:
            Tl[i + 1] = 100.0
        if Tm[i + 1] > 100.0:
            Tm[i + 1] = 100.0
        if Td[i + 1] > 100.0:
            Td[i + 1] = 100.0
    return Tl, Tm, Td


@st.cache_data
def run_model(params, init_temp, end_year=2025, end_co2=427):
    lambda_base, aer_mult, k_lo, enso_amp = params
    current_years_count = int(end_year - START_YEAR + 1)
    total_steps = current_years_count * 365
    y_lin = np.linspace(START_YEAR, end_year, total_steps)
    co2_path = np.interp(
        y_lin,
        np.array([1925.0, 2025.0, float(max(2025, end_year))]),
        np.array([306.0, 427.0, float(end_co2)]),
    )
    Tl, Tm, Td = fast_core(
        total_steps, START_YEAR, dt, land_frac, ocean_frac,
        C_land, C_mixed, C_deep, init_temp,
        lambda_base, aer_mult, k_lo, enso_amp, co2_path,
    )
    daily_res = land_frac * Tl + ocean_frac * Tm
    return (
        daily_res.reshape(current_years_count, 365).mean(axis=1),
        Tl.reshape(current_years_count, 365).mean(axis=1),
        Tm.reshape(current_years_count, 365).mean(axis=1),
        Td.reshape(current_years_count, 365).mean(axis=1),
        daily_res,
    )


@st.cache_data
def get_optimized_params(obs_data):
    init_temp = obs_data[0]

    def objective(params):
        m, _, _, _, _ = run_model(params, init_temp)
        return np.mean((m - obs_data) ** 2)

    res = minimize(
        objective,
        [1.5, 1.0, 2.0, 0.12],
        bounds=[(0.7, 2.3), (0.5, 2.0), (0.5, 3.5), (0.05, 0.25)],
        method="L-BFGS-B",
        options={"maxiter": 15, "ftol": 1e-4},
    )
    return res.x


@st.cache_data
def load_report_file():
    report_candidates = [
        Path("기후모델 웹사이트 분석 리포트.docx"),
        Path("./기후모델 웹사이트 분석 리포트.docx"),
    ]
    for path in report_candidates:
        if path.exists():
            return path.name, path.read_bytes()
    return None, None


# ── Matplotlib 공통 스타일 ────────────────────────────────────────────────────
def _apply_chart_style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor("#f8fafc")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#d6e2f0")
    ax.spines["bottom"].set_color("#d6e2f0")
    ax.tick_params(colors="#64748b", labelsize=9)
    ax.grid(True, color="#d6e2f0", linewidth=0.7, linestyle="--", alpha=0.6)
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold", color="#0f2744", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, color="#64748b")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color="#64748b")


def _styled_fig(nrows=1, ncols=1, figsize=(12, 5)):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    fig.patch.set_facecolor("#f8fafc")
    return fig, axes


# ── UI 헬퍼 ──────────────────────────────────────────────────────────────────
def render_metric(label, val, unit="", note=""):
    st.markdown(
        f"""
<div class="mcard">
  <div class="mcard-label">{label}</div>
  <div class="mcard-val">{val}<span class="mcard-unit">{unit}</span></div>
  <div class="mcard-note">{note}</div>
</div>""",
        unsafe_allow_html=True,
    )


def render_infobox(title, body):
    st.markdown(
        f"""
<div class="infobox">
  <div class="infobox-title">{title}</div>
  <div class="infobox-body">{body}</div>
</div>""",
        unsafe_allow_html=True,
    )


def sec(label):
    st.markdown(f'<div class="sec-title">{label}</div>', unsafe_allow_html=True)


def page_header(title, subtitle=""):
    st.markdown(f'<div class="page-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(
            f'<div class="page-subtitle">{subtitle}</div>', unsafe_allow_html=True
        )


def grid_gap(height="1.2rem"):
    st.markdown(f'<div style="height:{height};"></div>', unsafe_allow_html=True)


# ── 페이지 / 네비게이션 상태 ───────────────────────────────────────────────────
ALL_PAGES = [
    "시작 페이지",
    "시나리오 기반 기후 변화 예측",
    "기후 시스템 파라미터 실험",
    "모델 적합도 및 관측자료 비교",
    "모델 검증 및 불확실성 정량화",
    "기후 모델링 용어 및 개념 정의",
    "연구 요약 및 보고서",
]
NAV_PAGES = ALL_PAGES[1:]

slug_to_page = {
    "home": "시작 페이지",
    "scenario": "시나리오 기반 기후 변화 예측",
    "experiment": "기후 시스템 파라미터 실험",
    "fit": "모델 적합도 및 관측자료 비교",
    "uncertainty": "모델 검증 및 불확실성 정량화",
    "glossary": "기후 모델링 용어 및 개념 정의",
    "summary": "연구 요약 및 보고서",
}
page_to_slug = {v: k for k, v in slug_to_page.items()}

if "page" not in st.session_state:
    st.session_state["page"] = "시작 페이지"

q = st.query_params.get("module")
if q in slug_to_page:
    st.session_state["page"] = slug_to_page[q]

page = st.session_state["page"]
policy_q = st.query_params.get("policy")
policy_map = {
    "netzero": "탄소중립",
    "low": "저배출",
    "current": "현재정책",
    "high": "고배출",
    "extreme": "극단배출",
}
if policy_q in policy_map:
    st.session_state["main_policy"] = policy_map[policy_q]


def set_page(target):
    st.session_state["page"] = target
    st.query_params["module"] = page_to_slug.get(target, "home")
    st.rerun()


# ── Left Panel ────────────────────────────────────────────────────────────────
def render_left_panel():
    current = st.session_state.get("page", "시작 페이지")

    nav_items = [
        ("시작 페이지", "시작 페이지", "home"),
        ("시나리오 예측", "시나리오 기반 기후 변화 예측", "scenario"),
        ("파라미터 실험", "기후 시스템 파라미터 실험", "experiment"),
        ("관측 비교", "모델 적합도 및 관측자료 비교", "fit"),
        ("모델 검증", "모델 검증 및 불확실성 정량화", "uncertainty"),
        ("용어 정의", "기후 모델링 용어 및 개념 정의", "glossary"),
        ("연구 요약", "연구 요약 및 보고서", "summary"),
    ]

    nav_html = ['<div class="nav-panel">']
    nav_html.append('<div class="nav-panel-title">탐색 메뉴</div>')
    nav_html.append('<div class="nav-links">')

    for label, page_name, slug in nav_items:
        active_class = " active" if current == page_name else ""
        nav_html.append(
            f'<a class="nav-link{active_class}" href="?module={slug}" target="_self">{label}</a>'
        )

    nav_html.append("</div></div>")

    st.markdown("".join(nav_html), unsafe_allow_html=True)

    source_html = (
        '<div class="source-card">'
        '<details class="source-details">'
        '<summary>'
        '<span class="source-arrow">›</span>'
        '<span class="source-title">자료 출처</span>'
        '</summary>'
        '<div class="source-content">'
        '<div class="source-note">본 모델은 주요 공인 데이터를 바탕으로 구성되었습니다.</div>'
        '<a class="source-item" href="https://data.giss.nasa.gov/gistemp/" target="_blank"><span class="source-name">NASA GISS GISTEMP v4</span><span class="source-desc">역사적 전지구 평균기온 데이터</span></a>'
        '<a class="source-item" href="https://www.ipcc.ch/report/ar6/wg1/" target="_blank"><span class="source-name">IPCC AR6</span><span class="source-desc">기후 변화 과학적 기준</span></a>'
        '<a class="source-item" href="https://gml.noaa.gov/ccgg/trends/" target="_blank"><span class="source-name">NOAA CO₂</span><span class="source-desc">대기 중 CO₂ 농도 추세</span></a>'
        '<a class="source-item" href="https://volcano.si.edu/" target="_blank"><span class="source-name">Smithsonian Volcano</span><span class="source-desc">화산 강제력 데이터</span></a>'
        '<a class="source-item" href="https://sealevel.nasa.gov/" target="_blank"><span class="source-name">NASA Sea Level</span><span class="source-desc">해수면 상승 데이터</span></a>'
        '</div>'
        '</details>'
        '</div>'
    )
    st.markdown(source_html, unsafe_allow_html=True)
    
# ── Settings Panel (per page) ─────────────────────────────────────────────────
def render_settings(current_page):
    controls = {}
    if current_page == "시나리오 기반 기후 변화 예측":
        current_policy = st.session_state.get("main_policy", "현재정책")
    
        scenario_meta = {
            "탄소중립": {
                "slug": "netzero",
                "hero_desc": "탄소 배출을 빠르게 감축해 장기 온난화 위험을 최소화하는 경로입니다.",
                "co2": 280,
            },
            "저배출": {
                "slug": "low",
                "hero_desc": "감축 정책이 비교적 잘 작동해 온도 상승을 완화하는 시나리오입니다.",
                "co2": 380,
            },
            "현재정책": {
                "slug": "current",
                "hero_desc": "현재 정책이 크게 강화되지 않는다는 가정 아래의 기준 시나리오입니다.",
                "co2": 550,
            },
            "고배출": {
                "slug": "high",
                "hero_desc": "배출 억제가 충분히 이루어지지 않아 온난화 속도가 더 커지는 경로입니다.",
                "co2": 850,
            },
            "극단배출": {
                "slug": "extreme",
                "hero_desc": "배출이 거의 제어되지 않아 가장 큰 기후 리스크가 나타나는 경로입니다.",
                "co2": 1500,
            },
        }
    
        order = ["탄소중립", "저배출", "현재정책", "고배출", "극단배출"]
        current_idx = order.index(current_policy)
        fill_pct = (current_idx / (len(order) - 1)) * 100
    
        html = [
            '<div class="settings-shell">',
            '<div class="settings-title">배출 시나리오 설정</div>',
            f'<div class="scenario-hero">'
            f'<div class="scenario-hero-top">현재 선택 시나리오</div>'
            f'<div class="scenario-hero-name">{current_policy}</div>'
            f'<div class="scenario-hero-desc">{scenario_meta[current_policy]["hero_desc"]}</div>'
            f'</div>',
            '<div class="scenario-scale-title">시나리오 강도</div>',
            '<div class="scenario-scale">',
            '<div class="scenario-scale-line"></div>',
            f'<div class="scenario-scale-fill" style="width: calc({fill_pct}% - 12px);"></div>',
            '<div class="scenario-scale-steps">'
        ]
    
        for label in order:
            meta = scenario_meta[label]
            active = " active" if label == current_policy else ""
            html.append(
                f'<a class="scenario-step{active}" href="?module=scenario&policy={meta["slug"]}" target="_self">'
                f'<span class="scenario-step-inner">'
                f'<div class="scenario-dot"></div>'
                f'<div class="scenario-step-label">{label}</div>'
                f'</span>'
                f'</a>'
            )
    
        html.extend([
            '</div>',
            '<div class="scenario-scale-foot"><span class="low">낮음</span><span class="mid">현재 수준</span><span class="high">높음</span></div>',
            '</div>',
            f'<div class="scenario-current">현재 선택: <strong>{current_policy}</strong> · 2100년 CO₂ {scenario_meta[current_policy]["co2"]} ppm</div>',
            '</div>'
        ])
    
        st.markdown("".join(html), unsafe_allow_html=True)
        controls["policy"] = current_policy
        
    elif current_page == "기후 시스템 파라미터 실험":
        st.markdown(
            """
    <div class="param-shell">
      <div class="param-head">
        <div class="param-title">파라미터 설정</div>
        <div class="param-subtitle">기후 시스템 입력값을 조정해 장기 온난화 반응을 실험합니다.</div>
      </div>
      <div class="param-reset-wrap">
            """,
            unsafe_allow_html=True,
        )
    
        if st.button("↻ 초기화", use_container_width=True, key="main_reset_experiment"):
            for k, v in {
                "main_exp_co2": 550,
                "main_exp_lambda": 1.5,
                "main_exp_aer": 1.0,
            }.items():
                st.session_state[k] = v
            st.rerun()
    
        st.markdown("</div>", unsafe_allow_html=True)
    
        # CO2
        current_co2 = int(st.session_state.get("main_exp_co2", 550))
        st.markdown(
            f"""
    <div class="param-card">
      <div class="param-card-top">
        <div>
          <div class="param-label">2100년 CO₂ 농도</div>
          <div class="param-desc">높을수록 복사강제력이 커져 온난화가 증가합니다.</div>
        </div>
        <div class="param-value">
          <div class="param-value-main">{current_co2}</div>
          <div class="param-value-unit">ppm</div>
        </div>
      </div>
            """,
            unsafe_allow_html=True,
        )
    
        st.markdown(
            """
      <div class="param-range">
        <span>250</span>
        <span class="param-pill">권장 400–1000 ppm</span>
        <span>1500</span>
      </div>
    </div>
            """,
            unsafe_allow_html=True,
        )
    
        # Lambda
        current_lambda = float(st.session_state.get("main_exp_lambda", 1.5))
        st.markdown(
            f"""
    <div class="param-card">
      <div class="param-card-top">
        <div>
          <div class="param-label">기후 피드백 파라미터 (λ)</div>
          <div class="param-desc">값이 클수록 기후 시스템의 복원력이 강해집니다.</div>
        </div>
        <div class="param-value">
          <div class="param-value-main">{current_lambda:.2f}</div>
          <div class="param-value-unit">W/m²/°C</div>
        </div>
      </div>
            """,
            unsafe_allow_html=True,
        )
    
    
        st.markdown(
            """
      <div class="param-range">
        <span>0.5</span>
        <span class="param-pill">기준 1.50</span>
        <span>3.0</span>
      </div>
    </div>
            """,
            unsafe_allow_html=True,
        )
    
        # Aerosol
        current_aer = float(st.session_state.get("main_exp_aer", 1.0))
        st.markdown(
            f"""
    <div class="param-card">
      <div class="param-card-top">
        <div>
          <div class="param-label">에어로졸 강도</div>
          <div class="param-desc">값이 클수록 에어로졸 냉각 효과가 강해집니다.</div>
        </div>
        <div class="param-value">
          <div class="param-value-main">{current_aer:.2f}</div>
          <div class="param-value-unit">배율</div>
        </div>
      </div>
            """,
            unsafe_allow_html=True,
        )
    
    
        st.markdown(
            """
      <div class="param-range">
        <span>0.0</span>
        <span class="param-pill">기준 1.00</span>
        <span>3.0</span>
      </div>
    </div>
    </div>
            """,
            unsafe_allow_html=True,
        )
    
    elif current_page == "모델 적합도 및 관측자료 비교":
        st.markdown(
            '<div class="settings-shell"><div class="settings-title">데이터셋 선택</div>',
            unsafe_allow_html=True,
        )
        obs_list = list(obs_datasets.keys())
        controls["obs_choice"] = st.selectbox(
            "관측 데이터셋",
            obs_list,
            index=obs_list.index(st.session_state.get("main_obs_choice", obs_list[0])),
            key="main_obs_choice",
        )
        controls["current_obs_data"] = np.interp(
            years_axis,
            list(obs_datasets[controls["obs_choice"]].keys()),
            list(obs_datasets[controls["obs_choice"]].values()),
        )
        st.markdown("</div>", unsafe_allow_html=True)

    elif current_page == "모델 검증 및 불확실성 정량화":
        st.markdown(
            '<div class="settings-shell"><div class="settings-title">검증 데이터셋</div>',
            unsafe_allow_html=True,
        )
        obs_list = list(obs_datasets.keys())
        controls["diag_obs_choice"] = st.selectbox(
            "검증용 데이터셋",
            obs_list,
            index=obs_list.index(st.session_state.get("main_diag_obs_choice", obs_list[0])),
            key="main_diag_obs_choice",
        )
        controls["diag_obs_data"] = np.interp(
            years_axis,
            list(obs_datasets[controls["diag_obs_choice"]].keys()),
            list(obs_datasets[controls["diag_obs_choice"]].values()),
        )
        st.markdown("</div>", unsafe_allow_html=True)

    return controls


# ═══════════════════════════════════════════════════════════════════════════════
# 페이지: 시작 페이지
# ═══════════════════════════════════════════════════════════════════════════════
if page == "시작 페이지":
    st.query_params["module"] = "home"

    st.markdown(
        """
<div class="hero">
  <div class="hero-badge">Research Presentation Interface</div>
  <div class="hero-title">기후 모델링 연구 대시보드</div>
  <div class="hero-desc">
    초기 물리 기반 기후 모델에 머신러닝을 결합하여 역사적 온도 변화를 재현, 
    미래 전지구 평균기온의 장기 경로를 탐색하기 위한 연구형 대시보드 
    
  </div>
  <div class="hero-chips">
    <div class="hero-chip">물리 기반 모델</div>
    <div class="hero-chip">관측자료 비교</div>
    <div class="hero-chip">불확실성 분석</div>
    <div class="hero-chip">시나리오 예측</div>
  </div>
</div>""",
        unsafe_allow_html=True,
    )

    sec("분석 모듈 바로가기")

    def modcard(num, title, desc, slug):
        href = f"?module={quote(slug)}"
        st.markdown(
            f"""
<a class="modcard" href="{href}" target="_self">
  <div class="modcard-num">Module {num:02d}</div>
  <div class="modcard-title">{title}</div>
  <div class="modcard-desc">{desc}</div>
</a>""",
            unsafe_allow_html=True,
        )

    row_gap = "1.2rem"

    r1c1, r1c2 = st.columns(2, gap="small")
    with r1c1:
        modcard(
            1,
            "시나리오 기반 기후 변화 예측",
            "탄소중립부터 고배출 경로까지 다양한 배출 시나리오로 2100년까지의 전지구 평균기온과 해수면 상승을 비교합니다.",
            "scenario",
        )
    with r1c2:
        modcard(
            2,
            "기후 시스템 파라미터 실험",
            "기후 피드백, 에어로졸 강도, 해양 열흡수 계수, ENSO 진폭을 직접 조정해 모델 응답이 어떻게 달라지는지 실험합니다.",
            "experiment",
        )

    grid_gap(row_gap)

    r2c1, r2c2 = st.columns(2, gap="small")
    with r2c1:
        modcard(
            3,
            "모델 적합도 및 관측자료 비교",
            "관측 자료와 모델 출력의 차이를 시계열, 상대오차, 강제력 기여 요소로 분해하여 역사적 기후 변화를 평가합니다.",
            "fit",
        )
    with r2c2:
        modcard(
            4,
            "모델 검증 및 불확실성 정량화",
            "잔차 진단, 불확실성 범위, 민감도 분석을 통해 예측 신뢰성과 구조적 특성을 검토합니다.",
            "uncertainty",
        )

    grid_gap(row_gap)

    r3c1, r3c2 = st.columns(2, gap="small")
    with r3c1:
        modcard(
            5,
            "기후 모델링 용어 및 개념 정의",
            "본 모델에서 사용되는 주요 기후학 개념, 물리 파라미터, 검증 지표를 정리한 참고 페이지입니다.",
            "glossary",
        )
    with r3c2:
        modcard(
            6,
            "연구 요약 및 보고서",
            "프로젝트의 연구 목적, 모델 구조, 해석 주의점, 연구 의의를 정리하고 분석 리포트 다운로드를 제공합니다.",
            "summary",
        )

    st.caption("모듈 카드를 클릭하면 해당 분석 페이지로 바로 이동합니다.")


# ═══════════════════════════════════════════════════════════════════════════════
# 페이지: 시나리오 기반 기후 변화 예측
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "시나리오 기반 기후 변화 예측":
    st.query_params["module"] = "scenario"
    left_col, main_col = st.columns([1.05, 4.2], gap="large")

    with left_col:
        render_left_panel()

    with main_col:
        policy = controls["policy"]
        page_header("시나리오 기반 기후 변화 예측",
                    "배출 경로별 장기 온난화 궤적 비교 · 1.5 / 2.0°C 임계선 관계 분석")
        emission_map = {
            "탄소중립": 280,
            "저배출": 380,
            "현재정책": 550,
            "고배출": 850,
            "극단배출": 1500,
        }
        res_full, _, _, _, _ = run_model(
            [1.5, 1.0, 2.0, 0.12], -0.22, end_year=2100, end_co2=emission_map[policy]
        )
        p_2100 = res_full[-1]
        trend_21c = np.polyfit(np.arange(1925, 2101), res_full, 1)[0]

        render_infobox(
            "분석 목적",
            "서로 다른 배출 시나리오에 따라 장기 온난화 경로가 어떻게 달라지는지 비교합니다. "
            "1.5°C · 2.0°C 임계선과의 관계를 함께 제시하여 각 시나리오의 상대적 기후 위험 수준을 해석할 수 있도록 구성했습니다.",
            )
            
    st.markdown("""
    <div style="
        background:#ffffff;
        border:1px solid #d6e2f0;
        border-radius:20px;
        padding:1.3rem 1.4rem;
        box-shadow:0 4px 14px rgba(26,86,160,0.08);
        margin-bottom:1.3rem;
    ">
    <div style="font-size:1.05rem;font-weight:800;color:#0f2744;margin-bottom:0.9rem;">
    파라미터 조정
    </div>
    """, unsafe_allow_html=True)
    
    # 초기화 버튼
    if st.button("초기화", use_container_width=True):
        st.session_state["main_exp_co2"] = 550
        st.session_state["main_exp_lambda"] = 1.5
        st.session_state["main_exp_aer"] = 1.0
        st.rerun()
    
    # CO2
    co2 = st.slider(
        "2100년 CO₂ 농도 (ppm)",
        250, 1500,
        int(st.session_state.get("main_exp_co2", 550)),
        step=10,
        key="main_exp_co2",
    )
    
    # lambda
    lam = st.slider(
        "기후 피드백 파라미터 (λ)",
        0.5, 3.0,
        float(st.session_state.get("main_exp_lambda", 1.5)),
        step=0.1,
        key="main_exp_lambda",
    )
    
    # aerosol
    aer = st.slider(
        "에어로졸 강도",
        0.0, 3.0,
        float(st.session_state.get("main_exp_aer", 1.0)),
        step=0.1,
        key="main_exp_aer",
    )
    
    st.markdown("</div>", unsafe_allow_html=True)

    target_co2 = emission_map[policy]
    st.markdown(
            f"""
<div class="cond-bar">
  <div class="cond-item">
    <div class="cond-label">선택 시나리오</div>
    <div class="cond-val" style="font-size:0.93rem">{policy}</div>
  </div>
  <div class="cond-item">
    <div class="cond-label">2100년 목표 CO₂</div>
    <div class="cond-val">{target_co2} <span style="font-size:0.8rem;font-weight:500;color:#94a3b8">ppm</span></div>
  </div>
  <div class="cond-item">
    <div class="cond-label">2100년 예상 온도</div>
    <div class="cond-val">+{p_2100:.2f} <span style="font-size:0.8rem;font-weight:500;color:#94a3b8">°C</span></div>
  </div>
</div>""",
            unsafe_allow_html=True,
        )

    sec("핵심 결과")
    c1, c2, c3 = st.columns(3)
    with c1:
        render_metric("2100년 예상 온난화", f"+{p_2100:.2f}", "°C",
                      "선택한 시나리오 기준 장기 예측값")
    with c2:
        render_metric("2100년 예상 해수면 상승", f"+{p_2100*35:.1f}", "cm",
                      "단순 비례 가정에 따른 참고 지표")
    with c3:
        render_metric("평균 온난화 속도", f"{trend_21c:.3f}", "°C/yr",
                      "1925–2100 전체 구간 평균 추세")

    sec("장기 기온 궤적")

    obs_vals = np.interp(
        years_axis,
        list(obs_datasets["NASA GISS (GISTEMP v4)"].keys()),
        list(obs_datasets["NASA GISS (GISTEMP v4)"].values()),
    )
    
    years_full = np.arange(1925, 2101)
    future_years = years_full[len(years_axis) - 1:]
    future_vals = res_full[len(years_axis) - 1:]
    
    step_size = 1
    
    frame_indices = list(range(1, len(future_years) + 1, step_size))
    if frame_indices[-1] != len(future_years):
        frame_indices.append(len(future_years))
    
    frames = []
    
    for i in frame_indices:
        frames.append(
            go.Frame(
                data=[
                    go.Scatter(
                        x=years_axis,
                        y=obs_vals,
                        mode="lines",
                        name="Historical Observation",
                        line=dict(width=3, color="#0f2744"),
                    ),
                    go.Scatter(
                        x=future_years[:i],
                        y=future_vals[:i],
                        mode="lines",
                        name="Projected Response",
                        line=dict(width=3, color="#1a56a0", dash="dash"),
                    ),
                    go.Scatter(
                        x=list(future_years[:i]) + list(future_years[:i][::-1]),
                        y=list(future_vals[:i]) + [0] * i,
                        fill="toself",
                        fillcolor="rgba(26, 86, 160, 0.14)",
                        line=dict(color="rgba(255,255,255,0)"),
                        hoverinfo="skip",
                        showlegend=False,
                    ),
                ],
                name=str(i),
            )
        )

    fig = go.Figure(
        data=[
            go.Scatter(
                x=years_axis,
                y=obs_vals,
                mode="lines",
                name="Historical Observation",
                line=dict(width=3, color="#0f2744"),
            ),
            go.Scatter(
                x=[future_years[0]],
                y=[future_vals[0]],
                mode="lines",
                name="Projected Response",
                line=dict(width=3, color="#1a56a0", dash="dash"),
            ),
            go.Scatter(
                x=[future_years[0], future_years[0]],
                y=[future_vals[0], 0],
                fill="toself",
                fillcolor="rgba(26, 86, 160, 0.14)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=False,
            ),
        ],
        frames=frames,
    )
    
    fig.add_hline(
        y=1.5,
        line_dash="dot",
        line_color="#f59e0b",
        annotation_text="1.5°C Threshold",
        annotation_position="top left",
    )
    
    fig.add_hline(
        y=2.0,
        line_dash="dot",
        line_color="#ef4444",
        annotation_text="2.0°C Threshold",
        annotation_position="top left",
    )
    
    fig.add_vline(
        x=2025,
        line_dash="dash",
        line_color="#94a3b8",
        annotation_text="2025",
        annotation_position="bottom right",
    )
    
    fig.update_layout(
        title=dict(
            text="Projected Global Temperature Trajectory",
            x=0.5,
            font=dict(size=18, color="#0f2744"),
        ),
        xaxis_title="Year",
        yaxis_title="Temperature Anomaly (°C)",
        plot_bgcolor="#f8fafc",
        paper_bgcolor="#f8fafc",
        font=dict(color="#64748b"),
        height=520,
        margin=dict(l=40, r=30, t=70, b=50),
        legend=dict(
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#d6e2f0",
            borderwidth=1,
        ),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                x=0.02,
                y=1.12,
                buttons=[
                    dict(
                        label="▶ 예측 애니메이션",
                        method="animate",
                        args=[
                            None,
                            {
                                "frame": {"duration": 22, "redraw": False},
                                "fromcurrent": True,
                                "transition": {"duration": 12},
                            },
                        ],
                    ),
                    dict(
                        label="⏸ 정지",
                        method="animate",
                        args=[
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    ),
                ],
            )
        ],
    )
        
    fig.update_xaxes(
        range=[1925, 2100],
        gridcolor="#d6e2f0",
        showgrid=True,
    )
    
    y_min = min(-0.4, float(np.min(obs_vals)) - 0.2)
    y_max = max(2.3, float(np.max(future_vals)) + 0.25)
    
    fig.update_yaxes(
        range=[y_min, y_max],
        gridcolor="#d6e2f0",
        showgrid=True,
    )        
    plotly_html = fig.to_html(
        full_html=False,
        include_plotlyjs="cdn",
        config={
            "displayModeBar": False,
            "scrollZoom": False,
        },
        auto_play=False,
    )
    
    auto_play_script = """
    <script>
    setTimeout(() => {
        const graph = document.querySelector('.plotly-graph-div');
        if (graph) {
            Plotly.animate(graph, null, {
                frame: {duration: 10, redraw: false},
                transition: {duration: 6},
                fromcurrent: true,
                mode: 'immediate'
            });
        }
    }, 300);
    </script>
    """
    
    components.html(
        plotly_html + auto_play_script,
        height=560,
        scrolling=False,
    )
    render_infobox(
        "해석",
        "고배출에 가까운 경로일수록 온도 상승 속도가 빠르게 커지며, 임계 온도 도달 시점도 앞당겨집니다. "
        "본 결과는 전지구 평균 기반 단순화 모델에서 도출된 것으로, "
        "정밀 예측보다 시나리오 간 상대적 차이와 장기 경향 비교용으로 해석하는 것이 적절합니다.",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 페이지: 기후 시스템 파라미터 실험
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "기후 시스템 파라미터 실험":
    st.query_params["module"] = "experiment"
    left_col, main_col = st.columns([1.05, 4.2], gap="large")

    with left_col:
        render_left_panel()

    with main_col:
        page_header(
            "기후 시스템 파라미터 실험",
            "핵심 기후 파라미터를 직접 조정하여 장기 온난화 경로 변화를 탐색합니다"
        )

        render_infobox(
            "분석 목적",
            "기후 피드백 강도, 에어로졸 냉각, 해양 열흡수, 내부 변동성과 같은 핵심 파라미터가 "
            "장기 온난화 결과에 어떤 영향을 주는지 탐색하기 위해 구성되었습니다.",
        )


with st.container(border=True, key="param_panel"):
    h1, h2 = st.columns([5, 1])

    with h1:
        st.markdown(
            """
<div style="font-size:1.18rem;font-weight:900;color:#0f2744;letter-spacing:-0.03em;">
    파라미터 조정
    </div>
    <div style="font-size:0.82rem;font-weight:650;color:#7a8da8;margin-top:0.25rem;margin-bottom:1.1rem;">
    입력값을 조절하면 아래 결과와 그래프가 즉시 갱신됩니다.
    </div>
    """,
                unsafe_allow_html=True,
            )
    
        with h2:
            if st.button("↻ 초기화", use_container_width=True, key="param_reset_btn"):
                st.session_state["main_exp_co2"] = 550
                st.session_state["main_exp_lambda"] = 1.5
                st.session_state["main_exp_aer"] = 1.0
                st.rerun()
    
        p1, p2, p3 = st.columns(3, gap="large")
    
        with p1:
            co2 = st.slider(
                "2100년 CO₂ 농도 (ppm)",
                250, 1500,
                int(st.session_state.get("main_exp_co2", 550)),
                step=10,
                key="main_exp_co2",
            )
    
        with p2:
            lam = st.slider(
                "기후 피드백 파라미터 (λ)",
                0.5, 3.0,
                float(st.session_state.get("main_exp_lambda", 1.5)),
                step=0.1,
                key="main_exp_lambda",
            )
    
        with p3:
            aer = st.slider(
                "에어로졸 강도",
                0.0, 3.0,
                float(st.session_state.get("main_exp_aer", 1.0)),
                step=0.1,
                key="main_exp_aer",
            )
        
        st.markdown("</div>", unsafe_allow_html=True)

        exp_co2 = co2
        exp_lambda = lam
        exp_aer = aer
        exp_klo = 2.0
        exp_enso = 0.12

        custom_params = [exp_lambda, exp_aer, exp_klo, exp_enso]
        res_exp, tl_exp, tm_exp, td_exp, _ = run_model(
            custom_params, -0.22, end_year=2100, end_co2=exp_co2
        )
            
    cond_html = "\n".join([
        '<div class="cond-bar">',
        '  <div class="cond-item">',
        '    <div class="cond-label">CO2 (2100)</div>',
        f'    <div class="cond-val">{exp_co2} <span style="font-size:0.8rem;color:#94a3b8">ppm</span></div>',
        '    <div class="cond-base">기준: 550 ppm</div>',
        '  </div>',
    
        '  <div class="cond-item">',
        '    <div class="cond-label">Aerosol</div>',
        f'    <div class="cond-val">{exp_aer:.2f} <span style="font-size:0.8rem;color:#94a3b8">배율</span></div>',
        '    <div class="cond-base">기준: 1.00</div>',
        '  </div>',
    
        '  <div class="cond-item">',
        '    <div class="cond-label">Feedback</div>',
        f'    <div class="cond-val">{exp_lambda:.2f}</div>',
        '    <div class="cond-base">기준: 1.50</div>',
        '  </div>',
    
        '  <div class="cond-item">',
        '    <div class="cond-label">Ocean Heat</div>',
        f'    <div class="cond-val">{exp_klo:.2f}</div>',
        '    <div class="cond-base">고정값</div>',
        '  </div>',
    
        '  <div class="cond-item">',
        '    <div class="cond-label">ENSO</div>',
        f'    <div class="cond-val">{exp_enso:.2f}</div>',
        '    <div class="cond-base">고정값</div>',
        '  </div>',
        '</div>',
    ])
    
    st.markdown(cond_html, unsafe_allow_html=True)

        sec("핵심 결과")
        c1, c2, c3 = st.columns(3)

        with c1:
            render_metric(
                "2100년 육지 온도 상승",
                f"+{tl_exp[-1]:.2f}",
                "°C",
                "육지는 상대적으로 빠르게 반응",
            )

        with c2:
            render_metric(
                "2100년 해양 표층 온도 상승",
                f"+{tm_exp[-1]:.2f}",
                "°C",
                "표층 해양의 완충 효과 반영",
            )

        with c3:
            render_metric(
                "2100년 심해 온도 상승",
                f"+{td_exp[-1]:.2f}",
                "°C",
                "심해는 가장 느리게 반응",
            )

        sec("실험 결과 시계열")

        fig, ax = _styled_fig(figsize=(12, 5.2))

        obs_vals = np.interp(
            years_axis,
            list(obs_datasets["NASA GISS (GISTEMP v4)"].keys()),
            list(obs_datasets["NASA GISS (GISTEMP v4)"].values()),
        )

        years_exp = np.arange(1925, 2101)

        ax.plot(
            years_axis,
            obs_vals,
            color="#0f2744",
            lw=1.8,
            label="Observed Temperature",
        )

        ax.fill_between(years_exp, res_exp, alpha=0.1, color="#1a56a0")

        ax.plot(
            years_exp,
            res_exp,
            color="#1a56a0",
            lw=2.4,
            label="Experimental Simulation",
        )

        ax.axhline(0, color="#94a3b8", lw=0.8, ls="--")
        ax.axhline(1.5, color="#f59e0b", ls=":", lw=1.6, label="1.5°C Threshold")
        ax.axvline(2025, color="#94a3b8", ls="--", lw=1, alpha=0.6)

        _apply_chart_style(
            ax,
            title=f"Projected Warming — User-Defined Parameters | 2100: +{res_exp[-1]:.2f}°C",
            xlabel="Year",
            ylabel="Temperature Anomaly (°C)",
        )

        ax.legend(fontsize=9, framealpha=0.85, edgecolor="#d6e2f0")
        plt.tight_layout()
        st.pyplot(fig)

        render_infobox(
            "해석",
            "같은 배출 조건에서도 파라미터 선택에 따라 최종 온도 상승폭과 경로가 달라질 수 있습니다. "
            "특히 해양은 열용량이 크기 때문에 육지보다 더 느리게 반응하며, "
            "이 반응 속도 차이는 장기 기후 변화 해석에서 중요한 의미를 가집니다.",
        )

# ═══════════════════════════════════════════════════════════════════════════════
# 페이지: 모델 적합도 및 관측자료 비교
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "모델 적합도 및 관측자료 비교":
    st.query_params["module"] = "fit"
    left_col, main_col = st.columns([1.05, 4.2], gap="large")

    with left_col:
        render_left_panel()
        controls = render_settings(page)

    with main_col:
        obs_choice = controls["obs_choice"]
        current_obs_data = controls["current_obs_data"]

        page_header("모델 적합도 및 관측자료 비교",
                    "관측값 vs 모의값 차이 정량화 · 강제력 기여 요소 분해")

        render_infobox(
            "분석 목적",
            "모델이 실제 관측자료의 장기 기온 변화를 어느 정도 재현할 수 있는지 평가합니다. "
            "최적화된 파라미터로 관측값과 모의값의 차이를 정량화하고, "
            "인위적 요인과 자연적 요인의 상대적 기여를 함께 해석할 수 있도록 구성했습니다.",
        )

        with st.spinner(f"{obs_choice} 데이터에 맞춰 모델을 최적화하는 중입니다..."):
            best_params = get_optimized_params(current_obs_data)
            best_global, best_tl, best_tm, best_td, daily_all = run_model(
                best_params, current_obs_data[0]
            )

        err = np.where(
            np.abs(current_obs_data) > 0.1,
            ((best_global - current_obs_data) / np.abs(current_obs_data)) * 100,
            0,
        )
        avg_err = np.mean(np.abs(err))
        rmse = np.sqrt(np.mean((best_global - current_obs_data) ** 2))
        mae = np.mean(np.abs(best_global - current_obs_data))
        bias = np.mean(best_global - current_obs_data)

        sec("핵심 지표")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            render_metric("RMSE", f"{rmse:.3f}", "°C", "제곱근 평균 오차")
        with c2:
            render_metric("MAE", f"{mae:.3f}", "°C", "절대값 평균 오차")
        with c3:
            render_metric("Bias", f"{bias:.3f}", "°C", "과대/과소 예측 경향")
        with c4:
            render_metric("Mean % Error", f"{avg_err:.2f}", "%", "상대 오차 평균")

        sec("상세 분석 차트")
        fig, axes = _styled_fig(nrows=3, ncols=2, figsize=(16, 18))
        plt.subplots_adjust(hspace=0.42, wspace=0.32)

        axes[0, 0].plot(years_axis, best_tl, color="#c2410c", lw=1.8, label="Land Surface")
        axes[0, 0].plot(years_axis, best_tm, color="#1d4ed8", lw=1.8, label="Ocean Mixed Layer")
        axes[0, 0].plot(years_axis, best_td, color="#1e3a8a", lw=2, label="Deep Ocean")
        _apply_chart_style(axes[0, 0], title="Layer-Specific Thermal Response",
                           xlabel="Year", ylabel="Anomaly (°C)")
        axes[0, 0].legend(fontsize=8, framealpha=0.85, edgecolor="#d6e2f0")

        axes[0, 1].fill_between(years_axis, best_global, current_obs_data, alpha=0.15, color="#1a56a0")
        axes[0, 1].plot(years_axis, best_global, color="#1a56a0", lw=2.2, label="Model Simulation")
        axes[0, 1].plot(years_axis, current_obs_data, color="#0f2744", lw=1.6, ls="--",
                        alpha=0.75, label="Observed Series")
        _apply_chart_style(axes[0, 1], title="Observed vs Simulated Temperature Anomaly",
                           xlabel="Year", ylabel="Anomaly (°C)")
        axes[0, 1].legend(fontsize=8, framealpha=0.85, edgecolor="#d6e2f0")

        bar_colors = ["#ef4444" if x > 0 else "#3b82f6" for x in err]
        axes[1, 0].bar(years_axis, err, color=bar_colors, alpha=0.8, width=0.85)
        axes[1, 0].axhline(0, color="#0f2744", lw=0.8)
        _apply_chart_style(
            axes[1, 0],
            title=f"Relative Error  |  Mean: {avg_err:.2f}%  RMSE: {rmse:.3f}°C  MAE: {mae:.3f}°C  Bias: {bias:.3f}°C",
            xlabel="Year", ylabel="Relative Error (%)",
        )

        f_co2 = [
            co2_forcing(np.interp(y, [1925, 2025], [306, 427]), best_global[int(y - 1925)])
            for y in years_axis
        ]
        f_non_co2 = [0.75 * ((y - 1925) / 100) ** 2.2 for y in years_axis]
        f_aero = [aerosol_effect(y, best_params[1]) for y in years_axis]
        axes[1, 1].stackplot(
            years_axis, f_co2, f_non_co2,
            labels=["CO2 Forcing", "Other Anthropogenic"],
            colors=["#1a56a0", "#60a5fa"], alpha=0.75,
        )
        axes[1, 1].plot(years_axis, f_aero, color="#ef4444", lw=2, label="Aerosol Cooling")
        _apply_chart_style(axes[1, 1], title="Anthropogenic Forcing Components",
                           xlabel="Year", ylabel="Forcing (W/m²)")
        axes[1, 1].legend(fontsize=8, framealpha=0.85, edgecolor="#d6e2f0", loc="upper left")

        f_volc = [
            sum(
                s * np.exp(-(y - ys) / d)
                for ys, s, d in [(1963.2, -0.8, 1.2), (1982.3, -1.3, 1.5), (1991.4, -1.8, 1.8)]
                if y >= ys
            )
            for y in years_axis
        ]
        f_osc = [
            best_params[3] * (np.sin(2 * np.pi * y / 3.8) + np.sin(2 * np.pi * y / 5.5))
            for y in years_axis
        ]
        axes[2, 0].fill_between(years_axis, 0, f_volc, color="#64748b", alpha=0.55, label="Volcanic Forcing")
        axes[2, 0].plot(years_axis, f_osc, color="#f59e0b", lw=1.8, label="Internal Oscillation")
        _apply_chart_style(axes[2, 0], title="Natural Forcing Components",
                           xlabel="Year", ylabel="Forcing (W/m²)")
        axes[2, 0].legend(fontsize=8, framealpha=0.85, edgecolor="#d6e2f0")

        anthro = np.array(f_co2) + np.array(f_non_co2) + np.array(f_aero)
        natural = np.array(f_volc) + np.array(f_osc)
        axes[2, 1].plot(years_axis, anthro, color="#1a56a0", lw=2.2, label="Anthropogenic Contribution")
        axes[2, 1].plot(years_axis, natural, color="#0f2744", lw=1.6, ls="--", label="Natural Contribution")
        axes[2, 1].axhline(0, color="#94a3b8", lw=0.8, ls="--")
        _apply_chart_style(
            axes[2, 1],
            title="Relative Contribution: Anthropogenic vs Natural",
            xlabel="Year", ylabel="Forcing (W/m²)",
        )
        axes[2, 1].legend(fontsize=8, framealpha=0.85, edgecolor="#d6e2f0")

        plt.tight_layout()
        st.pyplot(fig)

        render_infobox(
            "해석",
            "모델은 전체적인 장기 온난화 추세를 비교적 잘 재현하지만, 일부 시기에는 과대·과소예측이 나타납니다. "
            "이는 단순화된 강제력 입력과 내부 변동성 표현의 한계에서 비롯될 수 있으며, "
            "장기 추세 설명에는 유용하나 단기 변동 재현에는 제한이 있음을 보여줍니다.",
        )

        sec("일별 상세 보기")
        st.markdown(
            """
st.markdown(
    """
<div style="background:#ffffff; border:1px solid #d0dff0; border-radius:14px; padding:1rem 1.2rem 0.9rem 1.2rem; box-shadow:0 2px 10px rgba(26,86,160,0.06); margin-bottom:1rem;">
<div style="font-size:0.8rem;font-weight:700;color:#64748b;
    text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.6rem;">
    일별 상세 분석 연도 선택
</div>""",
            unsafe_allow_html=True,
        )
        sl_col, btn_col = st.columns([4, 1], gap="medium")
        with sl_col:
            target_year = st.slider(
                "연도", 1925, 2025,
                int(st.session_state.get("main_target_year", 2024)),
                key="main_target_year",
                label_visibility="collapsed",
            )
        with btn_col:
            df_export = pd.DataFrame({
                "Year": years_axis,
                "Observed": current_obs_data,
                "Model_Anomaly": best_global,
                "Deep_Ocean_Temp": best_td,
            })
            csv = df_export.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="CSV 다운로드",
                data=csv,
                file_name="climate_data.csv",
                mime="text/csv",
                use_container_width=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

        y_idx = int(target_year - START_YEAR)
        daily_segment = daily_all[y_idx * 365:(y_idx + 1) * 365]
        fig_d, ax_d = _styled_fig(figsize=(12, 4.5))
        ax_d.fill_between(range(1, 366), daily_segment, np.mean(daily_segment),
                          alpha=0.15, color="#1a56a0")
        ax_d.plot(range(1, 366), daily_segment, color="#1a56a0", lw=1.5)
        ax_d.axhline(np.mean(daily_segment), color="#ef4444", ls="--", lw=1.5,
                     label=f"Annual Mean: {np.mean(daily_segment):.3f}°C")
        _apply_chart_style(
            ax_d,
            title=f"{target_year} Daily Temperature Evolution",
            xlabel="Day of Year",
            ylabel="Temperature Anomaly (°C)",
        )
        ax_d.legend(fontsize=9, framealpha=0.85, edgecolor="#d6e2f0")
        plt.tight_layout()
        st.pyplot(fig_d)


# ═══════════════════════════════════════════════════════════════════════════════
# 페이지: 모델 검증 및 불확실성 정량화
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "모델 검증 및 불확실성 정량화":
    st.query_params["module"] = "uncertainty"
    left_col, main_col = st.columns([1.05, 4.2], gap="large")

    with left_col:
        render_left_panel()
        controls = render_settings(page)

    with main_col:
        diag_obs_choice = controls["diag_obs_choice"]
        diag_obs_data = controls["diag_obs_data"]

        page_header("모델 검증 및 불확실성 정량화",
                    "잔차 진단 · 파라미터 불확실성 범위 · 민감도 분석")

        render_infobox(
            "분석 목적",
            "모델의 잔차 구조, 파라미터 변화에 따른 예측 범위, "
            "특정 파라미터의 민감도를 함께 확인하여 모델 결과의 안정성과 해석 가능성을 점검합니다.",
        )

        with st.spinner(f"{diag_obs_choice} 자료를 기준으로 검증을 수행하는 중입니다..."):
            diag_best_params = get_optimized_params(diag_obs_data)
            diag_best_global, _, _, _, _ = run_model(diag_best_params, diag_obs_data[0])

        residuals = diag_best_global - diag_obs_data
        rmse_diag = np.sqrt(np.mean((diag_best_global - diag_obs_data) ** 2))
        mae_diag = np.mean(np.abs(diag_best_global - diag_obs_data))
        bias_diag = np.mean(diag_best_global - diag_obs_data)

        sec("핵심 지표")
        c1, c2, c3 = st.columns(3)
        with c1:
            render_metric("RMSE", f"{rmse_diag:.3f}", "°C", "예측과 관측의 전체 오차 수준")
        with c2:
            render_metric("MAE", f"{mae_diag:.3f}", "°C", "평균 절대 오차")
        with c3:
            render_metric("Bias", f"{bias_diag:.3f}", "°C", "전반적 편향")

        sec("잔차 진단")
        c1, c2 = st.columns(2)
        with c1:
            fig_rl, ax_rl = _styled_fig(figsize=(10, 4.5))
            ax_rl.fill_between(years_axis, residuals, 0, alpha=0.25,
                               color=["#1a56a0" if r > 0 else "#ef4444" for r in residuals])
            ax_rl.plot(years_axis, residuals, color="#1a56a0", lw=1.8)
            ax_rl.axhline(0, color="#0f2744", lw=0.9)
            _apply_chart_style(ax_rl, title="Residual Time Series",
                               xlabel="Year", ylabel="Residual (°C)")
            plt.tight_layout()
            st.pyplot(fig_rl)
        with c2:
            fig_rh, ax_rh = _styled_fig(figsize=(10, 4.5))
            ax_rh.hist(residuals, bins=15, color="#1a56a0", edgecolor="#ffffff",
                       alpha=0.85, linewidth=0.8)
            ax_rh.axvline(0, color="#ef4444", lw=1.5, ls="--")
            _apply_chart_style(ax_rh, title="Residual Distribution",
                               xlabel="Residual (°C)", ylabel="Frequency")
            plt.tight_layout()
            st.pyplot(fig_rh)

        sec("불확실성 범위")
        rng = np.random.default_rng(42)
        samples = []
        for _ in range(20):
            noisy_params = np.array(diag_best_params) + rng.normal(
                0, [0.08, 0.08, 0.10, 0.01], size=4
            )
            noisy_params[0] = np.clip(noisy_params[0], 0.7, 2.3)
            noisy_params[1] = np.clip(noisy_params[1], 0.5, 2.0)
            noisy_params[2] = np.clip(noisy_params[2], 0.5, 3.5)
            noisy_params[3] = np.clip(noisy_params[3], 0.05, 0.25)
            res_tmp, _, _, _, _ = run_model(noisy_params.tolist(), diag_obs_data[0])
            samples.append(res_tmp)

        samples = np.array(samples)
        mean_path = samples.mean(axis=0)
        std_path = samples.std(axis=0)

        fig_unc, ax_unc = _styled_fig(figsize=(12, 5.2))
        ax_unc.fill_between(years_axis, mean_path - std_path, mean_path + std_path,
                            color="#1a56a0", alpha=0.2, label="Uncertainty Band (±1σ)")
        ax_unc.fill_between(years_axis, mean_path - 2 * std_path, mean_path + 2 * std_path,
                            color="#1a56a0", alpha=0.08, label="Uncertainty Band (±2σ)")
        ax_unc.plot(years_axis, diag_obs_data, color="#0f2744", lw=1.6, ls="--",
                    alpha=0.75, label="Observed")
        ax_unc.plot(years_axis, mean_path, color="#1a56a0", lw=2.2, label="Mean Prediction")
        _apply_chart_style(ax_unc, title="Prediction Uncertainty Envelope",
                           xlabel="Year", ylabel="Temperature Anomaly (°C)")
        ax_unc.legend(fontsize=9, framealpha=0.85, edgecolor="#d6e2f0")
        plt.tight_layout()
        st.pyplot(fig_unc)

        render_infobox(
            "해석",
            "파라미터를 조금씩 변화시켜도 평균 경로는 유지되지만, 후반부로 갈수록 예측 범위가 넓어지는 경향이 나타납니다. "
            "이는 장기 예측일수록 파라미터 선택에 따른 민감도가 커진다는 의미이며, "
            "단일 경로보다 불확실성 범위를 함께 제시하는 것이 더 정직한 해석 방식임을 보여줍니다.",
        )

        sec("민감도 분석")
        sens_options = ["기후 피드백 파라미터", "에어로졸 강도", "해양 열흡수 계수", "ENSO 진폭"]
        sens_param = st.selectbox(
            "민감도 분석 파라미터",
            sens_options,
            index=sens_options.index(st.session_state.get("main_sens_param", sens_options[0])),
            key="main_sens_param",
        )

        param_config = {
            "기후 피드백 파라미터": (np.linspace(0.7, 2.3, 12), 0),
            "에어로졸 강도": (np.linspace(0.5, 2.0, 12), 1),
            "해양 열흡수 계수": (np.linspace(0.5, 3.5, 12), 2),
            "ENSO 진폭": (np.linspace(0.05, 0.25, 12), 3),
        }
        test_range, idx_change = param_config[sens_param]

        sens_results = []
        for val in test_range:
            params = list(diag_best_params)
            params[idx_change] = val
            res_tmp, _, _, _, _ = run_model(params, diag_obs_data[0], end_year=2100, end_co2=550)
            sens_results.append(res_tmp[-1])

        fig_s, ax_s = _styled_fig(figsize=(12, 4.8))
        ax_s.fill_between(test_range, min(sens_results), sens_results, alpha=0.15, color="#1a56a0")
        ax_s.plot(test_range, sens_results, color="#1a56a0", lw=2.2, marker="o",
                  markersize=7, markerfacecolor="#ffffff", markeredgecolor="#1a56a0",
                  markeredgewidth=1.8)
        _apply_chart_style(
            ax_s,
            title=f"Sensitivity of Projected 2100 Warming — {sens_param}",
            xlabel="Parameter Value",
            ylabel="Projected Temperature in 2100 (°C)",
        )
        plt.tight_layout()
        st.pyplot(fig_s)


# ═══════════════════════════════════════════════════════════════════════════════
# 페이지: 기후 모델링 용어 및 개념 정의
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "기후 모델링 용어 및 개념 정의":
    st.query_params["module"] = "glossary"
    left_col, main_col = st.columns([1.05, 4.2], gap="large")

    with left_col:
        render_left_panel()

    with main_col:
        page_header("기후 모델링 용어 및 개념 정의",
                    "모델에서 사용되는 주요 기후학 개념 · 물리 파라미터 · 검증 지표 참고")

        render_infobox(
            "페이지 안내",
            "본 페이지는 모델에서 사용되는 주요 기후학 개념, 물리 파라미터, 검증 지표를 정리한 참고용 자료입니다. "
            "그래프 해석 전에 필요한 개념을 빠르게 확인할 수 있도록 구성했습니다.",
        )

        sec("기후 변화의 원인")
        glossary_forcing = [
            ("온실가스 복사 강제력 (Greenhouse Gas Radiative Forcing)",
             "대기 중 온실가스가 지구 복사에너지의 우주 방출을 억제하여 지표를 따뜻하게 만드는 효과입니다. "
             "이산화탄소 농도가 증가할수록 강제력이 커지며, 본 모델에서는 로그 함수 형태로 반영됩니다.", True),
            ("에어로졸 효과 (Aerosol Effect)",
             "대기 중 에어로졸 입자가 태양복사를 반사하거나 구름의 반사도를 높여 지표를 냉각시키는 효과입니다. "
             "온난화를 일부 상쇄하지만 시기와 지역에 따라 영향이 달라집니다.", False),
            ("화산 강제력 (Volcanic Forcing)",
             "대규모 화산 분출 이후 성층권에 주입된 입자가 태양복사를 차단하여 단기 냉각을 유도하는 효과입니다. "
             "본 모델은 시간에 따라 약해지는 지수 감쇠 형태로 이를 표현합니다.", False),
            ("엘니뇨-남방진동 (ENSO, El Nino-Southern Oscillation)",
             "열대 태평양의 해수면 온도와 대기 순환이 수년 주기로 변동하는 자연 내부 변동성입니다. "
             "장기 온난화 추세와 별개로 특정 연도의 기온을 높이거나 낮출 수 있습니다.", False),
        ]
        for title, body, expanded in glossary_forcing:
            with st.expander(title, expanded=expanded):
                st.write(body)

        sec("기후 시스템의 물리적 반응")
        glossary_physics = [
            ("기온 편차 (Temperature Anomaly)",
             "절대기온이 아니라 기준 기간 평균으로부터 얼마나 벗어났는지를 나타내는 값입니다. "
             "서로 다른 시기와 자료를 비교할 때 널리 사용됩니다."),
            ("열용량 (Heat Capacity)",
             "물질의 온도를 1도 높이는 데 필요한 에너지 양입니다. "
             "바다는 열용량이 커서 육지보다 천천히 가열되고 천천히 냉각됩니다."),
            ("해양 층위 분리 (Ocean Layering)",
             "모델에서 해양을 혼합층과 심해로 구분하여 열 저장과 전달 과정을 표현하는 방식입니다. "
             "이는 해양의 열관성과 장기 온난화 반응을 설명하는 데 중요합니다."),
            ("열 교환 계수 (Heat Exchange Rate)",
             "육지와 바다, 혹은 해양 표층과 심해 사이에 열이 얼마나 빠르게 이동하는지를 나타내는 계수입니다."),
            ("기후 피드백 파라미터 (Climate Feedback Parameter)",
             "기후 시스템이 따뜻해질수록 얼마나 강하게 복사 냉각으로 되돌리려 하는지를 나타내는 계수입니다. "
             "값이 클수록 온난화 억제 효과가 큽니다."),
        ]
        for title, body in glossary_physics:
            with st.expander(title):
                st.write(body)

        sec("모델 평가와 검증 지표")
        glossary_metrics = [
            ("기계학습 최적화 알고리즘 (L-BFGS-B)",
             "관측값과 모델값의 차이가 최소가 되도록 파라미터를 자동 조정하는 수치 최적화 알고리즘입니다."),
            ("오차율 (Error Percentage)",
             "모델 예측이 관측값에 비해 상대적으로 얼마나 높거나 낮은지를 백분율로 나타낸 지표입니다. "
             "관측값이 0에 가까운 구간에서는 왜곡이 커질 수 있습니다."),
            ("평균제곱근오차 (RMSE, Root Mean Squared Error)",
             "예측값과 관측값 차이의 제곱 평균에 제곱근을 취한 지표입니다. "
             "큰 오차에 민감하므로 모델이 특정 시점에서 크게 빗나가는지 평가하는 데 유용합니다."),
            ("평균절대오차 (MAE, Mean Absolute Error)",
             "예측값과 관측값 차이의 절댓값 평균입니다. "
             "평균적으로 몇 도 정도 오차가 나는지를 직관적으로 해석하기 좋습니다."),
            ("편향 (Bias)",
             "예측값에서 관측값을 뺀 차이의 평균입니다. "
             "양수이면 전반적으로 높게, 음수이면 낮게 예측하는 경향을 의미합니다."),
            ("잔차 (Residual)",
             "각 시점에서 모델 예측값과 실제 관측값의 차이입니다. "
             "잔차 패턴을 보면 특정 시기에 구조적인 과대예측 또는 과소예측이 존재하는지 확인할 수 있습니다."),
            ("불확실성 범위 (Uncertainty Band)",
             "모델 파라미터를 조금씩 변화시켜 여러 번 계산했을 때 나타나는 예측 범위입니다. "
             "단일 예측선보다 모델의 불확실성을 더 정직하게 보여줍니다."),
            ("민감도 분석 (Sensitivity Analysis)",
             "하나의 파라미터를 바꾸었을 때 결과가 얼마나 크게 달라지는지 평가하는 분석입니다. "
             "어떤 변수가 모델 출력에 가장 큰 영향을 주는지 파악할 수 있습니다."),
            ("인위적 요인과 자연적 요인의 상대 기여 (Forcing Dominance)",
             "온실가스, 에어로졸, 화산, 내부 변동 등 여러 강제력 요소를 비교하여 "
             "온도 변화에 어떤 요인이 더 크게 작용하는지 해석하는 개념입니다."),
        ]
        for title, body in glossary_metrics:
            with st.expander(title):
                st.write(body)

        st.info(
            "그래프 내부 제목과 축 라벨은 글꼴 호환성을 위해 영어로 유지했고, "
            "UI와 설명문은 한국어로 정리했습니다."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 페이지: 연구 요약 및 보고서
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "연구 요약 및 보고서":
    st.query_params["module"] = "summary"
    left_col, main_col = st.columns([1.05, 4.2], gap="large")

    with left_col:
        render_left_panel()

    with main_col:
        page_header("연구 요약 및 보고서",
                    "연구 목적 · 모델 구조 · 해석 주의점 · 연구 의의")

        render_infobox(
            "연구 목적",
            "본 프로젝트는 관측 자료와 단순화된 물리 기반 기후 모델을 결합하여 역사적 기온 변화를 재현하고, "
            "배출 시나리오와 주요 물리 파라미터 변화에 따라 미래 온난화 경로가 어떻게 달라지는지를 해석하는 것을 목표로 합니다.",
        )
        render_infobox(
            "모델 구조와 물리적 가정",
            "모델은 육지, 해양 혼합층, 심해의 세 층으로 구성된 간이 에너지 균형 구조를 사용합니다. "
            "이산화탄소 복사 강제력, 에어로졸 냉각, 비이산화탄소 인위적 강제력, 화산 강제력, 내부 변동성을 포함하며, "
            "전지구 평균 규모에서 장기 추세와 주요 기여 요인을 해석하는 데 초점을 둡니다.",
        )

        sec("연구 요약")
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            st.markdown(
                """
<div class="pcard">
  <div class="pcard-tag">Summary</div>
  <div class="pcard-title">핵심 분석 구성</div>
  <div class="pcard-body">
    시나리오 기반 장기 온난화 예측, 기후 피드백 및 해양 열흡수 파라미터 실험,
    관측자료와의 적합도 비교, 잔차와 불확실성 범위 검토, 민감도 분석을 하나의 흐름으로 통합했습니다.
  </div>
</div>""",
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                """
<div class="pcard">
  <div class="pcard-tag">Interpretation</div>
  <div class="pcard-title">해석상의 주의점</div>
  <div class="pcard-body">
    본 모델은 정밀 예측 모델이 아니라 해석 중심의 교육·연구용 모델입니다.
    지역별 차이보다 전지구 평균 경향을 다루며, 일부 강제력과 내부 변동성은 단순화된 함수로 표현됩니다.
  </div>
</div>""",
                unsafe_allow_html=True,
            )

        sec("연구 의의")
        st.markdown(
            """
<div class="abstract-box">
  <div class="abstract-label">Research Significance</div>
  <div class="abstract-text">
    이 대시보드는 단순한 시각화 도구를 넘어, 기후 시스템의 핵심 강제력과 반응 변수를
    하나의 흐름 안에서 탐색할 수 있도록 구성되어 있습니다.
    특히 관측자료 비교, 파라미터 실험, 불확실성 정량화를 하나의 인터페이스에 통합함으로써
    학부 수준에서 기후 모델링의 기본 구조와 검증 과정을 체계적으로 학습할 수 있도록 설계되었습니다.
  </div>
</div>""",
            unsafe_allow_html=True,
        )

        sec("보고서 다운로드")
        report_name, report_bytes = load_report_file()
        if report_bytes is not None:
            st.download_button(
                label="분석 리포트 다운로드 (.docx)",
                data=report_bytes,
                file_name=report_name,
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True,
            )
        else:
            st.info(
                "리포트 파일을 찾지 못했습니다. "
                "app.py와 같은 위치에 .docx 파일이 있는지 확인하세요."
            )
