import streamlit as st
import pandas as pd
import pandasql as ps
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import matplotlib.pyplot as plt

# --- Global chart theme: every matplotlib chart in this app (radar, bar charts,
# boxplots, etc.) uses this dark palette instead of matplotlib's white default,
# so charts match the app's dark theme instead of popping up as white cards. ---
plt.rcParams.update({
    "figure.facecolor": "#0F1216",
    "axes.facecolor": "#0F1216",
    "savefig.facecolor": "#0F1216",
    "figure.edgecolor": "#0F1216",
    "axes.edgecolor": "#EDEEF0",
    "axes.labelcolor": "#EDEEF0",
    "xtick.color": "#EDEEF0",
    "ytick.color": "#EDEEF0",
    "text.color": "#EDEEF0",
    "grid.color": "#2A2F3A",
    "legend.facecolor": "#1B2029",
    "legend.edgecolor": "#2A2F3A",
    "legend.labelcolor": "#EDEEF0",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "axes.titleweight": "semibold",
})
import numpy as np
import logging
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors



# Bump this whenever the file changes — it's rendered in the footer, so you can
# tell at a glance whether Streamlit Cloud is actually serving the latest commit
# or still running a previous deploy.
APP_BUILD = "2026-09-15c"

# --- Page Config ---
st.set_page_config(
    layout="wide",
    page_title="IMDb Data & AI Playground",
    page_icon="🎬",
    initial_sidebar_state="collapsed",
)

# --- Theme: "Screening Room" (charcoal-navy + marquee gold, ticket-stub motif) ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg: #0F1216;
    --surface: #171C24;
    --surface-2: #1E242E;
    --border: #2B323E;
    --border-soft: #242A35;
    --text: #ECEEF1;
    --text-muted: #949DAC;
    --gold: #E3A857;
    --gold-soft: rgba(227, 168, 87, 0.13);
    --teal: #4FA88F;
    --brick: #C1524B;
    --radius: 12px;
}

html, body, [data-testid="stAppViewContainer"], .main,
[data-testid="stHeader"], [data-testid="stBottomBlockContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif;
}
[data-testid="stHeader"] { background: transparent !important; }

[data-testid="stAppViewContainer"] .block-container {
    padding-top: 2.1rem;
    padding-bottom: 4rem;
    max-width: 1480px;
}

h1, h2, h3, h4, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    font-family: 'Fraunces', serif !important;
    font-weight: 600;
    letter-spacing: -0.01em;
    color: var(--text) !important;
}
p, span, label, li { color: var(--text); }

/* ------------------------------------------------------------------ *
 * Hero
 * ------------------------------------------------------------------ */
.hero {
    display: flex;
    flex-wrap: wrap;
    align-items: flex-end;
    justify-content: space-between;
    gap: 1.2rem 2rem;
    margin-bottom: 0.9rem;
}
.hero-eyebrow {
    font-size: 0.72rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--gold);
    font-weight: 600;
    margin-bottom: 0.35rem;
}
.hero-title {
    font-family: 'Fraunces', serif;
    font-size: 2.45rem;
    line-height: 1.05;
    font-weight: 700;
    margin: 0 0 0.35rem 0;
    color: var(--text);
}
.hero-subtitle {
    color: var(--text-muted);
    font-size: 0.98rem;
    max-width: 72ch;
    margin: 0;
}
.hero-stats { display: flex; gap: 0.55rem; flex-wrap: wrap; }
.stat {
    background: var(--surface);
    border: 1px solid var(--border-soft);
    border-left: 3px dashed var(--gold);
    border-radius: 0 10px 10px 0;
    padding: 0.5rem 0.85rem;
    min-width: 96px;
}
.stat-value {
    font-family: 'Fraunces', serif;
    font-size: 1.25rem;
    font-weight: 600;
    line-height: 1.1;
    color: var(--text);
}
.stat-label {
    font-size: 0.68rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-top: 0.15rem;
}
.hero-rule {
    height: 2px;
    background: repeating-linear-gradient(90deg, var(--gold) 0 10px, transparent 10px 18px);
    opacity: 0.5;
    margin: 0.2rem 0 0.6rem 0;
    border: none;
}

/* ------------------------------------------------------------------ *
 * Tab-bar navigation (radio widgets restyled as tabs / chips)
 * Radios are used under the hood on purpose: st.tabs renders *every*
 * tab's body on every rerun, which here would mean training models and
 * calling OMDb for all 20 scenarios on each page load.
 * ------------------------------------------------------------------ */
div[role="radiogroup"] {
    gap: 0.1rem !important;
    flex-wrap: wrap;
    border-bottom: 1px solid var(--border);
    margin: 0;
    padding: 0;
}
/* Hide the radio dot, keep the text. Targets whatever sits immediately before
   the label text, so it works across Streamlit's different radio internals. */
div[role="radiogroup"] div:has(+ div[data-testid="stMarkdownContainer"]) { display: none !important; }
div[role="radiogroup"] input[type="radio"] { display: none !important; }

div[role="radiogroup"] > label {
    margin: 0 !important;
    padding: 0.5rem 0.95rem;
    border-radius: 8px 8px 0 0;
    border-bottom: 2px solid transparent;
    cursor: pointer;
    transition: background 0.15s ease, border-color 0.15s ease;
}
div[role="radiogroup"] > label p {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem;
    font-weight: 500;
    color: var(--text-muted) !important;
    margin: 0 !important;
    white-space: nowrap;
}
div[role="radiogroup"] > label:hover { background: rgba(255,255,255,0.035); }
div[role="radiogroup"] > label:hover p { color: var(--text) !important; }
div[role="radiogroup"] > label[data-selected="true"],
div[role="radiogroup"] > label:has(input:checked) {
    border-bottom-color: var(--gold);
    background: linear-gradient(180deg, var(--gold-soft), transparent);
}
div[role="radiogroup"] > label[data-selected="true"] p,
div[role="radiogroup"] > label:has(input:checked) p {
    color: var(--gold) !important;
    font-weight: 600;
}

/* second level: pill chips instead of underlined tabs */
.st-key-nav_feature div[role="radiogroup"] {
    border-bottom: none;
    gap: 0.35rem !important;
    padding-top: 0.65rem;
}
.st-key-nav_feature div[role="radiogroup"] > label {
    border: 1px solid var(--border-soft);
    border-radius: 999px;
    background: var(--surface);
    padding: 0.3rem 0.85rem;
}
.st-key-nav_feature div[role="radiogroup"] > label p { font-size: 0.82rem; }
.st-key-nav_feature div[role="radiogroup"] > label[data-selected="true"],
.st-key-nav_feature div[role="radiogroup"] > label:has(input:checked) {
    background: var(--gold-soft);
    border-color: var(--gold);
}

/* ------------------------------------------------------------------ *
 * Scenario headers
 * ------------------------------------------------------------------ */
.sc-head { margin: 1.1rem 0 0 0; }
.sc-eyebrow {
    display: inline-block;
    font-size: 0.68rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    font-weight: 600;
    color: var(--gold);
    background: var(--gold-soft);
    border: 1px solid rgba(227,168,87,0.28);
    border-radius: 999px;
    padding: 0.15rem 0.6rem;
    margin-bottom: 0.5rem;
}
.sc-title {
    font-family: 'Fraunces', serif;
    font-size: 1.72rem;
    font-weight: 600;
    line-height: 1.15;
    color: var(--text);
    margin-bottom: 0.15rem;
}
.sc-divider {
    height: 1px;
    background: var(--border);
    border: none;
    margin: 0.7rem 0 0.1rem 0;
}
/* the blurb that follows a scenario header */
[data-testid="stElementContainer"]:has(.sc-head) + [data-testid="stElementContainer"] p,
[data-testid="stElementContainer"]:has(.sc-head) + [data-testid="stElementContainer"] li {
    color: var(--text-muted) !important;
    font-size: 0.93rem;
}

.section-label {
    font-size: 0.72rem;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    font-weight: 600;
    color: var(--text-muted);
    margin: 1.4rem 0 0.5rem 0;
}

/* ------------------------------------------------------------------ *
 * Widgets
 * ------------------------------------------------------------------ */

/* Table grids follow the dark theme set in .streamlit/config.toml — they just
   get a card frame here so they don't float loose on the page background. */
[data-testid="stDataFrame"], [data-testid="stTable"] {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid var(--border);
    background-color: var(--surface);
}

[data-testid="stMetric"] {
    height: 100%;
    background: var(--surface);
    border: 1px solid var(--border-soft);
    border-left: 3px dashed var(--gold);
    border-radius: 0 10px 10px 0;
    padding: 0.85rem 1rem;
}
[data-testid="stMetricLabel"] p {
    color: var(--text-muted) !important;
    font-size: 0.74rem !important;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
[data-testid="stMetricValue"] { color: var(--text) !important; font-family: 'Fraunces', serif; }

.stButton > button {
    background-color: var(--gold);
    color: #1B140A;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.9rem;
    padding: 0.5rem 1.2rem;
    transition: filter 0.15s ease, transform 0.15s ease;
}
.stButton > button p, .stButton > button div, .stButton > button span {
    color: #1B140A !important;
    font-weight: 600;
}
.stButton > button:hover { filter: brightness(1.08); color: #1B140A; transform: translateY(-1px); }
.stButton > button:focus:not(:active) { color: #1B140A; }

/* Input surfaces. Two selector families on purpose: the first matches current
   Streamlit builds (react-aria), the second older baseweb-based ones. */
[data-testid="stTextInput"] div[role="group"],
[data-testid="stNumberInput"] div[role="group"],
[data-testid="stSelectbox"] div[role="group"],
[data-testid="stMultiSelect"] div[role="group"],
.stTextArea textarea, .stTextInput input,
.stSelectbox div[data-baseweb="select"] > div,
.stMultiSelect div[data-baseweb="select"] > div {
    background-color: var(--surface) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}
[data-testid="stSelectbox"] input, [data-testid="stMultiSelect"] input,
[data-testid="stTextInput"] input {
    background: transparent !important;
    color: var(--text) !important;
}
.stTextArea textarea {
    font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace !important;
    font-size: 0.79rem !important;
    line-height: 1.5 !important;
}

/* Cards: st.container(border=True), expanders, alerts */
[data-testid="stExpander"], [data-testid="stVerticalBlockBorderWrapper"] > div > [data-testid="stVerticalBlock"] {
    background-color: var(--surface);
    border-radius: var(--radius);
}
[data-testid="stExpander"] {
    border: 1px solid var(--border-soft);
    overflow: hidden;
}
[data-testid="stExpander"] summary:hover p { color: var(--gold) !important; }
[data-testid="stExpander"] summary p { font-size: 0.88rem; font-weight: 500; }

[data-testid="stAlert"] { border-radius: 10px; border: 1px solid var(--border); }
[data-testid="stCaptionContainer"] p, .stCaption p { color: var(--text-muted) !important; }
code { color: var(--gold) !important; background: rgba(227,168,87,0.08) !important; }
hr { border-color: var(--border); }

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #12161D;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
.side-title {
    font-family: 'Fraunces', serif;
    font-size: 1.05rem;
    color: var(--gold) !important;
    margin: 0.2rem 0 0.6rem 0;
}
.side-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.82rem;
    padding: 0.32rem 0;
    border-bottom: 1px dashed var(--border-soft);
}
.side-row span:last-child { color: var(--gold) !important; font-weight: 600; }
.side-note { color: var(--text-muted) !important; font-size: 0.78rem; line-height: 1.5; }

.app-footer {
    margin-top: 3rem;
    padding-top: 1rem;
    border-top: 1px dashed var(--border);
    color: var(--text-muted);
    font-size: 0.78rem;
    display: flex;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# --- Load Excel files ---
_load_error = None
try:
    IMDB_Ratings = pd.read_excel("imdbratings.xlsx")
    IMDB_Ratings_2019 = pd.read_excel("imdbratings2019onwards.xlsx")  # New workbook
    My_Ratings = pd.read_excel("myratings.xlsx")
    Votes = pd.read_excel("votes.xlsx")  # Optional votes source
except Exception as e:
    _load_error = str(e)
    IMDB_Ratings = pd.DataFrame()
    IMDB_Ratings_2019 = pd.DataFrame()
    My_Ratings = pd.DataFrame()
    Votes = pd.DataFrame()

# --- Clean unnamed columns ---
def clean_unnamed_columns(df):
    return df.loc[:, ~df.columns.str.contains('^Unnamed')]

IMDB_Ratings = clean_unnamed_columns(IMDB_Ratings)
IMDB_Ratings_2019 = clean_unnamed_columns(IMDB_Ratings_2019)
My_Ratings = clean_unnamed_columns(My_Ratings)
Votes = clean_unnamed_columns(Votes)

# --- Append and remove duplicates ---
if not IMDB_Ratings_2019.empty:
    IMDB_Ratings = pd.concat([IMDB_Ratings, IMDB_Ratings_2019], ignore_index=True)
    IMDB_Ratings = IMDB_Ratings.drop_duplicates(subset=["Movie ID"], keep="last")

# --- Merge votes ---
if not Votes.empty:
    IMDB_Ratings = IMDB_Ratings.merge(Votes, on="Movie ID", how="left")
    My_Ratings = My_Ratings.merge(Votes, on="Movie ID", how="left")

# --- Vote-count floor: this app only covers films with a meaningful number
# of IMDb votes, so a handful of stray low-vote titles don't skew stats or
# ML training. Applied globally, right after the votes join, so every
# scenario (and the Dashboard) works off the same filtered dataset. A film
# with no vote count at all (not present in votes.xlsx) is excluded too,
# since there's no way to confirm it clears the bar. ---
MIN_VOTES = 10000
if "Num Votes" in IMDB_Ratings.columns:
    IMDB_Ratings = IMDB_Ratings[IMDB_Ratings["Num Votes"] > MIN_VOTES]
if "Num Votes" in My_Ratings.columns:
    My_Ratings = My_Ratings[My_Ratings["Num Votes"] > MIN_VOTES]

# --- Hero: title on the left, headline numbers from the loaded data on the right ---
_n_rated = int(My_Ratings["Your Rating"].notna().sum()) if "Your Rating" in My_Ratings.columns else 0
_n_catalog = len(IMDB_Ratings)
_avg_rating = (
    f"{My_Ratings['Your Rating'].mean():.1f}"
    if "Your Rating" in My_Ratings.columns and _n_rated else "—"
)

st.markdown(f"""
<div class="hero">
  <div>
    <div class="hero-eyebrow">A personal screening room</div>
    <div class="hero-title">IMDb Data &amp; AI Playground 🎬</div>
    <p class="hero-subtitle">Twenty ways to browse, analyse and predict my own movie ratings —
    SQL, statistics, machine learning, NLP and a live ratings monitor, all over the same two spreadsheets.</p>
  </div>
  <div class="hero-stats">
    <div class="stat"><div class="stat-value">{_n_rated:,}</div><div class="stat-label">Films rated</div></div>
    <div class="stat"><div class="stat-value">{_avg_rating}</div><div class="stat-label">My average</div></div>
    <div class="stat"><div class="stat-value">{_n_catalog:,}</div><div class="stat-label">In catalogue</div></div>
  </div>
</div>
<hr class="hero-rule">
""", unsafe_allow_html=True)

if _load_error:
    st.error(f"Error loading Excel files: {_load_error}")

# --- Sidebar: reference panel only. Every control now lives next to the
# scenario it belongs to, so nothing appears or disappears up there
# depending on which scenario happens to be open. ---
st.sidebar.markdown('<div class="side-title">🎟️ About this playground</div>', unsafe_allow_html=True)
st.sidebar.markdown(f"""
<div class="side-row"><span>Films I've rated</span><span>{_n_rated:,}</span></div>
<div class="side-row"><span>Titles in catalogue</span><span>{_n_catalog:,}</span></div>
<div class="side-row"><span>Vote floor</span><span>{MIN_VOTES:,}</span></div>
""", unsafe_allow_html=True)
st.sidebar.markdown(
    '<p class="side-note">Every scenario works off the same filtered dataset: films with more '
    'than {:,} IMDb votes, so a handful of obscure titles can\'t skew the statistics or the '
    'model training.</p>'.format(MIN_VOTES),
    unsafe_allow_html=True,
)
st.sidebar.markdown(
    '<p class="side-note">Sources: my IMDb ratings export, the IMDb catalogue export, and the '
    'OMDb API for live ratings, plots and posters.</p>',
    unsafe_allow_html=True,
)

# --- Scenarios: grouped into a two-level tab bar across the top.
# NOTE: the underlying option strings are left exactly as before so every
# `if scenario == "..."` check further down the file keeps working unchanged —
# only the labels shown on the tabs are shortened. ---
SCENARIO_CATEGORIES = {
    "🏠  Dashboard": [
        "0 – Dashboard",
    ],
    "🔍  Discover": [
        "1 – Highlight Disagreements",
        "2 – Hybrid Recommendations",
        "3 – Top Unseen Films by Decade",
        "9 – Natural-Language Film Q&A Assistant",
        "15 – Personalized Watchlist Ranker",
        "16 – Similar Films Finder",
        "19 – Tonight's Pick Roulette",
    ],
    "📊  Stats & Insights": [
        "4 – Statistical Insights by Genre (Agreement)",
        "5 – Statistical Insights by Director (t-test)",
        "6 – Review Analysis (Sentiment, Subjectivity)",
        "17 – Taste Profile Radar",
        "20 – Ratings Timeline by Release Decade",
    ],
    "🤖  ML & AI": [
        "10 – Predict My Ratings (ML)",
        "11 – Model Evaluation (Feature Importance)",
        "12 – Feature Hypothesis Testing",
        "13 – Semantic Genre & Recommendations (Deep Learning / NLP)",
        "18 – Prediction Outlier Detector",
    ],
    "🕸️  Media & Graph": [
        "7 – Poster Image Analysis (OMDb API)",
        "8 – Graph Based Movie Relationships",
    ],
    "📡  Live Monitor": [
        "14 – Live Ratings Monitor (Scheduled + on-demand)",
    ],
}

# Short labels for the second-level chips — the full names are far too long to
# sit in a row, but the canonical strings above stay untouched.
SCENARIO_LABELS = {
    "1 – Highlight Disagreements": "1 · Disagreements",
    "2 – Hybrid Recommendations": "2 · Recommendations",
    "3 – Top Unseen Films by Decade": "3 · Unseen by decade",
    "4 – Statistical Insights by Genre (Agreement)": "4 · Genre agreement",
    "5 – Statistical Insights by Director (t-test)": "5 · Director t-test",
    "6 – Review Analysis (Sentiment, Subjectivity)": "6 · Review sentiment",
    "7 – Poster Image Analysis (OMDb API)": "7 · Poster mood",
    "8 – Graph Based Movie Relationships": "8 · Relationship graph",
    "9 – Natural-Language Film Q&A Assistant": "9 · Film Q&A",
    "10 – Predict My Ratings (ML)": "10 · Predict ratings",
    "11 – Model Evaluation (Feature Importance)": "11 · Feature importance",
    "12 – Feature Hypothesis Testing": "12 · Hypothesis testing",
    "13 – Semantic Genre & Recommendations (Deep Learning / NLP)": "13 · Semantic genre",
    "14 – Live Ratings Monitor (Scheduled + on-demand)": "14 · Live monitor",
    "15 – Personalized Watchlist Ranker": "15 · Watchlist ranker",
    "16 – Similar Films Finder": "16 · Similar films",
    "17 – Taste Profile Radar": "17 · Taste radar",
    "18 – Prediction Outlier Detector": "18 · Outliers",
    "19 – Tonight's Pick Roulette": "19 · Roulette",
    "20 – Ratings Timeline by Release Decade": "20 · By decade",
}

category = st.radio(
    "Section",
    list(SCENARIO_CATEGORIES.keys()),
    key="nav_category",
    horizontal=True,
    label_visibility="collapsed",
)

_options = SCENARIO_CATEGORIES[category]
if len(_options) == 1:
    # A category with a single entry (the Dashboard) doesn't need a second row.
    scenario = _options[0]
else:
    _labels = [SCENARIO_LABELS[s] for s in _options]
    # Switching category leaves the previous chip selection pointing at an
    # option that no longer exists — drop it so the first chip is picked.
    if st.session_state.get("nav_feature") not in _labels:
        st.session_state.pop("nav_feature", None)
    _chosen = st.radio(
        "Feature",
        _labels,
        key="nav_feature",
        horizontal=True,
        label_visibility="collapsed",
    )
    scenario = _options[_labels.index(_chosen)]


def scenario_header(number, title, blurb=None):
    """Consistent title block for every scenario: numbered badge, display
    title, and a thin rule separating the heading from the controls."""
    eyebrow = f"Scenario {number}" if str(number).isdigit() else str(number)
    st.markdown(
        f'<div class="sc-head">'
        f'<div class="sc-eyebrow">{eyebrow}</div>'
        f'<div class="sc-title">{title}</div>'
        f'</div>'
        f'<hr class="sc-divider">',
        unsafe_allow_html=True,
    )
    if blurb:
        st.markdown(f'<p style="color:var(--text-muted);font-size:0.93rem;">{blurb}</p>',
                    unsafe_allow_html=True)


def section_label(text):
    """Small uppercase label used to break a long scenario into sections."""
    st.markdown(f'<div class="section-label">{text}</div>', unsafe_allow_html=True)


# --- Dashboard (start screen): KPI cards, filterable charts, and the raw
# data tables. It's the first category/scenario in the sidebar, so it's what
# shows before you click into any numbered scenario (e.g. "1 – Highlight
# Disagreements") — clicking a scenario switches `scenario` away from
# "0 – Dashboard" and this whole block is skipped in favor of that scenario's
# own block further down. ---
if scenario == "0 – Dashboard":
    # Default table views (unfiltered) - overridden below with the filtered
    # versions once the filter widgets exist, so the tables at the bottom
    # of the page always reflect whatever's currently selected above.
    _imdb_view = IMDB_Ratings.copy()
    _my_view = My_Ratings.copy()

    if not My_Ratings.empty and "Your Rating" in My_Ratings.columns:
        # KPIs and charts here come only from myratings.xlsx — it already
        # carries IMDb Rating, Genre, Director, Year and Runtime per title,
        # so there's no merge against the larger IMDB_Ratings catalog (that
        # merge was also the source of the earlier count mismatches).
        _compare = My_Ratings[My_Ratings["Your Rating"].notna()].copy()
        total_rated = len(_compare)
        if total_rated:
            # Declared here so the KPI cards render at the top of the page,
            # but filled in further down once `_filtered` is known — that
            # way the numbers react to the filters below instead of always
            # showing the full unfiltered total.
            kpi_placeholder = st.container()

            section_label("Explore ratings")
            st.caption("Filter your rated films — the KPIs, charts, and tables below update together.")

            _all_genres = sorted({
                g.strip() for sub in _compare["Genre"].dropna().str.split(",") for g in sub if g.strip()
            })
            _all_directors = sorted(_compare["Director"].dropna().unique().tolist())
            _year_series = pd.to_numeric(_compare["Year"], errors="coerce").dropna()
            _min_year = int(_year_series.min()) if not _year_series.empty else 1950
            _max_year = int(_year_series.max()) if not _year_series.empty else 2026
            _min_rating = int(_compare["Your Rating"].min())
            _max_rating = int(_compare["Your Rating"].max())

            with st.container(border=True):
                f1, f2, f3, f4 = st.columns(4)
                with f1:
                    landing_genres = st.multiselect(
                        "Genre", _all_genres, default=[], key="landing_genre_filter",
                        placeholder="All genres",
                    )
                with f2:
                    landing_directors = st.multiselect(
                        "Director", _all_directors, default=[], key="landing_director_filter",
                        placeholder="All directors",
                    )
                with f3:
                    landing_year_range = st.slider(
                        "Release year", _min_year, _max_year, (_min_year, _max_year),
                        key="landing_year_filter"
                    )
                with f4:
                    landing_rating_range = st.slider(
                        "My rating", _min_rating, _max_rating, (_min_rating, _max_rating),
                        key="landing_rating_filter"
                    )

            _filtered = _compare.copy()
            _filtered["Year"] = pd.to_numeric(_filtered["Year"], errors="coerce")
            _filtered = _filtered[
                (_filtered["Year"] >= landing_year_range[0]) & (_filtered["Year"] <= landing_year_range[1])
            ]
            _filtered = _filtered[
                (_filtered["Your Rating"] >= landing_rating_range[0])
                & (_filtered["Your Rating"] <= landing_rating_range[1])
            ]
            if landing_genres:
                _pattern = "|".join(g for g in landing_genres)
                _filtered = _filtered[_filtered["Genre"].str.contains(_pattern, case=False, na=False)]
            if landing_directors:
                _filtered = _filtered[_filtered["Director"].isin(landing_directors)]

            with kpi_placeholder:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Films rated", f"{len(_filtered):,}")
                if len(_filtered):
                    _kpi_avg = _filtered["Your Rating"].mean()
                    _kpi_imdb = _filtered["IMDb Rating"].mean()
                    _kpi_gap = _kpi_avg - _kpi_imdb
                    _kpi_agreement = ((_filtered["Your Rating"] - _filtered["IMDb Rating"]).abs() <= 1).mean() * 100
                    c2.metric("My average", f"{_kpi_avg:.1f}", delta=f"{_kpi_gap:+.1f} vs IMDb")
                    c3.metric("IMDb average", f"{_kpi_imdb:.1f}")
                    c4.metric("Within 1 point of IMDb", f"{_kpi_agreement:.0f}%")
                else:
                    c2.metric("My average", "—")
                    c3.metric("IMDb average", "—")
                    c4.metric("Within 1 point of IMDb", "—")

            if _filtered.empty:
                st.info("No films match these filters — widen a range or clear a filter.")
            else:
                section_label("Top 15 watched directors, by average rating")
                _dir_stats = (
                    _filtered.dropna(subset=["Director"]).groupby("Director")["Your Rating"]
                    .agg(["mean", "count"]).reset_index()
                )
                # Require at least 2 rated films so a single high score from
                # a one-off director doesn't dominate the ranking, then rank
                # by how many films were watched (not by score) since this
                # is framed as "most watched", with average rating as the
                # value plotted on the radar.
                _dir_stats = _dir_stats[_dir_stats["count"] >= 2]
                _dir_stats = _dir_stats.sort_values("count", ascending=False).head(15)
                _dir_stats = _dir_stats.sort_values("mean", ascending=False)

                if _dir_stats.empty:
                    st.info("No director has 2+ rated films in this selection yet.")
                else:
                    _categories = _dir_stats["Director"].tolist()
                    _values = _dir_stats["mean"].tolist()

                    # Zoom the radial axis into the actual spread of the data
                    # instead of a fixed 0-10 scale, same approach as the
                    # Taste Profile Radar scenario, so differences between
                    # directors are actually visible.
                    _lo = max(0, min(_values) - 1)
                    _hi = min(10, max(_values) + 1)

                    _values_closed = _values + _values[:1]
                    _angles = np.linspace(0, 2 * np.pi, len(_categories), endpoint=False).tolist()
                    _angles_closed = _angles + _angles[:1]

                    _fig, _ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True))
                    _ax.plot(_angles_closed, _values_closed, color="#4FA88F", linewidth=2)
                    _ax.fill(_angles_closed, _values_closed, color="#4FA88F", alpha=0.22)
                    _ax.set_xticks(_angles)
                    _ax.set_xticklabels(_categories, fontsize=8)
                    _ax.set_ylim(_lo, _hi)
                    _ax.tick_params(axis='y', labelsize=6.5)
                    _ax.spines['polar'].set_color("#2A2F3A")
                    _ax.grid(color="#2A2F3A", linewidth=0.7)

                    for _angle, _value in zip(_angles, _values):
                        _ax.annotate(
                            f"{_value:.1f}", xy=(_angle, _value), fontsize=6.5, color="#EDEEF0",
                            ha="center", va="bottom"
                        )

                    _fig.tight_layout()

                    _chart_col, _table_col = st.columns([1.15, 1])
                    with _chart_col:
                        st.pyplot(_fig)
                        st.caption(
                            f"Axis zoomed to {_lo:.1f}–{_hi:.1f} (not 0–10) — ranked among the 15 "
                            f"directors you've watched the most (2+ films), ordered by your average "
                            f"rating of them."
                        )
                    with _table_col:
                        st.dataframe(
                            _dir_stats.rename(
                                columns={"mean": "Avg rating", "count": "Films watched"}
                            ).round(2).reset_index(drop=True),
                            width="stretch", height=430,
                        )
                    plt.close(_fig)

            # Carry the same genre/director/year filters over to the raw
            # tables below (the rating filter only applies to My Ratings —
            # IMDB_Ratings covers the whole catalog, not just films you've
            # rated).
            _my_view = _filtered.copy()

            if not _imdb_view.empty:
                if "Year" in _imdb_view.columns:
                    _imdb_view["Year"] = pd.to_numeric(_imdb_view["Year"], errors="coerce")
                    _imdb_view = _imdb_view[
                        (_imdb_view["Year"] >= landing_year_range[0])
                        & (_imdb_view["Year"] <= landing_year_range[1])
                    ]
                if landing_genres and "Genre" in _imdb_view.columns:
                    _imdb_view = _imdb_view[_imdb_view["Genre"].str.contains(_pattern, case=False, na=False)]
                if landing_directors and "Director" in _imdb_view.columns:
                    _imdb_view = _imdb_view[_imdb_view["Director"].isin(landing_directors)]

    # --- Data tables (reflect the filters above, if any were applied) ---
    with st.expander("📋 Browse IMDb Ratings table"):
        if not IMDB_Ratings.empty:
            st.dataframe(
                _imdb_view.drop(columns=["Movie URL"], errors="ignore"),
                width="stretch", height=400
            )
        else:
            st.warning("IMDb Ratings table is empty or failed to load.")

    with st.expander("📋 Browse My Ratings table"):
        if not My_Ratings.empty:
            _my_view = _my_view.copy()
            _my_view['Year_Sort'] = pd.to_numeric(_my_view['Year'], errors='coerce')
            _my_view_sorted = _my_view.sort_values(by="Year_Sort", ascending=False)
            # Rename column only for display
            display_ratings = _my_view_sorted.rename(columns={"Your Rating": "My Ratings"})
            display_ratings = display_ratings.drop(columns=['Year_Sort', 'URL'], errors='ignore')
            st.dataframe(display_ratings, width="stretch", height=400)
        else:
            st.warning("My Ratings table is empty or failed to load.")


# --- Scenario 1: SQL Playground ---
if scenario == "1 – Highlight Disagreements":
    scenario_header("1", "Highlight Disagreements")
    st.write("Movies where my rating differs from IMDb by more than 2 points.")

    default_query_1 = """SELECT 
       pr.Title,
       pr.[Your Rating] AS [My Rating],
       ir.[IMDb Rating],
       ABS(CAST(pr.[Your Rating] AS FLOAT) - CAST(ir.[IMDb Rating] AS FLOAT)) AS Rating_Diff,
       CASE 
            WHEN pr.[Your Rating] > ir.[IMDb Rating] THEN 'I Liked More'
            ELSE 'I Liked Less'
       END AS Disagreement_Type
FROM My_Ratings pr
JOIN IMDB_Ratings ir
    ON pr.[Movie ID] = ir.[Movie ID]
WHERE ABS(CAST(pr.[Your Rating] AS FLOAT) - CAST(ir.[IMDb Rating] AS FLOAT)) > 2
ORDER BY Rating_Diff DESC, ir.[Num Votes] DESC
LIMIT 1000;"""

    with st.expander("🛠️ View / edit the underlying code", expanded=False):
        user_query = st.text_area("Enter SQL query:", default_query_1, height=360, key="sql1")
    if st.button("Run SQL Query – Find my disagreements", key="run_sql1"):
        try:
            result = ps.sqldf(user_query, {"IMDB_Ratings": IMDB_Ratings, "My_Ratings": My_Ratings})
            st.dataframe(result, width="stretch", height=560)
        except Exception as e:
            st.error(f"Error in SQL query: {e}")

# --- Scenario 2: SQL Playground ---
if scenario == "2 – Hybrid Recommendations":
    scenario_header("2", "Hybrid Recommendations")
    st.write("""
    Recommend movies I haven't seen yet with a bonus point system:  
    - Director I liked before → +1 point  
    - Genre is Comedy or Drama → +0.5  
    - Other genres → +0.2
    """)

    default_query_2 = """SELECT ir.Title,
       ir.[IMDb Rating],
       ir.Director,
       ir.Genre,
       ir.Year,
       CASE WHEN ir.Director IN (SELECT DISTINCT Director FROM My_Ratings WHERE [Your Rating] >= 7) THEN 1 ELSE 0 END AS Director_Bonus,
       CASE WHEN ir.Genre IN ('Comedy','Drama') THEN 0.5 ELSE 0.2 END AS Genre_Bonus,
       ir.[IMDb Rating] 
       + CASE WHEN ir.Director IN (SELECT DISTINCT Director FROM My_Ratings WHERE [Your Rating] >= 7) THEN 1 ELSE 0 END
       + CASE WHEN ir.Genre IN ('Comedy','Drama') THEN 0.5 ELSE 0.2 END AS Recommendation_Score
FROM IMDB_Ratings ir
LEFT JOIN My_Ratings pr
    ON ir.[Movie ID] = pr.[Movie ID]
WHERE pr.[Your Rating] IS NULL
  AND ir.[Num Votes] > 40000
ORDER BY Recommendation_Score DESC
LIMIT 10000;"""

    with st.expander("🛠️ View / edit the underlying code", expanded=False):
        user_query = st.text_area("Enter SQL query:", default_query_2, height=360, key="sql2")
    if st.button("Run SQL Query – Recommend movies", key="run_sql2"):
        try:
            result = ps.sqldf(user_query, {"IMDB_Ratings": IMDB_Ratings, "My_Ratings": My_Ratings})
            st.dataframe(result, width="stretch", height=560)
        except Exception as e:
            st.error(f"Error in SQL query: {e}")



# --- Scenario 3: SQL Playground ---
if scenario == "3 – Top Unseen Films by Decade":
    scenario_header("3", "Top Unseen Films by Decade")
    st.write("""
    Shows the highest-rated unseen films grouped by decade.  
    Uses Python deduplication and limits results to the top 20 per decade.
    """)

    # Cleaner SQL – no redundant CTE
    default_query_3 = """
SELECT *
FROM (
    SELECT ir.[Movie ID], 
           ir.Title,
           ir.[IMDb Rating],
           ir.[Num Votes],
           ir.Genre,
           ir.Director,
           ir.Year,
           (ir.Year / 10) * 10 AS Decade,
           ROW_NUMBER() OVER (
               PARTITION BY (ir.Year / 10) * 10 
               ORDER BY ir.[IMDb Rating] DESC, ir.[Num Votes] DESC
           ) AS RankInDecade
    FROM IMDB_Ratings ir
    LEFT JOIN My_Ratings pr
        ON ir.[Movie ID] = pr.[Movie ID]
    WHERE pr.[Your Rating] IS NULL
      AND ir.[Num Votes] > 50000
) ranked
WHERE RankInDecade <= 20
ORDER BY Decade, [IMDb Rating] DESC, [Num Votes] DESC;
"""

    # Text area to allow user edits
    with st.expander("🛠️ View / edit the underlying code", expanded=False):
        user_query = st.text_area("Enter SQL query:", default_query_3, height=380, key="sql3")

    # Run button
    if st.button("Run SQL Query – Top unseen films", key="run_sql3"):
        try:
            result = ps.sqldf(user_query, {"IMDB_Ratings": IMDB_Ratings, "My_Ratings": My_Ratings})
            st.dataframe(result, width="stretch", height=560)
        except Exception as e:
            st.error(f"Error in SQL query: {e}")



# --- Scenario 9: Python ML ---
if scenario == "10 – Predict My Ratings (ML)":
    scenario_header("10", "Predict My Ratings (ML)")
    st.write("""
    Predict my ratings for unseen movies using a machine learning model.

    **How it works:**
    1. The model uses my existing ratings (`My_Ratings`) as training data.
    2. Features used include:  
       - IMDb Rating  
       - Genre  
       - Director  
       - Year of release  
       - Number of votes
    3. A Random Forest Regressor learns patterns from the movies I've already rated.
    4. The model predicts how I might rate movies I haven't seen yet (`Predicted Rating`).

    """)

    ml_code = '''
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

df_ml = IMDB_Ratings.merge(My_Ratings[['Movie ID','Your Rating']], on='Movie ID', how='left')
train_df = df_ml[df_ml['Your Rating'].notna()]
predict_df = df_ml[df_ml['Your Rating'].isna()]

categorical_features = ['Genre', 'Director']
numerical_features = ['IMDb Rating', 'Num Votes', 'Year']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', Pipeline([
            ('impute', SimpleImputer(strategy='constant', fill_value='Unknown')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features),
        ('num', SimpleImputer(strategy='median'), numerical_features)
    ]
)

model = Pipeline([
    ('prep', preprocessor),
    ('reg', RandomForestRegressor(n_estimators=100, random_state=42))
])


X_train = train_df[categorical_features + numerical_features]
y_train = train_df['Your Rating']
model.fit(X_train, y_train)
X_pred = predict_df[categorical_features + numerical_features]
predict_df['Predicted Rating'] = model.predict(X_pred)
predict_df
'''

    section_label("Model options")
    _o1, _o2 = st.columns(2)
    with _o1:
        min_votes = st.slider("Minimum IMDb votes", 0, 500000, 50000, step=5000, key="s10_min_votes")
    with _o2:
        top_n = st.slider("Number of top predictions", 5, 50, 30, step=5, key="s10_top_n")

    with st.expander("🛠️ View / edit the underlying code", expanded=False):
        user_ml_code = st.text_area("Python ML Code (editable)", ml_code, height=420)

    if st.button("Run Python ML Code", key="run_ml"):
        try:
            local_vars = {"IMDB_Ratings": IMDB_Ratings, "My_Ratings": My_Ratings}
            exec(user_ml_code, {}, local_vars)
            predict_df = local_vars['predict_df']
            predict_df = predict_df[predict_df['Num Votes'] >= min_votes]
            st.dataframe(
                predict_df[['Title','IMDb Rating','Genre','Director','Predicted Rating']]
                .sort_values(by='Predicted Rating', ascending=False)
                .head(top_n)
                .reset_index(drop=True)
            )
        except Exception as e:
            st.error(f"Error running ML code: {e}")




# --- Scenario 4: Statistical Insights ---
if scenario == "4 – Statistical Insights by Genre (Agreement)":
    scenario_header("4", "Statistical Insights by Genre")
    st.write("""
    This analysis measures how often my ratings align with IMDb ratings **within a tolerance band of ±1 point**.  
    Results are grouped by genre, showing agreements, disagreements, and overall percentages.
    """)

    stats_code = '''
df_compare = IMDB_Ratings.merge(
    My_Ratings[['Movie ID','Your Rating']],
    on='Movie ID', how='inner'
)

df_compare['Agreement'] = (
    (df_compare['Your Rating'] - df_compare['IMDb Rating']).abs() <= 1
)

genre_agreement = (
    df_compare.groupby('Genre')
    .agg(
        Total_Movies=('Movie ID','count'),
        Agreements=('Agreement','sum')
    )
    .reset_index()
)

genre_agreement['Disagreements'] = (
    genre_agreement['Total_Movies'] - genre_agreement['Agreements']
)
genre_agreement['Agreement_%'] = (
    genre_agreement['Agreements'] / genre_agreement['Total_Movies'] * 100
).round(2)

genre_agreement.sort_values(by='Agreement_%', ascending=False)
'''

    # Editable code box
    with st.expander("🛠️ View / edit the underlying code", expanded=False):
        user_stats_code = st.text_area("Python Statistical Code (editable)", stats_code, height=380)

    if st.button("Run Statistical Analysis", key="run_stats5"):
        try:
            # Run the code entered in the text area
            local_vars = {"IMDB_Ratings": IMDB_Ratings, "My_Ratings": My_Ratings}
            exec(user_stats_code, {}, local_vars)

            # Retrieve dataframe if created
            if "genre_agreement" in local_vars:
                st.dataframe(local_vars["genre_agreement"], width="stretch", height=460)
            else:
                st.warning("No output dataframe named 'genre_agreement' was produced. Please check your code.")

        except Exception as e:
            st.error(f"Error running Statistical Analysis code: {e}")




# --- Scenario 5: Statistical Insights (t-test per Director) ---
if scenario == "5 – Statistical Insights by Director (t-test)":
    scenario_header("5", "Statistical Insights by Director (t-test)")
    st.write("""
This analysis compares my ratings with IMDb ratings on a director-by-director basis using a **paired t-test**.  
The test checks whether the differences between my ratings and IMDb’s are statistically significant for each director.  

- **t-statistic**: shows the size and direction of the difference (positive = I rate higher than IMDb, negative = I rate lower).  
- **p-value**: shows whether the difference is statistically significant or could be due to chance. p < 0.05 (significant) → Unlikely the difference is due to chance. I consistently rate this director higher or lower than IMDb. 
""")

    # Kept next to the scenario it belongs to rather than in the sidebar.
    section_label("Test options")
    min_movies = st.slider(
        "Minimum movies per director for t-test", 2, 10, 5, key="s5_min_movies"
    )

    # Editable t-test code
    ttest_code_director = f'''
from scipy.stats import ttest_rel
import numpy as np
import pandas as pd

df_ttest = IMDB_Ratings.merge(
    My_Ratings[['Movie ID','Your Rating']],
    on='Movie ID', how='inner'
)

results = []

for director, group in df_ttest.groupby('Director'):
    n = len(group)
    if n >= {min_movies}:
        differences = group['Your Rating'] - group['IMDb Rating']

        
        if differences.std() == 0:
            stat, pval = np.nan, np.nan
            interpretation = "All differences identical — t-test undefined"
        else:
            stat, pval = ttest_rel(group['Your Rating'], group['IMDb Rating'])
            if pval < 0.05:
                if n <= 2*{min_movies}:
                    interpretation = "Significant (p < 0.05) — small sample, interpret cautiously"
                else:
                    interpretation = "Significant (p < 0.05)"
            else:
                interpretation = "Not Significant"

        results.append({{
            "Director": director,
            "Num_Movies": n,
            "Mean_IMDb": group['IMDb Rating'].mean().round(2),
            "Mean_Mine": group['Your Rating'].mean().round(2),
            "t_statistic": round(stat, 3) if not np.isnan(stat) else np.nan,
            "p_value": round(pval, 4) if not np.isnan(pval) else np.nan,
            "Interpretation": interpretation
        }})


df_results = pd.DataFrame(results)
df_results = df_results.sort_values(by="p_value")
'''

    with st.expander("🛠️ View / edit the underlying code", expanded=False):
        user_ttest_code_director = st.text_area("Python t-test per Director Code (editable)", ttest_code_director, height=400)

    if st.button("Run t-test Analysis", key="run_ttest_director6"):
        try:
            local_vars = {"IMDB_Ratings": IMDB_Ratings, "My_Ratings": My_Ratings}
            exec(user_ttest_code_director, {}, local_vars)

            if "df_results" in local_vars:
                st.dataframe(local_vars["df_results"], width="stretch", height=460)
            else:
                st.warning("No dataframe named 'df_results' was produced. Please check your code.")

        except Exception as e:
            st.error(f"Error running t-test analysis: {e}")




# --- SCENARIO 6 ---

from textblob import TextBlob
import pandas as pd
import streamlit as st


if scenario == "6 – Review Analysis (Sentiment, Subjectivity)":
    scenario_header("6", "Review Analysis — Sentiment &amp; Subjectivity")

    # --- Short explanation ---
    st.markdown("""
    This scenario analyzes **audience reviews** of *Mother! (2017)*.  
    Each review is processed with natural language techniques to calculate:
    - **Sentiment** (negative to positive tone, -1 → +1)  
    - **Subjectivity** (objective → opinionated, 0 → 1)  
    The results include a summary table, aggregate metrics, and sample snippets.  
    """)

    # --- All reviews stored in a multi-line string ---
    reviews_text = """
Religious allegories abound but really it's just pretentious nonsense
Now I'm not one to disparage the director, I liked Requiem for a Dream and loved Black Swan, but this is a stinker and just simply boring. It's all just packed full of cod biblical allegories spread thickly throughout which tries to twist between different types of horror genres, but leaved me unintrigued. 
Granted the settings, claustrophobic direction and acting are top notch but it shouldn't mask for what otherwise is a poor uninteresting movie. It unsettles and bores, way too much to care, and as the ending dragged on I was left increasingly frustrated as it refused to just shut up shop. 
It's totally split opinion from what I've seen so far, and you'll struggle to find anyone in the middle on this one. In fairness, some credit to the film studios for risking this effort in launching it into mainstream cinemas but without the director it would have rightfully languished on cable late night showings.
There's no point going anymore into this. I simply hated it, and that despite being a major admirer of offbeat horror and psychological movies, but this isn't in the same league as for example Raw or Get Out, which is a shame. I'd recommend you pass on this there are far better films out there to go watch.
Aronofsky's mother! will be hated by many, but loved by a precious few

Horrifying. Just.. horrifying. Aronofsky really got me with this one. Not only did he manage to grab me on an intellectual level, but also on an emotional one...
# (Include all remaining reviews here, each separated by an empty line)

Usually this is where I put my plot description but it's best that you go into Darren Aronofsky's latest knowing as little as possible. Lets just say that Jennifer Lawrence and Javier Bardem are living in a large house all alone when a surprise visit sets them off into madness.
It really shocks me that Paramount would try to push MOTHER! onto the masses. For starters, the majority of moviegoers today do not want to think and they certainly don't want to see a movie where everything isn't explained. In fact, most people need everything explained in the trailer before they'll even go see a movie. A movie like MOTHER! is something that never explains itself and it constantly keeps you guessing from one scene to the next. What's it about? It's really hard to say as every viewer is going to come away with something different. With all of that said, it's easy to see why the film bombed at the box office and why those who did see it gave it a F rating.
what I loved most about this movie is that the setting is just so perfect. You've got a large beautiful house out in the middle of nowhere and it's surrounded by beautiful grass and trees. From the very first scene we can just tell that something isn't quite right and Aronofsky puts us in this beautiful place with confusing surroundings. What makes the film so special is the fact that nothing is ever explained and with each new plot twist your brain just becomes more confused as to what's going on. We know something is happening and we know something bad is going to happen but you're constantly trying to guess what.
Of course, a movie like this wouldn't work without a terrific cast to pull it off. Lawrence turns in another terrific performance and I thought she as fabulous at showing how fractured this character was. We're often questioning her mental state and I thought Lawrence managed to make you feel for the character and go along with her confusion to everything that is happening. Bardem actually steals the show with his fiery performance and I really loved the rage and anger he brought to the film as well as another side that I won't spoil to prevent giving away aspects of the plot. Both Ed Harris and Michelle Pfeiffer were also terrific but, again, I'll hold off commenting more to prevent plot points.
The cinematography is terrific and on a technical level the film is quite flawless. The story is a very interesting one and one that keeps you guessing throughout. The performances just seal the deal. With that said, the film certainly goes downright insane at times and the ending is just one that will have you staggering out of the theater. I must say that I thought the finale went on a bit too long and that it would have worked better had it been edited down a bit. Still, MOTHER! is a film that I really loved and one that I really respected but at the same time I'm not sure who I'd recommend it to.

Went to the first matinée available locally and I am still thinking the picture over. Will definitely see this one again, if it hasn't left the theatre abruptly. I was certainly horrified by the film, which is a good thing, as I had assumed it was a horror picture. It is, of course, much more than that. Nonetheless, it is NOT The Conjuring or Get Out (both good films, for sure), so just be warned.
By now you are aware that the film has been controversial, also a good thing. Jennifer Lawrence does a fine job and her career is certainly not going to suffer for her performance. I am not exactly a JLaw "fan" (could live without the Hunger Games), although I will pay closer attention to her future performances, especially if she pulls off more roles like this one (really liked Winter's Bone, by the way). As I understand the Hollywood scene, it is a respectable personal decision to take on a challenging role in an avant garde picture, especially if you have already banked serious money from popular roles in blockbusters. Javier Bardem, Michelle Pfeiffer, and Ed Harris also do their respective parts justice--a well-acted film by A-listers, overall. Camera work and special effects are also impressive.
The story is genuinely disturbing in a Requiem for a Dream way, so don't go if you can't handle that sort of thing. Some of the violence is, indeed, OVER THE TOP. Seriously, not for the faint of heart. Aside from the biblical allegory stuff, I found the character portrayals creepy as hell in a (sur?)realistic David Lynch-esque way. Hell is other people!
I applaud Mr. Aronofsky for keeping his vision intact all the way to the big screen. For reference, I just don't need any more movies based on superheros, comic books (except The Tenth or Gen 13), children's cartoons, vampires fighting werewolves, or horror stick about unfriending weirdos on facebook. 
You will have to make up your own mind on this one, so please do just that. Even if you end up despising the film, try to remember that, to quote Rob Zombie, "Art's Not Safe."

A married couple live in an isolated country house. He is a celebrated poet, suffering from writer's block, and she is working on renovating the house. Then a guest, a stranger, suddenly drops in and nothing will ever be the same again. 
Written and directed by Darren Aronofsky who gave us masterpieces like 'The Wrestler' and 'Requiem for a Dream', as well as the excellent 'Black Swan'. The fact that he wrote and directed this was the only reason I watched it, hoping that he was back to the form of those movies as his previous movie was the craptacular-beyond-belief 'Noah'.
Unfortunately, no, he isn't, though initially there was a glimmer of hope. The movie started interestingly enough, with some decent character development and some interesting themes. However, from the outset it was slow, plus there were signs this wasn't going to be a character-based drama but something symbolic, and pretentious.
Plus it was annoying. The only likeable character was Jennifer Lawrence's. Javier Bardem's was selfish and egotistical and every single other character was incredibly irritating. 
Still, I was hoping this would all develop into something interesting and profound. Wrong again. It develops into anarchy and some sort of badly-thought-out horror movie, and the annoyance factor gets pushed to the max. Of course, it's all meant to be symbolic, but figuring out everything would require you to think about the movie, and do so you would have had to have concentrated all through the tidal wave of excrement that was the movie.
Pretentious and annoying, and evidence that, sadly, Darren Aronofsky has run out of ideas.

I have been going to the movies for 45 years. This is, hands down, the worst movie I have ever seen. I mean, I hated this movie. Plan 9 From Outer Space and The Room were at least entertaining. This is like being locked in a cell with a stoned college student who can't shut up and thinks that every opinion they have, is the final word on a subject for 2 hours. Jennifer Lawrence should stick to roles that require her to paint herself blue or shoot arrows. Darren Aronofsky wants to be Luis Buñuel but he's closer to Uwe Boll. He cites The Exterminating Angel as the inspiration for Mother! I agree, in the sense that I did feel like one of the dinner guests who can't leave in Buñuel's classic during the course of watching Mother after paying 13 bucks to see this pretentious, heavy handed waste of time. Do yourself a favor, don't go see this movie, you won't get the 2 hours of your life back if you do. When it shows up on The Movie Channel playing at 3 in the morning in a couple of months, don't even set your DVR to record it. There are infomercials about gardening tools on at the same time, that are much more entertaining to watch that this.

I thought this was worth its salt even though it did tend towards cliché as it wore on. The disappointing aspect of this film is that Jennifer Lawrence somehow portrays an ego that is beyond the character. It's a kind of "you know that I know I'm only acting this and the real movie is me" that seems to have perpetuated in every film she had made since Silver linings Playbook, bar X-Men (when she was covered in paint and having to "live in" the previous "humble" shoes of Rebecca Romijn) and American Hustle (where she was greedy White Trash). She needs a director who can "humble her down", in the same way Eastwood did for Jolie in Changeling, so that her ego is less of a distraction for her acting.

Where to start - I've literally just finished watching this and spent the last hour questioning if I had been transported to another universe.
This movie had all the potential to be something great, from the cast to the secluded creepy setting - but no, we got almost 2 hours of the what could only be described as one of those brainwashing experimental videos where you have no idea what's going on.
If you like movies which make you feel uneasy, and make you think you're going mad then this might be the movies for you. Otherwise, I wouldn't bother.
Edit:So after having a day or so to ponder over the meaning of this movie - I've changed my rating and edited my review based on what I have come to know.
I can now say that once you understand the characters and why they represent, you'll understand the meaning and it could change your entire view of this movie.

If you just watch the pictures, the are confusing, disturbing, chaotic but their actual meaning is the representation of a person giving everything for someone else and still is not enough and everything she gives is topped by something else, every precious moment is taken to be shared.
I don't want to go into details of the meaning of every scene and why the movie develops the way it does, it would probably take a book to describe.
Every scene is symbolic, really well done.

This is a phenomenal film, full of details, full of symbolism and references to the Bible, to man's relationship to the mother Earth, to the state of consciousness in which we find ourselves as humanity. The atmosphere is superb and the actors are exceptional. I think this is my favorite Darren Aronofsky's movie. And I'm a bit sad because people in their reviews give it worst grade just because the movie does not have "enough action", just because is slow or not fun enough, just because the super heroes in it doesn't shoot and fly in the air and perform all kinds of spectacular things. I think, there is internet, there is IMDB and before you go to see a movie (especially if it is a an intellectually demanding like this one) read about it and see, decide if you are interested in such a thing. If it is not for you, there will always be something else to your liking out there An extraordinary movie in every aspect.

I have never watch a movie about it :).Dont try to learn something about the film before watching. Actually, it tells very good the whole life, and theatral aspect was wonderful in the movie. I strongly suggest that movie but, first, you have to leave your superstitions and prejudice . Just watch as an art and movie. But this movie, is not for superhero lovers and childs.
    """

    # --- Full editable code block for this scenario ---
    review_code = f'''
from textblob import TextBlob
import pandas as pd

# --- Reviews input ---
reviews_text = """{reviews_text.strip()}"""

# Convert multi-line text to list of reviews
reviews = [r.strip() for r in reviews_text.split("\\n\\n") if r.strip()]

review_records = []
review_counter = 1
for review in reviews:
    words = review.split()
    if len(words) < 5:
        continue  

    tb = TextBlob(review)
    sentiment = tb.sentiment.polarity
    subjectivity = tb.sentiment.subjectivity

    snippet = review[:500].strip()
    if not snippet:
        continue

    review_records.append({{
        "ReviewID": review_counter,
        "Words": len(words),
        "Sentiment": round(sentiment, 3),
        "Subjectivity": round(subjectivity, 3),
        "Snippet": snippet + ("..." if len(review) > 500 else "")
    }})
    review_counter += 1

df_reviews = pd.DataFrame(review_records)
df_reviews.reset_index(drop=True, inplace=True)
df_reviews['ReviewID'] = df_reviews.index + 1
'''

    # --- Editable code input (like Scenario 5) ---
    with st.expander("🛠️ View / edit the underlying code", expanded=False):
        user_review_code = st.text_area(
            "Python Review Sentiment Code (editable)",
            review_code,
            height=420
        )

    # --- Run button ---
    if st.button("Run Sentiment Analysis", key="run_sentiment6"):
        try:
            local_vars = {}
            exec(user_review_code, {}, local_vars)

            if "df_reviews" in local_vars:
                df_reviews = local_vars["df_reviews"]

                section_label("Reviews overview")
                st.dataframe(df_reviews, width="stretch", height=400)

                section_label("Aggregate insights")
                st.write(f"**Average sentiment:** {df_reviews['Sentiment'].mean():.3f}")
                st.write(f"**Average subjectivity:** {df_reviews['Subjectivity'].mean():.3f}")

                st.markdown("""
                **What these metrics mean:**
                - **Sentiment**: ranges from -1 (negative) to +1 (positive).  
                - **Subjectivity**: ranges from 0 (objective) to 1 (subjective/opinionated).  
                - **Snippet**: first 500 characters of the review.  
                """)

                st.markdown("""
                ---
                **How TextBlob works (in simple terms):**  
                - TextBlob uses a built-in **lexicon** (a dictionary of words) where each word has a sentiment score  
                  (e.g., *"great"* → +0.8, *"boring"* → -0.6).  
                - When it processes a review, it breaks the text into words and phrases, looks them up in the lexicon,  
                  and then averages the scores to estimate overall **sentiment**.  
                - For **subjectivity**, it checks how opinion-based the words are. Words like *"amazing"* or *"terrible"*  
                  are subjective, while factual words like *"movie length"* are objective.  
                - The result is a quick, automated way of measuring tone and bias without needing manual labeling.  

                ⚠️ **Note:** TextBlob is rule-based and doesn’t “understand” context deeply.  
                For example, sarcasm or irony might confuse it (e.g., *"What a masterpiece..."* said negatively will still be read as **positive**). 
                """)

                # --- Full reviews ---
                st.markdown("---")
                with st.expander("Full Reviews (click to expand)"):
                    for r in local_vars["reviews"]:
                        if len(r.split()) >= 5:
                            st.markdown(f"<div style='color:gray; padding:5px;'>{r}</div>", unsafe_allow_html=True)
            else:
                st.warning("No dataframe named 'df_reviews' was produced. Please check your code.")

        except Exception as e:
            st.error(f"Error running sentiment analysis: {e}")




# --- Scenario 11---
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Scenario 11 ---
if scenario == "11 – Model Evaluation (Feature Importance)":
    scenario_header("11", "Model Evaluation — Feature Importance")

    st.write("""
    We analyze which features matter most for predicting **my movie ratings** using a Random Forest model.  

    **Feature Importance:**  
    - Higher score → stronger influence on predictions.  
    - Lower score → weaker influence.  

    *(Trains its own Random Forest model below — no need to visit another scenario first.)*
    """)

    s11_code_view = '''
df_ml = IMDB_Ratings.merge(My_Ratings[['Movie ID','Your Rating']], on='Movie ID', how='left')
train_df = df_ml[df_ml['Your Rating'].notna()]

categorical_features = ['Genre', 'Director', 'Year']
numerical_features = ['IMDb Rating', 'Num Votes']

preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ('num', 'passthrough', numerical_features)
])
model = Pipeline([
    ('prep', preprocessor),
    ('reg', RandomForestRegressor(n_estimators=100, random_state=42))
])
model.fit(train_df[categorical_features + numerical_features], train_df['Your Rating'])

# Feature importance, individually and rolled up by category (Genre/Director/Year/etc)
cat_features = model.named_steps['prep'].named_transformers_['cat'].get_feature_names_out(categorical_features)
all_features = np.concatenate([cat_features, numerical_features])
importances = model.named_steps['reg'].feature_importances_
fi_df = pd.DataFrame({'Feature': all_features, 'Importance': importances}).sort_values('Importance', ascending=False)
fi_df['Category'] = fi_df['Feature'].str.split('_').str[0]
agg_df = fi_df.groupby('Category')['Importance'].sum().sort_values(ascending=False)
'''
    with st.expander("🛠️ View the underlying code", expanded=False):
        st.text_area(
            "Python code (view only — this box is not wired back into execution)",
            s11_code_view, height=360, key="s11_code_view",
        )
        st.caption("This is a reference copy of the code that actually runs below. Editing this box won't change the output.")

    # --- Train / retrain model (always available, not just first run) ---
    retrain_clicked = st.button(
        "🔄 Retrain model now" if 'model' in st.session_state else "▶️ Train model now"
    )
    if 'model' not in st.session_state and not retrain_clicked:
        st.info("No model trained yet for this session — click the button above to train one on your ratings.")

    if retrain_clicked:
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline

        df_ml = IMDB_Ratings.merge(My_Ratings[['Movie ID','Your Rating']], on='Movie ID', how='left')
        train_df = df_ml[df_ml['Your Rating'].notna()]

        # Treat Year as categorical
        categorical_features = ['Genre', 'Director', 'Year']
        numerical_features = ['IMDb Rating', 'Num Votes']

        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
                ('num', 'passthrough', numerical_features)
            ]
        )

        model = Pipeline([
            ('prep', preprocessor),
            ('reg', RandomForestRegressor(n_estimators=100, random_state=42))
        ])

        X_train = train_df[categorical_features + numerical_features]
        y_train = train_df['Your Rating']
        model.fit(X_train, y_train)

        st.session_state['model'] = model
        st.success("Model trained successfully! Feature importance is shown below.")

    # --- Show feature importance if model exists ---
    if 'model' in st.session_state:
        trained_model = st.session_state['model']
        rf = trained_model.named_steps['reg']
        preproc = trained_model.named_steps['prep']

        # Feature names
        cat_features = preproc.named_transformers_['cat'].get_feature_names_out(['Genre','Director','Year'])
        numerical_features = ['IMDb Rating', 'Num Votes']
        all_features = np.concatenate([cat_features, numerical_features])
        importances = rf.feature_importances_

        fi_df = pd.DataFrame({
            'Feature': all_features,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        # --- Top N individual features ---
        top_n = 20
        fi_top = fi_df.head(top_n)

        section_label(f"Top {top_n} individual features")
        fig_fi, ax_fi = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=fi_top, color="#E3A857", ax=ax_fi)
        ax_fi.set_title("Top Feature Importances", color="#EDEEF0")
        ax_fi.grid(axis='x', color="#2A2F3A", linewidth=0.6)
        ax_fi.set_axisbelow(True)
        for _s in ("top", "right"):
            ax_fi.spines[_s].set_visible(False)
        fig_fi.tight_layout()
        st.pyplot(fig_fi)
        plt.close(fig_fi)

        # --- Automatic explanation for top Director ---
        director_features = fi_df[fi_df['Feature'].str.startswith('Director')]
        if not director_features.empty:
            top_director = director_features.sort_values(by='Importance', ascending=False).iloc[0]
            feature = top_director['Feature']
            importance = top_director['Importance']
            director_name = feature.replace('Director_','')

            st.write("**Specific Insight:**")
            st.write(f"""
            **{feature}** (importance {importance:.3f}):

            **What the feature represents:**  
            For `{feature}`, the model uses a one-hot encoded feature to distinguish {director_name} movies from all other movies.  
            In other words, whether a movie is directed by {director_name} significantly affects the model's predictions.  
            My rating behavior for {director_name} movies is distinct from my average ratings, and therefore the model relies on this pattern to make predictions.
            """)

        # --- Aggregated by category ---
        fi_df['Category'] = fi_df['Feature'].str.split('_').str[0]
        agg_df = fi_df.groupby('Category')['Importance'].sum().sort_values(ascending=False)

        section_label("Feature importance by category")
        fig_agg, ax_agg = plt.subplots(figsize=(8, 4))
        sns.barplot(x=agg_df.values, y=agg_df.index, color="#4FA88F", ax=ax_agg)
        ax_agg.set_title("Aggregated Importances", color="#EDEEF0")
        ax_agg.grid(axis='x', color="#2A2F3A", linewidth=0.6)
        ax_agg.set_axisbelow(True)
        for _s in ("top", "right"):
            ax_agg.spines[_s].set_visible(False)
        fig_agg.tight_layout()
        st.pyplot(fig_agg)
        plt.close(fig_agg)

        # --- Summary explanation (only shows when model exists) ---
        st.write("""
        **Interpretation:**  
        Aggregating features by category shows the bigger picture of what drives my ratings. If `Director` is high, it means certain directors consistently shape how I score movies.  

        **Why this matters for me:**  
        I bring my own personal insight into how I feel about directors — their style, storytelling, or reputation.  
        The model simply quantifies what I already sense: that my ratings often rise or fall depending on who directed the film.  

        **Why movies are my choice for all scenarios:**  
        Movies are personal. Unlike abstract datasets, I have close experience with films and directors.  
        This makes the insights richer — I can interpret the model’s patterns through my own perspective as a movie fan.  
        That connection is why I chose film as the subject matter to explore these scenarios.
        """)





# --- Scenario 12: Feature Hypothesis Testing ---
if scenario == "12 – Feature Hypothesis Testing":
    scenario_header("12", "Feature Hypothesis Testing &amp; Predictions")

    st.markdown("""
    Select features to test if they **improve model predictions** for your ratings.
    After running, you'll see:
    1. Statistical test results
    2. Detailed explanation of feature impact
    3. Example predicted ratings for unseen movies with reasoning
    4. Annotated RMSE comparison with interpretation
    """)

    # --- Feature selection ---
    candidate_features = ['Director', 'Genre', 'Year', 'Num Votes', 'IMDb Rating']
    selected_features = st.multiselect(
        "Select feature(s) to test", 
        candidate_features, 
        default=['Director'] 

    )
    if 'scenario10_result' not in st.session_state:
        st.session_state['scenario10_result'] = None

    if st.button("Run Test & Show Predictions"):
        import numpy as np
        from sklearn.model_selection import cross_val_score, KFold
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.ensemble import RandomForestRegressor
        from scipy.stats import ttest_rel
        import matplotlib.pyplot as plt
        from sklearn.impute import SimpleImputer

        # --- Prepare training data ---
        df_ml = IMDB_Ratings.merge(My_Ratings[['Movie ID','Your Rating']], on='Movie ID', how='left')
        train_df = df_ml[df_ml['Your Rating'].notna()]
        y = train_df['Your Rating']  # Target variable: your ratings
                        
        # --- Baseline model (numeric only) ---
        baseline_features = ['Num Votes','IMDb Rating']
        X_base = train_df[baseline_features]
        model_base = Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('reg', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        scores_base = -cross_val_score(model_base, X_base, y, cv=cv, scoring='neg_root_mean_squared_error')
        
        # --- Feature-added model ---
        categorical_features = [f for f in selected_features if f in ['Director','Genre','Year']]
        numerical_features = [f for f in selected_features if f in ['Num Votes','IMDb Rating']]
        features_to_use = categorical_features + numerical_features
        
        if features_to_use:
            preprocessor = ColumnTransformer(
                transformers=[
                    ('cat', Pipeline([
                        ('impute', SimpleImputer(strategy='constant', fill_value='Unknown')),
                        ('onehot', OneHotEncoder(handle_unknown='ignore'))
                    ]), categorical_features),
                    ('num', SimpleImputer(strategy='median'), numerical_features)
                ]
            )
            X_test = train_df[features_to_use]
            model_test = Pipeline([
                ('prep', preprocessor),
                ('reg', RandomForestRegressor(n_estimators=100, random_state=42))
            ])
            scores_test = -cross_val_score(model_test, X_test, y, cv=cv, scoring='neg_root_mean_squared_error')

            # --- Paired t-test ---
            t_stat, p_val = ttest_rel(scores_base, scores_test)

            # --- Retrain for predictions ---
            model_test.fit(X_test, y)

            # --- Predict all unseen movies ---
            unseen_df = df_ml[df_ml['Your Rating'].isna()]
            if not unseen_df.empty:
                X_unseen = unseen_df[features_to_use]
                preds = model_test.predict(X_unseen)
                pred_df = unseen_df[['Movie ID','Title','Year','IMDb Rating']].copy()
                pred_df['Predicted Rating'] = np.round(preds,1)

                # --- Features considered per movie ---
                features_list = []
                for idx, row in unseen_df.iterrows():
                    feature_values = {f: row.get(f,'?') for f in selected_features}
                    features_list.append(", ".join([f"{k}={v}" for k,v in feature_values.items()]))
                pred_df['Features Considered'] = features_list

                # --- Sort by Year descending ---
                pred_df = pred_df.sort_values(by='Year', ascending=False)
            else:
                pred_df = pd.DataFrame()

            # --- RMSE summary & automatic interpretation ---
            rmse_base_mean = np.mean(scores_base)
            rmse_test_mean = np.mean(scores_test)
            rmse_diff = rmse_base_mean - rmse_test_mean

            if p_val < 0.05:
                if rmse_diff > 0:
                    stat_explanation = (
                        f"✅ Adding {', '.join(selected_features)} improved the model.\n"
                        f"- Average RMSE decreased from {rmse_base_mean:.2f} → {rmse_test_mean:.2f}.\n"
                        f"- t-value = {t_stat:.3f}, p-value = {p_val:.4f} → statistically significant improvement."
                    )
                else:
                    stat_explanation = (
                        f"❌ Adding {', '.join(selected_features)} worsened the model.\n"
                        f"- Average RMSE increased from {rmse_base_mean:.2f} → {rmse_test_mean:.2f}.\n"
                        f"- t-value = {t_stat:.3f}, p-value = {p_val:.4f} → statistically significant deterioration."
                    )
            else:
                stat_explanation = (
                    f"ℹ️ Adding {', '.join(selected_features)} did NOT meaningfully change the model.\n"
                    f"- Average RMSE changed from {rmse_base_mean:.2f} → {rmse_test_mean:.2f}.\n"
                    f"- t-value = {t_stat:.3f}, p-value = {p_val:.4f} → no statistically significant difference."
                )

            st.session_state['scenario10_result'] = {
                't_stat': t_stat,
                'p_val': p_val,
                'stat_explanation': stat_explanation,
                'predictions': pred_df,
                'scores_base': scores_base,
                'scores_test': scores_test,
                'selected_features': selected_features
            }

    # --- Display results ---
    if st.session_state['scenario10_result']:
        result = st.session_state['scenario10_result']

        # --- Predictions table ---
        section_label("Predictions table — all unrated movies")
        if not result['predictions'].empty:
            st.dataframe(result['predictions'])

            # --- Statistical significance explanation ---
            section_label("Statistical significance of the improvement")
            st.info(result['stat_explanation'])

            # --- Explanation of predicted rating changes ---
            section_label("Why predicted ratings change")
            st.markdown(f"""
            The predicted ratings change when you modify the selected features because the model learns patterns from your past ratings.  

            **Current features used:** {', '.join(result['selected_features'])}  

            - **Director:** captures your preferences for specific directors.  
            - **Genre:** captures your preferences for specific types of films.  
            - **Year:** considers how your ratings vary over time.  
            - **IMDb Rating & Num Votes:** reflect general popularity and consensus quality.  

            When features are added or removed, the model adjusts the predictions based on the patterns it learned from your historical ratings.
            """)
        else:
            st.warning("No unseen movies available for prediction.")

        # --- Annotated RMSE boxplot ---
        rmse_base_mean = np.mean(result['scores_base'])
        rmse_test_mean = np.mean(result['scores_test'])
        fig_rmse, ax_rmse = plt.subplots(figsize=(7, 4))
        bp = ax_rmse.boxplot(
            [result['scores_base'], result['scores_test']],
            patch_artist=True, medianprops=dict(color="#EDEEF0", linewidth=1.4),
        )
        for patch, colour in zip(bp['boxes'], ["#8B93A1", "#E3A857"]):
            patch.set_facecolor(colour)
            patch.set_alpha(0.35)
            patch.set_edgecolor(colour)
        ax_rmse.set_xticks([1, 2], ['Baseline', 'With Feature(s)'])
        ax_rmse.set_ylabel("RMSE")
        ax_rmse.set_title("Cross-Validated RMSE Comparison", color="#EDEEF0")
        ax_rmse.grid(axis='y', color="#2A2F3A", linewidth=0.6)
        ax_rmse.set_axisbelow(True)
        for _s in ("top", "right"):
            ax_rmse.spines[_s].set_visible(False)
        ax_rmse.text(1, rmse_base_mean + 0.02, f"{rmse_base_mean:.2f}", ha='center', color="#8B93A1")
        ax_rmse.text(2, rmse_test_mean + 0.02, f"{rmse_test_mean:.2f}", ha='center', color="#E3A857")
        fig_rmse.tight_layout()
        st.pyplot(fig_rmse)
        plt.close(fig_rmse)

        # --- RMSE interpretation ---
        st.write("""
        **Interpretation of RMSE Boxplot and Model Comparison**

        **1: Baseline Model (Numeric Features Only)**
        - Uses only `IMDb Rating` and `Num Votes`.
        - Captures general popularity and average rating information.
        - Higher RMSE → predictions deviate more from your actual ratings.
        - Wide spread → inconsistent performance across movies.

        **2: Feature-Added Model (Selected Features Included)**
        - Includes additional features such as `Director`, `Genre`, `Year`.
        - Provides context about your personal preferences.
        - Lower RMSE → predictions closer to your actual ratings.
        - Tighter spread → more consistent performance.

        **Takeaway**
        - RMSE decrease + p-value < 0.05 → features improve model accuracy.
        - RMSE increase + p-value < 0.05 → features worsen predictions.
        - p-value ≥ 0.05 → no significant change.
        """)




# --- Scenario 8: Graph-Based Movie Relationships ---
if scenario == "8 – Graph Based Movie Relationships":
    scenario_header("8", "Graph-Based Movie Relationships")
    st.write("""
    This scenario models the dataset as a **graph**:
    - **Nodes**: Movies, Directors, Genres  
    - **Edges**: Relationships between them.  
    
    Use the filters below to narrow down by **Year**, **Director(s)**, and **Genre**, then run the graph builder.
    """)

    # --- Filters ---
    directors = sorted(IMDB_Ratings["Director"].dropna().unique()) if not IMDB_Ratings.empty else []
    genres = []
    if "Genre" in IMDB_Ratings.columns:
        genres = sorted({g.strip() for sublist in IMDB_Ratings["Genre"].dropna().str.split(",") for g in sublist})
    years = sorted(IMDB_Ratings["Year"].dropna().unique().astype(int).tolist()) if "Year" in IMDB_Ratings.columns else []

    # --- Default Selections ---
    default_year = "All"
    default_genre = "Drama"
    default_director = [d for d in ["Alfred Hitchcock", "Stanley Kubrick", "Francis Ford Coppola"] if d in directors]

    selected_year = st.selectbox(
        "Filter by Year",
        ["All"] + [str(y) for y in years],
        index=(["All"] + [str(y) for y in years]).index(default_year)
    )
    selected_directors = st.multiselect("Filter by Director(s)", directors, default=default_director)
    selected_genre = st.selectbox(
        "Filter by Genre",
        ["All"] + genres,
        index=(["All"] + genres).index(default_genre) if default_genre in genres else 0
    )

    # --- Editable code template ---
    graph_code = '''
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

df_graph = IMDB_Ratings.copy()

if selected_year != "All":
    df_graph = df_graph[df_graph["Year"] == int(selected_year)]
if selected_directors:
    df_graph = df_graph[df_graph["Director"].isin(selected_directors)]
if selected_genre != "All":
    df_graph = df_graph[df_graph["Genre"].str.contains(selected_genre, na=False)]

G = nx.Graph()
for _, row in df_graph.iterrows():
    movie = row.get("Title")
    director = row.get("Director")
    genre = row.get("Genre")

    if pd.notna(movie):
        G.add_node(movie, type="movie")
    if pd.notna(director):
        G.add_node(director, type="director")
        G.add_edge(director, movie)
    if pd.notna(genre):
        for g in str(genre).split(", "):
            G.add_node(g, type="genre")
            G.add_edge(movie, g)

fig, ax = plt.subplots(figsize=(12, 8))
pos = nx.spring_layout(G, k=0.3, iterations=25)
color_map = []
for node, data in G.nodes(data=True):
    if data["type"] == "movie":
        color_map.append("#4FA88F")
    elif data["type"] == "director":
        color_map.append("#E3A857")
    else:
        color_map.append("#C1524B")

nx.draw(G, pos, with_labels=True, node_size=750, node_color=color_map, font_size=8,
        font_color="#EDEEF0", edge_color="#3A414F", ax=ax)
st.pyplot(fig)
st.write(f"Graph built with **{len(G.nodes)} nodes** and **{len(G.edges)} edges**.")
'''

    with st.expander("🛠️ View / edit the underlying code", expanded=False):
        user_graph_code = st.text_area("Python Graph Code (editable)", graph_code, height=380)

    if st.button("Run Graph Analysis", key="run_graph11"):
        try:
            local_vars = {
                "IMDB_Ratings": IMDB_Ratings,
                "selected_year": selected_year,
                "selected_directors": selected_directors,
                "selected_genre": selected_genre,
                "st": st,
                "pd": pd
            }
            exec(user_graph_code, {}, local_vars)

            # --- Clean Explanation ---
            st.markdown("""
### Understanding the Graph

**Nodes:**  
- Movies  
- Directors  
- Genres  

**Edges:**  
- Director → Movie  
- Movie → Genre  

**Why this matters:**  
- Identify which directors specialize in which genres  
- Discover genre clusters with many movies  
- Explore connections between directors through shared genres or collaborations  

This visualization helps you explore the movie dataset’s structure and uncover patterns and relationships clearly.
""")
        except Exception as e:
            st.error(f"Error running Graph Analysis code: {e}")




# --- Scenario 7 Poster Analysis ---
if scenario == "7 – Poster Image Analysis (OMDb API)":
    scenario_header("7", "Poster Image &amp; Mood Analysis")
    st.markdown("""
    Select a movie, then click **Fetch Poster & Analyze** to display the poster, 
    dominant colors, and an easy-to-understand mood analysis.
    """)

    import requests
    from PIL import Image
    import numpy as np
    from sklearn.cluster import KMeans

    # --- Editable code block ---
    poster_code = '''

imdb_id = IMDB_Ratings.loc[IMDB_Ratings['Title'] == selected_film, 'Movie ID'].values[0]


url = f"http://www.omdbapi.com/?i={imdb_id}&apikey={OMDB_API_KEY}"
response = requests.get(url).json()
poster_url = response.get('Poster')

if poster_url and poster_url != "N/A":
    img = Image.open(requests.get(poster_url, stream=True).raw).convert("RGB")
    img_small = img.resize((150, 150))
    img_array = np.array(img_small).reshape(-1, 3)

    
    kmeans = KMeans(n_clusters=3, random_state=42).fit(img_array)
    dominant_colors = kmeans.cluster_centers_

    
    st.image(poster_url, width=300)

    
    st.write("🎨 Dominant Colors:")
    cols = st.columns(len(dominant_colors))
    for idx, color in enumerate(dominant_colors.astype(int)):
        hex_color = '#%02x%02x%02x' % tuple(color)
        cols[idx].markdown(
            "<div style='width:60px; height:60px; background:{}; border-radius:8px; border:1px solid #000'></div>".format(hex_color),
            unsafe_allow_html=True
        )

    
    brightness = np.mean(img_array)
    if brightness < 100:
        mood = "dark and moody"
        cluster_name = "Cluster 0 – Thriller / Horror style"
        mood_tag = "🌑 Dark Thriller vibes"
    elif brightness < 170:
        mood = "balanced"
        cluster_name = "Cluster 1 – Drama / Realistic style"
        mood_tag = "🎭 Dramatic tone"
    else:
        mood = "bright and vivid"
        cluster_name = "Cluster 2 – Comedy / Family style"
        mood_tag = "😂 Lighthearted & Fun"

    
    st.success("🎬 Poster assigned to: **{}**".format(cluster_name))
    st.info("The poster looks **{}**, suggesting **{}**.\\n\\n👉 Mood tag: **{}**".format(
        mood, cluster_name.split('–')[1].strip(), mood_tag
    ))
else:
    st.warning("Poster not found.")
'''

    # --- Editable text area ---
    with st.expander("🛠️ View / edit the underlying code", expanded=False):
        user_poster_code = st.text_area("Python Poster Analysis Code (editable)", poster_code, height=400)

    # --- Hidden API key ---
    OMDB_API_KEY = "cbbdb8f8"  # Keep this hidden in production

    # --- Movie selection ---
    film_list = IMDB_Ratings['Title'].dropna().unique().tolist()
    selected_film = st.selectbox("Select a movie to analyze poster:", film_list)

    # --- Run button ---
    if st.button("Fetch Poster & Analyze"):
        try:
            local_vars = {
                "IMDB_Ratings": IMDB_Ratings,
                "selected_film": selected_film,
                "OMDB_API_KEY": OMDB_API_KEY,
                "st": st,
                "np": np,
                "KMeans": KMeans,
                "requests": requests,
                "Image": Image
            }
            exec(user_poster_code, {}, local_vars)
        except Exception as e:
            st.error(f"Error running poster analysis: {e}")




# --- Scenario 13: Deep Learning Semantic Genre Analysis (Dynamic) ---
if scenario == "13 – Semantic Genre & Recommendations (Deep Learning / NLP)":
    scenario_header("13", "Semantic Genre &amp; Recommendations (Deep Learning / NLP)")
    st.markdown("""
    This scenario uses **sentence embeddings** to determine the main genre of films by analyzing the plot.  
    The table shows:
    - Film title
    - Plot snippet
    - OMDb listed genres
    - Embedding similarity with each genre
    - Predicted main genre
    """)

    # --- Dynamic list of directors from the dataset ---
    directors_list = IMDB_Ratings["Director"].dropna().unique().tolist()
    directors_list.sort()
    selected_director = st.selectbox("Choose a director:", directors_list)

    s13_code_view = '''
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer("all-MiniLM-L6-v2")

movies = IMDB_Ratings[IMDB_Ratings["Director"] == selected_director]["Title"].dropna().tolist()
results = []
for title in movies:
    movie_data = fetch_movie_data(title)          # OMDb lookup, cached per title
    plot_embedding = model.encode(movie_data["Plot"], convert_to_tensor=True)

    similarities = {}
    for g in movie_data["Genre"]:
        g_embedding = model.encode(g, convert_to_tensor=True)
        similarities[g] = round(util.cos_sim(plot_embedding, g_embedding).item(), 3)

    main_genre = max(similarities, key=similarities.get) if similarities else "Unknown"
    results.append({
        "Film": movie_data["Title"],
        "OMDb Genres": ", ".join(movie_data["Genre"]),
        "Embedding Similarity": similarities,
        "Main Genre (Predicted)": main_genre,
    })

df_results = pd.DataFrame(results)
'''
    with st.expander("🛠️ View the underlying code", expanded=False):
        st.text_area(
            "Python code (view only — this box is not wired back into execution)",
            s13_code_view, height=380, key="s13_code_view",
        )
        st.caption("This is a reference copy of the code that actually runs below. Editing this box won't change the output.")

    # --- Hidden OMDb API key ---
    OMDB_API_KEY = "72466310"  # keep this private

    # --- Cached function to fetch OMDb data ---
    @st.cache_data(show_spinner=False)
    def fetch_movie_data(title):
        import requests
        response = requests.get(f"http://www.omdbapi.com/?t={title}&apikey={OMDB_API_KEY}&plot=full").json()
        plot = response.get("Plot") or "Plot missing"
        genres = response.get("Genre").split(", ") if response.get("Genre") else ["Unknown"]
        return {"Title": response.get("Title") or title, "Plot": plot, "Genre": genres}

    # --- Run button ---
    if st.button("Run Deep Learning Genre Analysis"):
        # Get all movies for the selected director dynamically
        movies = IMDB_Ratings[IMDB_Ratings["Director"] == selected_director]["Title"].dropna().tolist()

        if not movies:
            st.warning(f"No movies found for {selected_director}")
        else:
            from sentence_transformers import SentenceTransformer, util
            model = SentenceTransformer("all-MiniLM-L6-v2")

            results = []

            for title in movies:
                movie_data = fetch_movie_data(title)
                plot = movie_data["Plot"]
                genres = movie_data["Genre"]

                # Compute plot embedding
                plot_embedding = model.encode(plot, convert_to_tensor=True)

                # Compute similarity for each genre
                similarities = {}
                for g in genres:
                    g_embedding = model.encode(g, convert_to_tensor=True)
                    sim = util.cos_sim(plot_embedding, g_embedding).item()
                    similarities[g] = round(sim, 3)

                # Main genre = highest similarity
                main_genre = max(similarities, key=similarities.get) if similarities else "Unknown"

                results.append({
                    "Film": movie_data["Title"],
                    "OMDb Genres": ", ".join(genres),
                    "Embedding Similarity": similarities,
                    "Main Genre (Predicted)": main_genre,
                    "Plot": plot[:200] + "..." if len(plot) > 200 else plot
                })

            df_results = pd.DataFrame(results)
            st.success(f"Analysis complete for {selected_director} ✅")
            st.dataframe(df_results, width="stretch")

            st.markdown("""
            **Explanation:**  
            - Each **plot** is converted into a vector (embedding).  
            - Each **genre** is also converted into a vector.  
            - **Cosine similarity** measures semantic closeness (0 to 1).  
            - The genre with the highest similarity is predicted as the **main genre**.  
            - This helps when OMDb lists multiple genres, showing the most semantically relevant one.
            """)


# --- Scenario 14: Live Ratings Monitor + Supervised ML Predictions (English only) ---
if scenario == "14 – Live Ratings Monitor (Scheduled + on-demand)":
    scenario_header("14", "Live Ratings Monitor")

    st.markdown("""
**What this scenario actually does**

- **Scheduled monitoring:** `scripts/refresh_live_ratings.py` runs daily via GitHub Actions,
  independent of this app — it fetches live OMDb ratings, compares them to the stored snapshot,
  and logs any changed titles to Supabase with a timestamp.
- **On-demand monitoring:** The button below runs the same check manually, with your own filters,
  and writes to the same Supabase table.
- **CI:** Every push runs `ci.yml` — syntax-checks this app and the scripts, and runs the unit
  test suite in `tests/`.
- **CD gate:** Branch protection on `main` requires that CI check to pass before merging, and
  Streamlit Cloud auto-deploys off `main` — so nothing untested reaches production.
- **MLOps — training & registry:** `scripts/train_model.py` cross-validates a candidate model,
  uploads it to Supabase Storage, and logs the run to a `model_runs` table. It's only **promoted**
  to production if its validated RMSE beats the current champion. This runs on a weekly schedule,
  on demand, whenever `myratings.xlsx` changes, and automatically right after CI passes on `main`.
- **MLOps — serving:** The prediction table below loads the current **registered** production
  model (see the panel above) instead of retraining from scratch on every click. If nothing's
  registered yet, it falls back to a temporary session-only model and says so explicitly.

See the setup notes at the bottom of this page for the Supabase table/bucket and GitHub secrets
this all depends on.
""")

    # --- OMDb API key(s): manual override > comma-separated keys in secrets >
    # shared fallback key. If a key hits its rate limit mid-run, the app
    # automatically switches to the next one for the rest of the run. ---
    with st.expander("🔑 OMDb API key (use this if you're seeing rate-limit errors)"):
        st.caption(
            "Paste your own free key here to use it for this session (get one in ~30 seconds at "
            "omdbapi.com/apikey.aspx). For a permanent fix, add `OMDB_API_KEY` to your Streamlit "
            "secrets — you can list several keys separated by commas there, and the app will "
            "automatically switch to the next one if one gets rate-limited mid-run."
        )
        st.text_input("Your OMDb API key", type="password", key="omdb_manual_key", placeholder="e.g. abcd1234")

    def _secret(name, default=None):
        """st.secrets raises (rather than returning a default) when there's no
        secrets.toml at all — which is the normal case when running locally."""
        try:
            return st.secrets.get(name, default)
        except Exception:
            return default

    def _get_omdb_keys():
        keys = []
        manual_key = st.session_state.get("omdb_manual_key", "").strip()
        if manual_key:
            keys.append(manual_key)
        secret_val = _secret("OMDB_API_KEY")
        if secret_val:
            keys.extend([k.strip() for k in str(secret_val).split(",") if k.strip()])
        keys.append("e9476c0a")  # last-resort shared demo key
        seen, ordered = set(), []
        for k in keys:
            if k not in seen:
                ordered.append(k)
                seen.add(k)
        return ordered

    omdb_keys = _get_omdb_keys()

    # --- Supabase client (reads secrets; degrades gracefully if not configured) ---
    @st.cache_resource
    def get_supabase_client():
        try:
            from supabase import create_client
        except ImportError:
            return None
        # Accept secrets either flat (SUPABASE_URL / SUPABASE_KEY at the top
        # level) or nested under a [supabase] section in secrets.toml.
        section = _secret("supabase", {}) or {}
        url = _secret("SUPABASE_URL") or section.get("SUPABASE_URL")
        key = _secret("SUPABASE_KEY") or section.get("SUPABASE_KEY")
        if not url or not key:
            return None
        return create_client(url, key)

    supabase = get_supabase_client()

    # --- Production model registry (see model_runs_schema.sql + scripts/train_model.py) ---
    current_model_row = None
    if supabase is not None:
        try:
            reg = (
                supabase.table("model_runs")
                .select("*")
                .eq("is_current", True)
                .order("trained_at", desc=True)
                .limit(1)
                .execute()
            )
            if reg.data:
                current_model_row = reg.data[0]
        except Exception as e:
            st.warning(f"Couldn't read the model registry (`model_runs` table): {e}")

    with st.expander("🏷️ Production model status", expanded=False):
        if current_model_row:
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Validated RMSE", f"{current_model_row['cv_rmse_mean']:.3f}")
            mc2.metric("Trained on", f"{current_model_row['n_training_movies']} movies")
            trained_at_display = str(current_model_row['trained_at'])[:19].replace("T", " ")
            mc3.metric("Trained at (UTC)", trained_at_display)
            st.caption(
                f"Features: {current_model_row['features']} · "
                f"Commit: {current_model_row.get('git_sha', 'n/a')[:8]}"
            )
        else:
            st.info(
                "No registered model yet. Run the **Train Ratings Model** GitHub Actions "
                "workflow once (Actions tab → Train Ratings Model → Run workflow) to train, "
                "validate, and register the first one. Until then, predictions below fall "
                "back to a temporary model trained just for this session."
            )

    @st.cache_resource
    def _load_production_model(storage_path: str):
        """Downloads and deserializes the registered model. Cached per
        storage_path, so a newly-promoted model (a different path) is
        downloaded fresh, while repeat runs against the same model reuse
        the cached object instead of re-downloading every rerun."""
        import joblib
        import tempfile
        data = supabase.storage.from_("models").download(storage_path)
        with tempfile.NamedTemporaryFile(suffix=".joblib") as tmp:
            tmp.write(data)
            tmp.flush()
            return joblib.load(tmp.name)

    # --- Filter which titles get checked ---
    section_label("Filter which titles to check")
    genres_available = sorted({
        g.strip() for sublist in IMDB_Ratings['Genre'].dropna().str.split(',') for g in sublist
    })
    default_genre = ["Horror"] if "Horror" in genres_available else []

    col1, col2, col3 = st.columns(3)
    with col1:
        selected_genres = st.multiselect("Genre(s)", genres_available, default=default_genre)
    with col2:
        min_rating_filter, max_rating_filter = st.slider(
            "IMDb rating range", 0.0, 9.5, (6.0, 9.5), 0.1
        )
    with col3:
        min_votes_filter, max_votes_filter = st.slider(
            "Votes (popularity) range", 0, 3000000, (20000, 3000000), step=5000
        )

    check_limit = st.slider("How many titles to check", 25, 250, 100, step=25)
    only_show_changed = st.checkbox(
        "Only show titles with a rating change in the results below", value=True
    )
    st.caption("Note: everything checked is still logged to Supabase either way — this only affects the table below, so trend history stays complete.")

    top250_films = IMDB_Ratings[
        (IMDB_Ratings['IMDb Rating'] >= min_rating_filter) &
        (IMDB_Ratings['IMDb Rating'] <= max_rating_filter) &
        (IMDB_Ratings['Num Votes'] >= min_votes_filter) &
        (IMDB_Ratings['Num Votes'] <= max_votes_filter)
    ]
    if selected_genres:
        import re as _re
        pattern = '|'.join(_re.escape(g) for g in selected_genres)
        top250_films = top250_films[top250_films['Genre'].str.contains(pattern, case=False, na=False)]

    top250_films = top250_films.sort_values(by="IMDb Rating", ascending=False).head(check_limit)
    st.caption(f"🎯 {len(top250_films)} titles match these filters and will be checked.")

    # --- Run Button ---
    if st.button("Run Live Ratings Check"):
        import requests
        from datetime import datetime, timezone
        import os
        import pandas as pd
        import numpy as np
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline

        history_file = "live_ratings_history.csv"
        timestamp = datetime.now(timezone.utc).isoformat()

        results = []
        omdb_errors = []       # sample of raw OMDb/network errors, for diagnostics
        non_english_skipped = 0
        key_idx = [0]          # mutable so the helper can advance it across calls

        def _fetch_from_omdb(movie_id):
            """Tries the current key; on a rate-limit-style error, advances to
            the next configured key and retries the same title with it."""
            for _ in range(len(omdb_keys)):
                idx = key_idx[0]
                try:
                    url = f"http://www.omdbapi.com/?i={movie_id}&apikey={omdb_keys[idx]}"
                    resp = requests.get(url, timeout=10).json()
                except Exception as e:
                    return None, str(e)

                if resp.get("Response") == "True":
                    return resp, None

                error_msg = resp.get("Error", "Unknown OMDb error")
                if "limit" in error_msg.lower() and idx + 1 < len(omdb_keys):
                    key_idx[0] += 1
                    continue
                return None, error_msg
            return None, "All configured OMDb keys are rate-limited"

        # --- Fetch live ratings from OMDb using Movie ID (IMDb ID) ---
        with st.spinner(f"Checking {len(top250_films)} titles against OMDb..."):
            for _, row in top250_films.iterrows():
                movie_id = row["Movie ID"]
                static_rating = row["IMDb Rating"]

                resp, error = _fetch_from_omdb(movie_id)
                if resp is not None:
                    # Normalize languages: split, strip, lowercase
                    languages = [lang.strip().lower() for lang in resp.get("Language", "").split(",")]
                    live_rating = float(resp.get("imdbRating", 0)) if resp.get("imdbRating") else None

                    if "english" not in languages:
                        non_english_skipped += 1
                        continue
                else:
                    live_rating = None
                    languages = []
                    if len(omdb_errors) < 3:
                        omdb_errors.append(error)

                rating_diff = live_rating - static_rating if live_rating is not None else None

                results.append({
                    "Title": row["Title"],
                    "IMDb Rating (Static)": static_rating,
                    "IMDb Rating (Live)": live_rating,
                    "Rating Difference": rating_diff,
                    "CheckedAt": timestamp,
                    "Movie ID": movie_id,
                    "Genre": row.get("Genre"),
                    "Director": row.get("Director"),
                    "Year": row.get("Year"),
                    "Num Votes": row.get("Num Votes"),
                    "Language": ", ".join([lang.capitalize() for lang in languages])
                })

        new_df = pd.DataFrame(results)

        # Keep every title that was successfully checked - including ones with
        # zero rating change. High-vote/popular films rarely move day-to-day
        # (a handful of new votes barely shifts a rounded average built on
        # hundreds of thousands of ratings), so dropping every zero-diff row
        # was silently hiding those films completely, and on runs where
        # *nothing* changed it left an empty table with nothing to log to
        # Supabase at all. We only drop titles OMDb couldn't be matched to a
        # live rating for.
        if not new_df.empty and "IMDb Rating (Live)" in new_df.columns:
            new_df = new_df[new_df["IMDb Rating (Live)"].notna()]
        else:
            new_df = pd.DataFrame()

        st.success("Live ratings check complete ✅")

        if key_idx[0] > 0:
            st.caption(f"🔁 Switched to backup OMDb key #{key_idx[0] + 1} of {len(omdb_keys)} partway through this run after hitting a rate limit.")

        # --- Diagnostics: make failures visible instead of a silently thin result ---
        checked_count = len(top250_films)
        matched_count = len(new_df)
        failed_count = checked_count - matched_count - non_english_skipped
        if checked_count > 0 and matched_count < checked_count * 0.5:
            hint = f' OMDb\'s own error message was: "{omdb_errors[0]}".' if omdb_errors else ""
            key_note = (
                f"All {len(omdb_keys)} configured OMDb key(s) appear rate-limited or invalid."
                if len(omdb_keys) > 1 else
                "This is almost always the OMDb API key being rate-limited or invalid — the key baked "
                "into this app is a shared public demo key with a low daily quota."
            )
            st.warning(
                f"Only {matched_count} of {checked_count} titles came back with a usable live rating "
                f"({non_english_skipped} skipped as non-English, {failed_count} failed outright).{hint} "
                f"{key_note} Add your own key via the \"🔑 OMDb API key\" box above, or add "
                "`OMDB_API_KEY` to your Streamlit secrets (comma-separate multiple keys for automatic "
                "fallback) — get a free one at omdbapi.com/apikey.aspx."
            )

        # --- Persist this run: Supabase if connected, otherwise a local CSV fallback ---
        if not new_df.empty:
            if supabase is not None:
                # pandas widens integer columns to float as soon as a merge
                # introduces a NaN, so "Num Votes" arrives here as 1075386.0 —
                # which Postgres rejects for a bigint column. Everything numeric
                # is coerced back to a plain Python int/float (never numpy
                # types, which aren't JSON-serialisable) before it's sent.
                # Deliberately tolerant: handles numpy scalars, strings like
                # "1075386.0", None and NaN alike.
                def _as_int(value):
                    try:
                        if value is None or pd.isna(value):
                            return None
                        return int(round(float(value)))
                    except (TypeError, ValueError):
                        return None

                def _as_float(value):
                    try:
                        if value is None or pd.isna(value):
                            return None
                        return float(value)
                    except (TypeError, ValueError):
                        return None

                records = [
                    {
                        "movie_id": r["Movie ID"],
                        "title": r["Title"],
                        "imdb_rating_static": _as_float(r["IMDb Rating (Static)"]),
                        "imdb_rating_live": _as_float(r["IMDb Rating (Live)"]),
                        "rating_diff": _as_float(r["Rating Difference"]),
                        "genre": r["Genre"],
                        "director": r["Director"],
                        "year": _as_int(r["Year"]),
                        "num_votes": _as_int(r["Num Votes"]),
                        "language": r["Language"],
                        "checked_at": r["CheckedAt"],
                    }
                    for r in new_df.to_dict(orient="records")
                ]
                try:
                    supabase.table("films").insert(records).execute()
                    st.caption(f"💾 Logged {len(records)} row(s) to Supabase (`films` table).")
                except Exception as e:
                    st.warning(f"Couldn't write to Supabase (`films` table): {e}")
                    # Show exactly what was sent — a type mismatch here is
                    # otherwise guesswork against Postgres's error message.
                    if records:
                        st.caption(
                            "First row sent: "
                            + ", ".join(
                                f"{k}={v!r} ({type(v).__name__})"
                                for k, v in records[0].items()
                            )
                        )
            else:
                if os.path.exists(history_file):
                    history_df = pd.read_csv(history_file)
                    combined = pd.concat([history_df, new_df], ignore_index=True)
                else:
                    combined = new_df
                combined.to_csv(history_file, index=False)
                st.caption("💾 Logged to a local CSV for this session (won't survive a redeploy — connect Supabase to persist).")

        # --- Show results, biggest changes first (nothing hidden by default) ---
        if not new_df.empty:
            section_label("📊 Current run — live ratings comparison")
            changed = int((new_df['Rating Difference'] != 0).sum())

            display_df = new_df[new_df['Rating Difference'] != 0].copy() if only_show_changed else new_df.copy()

            if display_df.empty:
                st.info("No titles changed this run — try unchecking 'Only show titles with a rating change' to see the full checked list.")
            else:
                display_df['Abs Change'] = display_df['Rating Difference'].abs()
                st.dataframe(
                    display_df.sort_values(by='Abs Change', ascending=False)
                    .drop(columns=['Abs Change']).reset_index(drop=True),
                    width="stretch"
                )
            st.caption(
                f"{changed} of {len(new_df)} checked titles show a different live rating today than the "
                f"stored snapshot (this is expected for high-vote films, whose rounded average barely "
                f"moves day-to-day)."
            )
        else:
            st.warning("No English-language films could be matched to a live rating this run — try again in a moment.")

        # --- Supervised ML: Predict My Ratings for Movies with Changed Live Ratings ---
        df_ml = IMDB_Ratings.merge(My_Ratings[['Movie ID','Your Rating']], on='Movie ID', how='left')
        df_ml = df_ml.merge(new_df[['Movie ID','Rating Difference']], on='Movie ID', how='left') if not new_df.empty else df_ml.assign(**{'Rating Difference': np.nan})

        # Only predict for unseen movies from the current top-250 subset that
        # actually had a live rating change (not just any successfully-checked
        # title - notna() alone let zero-diff rows through here).
        predict_df = df_ml[
            (df_ml['Movie ID'].isin(top250_films['Movie ID'])) &
            (df_ml['Rating Difference'].fillna(0) != 0) &
            (df_ml['Your Rating'].isna())
        ].copy()
        train_df = df_ml[df_ml['Your Rating'].notna()]

        categorical_features = ['Genre', 'Director']
        numerical_features = ['IMDb Rating', 'Num Votes', 'Year']

        if not predict_df.empty:
            X_pred = predict_df[categorical_features + numerical_features]

            if current_model_row is not None:
                try:
                    model = _load_production_model(current_model_row["storage_path"])
                    predict_df['Predicted Rating'] = model.predict(X_pred)
                    section_label("🤖 Predicted ratings — unseen films whose rating moved")
                    st.caption(
                        f"Served from the registered production model "
                        f"(validated RMSE {current_model_row['cv_rmse_mean']:.3f}) — "
                        f"not retrained on this click."
                    )
                except Exception as e:
                    st.warning(
                        f"Couldn't load the registered model ({e}) — falling back to a "
                        f"temporary session model instead."
                    )
                    current_model_row = None

            if current_model_row is None:
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
                        ('num', 'passthrough', numerical_features)
                    ]
                )
                fallback_model = Pipeline([
                    ('prep', preprocessor),
                    ('reg', RandomForestRegressor(n_estimators=100, random_state=42))
                ])
                X_train = train_df[categorical_features + numerical_features]
                y_train = train_df['Your Rating']
                fallback_model.fit(X_train, y_train)
                predict_df['Predicted Rating'] = fallback_model.predict(X_pred)
                section_label("🤖 Predicted ratings — unseen films whose rating moved")
                st.caption(
                    "⚠️ No registered production model — this is a temporary model trained "
                    "just for this session, not the validated registry model."
                )

            st.dataframe(
                predict_df[['Title','IMDb Rating','Genre','Director','Rating Difference','Predicted Rating']]
                .sort_values(by='Predicted Rating', ascending=False)
                .reset_index(drop=True),
                width="stretch"
            )
        else:
            st.info("No new movies available for prediction this run.")

    # --- Historical trend from Supabase (persists across runs and sessions) ---
    st.markdown("---")
    section_label("📈 Historical trend")
    if supabase is None:
        st.caption("Connect Supabase (see notice above) to see rating drift trends across every past run here.")
    else:
        try:
            hist = supabase.table("films").select("*").order("checked_at", desc=True).limit(5000).execute()
            hist_df = pd.DataFrame(hist.data)
            if not hist_df.empty:
                hist_df['checked_at'] = pd.to_datetime(hist_df['checked_at'])
                hist_df['Run Date'] = hist_df['checked_at'].dt.date.astype(str)

                n_runs = hist_df['Run Date'].nunique()
                c1, c2, c3 = st.columns(3)
                c1.metric("Rows logged", f"{len(hist_df):,}")
                c2.metric("Distinct run dates", n_runs)
                c3.metric("Titles with any live change", int((hist_df['rating_diff'] != 0).sum()))

                if n_runs < 2:
                    st.info(
                        "Only one run's worth of history so far, so there's nothing to trend yet — "
                        "a single point can't show drift over time. Run this again on a different day "
                        "(or let the scheduled CI/CD job build history) and a real trend line will "
                        "appear here."
                    )
                else:
                    trend = (
                        hist_df.groupby('Run Date')['rating_diff']
                        .mean().reset_index().sort_values('Run Date')
                    )
                    st.bar_chart(trend.set_index('Run Date'), color="#E3A857")
                    st.caption("Average live-vs-static rating difference per run date, oldest to newest.")

                with st.expander("📋 View all logged rows"):
                    st.dataframe(hist_df.drop(columns=['Run Date']), width="stretch", height=300)
            else:
                st.info("No rows logged yet — click **Run Live Ratings Check** above to start building history.")
        except Exception as e:
            st.warning(f"Couldn't read from Supabase (`films` table): {e}")

    # --- Explain how Python and packages make predictions ---
    st.markdown("""
**How the Predictions Work (Technical Explanation):**

1. **Data Preparation**
   - Features used: `Genre`, `Director` (categorical), `IMDb Rating`, `Num Votes`, `Year` (numerical).
   - `My Rating` is the target variable for supervised learning.

2. **Feature Encoding with `ColumnTransformer` and `OneHotEncoder`**
   - Categorical features are converted to **one-hot encoded vectors**.
   - Numerical features are passed through unchanged.

3. **Pipeline with `RandomForestRegressor`**
   - Combines preprocessing and model training.
   - Random forest is an **ensemble of decision trees**:
     - Each tree predicts independently.
     - The final prediction is the average across all trees.
     - This reduces overfitting and improves accuracy.

4. **Training**
   - Model learns patterns from movies I have rated (`Your Rating`).

5. **Prediction**
   - Model predicts ratings for movies I haven't rated based on learned patterns.

6. **Why this works**
   - Handles non-linear relationships and feature interactions naturally.
   - One-hot encoding allows categorical variables like directors and genres to be used.
   - Random forests are robust to overfitting and can generalize well to unseen movies.
""")

    with st.expander("⚙️ CI/CD + MLOps setup notes (Supabase + GitHub Actions)"):
        st.markdown("""
**1. `films` table** (live-ratings monitoring log) — unchanged from before:

```sql
create table if not exists films (
    id bigint generated always as identity primary key,
    movie_id text not null,
    title text,
    imdb_rating_static numeric,
    imdb_rating_live numeric,
    rating_diff numeric,
    genre text,
    director text,
    year int,
    num_votes bigint,
    language text,
    checked_at timestamptz not null default now()
);
```

**2. `model_runs` table** (model registry — see `model_runs_schema.sql`):

```sql
create table if not exists model_runs (
    id bigint generated always as identity primary key,
    trained_at timestamptz not null default now(),
    git_sha text,
    n_training_movies int,
    features text,
    cv_rmse_mean numeric,
    cv_rmse_std numeric,
    storage_path text not null,
    is_current boolean not null default false
);
```

**3. Storage bucket:** create a bucket named `models` (Supabase dashboard → Storage → New bucket,
private) — this is where trained model artifacts (`.joblib` files) live.

**4. Streamlit Cloud secrets** (Settings → Secrets): `SUPABASE_URL`, `SUPABASE_KEY` (anon key is
fine here — the app only *reads* `model_runs` and downloads from `models`), and `supabase` added
to `requirements.txt`.

**5. GitHub Actions secrets** (repo → Settings → Secrets and variables → Actions):
- `SUPABASE_URL`, `SUPABASE_KEY`, `OMDB_API_KEY` — used by `refresh_live_ratings.py` (unchanged).
- `SUPABASE_URL`, `SUPABASE_SERVICE_KEY` — used by `train_model.py`. This one needs the
  **service_role** key specifically, since it writes to `model_runs` and uploads to Storage
  regardless of RLS policy — never expose this key to the browser or put it in Streamlit secrets.

**6. Branch protection (the actual CD gate):** repo → Settings → Branches → add a rule for `main`
requiring the `CI` workflow's status check to pass before merging. This is a one-time UI setting,
not a file — it's what turns the `ci.yml` workflow from decorative into an actual gate on what
reaches production.

**7. First run:** once secrets are set, trigger **Train Ratings Model** manually once from the
Actions tab so a model gets registered — otherwise this scenario's predictions fall back to a
temporary session-only model (and will tell you so).
""")


# --- Scenario 9: Natural-Language Film Q&A Assistant (final version) ---

# --- Scenario 9: Natural-Language Film Q&A Assistant (final version, cleaned) ---

if scenario.startswith("9"):
    import streamlit as st
    import pandas as pd
    import textwrap
    import re
    import difflib

    scenario_header("9", "Natural-Language Film Q&amp;A Assistant")

    st.markdown("""
This scenario allows you to ask **natural-language questions** about my personal film ratings.

- When asking about directors, include only the **director’s surname** (last name) — small typos are okay.
- You can also filter by genre, using whatever genres actually appear in the data (comedy, fantasy, documentary, etc).
- Words like **top/highest/best** or **lowest/worst/bottom** control sorting.
- If nothing specific is recognized, you'll still get your full rated list back, sorted — never a blank page.
""")

    st.markdown(
        "**Example questions you can ask:**\n\n"
        "- Which Hitchcock films did I rate the highest?\n"
        "- Top films by Spielberg?\n"
        "- Which drama films did I rate the lowest?\n"
        "- Show me films by Cameron"
    )

    # (Both tables are already loaded, vote-filtered and merged at the top of
    # the file — this scenario used to re-read the raw workbooks here, which
    # quietly gave it a different dataset to every other scenario.)

    # --- Editable logic code ---
    logic_code = textwrap.dedent(r"""
        question_lower = user_question.lower()
        filtered = My_Ratings.copy()
        question_tokens = set(re.findall(r"\b[\w'-]+\b", question_lower))

        # Genre list is built from whatever genres actually appear in the data
        # (not a fixed shortlist), so Fantasy, Documentary, Crime, etc. all work.
        all_genres = sorted({
            g.strip().lower()
            for sub in My_Ratings['Genre'].dropna().str.split(',')
            for g in sub
        })
        matched_genres = [g for g in all_genres if g and g in question_lower]

        if matched_genres:
            pattern = '|'.join(re.escape(g) for g in matched_genres)
            filtered = filtered[filtered['Genre'].str.lower().str.contains(pattern, na=False)]

        # Director matching: exact surname first, then a fuzzy fallback so small
        # typos ("Nolen" instead of "Nolan") still resolve. Grouped as lists
        # because multiple directors can share a surname (e.g. James Cameron
        # and Cody Cameron) - a plain {surname: director} dict would silently
        # drop all but one of them.
        all_directors = My_Ratings['Director'].dropna().unique()
        surname_to_directors = {}
        for d in all_directors:
            last_name = d.split()[-1].lower()
            surname_to_directors.setdefault(last_name, []).append(d)

        director_matches = []
        for last_name, directors_sharing_name in surname_to_directors.items():
            if re.search(r'\b' + re.escape(last_name) + r'\b', question_lower):
                director_matches.extend(directors_sharing_name)

        if not director_matches:
            close = []
            for token in question_tokens:
                close += difflib.get_close_matches(token, surname_to_directors.keys(), n=1, cutoff=0.8)
            for c in dict.fromkeys(close):
                director_matches.extend(surname_to_directors[c])

        used_fallback = False
        if director_matches:
            filtered = filtered[filtered['Director'].isin(director_matches)]
        elif not matched_genres:
            # Nothing recognized in the question - show everything rather than
            # an empty table, so the assistant never comes back blank.
            used_fallback = True

        sort_col = "IMDb Rating" if "imdb" in question_tokens else "Your Rating"
        if any(w in question_tokens for w in ["highest", "top", "best"]):
            ascending = False
        elif any(w in question_tokens for w in ["lowest", "worst", "bottom"]):
            ascending = True
        else:
            ascending = False
    """)

    with st.expander("🛠️ View / edit the underlying code", expanded=False):
        editable_code = st.text_area("Modify logic if needed:", logic_code, height=400)

    user_question = st.text_input(
        "🎥 Ask a question:",
        placeholder="Which comedy films did I rate the highest?"
    )

    if user_question and not My_Ratings.empty:
        exec_ns = {"My_Ratings": My_Ratings, "user_question": user_question, "re": re, "difflib": difflib}
        try:
            exec(editable_code, exec_ns)
        except Exception as e:
            st.error(f"Error running logic: {e}")
            exec_ns.setdefault("filtered", My_Ratings.copy())
            exec_ns.setdefault("sort_col", "Your Rating")
            exec_ns.setdefault("ascending", False)
            exec_ns.setdefault("used_fallback", False)

        filtered = exec_ns.get("filtered", My_Ratings.copy())
        sort_col = exec_ns.get("sort_col", "Your Rating")
        ascending = exec_ns.get("ascending", False)
        used_fallback = exec_ns.get("used_fallback", False)

        if used_fallback:
            st.info("Couldn't pin down a specific director or genre in that question — showing your full rated list instead.")

        if not filtered.empty:
            filtered_sorted = filtered.sort_values(by=sort_col, ascending=ascending)
            st.dataframe(filtered_sorted)
        else:
            st.info("No films matched that director/genre combination. Try a different surname or genre keyword.")


# --- Scenario 15: Personalized Watchlist Ranker ---
if scenario == "15 – Personalized Watchlist Ranker":
    scenario_header("15", "Personalised Watchlist Ranker")
    st.write("""
    Rank unseen films with your own weighting — drag the sliders and the watchlist reorders live.
    Builds on the same idea as the Hybrid Recommendations scenario, but interactive instead of a fixed formula.
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        w_director = st.slider("Weight: liked directors", 0.0, 3.0, 1.0, 0.1)
    with col2:
        w_genre = st.slider("Weight: favorite genres", 0.0, 3.0, 1.0, 0.1)
    with col3:
        w_popularity = st.slider("Weight: popularity (votes)", 0.0, 3.0, 0.5, 0.1)

    min_votes = st.slider("Minimum IMDb votes", 0, 200000, 40000, step=5000)
    top_n = st.slider("How many to show", 5, 50, 15, step=5)

    s15_code_view = '''
liked_directors = set(My_Ratings.loc[My_Ratings['Your Rating'] >= 7, 'Director'].dropna())

rated_with_genre = IMDB_Ratings.merge(My_Ratings[['Movie ID', 'Your Rating']], on='Movie ID', how='inner')
favorite_genres = (
    rated_with_genre.loc[rated_with_genre['Your Rating'] >= 7, 'Genre']
    .dropna().str.split(',').explode().str.strip()
    .value_counts().head(5).index.tolist()
)

unseen = IMDB_Ratings.merge(My_Ratings[['Movie ID']], on='Movie ID', how='left', indicator=True)
unseen = unseen[unseen['_merge'] == 'left_only'].drop(columns=['_merge'])
unseen = unseen[unseen['Num Votes'] >= min_votes].copy()

def score_row(row):
    director_bonus = w_director if row['Director'] in liked_directors else 0
    genres = [g.strip() for g in str(row['Genre']).split(',')]
    genre_bonus = w_genre * sum(1 for g in genres if g in favorite_genres) / max(len(genres), 1)
    popularity_bonus = w_popularity * min(row['Num Votes'] / 200000, 1)
    return row['IMDb Rating'] + director_bonus + genre_bonus + popularity_bonus

unseen['Watchlist Score'] = unseen.apply(score_row, axis=1)
ranked = unseen.sort_values(by='Watchlist Score', ascending=False).head(top_n)
'''
    with st.expander("🛠️ View the underlying code", expanded=False):
        st.text_area(
            "Python code (view only — this box is not wired back into execution)",
            s15_code_view, height=360, key="s15_code_view",
        )
        st.caption("This is a reference copy of the code that actually runs below. Editing this box won't change the output.")

    if IMDB_Ratings.empty or My_Ratings.empty:
        st.warning("Need both tables loaded to build a watchlist.")
    else:
        liked_directors = set(My_Ratings.loc[My_Ratings['Your Rating'] >= 7, 'Director'].dropna())

        rated_with_genre = IMDB_Ratings.merge(My_Ratings[['Movie ID', 'Your Rating']], on='Movie ID', how='inner')
        favorite_genres = (
            rated_with_genre.loc[rated_with_genre['Your Rating'] >= 7, 'Genre']
            .dropna().str.split(',').explode().str.strip()
            .value_counts().head(5).index.tolist()
        )

        unseen = IMDB_Ratings.merge(My_Ratings[['Movie ID']], on='Movie ID', how='left', indicator=True)
        unseen = unseen[unseen['_merge'] == 'left_only'].drop(columns=['_merge'])
        unseen = unseen[unseen['Num Votes'] >= min_votes].copy()

        def score_row(row):
            director_bonus = w_director if row['Director'] in liked_directors else 0
            genres = [g.strip() for g in str(row['Genre']).split(',')]
            genre_bonus = w_genre * sum(1 for g in genres if g in favorite_genres) / max(len(genres), 1)
            popularity_bonus = w_popularity * min(row['Num Votes'] / 200000, 1)
            return row['IMDb Rating'] + director_bonus + genre_bonus + popularity_bonus

        unseen['Watchlist Score'] = unseen.apply(score_row, axis=1)
        ranked = unseen.sort_values(by='Watchlist Score', ascending=False).head(top_n)

        st.dataframe(
            ranked[['Title', 'IMDb Rating', 'Genre', 'Director', 'Year', 'Num Votes', 'Watchlist Score']]
            .round(2).reset_index(drop=True),
            width="stretch"
        )
        st.caption(
            f"Boosting directors you've rated 7+: {', '.join(sorted(liked_directors)) if liked_directors else 'none yet'} "
            f"· Favorite genres: {', '.join(favorite_genres) if favorite_genres else 'none yet'}"
        )


# --- Scenario 16: Similar Films Finder ---
if scenario == "16 – Similar Films Finder":
    scenario_header("16", "Similar Films Finder")
    st.write("""
    Pick a film and find unseen titles with the closest content profile — genre, director,
    era, and popularity — using nearest-neighbor search.
    """)

    if IMDB_Ratings.empty:
        st.warning("IMDb Ratings table is empty.")
    else:
        feature_df = IMDB_Ratings.dropna(
            subset=['Genre', 'Director', 'Year', 'IMDb Rating', 'Num Votes', 'Title']
        ).drop_duplicates(subset=['Title']).reset_index(drop=True)
        feature_df['Title'] = feature_df['Title'].astype(str)

        title_options = sorted(feature_df['Title'].unique().tolist())
        seed_title = st.selectbox("Find films similar to:", title_options)
        k = st.slider("How many matches", 3, 20, 8)

        s16_code_view = '''
from scipy.sparse import hstack, csr_matrix

encoder = OneHotEncoder(handle_unknown='ignore')
cat_encoded = encoder.fit_transform(feature_df[['Genre', 'Director']])

num_features = feature_df[['Year', 'IMDb Rating', 'Num Votes']].copy()
num_features = (num_features - num_features.mean()) / num_features.std().replace(0, 1)

X = hstack([cat_encoded, csr_matrix(num_features.values)]).tocsr()

nn = NearestNeighbors(n_neighbors=min(k + 1, len(feature_df)), metric='cosine')
nn.fit(X)

seed_idx = feature_df.index[feature_df['Title'] == seed_title][0]
distances, indices = nn.kneighbors(X[seed_idx])

matches = feature_df.iloc[indices[0][1:]].copy()
matches['Similarity'] = (1 - distances[0][1:]).round(3)
matches['Already Seen'] = matches['Movie ID'].isin(set(My_Ratings['Movie ID']))
'''
        with st.expander("🛠️ View the underlying code", expanded=False):
            st.text_area(
                "Python code (view only — this box is not wired back into execution)",
                s16_code_view, height=380, key="s16_code_view",
            )
            st.caption("This is a reference copy of the code that actually runs below. Editing this box won't change the output.")

        if st.button("🔎 Find similar films"):
            from scipy.sparse import hstack, csr_matrix

            encoder = OneHotEncoder(handle_unknown='ignore')
            cat_encoded = encoder.fit_transform(feature_df[['Genre', 'Director']])

            num_features = feature_df[['Year', 'IMDb Rating', 'Num Votes']].copy()
            num_features = (num_features - num_features.mean()) / num_features.std().replace(0, 1)

            X = hstack([cat_encoded, csr_matrix(num_features.values)]).tocsr()

            nn = NearestNeighbors(n_neighbors=min(k + 1, len(feature_df)), metric='cosine')
            nn.fit(X)

            seed_idx = feature_df.index[feature_df['Title'] == seed_title][0]
            distances, indices = nn.kneighbors(X[seed_idx])

            matches = feature_df.iloc[indices[0][1:]].copy()
            matches['Similarity'] = (1 - distances[0][1:]).round(3)

            seen_ids = set(My_Ratings['Movie ID']) if not My_Ratings.empty else set()
            matches['Already Seen'] = matches['Movie ID'].isin(seen_ids)

            st.dataframe(
                matches[['Title', 'Genre', 'Director', 'Year', 'IMDb Rating', 'Similarity', 'Already Seen']]
                .sort_values(by='Similarity', ascending=False)
                .reset_index(drop=True),
                width="stretch"
            )


# --- Scenario 17: Taste Profile Radar ---
if scenario == "17 – Taste Profile Radar":
    scenario_header("17", "Taste Profile Radar")
    st.write("A snapshot of which genres you rate highest and watch the most.")

    if My_Ratings.empty:
        st.warning("My Ratings table is empty.")
    else:
        genre_df = My_Ratings.assign(Genre=My_Ratings['Genre'].str.split(',')).explode('Genre')
        genre_df['Genre'] = genre_df['Genre'].str.strip()
        genre_stats = genre_df.groupby('Genre').agg(
            Avg_Rating=('Your Rating', 'mean'),
            Films_Watched=('Movie ID', 'count')
        ).reset_index()

        col_a, col_b = st.columns(2)
        with col_a:
            min_films = st.slider("Minimum films watched in genre", 1, 20, 3)
        with col_b:
            top_n_genres = st.slider("Show top N genres (by films watched)", 4, 15, 8)

        s17_code_view = '''
genre_df = My_Ratings.assign(Genre=My_Ratings['Genre'].str.split(',')).explode('Genre')
genre_df['Genre'] = genre_df['Genre'].str.strip()
genre_stats = genre_df.groupby('Genre').agg(
    Avg_Rating=('Your Rating', 'mean'),
    Films_Watched=('Movie ID', 'count')
).reset_index()

genre_stats = genre_stats[genre_stats['Films_Watched'] >= min_films]
genre_stats = genre_stats.sort_values('Films_Watched', ascending=False).head(top_n_genres)
genre_stats = genre_stats.sort_values('Avg_Rating', ascending=False)

# Radial axis is zoomed to the actual value spread, not a fixed 0-10 scale
lo = max(0, min(genre_stats['Avg_Rating']) - 1)
hi = min(10, max(genre_stats['Avg_Rating']) + 1)
'''
        with st.expander("🛠️ View the underlying code", expanded=False):
            st.text_area(
                "Python code (view only — this box is not wired back into execution)",
                s17_code_view, height=340, key="s17_code_view",
            )
            st.caption("This is a reference copy of the code that actually runs below. Editing this box won't change the output.")

        genre_stats = genre_stats[genre_stats['Films_Watched'] >= min_films]
        genre_stats = genre_stats.sort_values('Films_Watched', ascending=False).head(top_n_genres)
        genre_stats = genre_stats.sort_values('Avg_Rating', ascending=False)

        if genre_stats.empty:
            st.warning("Not enough data at this threshold — lower the minimum.")
        else:
            import numpy as np

            categories = genre_stats['Genre'].tolist()
            values = genre_stats['Avg_Rating'].tolist()
            counts = genre_stats['Films_Watched'].tolist()

            # Zoom the radial axis into the actual spread of the data instead of a
            # fixed 0-10 scale, so differences between genres are actually visible
            # rather than everything hugging the outer edge as a near-circle.
            lo = max(0, min(values) - 1)
            hi = min(10, max(values) + 1)

            values_closed = values + values[:1]
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            angles_closed = angles + angles[:1]

            fig, ax = plt.subplots(figsize=(6.0, 6.0), subplot_kw=dict(polar=True))
            ax.plot(angles_closed, values_closed, color='#E3A857', linewidth=2)
            ax.fill(angles_closed, values_closed, color='#E3A857', alpha=0.22)
            ax.set_xticks(angles)
            ax.set_xticklabels(categories, fontsize=8)
            ax.set_ylim(lo, hi)
            ax.tick_params(axis='y', labelsize=6.5)
            ax.spines['polar'].set_color("#2A2F3A")
            ax.grid(color="#2A2F3A", linewidth=0.7)

            for angle, value in zip(angles, values):
                ax.annotate(
                    f"{value:.1f}", xy=(angle, value), fontsize=6.5, color='#EDEEF0',
                    ha='center', va='bottom'
                )

            fig.tight_layout()

            chart_col, table_col = st.columns([1.15, 1])
            with chart_col:
                st.pyplot(fig)
                st.caption(
                    f"Axis is zoomed to {lo:.1f}–{hi:.1f} (not 0–10) so the shape reflects real "
                    f"differences between genres, not just how bunched-up your ratings are."
                )
            with table_col:
                st.dataframe(
                    genre_stats.round(2).reset_index(drop=True),
                    width="stretch", height=430,
                )
            plt.close(fig)


# --- Scenario 18: Prediction Outlier Detector ---
if scenario == "18 – Prediction Outlier Detector":
    scenario_header("18", "Prediction Outlier Detector")
    st.write("""
    Uses out-of-fold predictions to find films where your actual rating surprised the model most —
    the ones that broke your usual genre/director patterns.
    """)

    from sklearn.model_selection import KFold, cross_val_predict

    df_ml = IMDB_Ratings.merge(My_Ratings[['Movie ID', 'Your Rating']], on='Movie ID', how='inner')
    df_ml = df_ml.dropna(subset=['Genre', 'Director', 'Year', 'IMDb Rating', 'Num Votes', 'Your Rating'])

    if len(df_ml) < 10:
        st.warning("Need at least 10 rated films with complete data to run this.")
    else:
        categorical_features = ['Genre', 'Director']
        numerical_features = ['IMDb Rating', 'Num Votes', 'Year']

        preprocessor = ColumnTransformer(transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', 'passthrough', numerical_features)
        ])
        model = Pipeline([
            ('prep', preprocessor),
            ('reg', RandomForestRegressor(n_estimators=200, random_state=42))
        ])

        X = df_ml[categorical_features + numerical_features]
        y = df_ml['Your Rating']
        cv = KFold(n_splits=min(5, len(df_ml)), shuffle=True, random_state=42)
        oof_preds = cross_val_predict(model, X, y, cv=cv)

        df_ml = df_ml.copy()
        df_ml['Predicted (out-of-fold)'] = oof_preds.round(2)
        df_ml['Surprise'] = (df_ml['Your Rating'] - df_ml['Predicted (out-of-fold)']).round(2)
        df_ml['Abs Surprise'] = df_ml['Surprise'].abs()

        top_n = st.slider("How many outliers to show", 5, 30, 10)
        outliers = df_ml.sort_values('Abs Surprise', ascending=False).head(top_n)

        st.dataframe(
            outliers[['Title', 'Genre', 'Director', 'Your Rating', 'Predicted (out-of-fold)', 'Surprise']]
            .reset_index(drop=True),
            width="stretch"
        )
        st.caption("Positive Surprise = you liked it more than your usual pattern predicts. Negative = you liked it less.")


# --- Scenario 19: Tonight's Pick Roulette ---
if scenario == "19 – Tonight's Pick Roulette":
    scenario_header("19", "Tonight's Pick Roulette 🎰")
    st.write("Can't decide what to watch? Filter by mood, then spin.")

    if IMDB_Ratings.empty:
        st.warning("IMDb Ratings table is empty.")
    else:
        genres_available = sorted({
            g.strip() for sublist in IMDB_Ratings['Genre'].dropna().str.split(',') for g in sublist
        })
        mood_genre = st.multiselect("Mood / genre (optional)", genres_available)
        min_rating = st.slider("Minimum IMDb rating", 0.0, 9.5, 7.0, 0.1)
        unseen_only = st.checkbox("Only films I haven't rated yet", value=True)

        pool = IMDB_Ratings[IMDB_Ratings['IMDb Rating'] >= min_rating].copy()
        if mood_genre:
            pattern = '|'.join(mood_genre)
            pool = pool[pool['Genre'].str.contains(pattern, case=False, na=False)]
        if unseen_only and not My_Ratings.empty:
            pool = pool[~pool['Movie ID'].isin(My_Ratings['Movie ID'])]

        st.write(f"🎬 {len(pool)} films match your mood.")

        if st.button("🎲 Spin for a pick"):
            if pool.empty:
                st.warning("No films match — loosen a filter and try again.")
            else:
                weights = pool['IMDb Rating'] ** 2  # bias toward higher-rated picks
                pick = pool.sample(n=1, weights=weights).iloc[0]
                st.success(f"Tonight's pick: **{pick['Title']}** ({int(pick['Year']) if pd.notna(pick['Year']) else '—'})")
                c1, c2, c3 = st.columns(3)
                c1.metric("IMDb Rating", f"{pick['IMDb Rating']}")
                c2.metric("Director", pick['Director'] if pd.notna(pick['Director']) else "—")
                c3.metric("Genre", pick['Genre'] if pd.notna(pick['Genre']) else "—")


# --- Scenario 20: Ratings Timeline by Release Decade ---
if scenario == "20 – Ratings Timeline by Release Decade":
    scenario_header("20", "Ratings Timeline by Release Decade")
    st.write("How does your taste shift across different decades of film?")

    if IMDB_Ratings.empty or My_Ratings.empty:
        st.warning("Need both tables loaded for this.")
    else:
        compare = IMDB_Ratings.merge(My_Ratings[['Movie ID', 'Your Rating']], on='Movie ID', how='inner')
        compare = compare.dropna(subset=['Year'])
        compare['Decade'] = (compare['Year'].astype(int) // 10) * 10
        # Label as text ("1990s") rather than a bare int - otherwise Streamlit's
        # chart/dataframe auto-formatting adds thousands separators (e.g. "1,990").
        compare['Decade Label'] = compare['Decade'].astype(str) + "s"

        decade_stats = compare.groupby(['Decade', 'Decade Label']).agg(
            Films_Rated=('Movie ID', 'count'),
            Avg_My_Rating=('Your Rating', 'mean'),
            Avg_IMDb_Rating=('IMDb Rating', 'mean'),
        ).reset_index().sort_values('Decade')

        min_films = st.slider("Minimum films rated in decade", 1, 15, 2)
        decade_stats = decade_stats[decade_stats['Films_Rated'] >= min_films]

        if decade_stats.empty:
            st.warning("Not enough data at this threshold — lower the minimum.")
        else:
            chart_df = decade_stats.set_index('Decade Label')[['Avg_My_Rating', 'Avg_IMDb_Rating']]
            st.bar_chart(chart_df, color=["#E3A857", "#4FA88F"])
            st.dataframe(
                decade_stats[['Decade Label', 'Films_Rated', 'Avg_My_Rating', 'Avg_IMDb_Rating']]
                .rename(columns={'Decade Label': 'Decade'})
                .round(2).reset_index(drop=True),
                width="stretch"
            )
            st.caption("Where your bar rises above IMDb's, you rate that decade more generously than the crowd — and vice versa.")

            st.markdown("---")
            section_label("🔍 Drill into a decade")
            decade_choice = st.selectbox("Pick a decade", decade_stats['Decade Label'].tolist())
            decade_films = compare[compare['Decade Label'] == decade_choice].sort_values('Your Rating', ascending=False)
            st.dataframe(
                decade_films[['Title', 'Your Rating', 'IMDb Rating', 'Genre', 'Director']].reset_index(drop=True),
                width="stretch"
            )


# --- Footer ---
st.markdown(
    '<div class="app-footer">'
    '<span>IMDb Data &amp; AI Playground — built with Streamlit, pandas, scikit-learn and the OMDb API.</span>'
    '<span>Ratings data: my own IMDb export · Films with more than {:,} votes only · '
    'Build {}</span>'
    '</div>'.format(MIN_VOTES, APP_BUILD),
    unsafe_allow_html=True,
)
