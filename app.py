# app.py — Advanced Scouting + Notes + Comparison Radar + Similar Players + Club Fit
# Single file, drop-in. Requires: streamlit, pandas, numpy, matplotlib.
# scikit-learn is optional; a tiny StandardScaler fallback is included.

import os
import math
from pathlib import Path
import re

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime  # <-- add me
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, Wedge

# ---- Optional sklearn (fallback provided) ----
try:
    from sklearn.preprocessing import StandardScaler
except Exception:
    class StandardScaler:  # minimal drop-in
        def __init__(self): self.mean_ = None; self.scale_ = None
        def fit(self, X):
            X = np.asarray(X, dtype=float)
            self.mean_ = X.mean(axis=0)
            std = X.std(axis=0, ddof=0)
            std[std == 0] = 1.0
            self.scale_ = std
            return self
        def transform(self, X):
            X = np.asarray(X, dtype=float)
            return (X - self.mean_) / self.scale_
        def fit_transform(self, X):
            self.fit(X); return self.transform(X)

# ----------------- PAGE -----------------
st.set_page_config(page_title="Advanced Attacker Scouting System", layout="wide")
st.title("🔎 Advanced Attacker Scouting System")
st.caption("Use the sidebar to shape your pool. Each section explains what you’re seeing and why.")

# ----------------- CONFIG -----------------
INCLUDED_LEAGUES = [
    'England 1.', 'England 2.', 'England 3.', 'England 4.', 'England 5.',
    'England 6.', 'England 7.', 'England 8.', 'England 9.', 'England 10.',
    'Albania 1.', 'Algeria 1.', 'Andorra 1.', 'Argentina 1.', 'Armenia 1.',
    'Australia 1.', 'Austria 1.', 'Austria 2.', 'Azerbaijan 1.', 'Belgium 1.',
    'Belgium 2.', 'Bolivia 1.', 'Bosnia 1.', 'Brazil 1.', 'Brazil 2.', 'Brazil 3.',
    'Bulgaria 1.', 'Canada 1.', 'Chile 1.', 'Colombia 1.', 'Costa Rica 1.',
    'Croatia 1.', 'Cyprus 1.', 'Czech 1.', 'Czech 2.', 'Denmark 1.', 'Denmark 2.',
    'Ecuador 1.', 'Egypt 1.', 'Estonia 1.', 'Finland 1.', 'France 1.', 'France 2.',
    'France 3.', 'Georgia 1.', 'Germany 1.', 'Germany 2.', 'Germany 3.', 'Germany 4.',
    'Greece 1.', 'Hungary 1.', 'Iceland 1.', 'Israel 1.', 'Israel 2.', 'Italy 1.',
    'Italy 2.', 'Italy 3.', 'Japan 1.', 'Japan 2.', 'Kazakhstan 1.', 'Korea 1.',
    'Latvia 1.', 'Lithuania 1.', 'Malta 1.', 'Mexico 1.', 'Moldova 1.', 'Morocco 1.',
    'Netherlands 1.', 'Netherlands 2.', 'North Macedonia 1.', 'Northern Ireland 1.',
    'Norway 1.', 'Norway 2.', 'Paraguay 1.', 'Peru 1.', 'Poland 1.', 'Poland 2.',
    'Portugal 1.', 'Portugal 2.', 'Portugal 3.', 'Qatar 1.', 'Ireland 1.', 'Romania 1.',
    'Russia 1.', 'Saudi 1.', 'Scotland 1.', 'Scotland 2.', 'Scotland 3.', 'Serbia 1.',
    'Serbia 2.', 'Slovakia 1.', 'Slovakia 2.', 'Slovenia 1.', 'Slovenia 2.', 'South Africa 1.',
    'Spain 1.', 'Spain 2.', 'Spain 3.', 'Sweden 1.', 'Sweden 2.', 'Switzerland 1.',
    'Switzerland 2.', 'Tunisia 1.', 'Turkey 1.', 'Turkey 2.', 'Ukraine 1.', 'UAE 1.',
    'USA 1.', 'USA 2.', 'Uruguay 1.', 'Uzbekistan 1.', 'Venezuela 1.', 'Wales 1.'
]

PRESET_LEAGUES = {
    "Top 5 Europe": {'England 1.', 'France 1.', 'Germany 1.', 'Italy 1.', 'Spain 1.'},
    "Top 20 Europe": {
        'England 1.','Italy 1.','Spain 1.','Germany 1.','France 1.',
        'England 2.','Portugal 1.','Belgium 1.','Turkey 1.','Germany 2.','Spain 2.','France 2.',
        'Netherlands 1.','Austria 1.','Switzerland 1.','Denmark 1.','Croatia 1.','Italy 2.','Czech 1.','Norway 1.'
    },
    "EFL (England 2–4)": {'England 2.','England 3.','England 4.'}
}

FEATURES = [
    'Defensive duels per 90','Aerial duels won, %', 'Non-penalty goals per 90',
    'xG per 90','Shots per 90','Shots on target, %', 'Crosses per 90','Accurate crosses, %','Dribbles per 90',
    'Successful dribbles, %', 'Offensive duels per 90','Offensive duels won, %','Touches in box per 90',
    'Progressive runs per 90','Accelerations per 90','Passes per 90','Accurate passes, %', 'Forward passes per 90',
    'Long passes per 90','xA per 90','Smart passes per 90','Key passes per 90',
    'Passes to final third per 90','Accurate passes to final third, %',
    'Passes to penalty area per 90','Accurate passes to penalty area, %', 
    'Deep completions per 90','Progressive passes per 90','Accurate progressive passes, %' ]

POLAR_METRICS = [
"Non-penalty goals per 90","xG per 90","Shots per 90", "Dribbles per 90","Passes to penalty area per 90",
"Touches in box per 90", "Smart passes per 90","Progressive passes per 90",
"Passes per 90", "Accurate passes, %","xA per 90","Progressive runs per 90",
]

# -------- Position filter (attacker) --------
# Define the prefixes that truly need startswith
prefixes = ('RWF', 'LWF', 'LAMF', 'RAMF', 'AMF', 'RW, ', 'LW, ')

def position_filter(pos):
    pos_clean = str(pos).strip().upper()
    if pos_clean in ('RW', 'LW'):      # exact match for RW and LW
        return True
    if pos_clean.startswith(prefixes): # startswith for others
        return True
    return False
# -------------------------------------------

# Role buckets
ROLES = {
    'Playmaker': {
        'desc': "Primary creator: volume passing, xA, and delivery into dangerous zones.",
        'metrics': {
            'Passes per 90': 2,
            'xA per 90': 3,
            'Key passes per 90': 1,
            'Deep completions per 90': 1.5,
            'Smart passes per 90': 1.5,
            'Passes to penalty area per 90': 2,
        },
    },
    'Goal Threat': {
        'desc': "High goal expectation and shot volume with strong box presence.",
        'metrics': {
            'xG per 90': 3,
            'Non-penalty goals per 90': 3,
            'Shots per 90': 2,
            'Touches in box per 90': 2,
        },
    },
    'Ball Carrier': {
        'desc': "Progresses play by beating players and advancing the ball repeatedly.",
        'metrics': {
            'Dribbles per 90': 4,
            'Successful dribbles, %': 2,
            'Progressive runs per 90': 3,
            'Accelerations per 90': 3,
        },
    },
    'All In': {
        'desc': "Balanced scorer-creator blend across xG/xA, dribbling, and end product.",
        'metrics': {
            'xA per 90': 3,
            'Dribbles per 90': 3,
            'xG per 90': 2,
            'Non-penalty goals per 90': 3,
        },
    },
}

LEAGUE_STRENGTHS = {
    'England 1.':100.00,'Italy 1.':97.14,'Spain 1.':94.29,'Germany 1.':94.29,'France 1.':91.43,
    'Brazil 1.':82.86,'England 2.':71.43,'Portugal 1.':71.43,'Argentina 1.':71.43,
    'Belgium 1.':68.57,'Mexico 1.':68.57,'Turkey 1.':65.71,'Germany 2.':65.71,'Spain 2.':65.71,
    'France 2.':65.71,'USA 1.':65.71,'Russia 1.':65.71,'Colombia 1.':62.86,'Netherlands 1.':62.86,
    'Austria 1.':62.86,'Switzerland 1.':62.86,'Denmark 1.':62.86,'Croatia 1.':62.86,
    'Japan 1.':62.86,'Korea 1.':62.86,'Italy 2.':62.86,'Czech 1.':57.14,'Norway 1.':57.14,
    'Poland 1.':57.14,'Romania 1.':57.14,'Israel 1.':57.14,'Algeria 1.':57.14,'Paraguay 1.':57.14,
    'Saudi 1.':57.14,'Uruguay 1.':57.14,'Morocco 1.':57.00,'Brazil 2.':56.00,'Ukraine 1.':55.00,
    'Ecuador 1.':54.29,'Spain 3.':54.29,'Scotland 1.':58.00,'Chile 1.':51.43,'Cyprus 1.':51.43,
    'Portugal 2.':51.43,'Slovakia 1.':51.43,'Australia 1.':51.43,'Hungary 1.':51.43,'Egypt 1.':51.43,
    'England 3.':51.43,'France 3.':48.00,'Japan 2.':48.00,'Bulgaria 1.':48.57,'Slovenia 1.':48.57,
    'Venezuela 1.':48.00,'Germany 3.':45.71,'Albania 1.':44.00,'Serbia 1.':42.86,'Belgium 2.':42.86,
    'Bosnia 1.':42.86,'Kosovo 1.':42.86,'Nigeria 1.':42.86,'Azerbaijan 1.':50.00,'Bolivia 1.':50.00,
    'Costa Rica 1.':50.00,'South Africa 1.':50.00,'UAE 1.':50.00,'Georgia 1.':40.00,'Finland 1.':40.00,
    'Italy 3.':40.00,'Peru 1.':40.00,'Tunisia 1.':40.00,'USA 2.':40.00,'Armenia 1.':40.00,
    'North Macedonia 1.':40.00,'Qatar 1.':40.00,'Uzbekistan 1.':42.00,'Norway 2.':42.00,
    'Kazakhstan 1.':42.00,'Poland 2.':38.00,'Denmark 2.':37.00,'Czech 2.':37.14,'Israel 2.':37.14,
    'Netherlands 2.':37.14,'Switzerland 2.':37.14,'Iceland 1.':34.29,'Ireland 1.':34.29,'Sweden 2.':34.29,
    'Germany 4.':34.29,'Malta 1.':30.00,'Turkey 2.':31.43,'Canada 1.':28.57,'England 4.':28.57,
    'Scotland 2.':28.57,'Moldova 1.':28.57,'Austria 2.':25.71,'Lithuania 1.':25.71,'Brazil 3.':25.00,
    'England 7.':25.00,'Slovenia 2.':22.00,'Latvia 1.':22.86,'Serbia 2.':20.00,'Slovakia 2.':20.00,
    'England 9.':20.00,'England 8.':15.00,'Montenegro 1.':14.29,'Wales 1.':12.00,'Portugal 3.':11.43,
    'Northern Ireland 1.':11.43,'England 10.':10.00,'Scotland 3.':10.00,'England 6.':10.00
}
REQUIRED_BASE = {"Player","Team","League","Age","Position","Minutes played","Market value","Contract expires","Goals"}

# ----------------- DATA LOADER -----------------
@st.cache_data(show_spinner=False)
def load_df(csv_name="WORLDJUNE25.csv"):
    p = Path(__file__).with_name(csv_name)
    if p.exists():
        return pd.read_csv(p)
    up = st.file_uploader("Upload WORLDJUNE25.csv", type=["csv"])
    if not up:
        st.stop()
    return pd.read_csv(up)

df = load_df()

# ----------------- SIDEBAR FILTERS -----------------
# ------- in the sidebar, replace your current preset UI -------
with st.sidebar:
    st.header("Filters")

    # segmented preset choice
    preset_choice = st.segmented_control(
        "League preset",
        options=["All", "Top-5", "Top-20", "EFL", "Custom"],
        help="Pick a preset, then prune or add leagues below.",
        key="league_preset_seg"
    )

    # resolve presets -> seed set
    if preset_choice == "All":
        seed = set(INCLUDED_LEAGUES)
    elif preset_choice == "Top-5":
        seed = set(PRESET_LEAGUES["Top 5 Europe"])
    elif preset_choice == "Top-20":
        seed = set(PRESET_LEAGUES["Top 20 Europe"])
    elif preset_choice == "EFL":
        seed = set(PRESET_LEAGUES["EFL (England 2–4)"])
    else:
        seed = set()  # Custom starts empty

    # multiselect shows chips you can prune (like the screenshot)
    leagues_avail = sorted(set(INCLUDED_LEAGUES) | set(df.get("League", pd.Series([])).dropna().unique()))
    leagues_sel = st.multiselect("Leagues (add or prune)", leagues_avail, default=sorted(seed) if seed else leagues_avail)

    st.caption(f"Selected: **{len(leagues_sel)}** league(s)")


    # numeric coercions
    df["Minutes played"] = pd.to_numeric(df["Minutes played"], errors="coerce")
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    min_minutes, max_minutes = st.slider("Minutes played", 0, 5000, (1000, 5000))
    age_min_data = int(np.nanmin(df["Age"])) if df["Age"].notna().any() else 14
    age_max_data = int(np.nanmax(df["Age"])) if df["Age"].notna().any() else 45
    min_age, max_age = st.slider("Age", age_min_data, age_max_data, (16, 40))

    pos_text = st.text_input("Position startswith", "CF")

    # Defaults OFF; league beta default shown as 0.40 but toggle unticked
    apply_contract = st.checkbox("Filter by contract expiry", value=False)
    cutoff_year = st.slider("Max contract year (inclusive)", 2025, 2030, 2026)

    min_strength, max_strength = st.slider("League quality (strength)", 0, 101, (0, 101))
    use_league_weighting = st.checkbox("Use league weighting in role score", value=False)
    beta = st.slider("League weighting beta", 0.0, 1.0, 0.40, 0.05,
                     help="0 = ignore league strength; 1 = only league strength")

    # Market value
    df["Market value"] = pd.to_numeric(df["Market value"], errors="coerce")
    mv_col = "Market value"
    mv_max_raw = int(np.nanmax(df[mv_col])) if df[mv_col].notna().any() else 50_000_000
    mv_cap = int(math.ceil(mv_max_raw / 5_000_000) * 5_000_000)
    st.markdown("**Market value (€)**")
    use_m = st.checkbox("Adjust in millions", True)
    if use_m:
        max_m = int(mv_cap // 1_000_000)
        mv_min_m, mv_max_m = st.slider("Range (M€)", 0, max_m, (0, max_m))
        min_value = mv_min_m * 1_000_000
        max_value = mv_max_m * 1_000_000
    else:
        min_value, max_value = st.slider("Range (€)", 0, mv_cap, (0, mv_cap), step=100_000)
    value_band_max = st.number_input("Value band (tab 4 max €)", min_value=0,
                                     value=min_value if min_value>0 else 5_000_000, step=250_000)

    st.subheader("Minimum performance thresholds")
    enable_min_perf = st.checkbox("Require minimum percentile on selected metrics", value=False)
    sel_metrics = st.multiselect("Metrics to threshold", FEATURES[:],
                                 default=['Non-penalty goals per 90','xG per 90'] if enable_min_perf else [])
    min_pct = st.slider("Minimum percentile (0–100)", 0, 100, 60)

    top_n = st.number_input("Top N per table", 5, 200, 50, 5)
    round_to = st.selectbox("Round output percentiles to", [0, 1], index=0)

# ----------------- VALIDATION -----------------
missing = [c for c in REQUIRED_BASE if c not in df.columns]
if missing:
    st.error(f"Dataset missing required base columns: {missing}")
    st.stop()
missing_feats = [c for c in FEATURES if c not in df.columns]
if missing_feats:
    st.error(f"Dataset missing required feature columns: {missing_feats}")
    st.stop()

# ----------------- FILTER POOL -----------------
df_f = df[df["League"].isin(leagues_sel)].copy()
df_f = df_f[df_f["Position"].astype(str).apply(position_filter)]
df_f = df_f[df_f["Minutes played"].between(min_minutes, max_minutes)]
df_f = df_f[df_f["Age"].between(min_age, max_age)]
df_f = df_f.dropna(subset=FEATURES)

df_f["Contract expires"] = pd.to_datetime(df_f["Contract expires"], errors="coerce")
if apply_contract:
    df_f = df_f[df_f["Contract expires"].dt.year <= cutoff_year]

df_f["League Strength"] = df_f["League"].map(LEAGUE_STRENGTHS).fillna(0.0)
df_f = df_f[(df_f["League Strength"] >= float(min_strength)) & (df_f["League Strength"] <= float(max_strength))]
df_f = df_f[(df_f["Market value"] >= min_value) & (df_f["Market value"] <= max_value)]
if df_f.empty:
    st.warning("No players after filters. Loosen filters.")
    st.stop()

# ----------------- PERCENTILES FOR TABLES (per league) -----------------
for c in FEATURES:
    df_f[c] = pd.to_numeric(df_f[c], errors="coerce")
df_f = df_f.dropna(subset=FEATURES)
for feat in FEATURES:
    df_f[f"{feat} Percentile"] = df_f.groupby("League")[feat].transform(lambda x: x.rank(pct=True) * 100.0)

# ----------------- ROLE SCORING (tables) -----------------
def compute_weighted_role_score(df_in: pd.DataFrame, metrics: dict, beta: float, league_weighting: bool) -> pd.Series:
    total_w = sum(metrics.values()) if metrics else 1.0
    wsum = np.zeros(len(df_in))
    for m, w in metrics.items():
        col = f"{m} Percentile"
        if col in df_in.columns:
            wsum += df_in[col].values * w
    player_score = wsum / total_w  # 0..100
    if league_weighting:
        league_scaled = (df_in["League Strength"].fillna(50) / 100.0) * 100.0
        return (1 - beta) * player_score + beta * league_scaled
    return player_score

for role_name, role_def in ROLES.items():
    df_f[f"{role_name} Score"] = compute_weighted_role_score(df_f, role_def["metrics"], beta=beta, league_weighting=use_league_weighting)

# ----------------- THRESHOLDS -----------------
if enable_min_perf and sel_metrics:
    keep_mask = np.ones(len(df_f), dtype=bool)
    for m in sel_metrics:
        pct_col = f"{m} Percentile"
        if pct_col in df_f.columns:
            keep_mask &= (df_f[pct_col] >= min_pct)
    df_f = df_f[keep_mask]
    if df_f.empty:
        st.warning("No players meet the minimum performance thresholds. Loosen thresholds.")
        st.stop()

# ----------------- HELPERS -----------------
def fmt_cols(df_in: pd.DataFrame, score_col: str) -> pd.DataFrame:
    out = df_in.copy()
    out[score_col] = out[score_col].round(round_to).astype(int if round_to == 0 else float)
    cols = ["Player","Team","League","Position", "Age","Contract expires","League Strength", score_col]
    return out[cols]

def top_table(df_in: pd.DataFrame, role: str, head_n: int) -> pd.DataFrame:
    col = f"{role} Score"
    ranked = df_in.dropna(subset=[col]).sort_values(col, ascending=False)
    ranked = fmt_cols(ranked, col).head(head_n).reset_index(drop=True)
    ranked.index = np.arange(1, len(ranked)+1)
    return ranked

def filtered_view(df_in: pd.DataFrame, *, age_max=None, contract_year=None, value_max=None):
    t = df_in.copy()
    if age_max is not None:
        t = t[t["Age"] <= age_max]
    if contract_year is not None:
        t = t[t["Contract expires"].dt.year <= contract_year]
    if value_max is not None:
        t = t[t["Market value"] <= value_max]
    return t

# ----------------- TABS (tables) -----------------
tabs = st.tabs(["Overall Top N", "U23 Top N", "Expiring Contracts", "Value Band (≤ max €)"])
for role, role_def in ROLES.items():
    with tabs[0]:
        st.subheader(f"{role} — Overall Top {int(top_n)}")
        st.caption(role_def.get("desc", ""))
        st.dataframe(top_table(df_f, role, int(top_n)), use_container_width=True)
        st.divider()
    with tabs[1]:
        u23_cutoff = st.number_input(f"{role} — U23 cutoff", min_value=16, max_value=30, value=23, step=1, key=f"u23_{role}")
        st.subheader(f"{role} — U{u23_cutoff} Top {int(top_n)}")
        st.caption(role_def.get("desc", ""))
        st.dataframe(top_table(filtered_view(df_f, age_max=u23_cutoff), role, int(top_n)), use_container_width=True)
        st.divider()
    with tabs[2]:
        exp_year = st.number_input(f"{role} — Expiring by year", min_value=2024, max_value=2030, value=cutoff_year, step=1, key=f"exp_{role}")
        st.subheader(f"{role} — Contracts expiring ≤ {exp_year}")
        st.caption(role_def.get("desc", ""))
        st.dataframe(top_table(filtered_view(df_f, contract_year=exp_year), role, int(top_n)), use_container_width=True)
        st.divider()
    with tabs[3]:
        v_max = st.number_input(f"{role} — Max value (€)", min_value=0, value=value_band_max, step=100_000, key=f"val_{role}")
        st.subheader(f"{role} — Value band ≤ €{v_max:,.0f}")
        st.caption(role_def.get("desc", ""))
        st.dataframe(top_table(filtered_view(df_f, value_max=v_max), role, int(top_n)), use_container_width=True)
        st.divider()

# ----------------- SINGLE PLAYER ROLE PROFILE (REPLACED) -----------------
st.subheader("🎯 Single Player Role Profile")
player_name = st.selectbox("Choose player", sorted(df_f["Player"].unique()))
player_row = df_f[df_f["Player"] == player_name].head(1)

# derive defaults from selected player (to propagate)
default_pos_prefix = str(player_row["Position"].iloc[0])[:2] if not player_row.empty else "CF"
default_league_for_pool = [player_row["League"].iloc[0]] if not player_row.empty else []

# Pool controls (for chart + notes only; NOT used for role scores)
st.caption("Percentiles & chart computed against the pool below (defaults to the player's league).")
with st.container():
    c1, c2, c3 = st.columns([2,1,1])
    leagues_pool = c1.multiselect("Comparison leagues", sorted(df["League"].dropna().unique()), default=default_league_for_pool)
    min_minutes_pool, max_minutes_pool = c2.slider("Pool minutes", 0, 5000, (1000, 5000))
    age_min_pool, age_max_pool = c3.slider("Pool age", 14, 45, (16, 40))  # default 16–40
    same_pos = st.checkbox("Limit pool to current position prefix", value=True)

def build_pool_df():
    if not leagues_pool:
        return pd.DataFrame([], columns=df.columns)
    pool = df[df["League"].isin(leagues_pool)].copy()
    pool["Minutes played"] = pd.to_numeric(pool["Minutes played"], errors="coerce")
    pool["Age"] = pd.to_numeric(pool["Age"], errors="coerce")
    pool = pool[pool["Minutes played"].between(min_minutes_pool, max_minutes_pool)]
    pool = pool[pool["Age"].between(age_min_pool, age_max_pool)]
    if same_pos and not player_row.empty:
        pool = pool[pool["Position"].astype(str).apply(position_filter)]
    pool = pool.dropna(subset=POLAR_METRICS)
    return pool

def clean_attacker_label(s: str) -> str:
    s = s.replace("Non-penalty goals per 90", "Non-Pen Goals")
    s = s.replace("xG per 90", "xG").replace("xA per 90", "xA")
    s = s.replace("Shots per 90", "Shots")
    s = s.replace("Passes per 90", "Passes")
    s = s.replace("Touches in box per 90", "Touches in box")
    s = s.replace("Smart passes per 90", "Smart Passes")
    s = s.replace("Progressive passes per 90", "Progressive Passes")
    s = s.replace("Progressive runs per 90", "Progressive runs")
    s = s.replace("Passes to penalty area per 90", "Passes to Pen area")
    s = s.replace("Accurate passes, %", "Pass %")
    return s

def percentiles_for_player_in_pool(pool_df: pd.DataFrame, ply_row: pd.Series) -> dict:
    if pool_df.empty:
        return {}
    pct_map = {}
    for m in POLAR_METRICS:
        if m not in pool_df.columns or pd.isna(ply_row[m]): 
            continue
        series = pd.to_numeric(pool_df[m], errors="coerce").dropna()
        if series.empty: 
            continue
        rank = (series < float(ply_row[m])).mean() * 100.0
        eq_share = (series == float(ply_row[m])).mean() * 100.0
        pct_map[m] = min(100.0, rank + 0.5 * eq_share)
    return pct_map

# Polar chart for attacker metrics
def plot_attacker_polar_chart(labels, vals):
    N = len(labels)
    color_scale = ["#be2a3e", "#e25f48", "#f88f4d", "#f4d166", "#90b960", "#4b9b5f", "#22763f"]
    cmap = LinearSegmentedColormap.from_list("custom_scale", color_scale)
    bar_colors = [cmap(v/100.0) for v in vals]

    angles = np.linspace(0, 2*np.pi, N, endpoint=False)[::-1]
    rotation_shift = np.deg2rad(75) - angles[0]
    ang = (angles + rotation_shift) % (2*np.pi)
    width = 2*np.pi / N

    fig = plt.figure(figsize=(8.2, 6.6), dpi=180)
    fig.patch.set_facecolor('#f3f4f6')
    ax = fig.add_axes([0.06, 0.08, 0.88, 0.74], polar=True)
    ax.set_facecolor('#f3f4f6')
    ax.set_rlim(0, 100)

    for i in range(N):
        ax.bar(ang[i], vals[i], width=width, color=bar_colors[i], edgecolor='black', linewidth=1.0, zorder=3)
        label_pos = max(12, vals[i] * 0.75)
        ax.text(ang[i], label_pos, f"{int(round(vals[i]))}", ha='center', va='center',
                fontsize=9, weight='bold', color='white', zorder=4)

    outer = plt.Circle((0, 0), 100, transform=ax.transData._b, color='black', fill=False, linewidth=2.2, zorder=5)
    ax.add_artist(outer)
    for i in range(N):
        sep_angle = (ang[i] - width/2) % (2*np.pi)
        is_cross = any(np.isclose(sep_angle, a, atol=0.01) for a in [0, np.pi/2, np.pi, 3*np.pi/2])
        ax.plot([sep_angle, sep_angle], [0, 100], color='black' if is_cross else '#b0b0b0',
                linewidth=1.6 if is_cross else 1.0, zorder=2)

    label_r = 120
    for i, lab in enumerate(labels):
        ax.text(ang[i], label_r, lab, ha='center', va='center', fontsize=8.5, weight='bold', color='#111827', zorder=6)

    ax.set_xticks([]); ax.set_yticks([])
    ax.spines['polar'].set_visible(False); ax.grid(False)
    return fig

# ---- render section ----
if player_row.empty:
    st.info("Pick a player above.")
else:
    ply = player_row.iloc[0]
    meta = player_row[["Team","League","Age","Contract expires","League Strength","Market value"]].iloc[0]
    st.caption(
        f"**{player_name}** — {meta['Team']} • {meta['League']} • Age {int(meta['Age'])} • "
        f"Contract: {pd.to_datetime(meta['Contract expires']).date() if pd.notna(meta['Contract expires']) else 'N/A'} • "
        f"League Strength {meta['League Strength']:.1f} • Value €{meta['Market value']:,.0f}"
    )

    # Build pool & compute player percentiles within that pool
    pool_df = build_pool_df()
    if pool_df.empty:
        st.warning("Comparison pool is empty. Add at least one league.")
        pct_map = {}
    else:
        pct_map = percentiles_for_player_in_pool(pool_df, ply)

    # ---------- 1) PERFORMANCE CHART FIRST ----------
    labels = [clean_attacker_label(m) for m in POLAR_METRICS if m in pct_map]
    vals   = [pct_map[m] for m in POLAR_METRICS if m in pct_map]
    if vals:
        fig = plot_attacker_polar_chart(labels, vals)
        team = str(ply["Team"]); league = str(ply["League"])
        fig.text(0.06, 0.94, f"{player_name} — Performance (pool size: {len(pool_df):,})",
                 fontsize=16, weight='bold', ha='left', color='#111827')
        fig.text(0.06, 0.915, f"Against selected pool • Team: {team} • Native league: {league}",
                 fontsize=9, ha='left', color='#6b7280')
        st.pyplot(fig, use_container_width=True)

   # ---------- 2) NOTES: Style / Strengths / Weaknesses ----------

EXTRA_METRICS = [
    'Defensive duels per 90','Aerial duels per 90','Aerial duels won, %', 'Non-penalty goals per 90',
    'xG per 90','Shots per 90','Goal conversion, %', 'Crosses per 90','Accurate crosses, %','Dribbles per 90','Successful dribbles, %',
    'Touches in box per 90',
    'Progressive runs per 90','Passes per 90','Accurate passes, %', 'xA per 90','Passes to penalty area per 90',
    'Deep completions per 90','Smart passes per 90'
]
STYLE_MAP = {
    'Defensive duels per 90': {
        'style': 'High work rate',
        'sw': 'Defensive Duels',
    },
    'Aerial duels won, %': {
        'style': None,
        'sw': 'Aerial Duels',
    },
    'Aerial duels per 90': {
        'style': 'Long Reference Point',
        'sw': None,
    },
    'Non-penalty goals per 90': {
        'style': None,
        'sw': 'Scoring Goals',
    },
    'xG per 90': {
        'style': 'Gets into good goal scoring positions',
        'sw': 'Attacking Positioning',
    },
    'Shots per 90': {
        'style': 'Takes many shots',
        'sw': 'Shot Volume',
    },
    'Goal conversion, %': {
        'style': None,
        'sw': 'Finishing',
    },
    'Crosses per 90': {
        'style': 'Wide crearor',
        'sw': 'Crossing',
    },
    'Dribbles per 90': {
        'style': 'Dribbler',
        'sw': 'Dribble Volume',
    },
    'Successful dribbles, %': {
        'style': None,
        'sw': 'Dribbling Efficiency',
    },
    'Touches in box per 90': {
        'style': 'Busy in the penalty box',
        'sw': 'Penalty-box Coverage',
    },
    'Progressive runs per 90': {
        'style': 'Gets team up the pitch via carries',
        'sw': 'Progressive Runs',
    },
    'Passes per 90': {
        'style': 'Involved in build-up',
        'sw': 'Involvement',
    },
    'Accurate passes, %': {
        'style': None,
        'sw': 'Retention',
    },
    'xA per 90': {
        'style': 'Creates goal scoring chances',
        'sw': 'Creativity',
    },
    'Passes to penalty area per 90': {
        'style': None,
        'sw': 'Passes to Penalty Area',
    },
    'Deep completions per 90': {
        'style': 'Gets ball into the box',
        'sw': None,
    },
    'Progressive passes per 90': {
        'style': 'Drops deep to build play',
        'sw': None,
    },
    'Smart passes per 90': {
        'style': 'Attempts through balls',
        'sw': None,
    },
}

HI, LO, STYLE_T = 70, 30, 65

def percentile_in_series(value, series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0 or pd.isna(value): 
        return np.nan
    rank = (s < float(value)).mean() * 100.0
    eq_share = (s == float(value)).mean() * 100.0
    return min(100.0, rank + 0.5 * eq_share)

def chips(items, color):
    if not items: return "_None identified._"
    spans = [
        f"<span style='background:{color};color:#111;padding:2px 6px;border-radius:10px;margin:0 6px 6px 0;display:inline-block'>{txt}</span>"
        for txt in items[:10]
    ]
    return " ".join(spans)

# Build pool-based percentiles for EXTRA_METRICS; fallback to league-table percentiles on the player row
pct_extra = {}
if isinstance(pool_df, pd.DataFrame) and not pool_df.empty:
    for m in EXTRA_METRICS:
        if m in df.columns and m in pool_df.columns and pd.notna(ply.get(m)):
            pct_extra[m] = percentile_in_series(ply[m], pool_df[m])
for m in EXTRA_METRICS:
    if m not in pct_extra or pd.isna(pct_extra[m]):
        col = f"{m} Percentile"
        if col in player_row.columns and pd.notna(player_row[col].iloc[0]):
            pct_extra[m] = float(player_row[col].iloc[0])

# Enforce style-only vs. strength/weakness-only via STYLE_MAP:
# - If 'sw' is None -> do NOT score strengths/weaknesses
# - If 'style' is None -> do NOT flag style
strengths, weaknesses, styles = [], [], []
for m, v in pct_extra.items():
    if pd.isna(v): 
        continue
    cfg = STYLE_MAP.get(m, {})
    sw_label = cfg.get('sw')          # keep None if absent
    style_tag = cfg.get('style')      # keep None if absent

    # Strengths/Weaknesses only if an sw label exists
    if sw_label:
        if v >= HI:
            strengths.append((sw_label, v))
        elif v <= LO:
            weaknesses.append((sw_label, v))

    # Style flag only if a style phrase exists
    if style_tag and v >= STYLE_T:
        styles.append((style_tag, v))

# De-dupe & sort nicely
if strengths:
    strength_best = {name: max(p for n,p in strengths if n==name) for name,_ in strengths}
    strengths = [name for name,_ in sorted(strength_best.items(), key=lambda kv: -kv[1])]
if weaknesses:
    weakness_worst = {name: min(p for n,p in weaknesses if n==name) for name,_ in weaknesses}
    weaknesses = [name for name,_ in sorted(weakness_worst.items(), key=lambda kv: kv[1])]
if styles:
    style_best = {name: max(p for n,p in styles if n==name) for name,_ in styles}
    styles = [name for name,_ in sorted(style_best.items(), key=lambda kv: -kv[1])]

# Summary + chips
st.markdown(
    f"**Profile:** {player_name} — {ply.get('Team','?')} ({ply.get('League','?')}), "
    f"age {int(ply['Age']) if pd.notna(ply.get('Age')) else '—'}, "
    f"minutes {int(ply['Minutes played']) if pd.notna(ply['Minutes played']) else '—'}."
)
st.markdown("**Style:**")
st.markdown(chips(styles, "#bfdbfe"), unsafe_allow_html=True)   # light blue
st.markdown("**Strengths:**")
st.markdown(chips(strengths, "#a7f3d0"), unsafe_allow_html=True)  # light green
st.markdown("**Weaknesses:**")
st.markdown(chips(weaknesses, "#fecaca"), unsafe_allow_html=True) # light red

# ---------- 3) ROLE SCORES (MATCH TABLES EXACTLY) ----------
def table_style_role_scores_from_row(row):
    """Use per-league percentiles from df_f (already computed) + sidebar league weighting."""
    rs = {}
    for role, rd in ROLES.items():
        total_w = sum(rd["metrics"].values()) or 1.0
        metric_score = 0.0
        for m, w in rd["metrics"].items():
            pct_col = f"{m} Percentile"
            if pct_col in row.index and pd.notna(row[pct_col]):
                metric_score += float(row[pct_col]) * w
        metric_score /= total_w
        if use_league_weighting:
            league_scaled = float(row.get("League Strength", 50.0))  # 0..100
            metric_score = (1 - beta) * metric_score + beta * league_scaled
        rs[role] = metric_score
    return rs

role_scores = table_style_role_scores_from_row(player_row.iloc[0])

# Best role line — choose ONLY among the first three roles in ROLES
if role_scores:
    role_list = list(ROLES.keys())[:3]
    candidates = [(r, role_scores.get(r, np.nan)) for r in role_list]
    candidates = [(r, v) for r, v in candidates if pd.notna(v)]
    if candidates:
        best_role = max(candidates, key=lambda kv: kv[1])[0]
        st.markdown(f"**Best role:** {best_role}.")

# Role table with gradient colors (show all roles)
def score_to_color(v: float) -> str:
    if pd.isna(v): return "background-color: #ffffff"
    if v <= 50:
        r1,g1,b1 = (190,42,62); r2,g2,b2 = (244,209,102); t = v/50
    else:
        r1,g1,b1 = (244,209,102); r2,g2,b2 = (34,197,94); t = (v-50)/50
    r = int(r1 + (r2-r1)*t); g = int(g1 + (g2-g1)*t); b = int(b1 + (b2-b1)*t)
    return f"background-color: rgb({r},{g},{b})"

rows = [{"Role": r, "Percentile": role_scores.get(r, np.nan)} for r in ROLES.keys()]
role_df = pd.DataFrame(rows).set_index("Role")
styled = (
    role_df.style
    .applymap(lambda x: score_to_color(float(x)) if pd.notna(x) else "background-color:#fff", subset=["Percentile"])
    .format({"Percentile": lambda x: f"{int(round(x))}" if pd.notna(x) else "—"})
)
st.dataframe(styled, use_container_width=True)

# ---------- right under your "🎯 Single Player Role Profile" chooser ----------
if "shortlist" not in st.session_state:
    st.session_state.shortlist = []  # list of dicts

def _add_to_shortlist(row: pd.Series):
    item = {
        "Player":  row["Player"],
        "Team":    row.get("Team", ""),
        "League":  row.get("League", ""),
        "Age":     int(row.get("Age", 0)) if pd.notna(row.get("Age")) else "",
        "Minutes": int(row.get("Minutes played", 0)) if pd.notna(row.get("Minutes played")) else "",
        "Value":   float(row.get("Market value", 0)) if pd.notna(row.get("Market value")) else 0.0,
        "Added":   datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    # dedupe by Player
    st.session_state.shortlist = [x for x in st.session_state.shortlist if x["Player"] != item["Player"]]
    st.session_state.shortlist.append(item)

if not player_row.empty and st.button("⭐ Add to shortlist", type="primary", use_container_width=False):
    _add_to_shortlist(player_row.iloc[0])
    st.success("Added!")

# ---------- render shortlist in the sidebar (bottom) ----------
with st.sidebar:
    st.markdown("---")
    st.subheader("Shortlist")
    if st.session_state.shortlist:
        sldf = pd.DataFrame(st.session_state.shortlist)
        st.dataframe(sldf, use_container_width=True, height=220)
        st.download_button(
            "⬇️ Download shortlist (CSV)",
            data=sldf.to_csv(index=False).encode("utf-8"),
            file_name="shortlist.csv", mime="text/csv", use_container_width=True
        )
        if st.button("🗑️ Clear shortlist", use_container_width=True):
            st.session_state.shortlist = []
    else:
        st.caption("_No players shortlisted yet._")

# ----------------- END SINGLE PLAYER ROLE PROFILE -----------------


# =====================================================================
# ============== BELOW THE NOTES: 3 EXTRA FEATURE BLOCKS ==============
# =====================================================================

# ----------------- (A) SCATTERPLOT — Goals vs xG -----------------
st.markdown("---")
st.header("📈 Scatter — Non-penalty Goals vs xG")

with st.expander("Scatter settings", expanded=False):
    # Axis metric picks (defaults as requested)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    x_default = "Non-penalty goals per 90"
    y_default = "xG per 90"
    x_metric = st.selectbox(
        "X-axis metric",
        [c for c in FEATURES if c in numeric_cols],
        index=(FEATURES.index(x_default) if x_default in FEATURES else 0),
        key="sc_x",
    )
    y_metric = st.selectbox(
        "Y-axis metric",
        [c for c in FEATURES if c in numeric_cols],
        index=(FEATURES.index(y_default) if y_default in FEATURES else 1),
        key="sc_y",
    )

    # Pool: default = player's league; presets + custom add-ons
    leagues_available_sc = sorted(df["League"].dropna().unique().tolist())
    player_league = player_row.iloc[0]["League"] if not player_row.empty else None

    preset_choices_sc = [
        "Player's league",
        "Top 5 Europe",
        "Top 20 Europe",
        "EFL (England 2–4)",
        "Custom",
    ]
    preset_sc = st.selectbox("League preset", preset_choices_sc, index=0, key="sc_preset")

    preset_map_sc = {
        "Player's league": {player_league} if player_league else set(),
        "Top 5 Europe": set(PRESET_LEAGUES.get("Top 5 Europe", [])),
        "Top 20 Europe": set(PRESET_LEAGUES.get("Top 20 Europe", [])),
        "EFL (England 2–4)": set(PRESET_LEAGUES.get("EFL (England 2–4)", [])),
        "Custom": set(),
    }
    preset_set = preset_map_sc.get(preset_sc, set())
    add_leagues_sc = st.multiselect("Add leagues", leagues_available_sc, default=[], key="sc_add_leagues")
    leagues_scatter = sorted(set(add_leagues_sc) | preset_set)

    # If user left it empty, fall back to player's league so the plot always works
    if not leagues_scatter and player_league:
        leagues_scatter = [player_league]

    same_pos_scatter = st.checkbox("Limit pool to current position prefix", value=True, key="sc_pos")

    # Filters: minutes, age, league strength (quality)
    df["Minutes played"] = pd.to_numeric(df["Minutes played"], errors="coerce")
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    min_minutes_s, max_minutes_s = st.slider("Minutes filter", 0, 5000, (1000, 5000), key="sc_min")
    age_min_bound = int(np.nanmin(df["Age"])) if df["Age"].notna().any() else 14
    age_max_bound = int(np.nanmax(df["Age"])) if df["Age"].notna().any() else 45
    min_age_s, max_age_s = st.slider("Age filter", age_min_bound, age_max_bound, (16, 40), key="sc_age")

    min_strength_s, max_strength_s = st.slider("League quality (strength)", 0, 101, (0, 101), key="sc_ls")

    # Label & inclusion toggles
    include_selected = st.toggle("Include selected player", value=True, key="sc_include")
    label_all = st.toggle("Label ALL players in chart", value=False, key="sc_labels_all")
    allow_overlap = st.toggle("Allow overlapping labels", value=False, key="sc_overlap")

    # Visual improvements
    show_medians = st.checkbox("Show median reference lines", value=True, key="sc_medians")
    shade_iqr = st.checkbox("Shade interquartile range (25–75%)", value=True, key="sc_iqr")
    point_alpha = st.slider("Point opacity", 0.2, 1.0, 0.85, 0.05, key="sc_alpha")

# ---- Build scatter pool ----
try:
    pool_sc = df[df["League"].isin(leagues_scatter)].copy()
    if same_pos_scatter and not player_row.empty:
        pool_sc = pool_sc[pool_sc["Position"].astype(str).apply(position_filter)]

    # numeric + filters
    pool_sc["Minutes played"] = pd.to_numeric(pool_sc["Minutes played"], errors="coerce")
    pool_sc["Age"] = pd.to_numeric(pool_sc["Age"], errors="coerce")
    pool_sc = pool_sc[pool_sc["Minutes played"].between(min_minutes_s, max_minutes_s)]
    pool_sc = pool_sc[pool_sc["Age"].between(min_age_s, max_age_s)]

    # league quality filter
    pool_sc["League Strength"] = pool_sc["League"].map(LEAGUE_STRENGTHS).fillna(0.0)
    pool_sc = pool_sc[
        (pool_sc["League Strength"] >= float(min_strength_s))
        & (pool_sc["League Strength"] <= float(max_strength_s))
    ]

    # Ensure metrics are numeric and present
    if x_metric not in pool_sc.columns or y_metric not in pool_sc.columns:
        st.info("Selected axis metrics are missing from the dataset.")
    else:
        pool_sc[x_metric] = pd.to_numeric(pool_sc[x_metric], errors="coerce")
        pool_sc[y_metric] = pd.to_numeric(pool_sc[y_metric], errors="coerce")
        pool_sc = pool_sc.dropna(subset=[x_metric, y_metric, "Player", "Team", "League"])

        # Selected player's name (regardless of inclusion toggle)
        selected_player_name = player_row.iloc[0]["Player"] if not player_row.empty else None

        # If excluded, make sure the selected player is NOT in the pool
        if not include_selected and selected_player_name is not None and not pool_sc.empty:
            pool_sc = pool_sc[pool_sc["Player"] != selected_player_name]

        # If included, ensure we add them even if filtered out above
        if include_selected and selected_player_name is not None:
            need_insert = True
            if not pool_sc.empty:
                need_insert = not (pool_sc["Player"] == selected_player_name).any()
            if need_insert:
                insertable = df[df["Player"] == selected_player_name].head(1).copy()
                if not insertable.empty:
                    insertable["League Strength"] = insertable["League"].map(LEAGUE_STRENGTHS).fillna(0.0)
                    insertable[x_metric] = pd.to_numeric(insertable[x_metric], errors="coerce")
                    insertable[y_metric] = pd.to_numeric(insertable[y_metric], errors="coerce")
                    pool_sc = pd.concat([pool_sc, insertable], ignore_index=True, sort=False)

        # ----- Plot -----
        if pool_sc.empty:
            st.info("No players in scatter pool after filters.")
        else:
            fig, ax = plt.subplots(figsize=(9.4, 6.6), dpi=200)
            # Grey page & axis backgrounds
            fig.patch.set_facecolor("#f3f4f6")     # page bg
            ax.set_facecolor("#eeeeee")            # plot bg

            # Compute limits with a small padding so labels/points don't clip
            x_vals = pool_sc[x_metric].values
            y_vals = pool_sc[y_metric].values
            def padded_limits(arr, pad_frac=0.06):
                a_min, a_max = float(np.nanmin(arr)), float(np.nanmax(arr))
                if a_min == a_max:
                    a_min -= 1e-6; a_max += 1e-6
                pad = (a_max - a_min) * pad_frac
                return a_min - pad, a_max + pad
            xlim = padded_limits(x_vals); ylim = padded_limits(y_vals)
            ax.set_xlim(*xlim); ax.set_ylim(*ylim)

            # Determine whether we have a selected player to highlight
            sel_name = selected_player_name if include_selected else None

            # Others (black)
            others = pool_sc[pool_sc["Player"] != sel_name] if sel_name is not None else pool_sc
            ax.scatter(
                others[x_metric], others[y_metric],
                s=30, c="black", alpha=float(point_alpha), linewidths=0.4, edgecolors="white", zorder=2
            )

            # Selected player (red) + label (only once) if included
            already_labeled = set()
            if sel_name is not None:
                sel = pool_sc[pool_sc["Player"] == sel_name]
                ax.scatter(
                    sel[x_metric], sel[y_metric],
                    s=90, c="#C81E1E", edgecolors="white", linewidths=1.0, zorder=4
                )
                for _, r in sel.iterrows():
                    ax.annotate(
                        r["Player"], (r[x_metric], r[y_metric]),
                        xytext=(8, 8), textcoords="offset points",
                        fontsize=9, fontweight="bold", color="#C81E1E", zorder=5
                    )
                    already_labeled.add(r["Player"])

            # ----- Visual improvements -----
            # 1) Optional IQR shading (helps quick orientation)
            if shade_iqr:
                x_q1, x_q3 = np.nanpercentile(x_vals, [25, 75])
                y_q1, y_q3 = np.nanpercentile(y_vals, [25, 75])
                ax.axvspan(x_q1, x_q3, color="#d1d5db", alpha=0.25, zorder=1)
                ax.axhspan(y_q1, y_q3, color="#d1d5db", alpha=0.25, zorder=1)

            # 2) Median reference lines (dashed), with unified label "Median"
            if show_medians:
                med_x = float(np.nanmedian(x_vals)); med_y = float(np.nanmedian(y_vals))
                ax.axvline(med_x, color="#6b7280", ls="--", lw=1.25, zorder=1.5)
                ax.axhline(med_y, color="#6b7280", ls="--", lw=1.25, zorder=1.5)
                ax.text(med_x, ylim[1], "Median", ha="right", va="bottom",
                        fontsize=8, color="#374151", backgroundcolor="white", zorder=3, clip_on=True)
                ax.text(xlim[0], med_y, "Median", ha="left", va="top",
                        fontsize=8, color="#374151", backgroundcolor="white", zorder=3, clip_on=True)

            # 3) Optional labeling of ALL players (without duplication)
            if label_all:
                label_df = pool_sc
                # Simple overlap-avoidance: skip labels too close to an already-labeled point
                x_tol = (xlim[1] - xlim[0]) * 0.02
                y_tol = (ylim[1] - ylim[0]) * 0.02
                placed_pts = []

                # seed with the selected player's position(s) if present & included
                if sel_name is not None:
                    sel_seed = pool_sc[pool_sc["Player"] == sel_name]
                    for _, r in sel_seed.iterrows():
                        placed_pts.append((float(r[x_metric]), float(r[y_metric])))

                offsets = [(6,6), (-6,6), (6,-6), (-6,-6)]
                for i, (_, r) in enumerate(label_df.iterrows()):
                    pname = r["Player"]
                    # don't duplicate the selected player's label
                    if pname in already_labeled:
                        continue

                    px, py = float(r[x_metric]), float(r[y_metric])
                    if not allow_overlap:
                        too_close = any((abs(px - qx) < x_tol and abs(py - qy) < y_tol) for (qx, qy) in placed_pts)
                        if too_close:
                            continue
                        placed_pts.append((px, py))

                    dx, dy = offsets[i % len(offsets)]
                    ax.annotate(
                        pname, (px, py),
                        xytext=(dx, dy), textcoords="offset points",
                        fontsize=8, color="#111827", zorder=3
                    )

            # Styling: bold axis labels, light grid, subtle spines
            ax.set_xlabel(x_metric, fontweight="bold")
            ax.set_ylabel(y_metric, fontweight="bold")
            ax.grid(True, which="major", linewidth=0.7, color="#d1d5db")
            ax.grid(True, which="minor", linewidth=0.45, color="#e5e7eb", alpha=0.7)
            ax.minorticks_on()
            for spine in ax.spines.values():
                spine.set_edgecolor("#9ca3af")

            # Caption with pool size & leagues
            leagues_shown = ", ".join(sorted(set(pool_sc["League"])))
            st.caption(f"Pool size: {len(pool_sc):,} • Leagues: {leagues_shown}")
            st.pyplot(fig, use_container_width=True)
except Exception as e:
    st.info(f"Scatter could not be drawn: {e}")
# ----------------------------------------------------------------------
# ----------------- (B) COMPARISON RADAR (SB-STYLE) -----------------
st.markdown("---")
st.header("📊 Player Comparison Radar")
with st.expander("Radar settings", expanded=False):
    # default pos prefix from selected player
    pos_scope = st.text_input("Position startswith (radar pool)", default_pos_prefix, key="rad_pos")
    df["Minutes played"] = pd.to_numeric(df["Minutes played"], errors="coerce")
    df["Age"]            = pd.to_numeric(df["Age"], errors="coerce")
    min_minutes_r, max_minutes_r = st.slider("Minutes filter (radar pool)", 0, 5000, (1000, 5000), key="rad_min")
    # default age 16–40 (changed from 16–33)
    age_min_r_bound = int(np.nanmin(df["Age"])) if df["Age"].notna().any() else 14
    age_max_r_bound = int(np.nanmax(df["Age"])) if df["Age"].notna().any() else 45
    min_age_r, max_age_r = st.slider(
        "Age filter (radar pool)",
        age_min_r_bound, age_max_r_bound,
        (16, 40), key="rad_age"
    )
    picker_pool = df[df["Position"].astype(str).apply(position_filter)].copy()
    players = sorted(picker_pool["Player"].dropna().unique().tolist())
    if len(players) < 2:
        st.warning("Not enough players for this filter.")
        players = sorted(df["Player"].dropna().unique().tolist())

    # default Player A = selected player if present
    try:
        pA_index = players.index(player_name)
    except Exception:
        pA_index = 0
    pA = st.selectbox("Player A (red)", players, index=pA_index, key="rad_a")

    # default Player B = next one (or index 1)
    pB_default_index = 1 if len(players) > 1 else 0
    if pA_index == pB_default_index and len(players) > 2:
        pB_default_index = 2
    pB = st.selectbox("Player B (blue)", players, index=pB_default_index, key="rad_b")

    DEFAULT_METRICS = [
        "Non-penalty goals per 90","xG per 90","Shots per 90",
        "Dribbles per 90","Successful dribbles, %","Touches in box per 90",
        "Aerial duels per 90","Aerial duels won, %","Passes per 90",
        "Accurate passes, %","xA per 90"
    ]
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    metrics_default = [m for m in DEFAULT_METRICS if m in df.columns]
    radar_metrics = st.multiselect("Radar metrics", [c for c in df.columns if c in numeric_cols], metrics_default, key="rad_ms")
    sort_by_gap = st.checkbox("Sort axes by biggest gap", False, key="rad_sort")
    show_avg    = st.checkbox("Show pool average (thin line)", True, key="rad_avg")

# Build radar pool and draw
def clean_label_r(s: str) -> str:
    s = s.replace("Non-penalty goals per 90", "Non-Pen Goals")
    s = s.replace("xG per 90", "xG").replace("xA per 90", "xA")
    s = s.replace("Shots per 90", "Shots").replace("Passes per 90", "Passes")
    s = s.replace("Touches in box per 90", "Touches in box").replace("Aerial duels per 90", "Aerial duels")
    s = s.replace("Successful dribbles, %", "Dribble %").replace("Accurate passes, %", "Pass %")
    s = re.sub(r"\s*per\s*90", "", s, flags=re.I); return s

if radar_metrics:
    try:
        rowA = df[df["Player"] == pA].iloc[0]; rowB = df[df["Player"] == pB].iloc[0]
        union_leagues = {rowA["League"], rowB["League"]}
        pool = df[(df["League"].isin(union_leagues)) &
                  (df["Position"].astype(str).apply(position_filter)) &
                  (df["Minutes played"].between(min_minutes_r, max_minutes_r)) &
                  (df["Age"].between(min_age_r, max_age_r))].copy()
        for m in radar_metrics: pool[m] = pd.to_numeric(pool[m], errors="coerce")
        pool = pool.dropna(subset=radar_metrics)
        if not pool.empty:
            labels = [clean_label_r(m) for m in radar_metrics]
            pool_pct = pool[radar_metrics].rank(pct=True) * 100.0
            def pct_for(player):
                sub_idx = pool[pool["Player"] == player].index
                if len(sub_idx)==0: return np.full(len(radar_metrics), np.nan)
                return pool_pct.loc[sub_idx, :].mean(axis=0).values
            A_r = pct_for(pA); B_r = pct_for(pB); AVG_r = np.full(len(radar_metrics), 50.0)

            axis_min = pool[radar_metrics].min().values
            axis_max = pool[radar_metrics].max().values
            pad = (axis_max - axis_min) * 0.07
            axis_min = axis_min - pad; axis_max = axis_max + pad
            ring_radii = np.linspace(10, 100, 11)
            axis_ticks = [np.linspace(axis_min[i], axis_max[i], 11) for i in range(len(labels))]

            if sort_by_gap:
                order = np.argsort(-np.abs(A_r - B_r))
                labels   = [labels[i] for i in order]
                A_r      = A_r[order]; B_r = B_r[order]; AVG_r = AVG_r[order]
                axis_ticks = [axis_ticks[i] for i in order]

            # draw
            COL_A = "#C81E1E"; COL_B = "#1D4ED8"
            FILL_A = (200/255, 30/255, 30/255, 0.60)
            FILL_B = (29/255, 78/255, 216/255, 0.60)
            PAGE_BG = AX_BG = "#FFFFFF"
            GRID_BAND_A = "#FFFFFF"; GRID_BAND_B = "#E5E7EB"; RING_COLOR = "#D1D5DB"; RING_LW = 1.0
            LABEL_COLOR = "#0F172A"; TITLE_FS = 26; SUB_FS = 12; AXIS_FS = 10
            TICK_FS = 7; TICK_COLOR = "#9CA3AF"; INNER_HOLE = 10

            def draw_radar(labels, A_r, B_r, ticks, headerA, subA, subA2, headerB, subB, subB2,
                           show_avg=False, AVG_r=None):
                N = len(labels)
                theta = np.linspace(0, 2*np.pi, N, endpoint=False)
                theta_closed = np.concatenate([theta, theta[:1]])
                Ar = np.concatenate([A_r, A_r[:1]]); Br = np.concatenate([B_r, B_r[:1]])
                fig = plt.figure(figsize=(13.2, 8.0), dpi=260); fig.patch.set_facecolor(PAGE_BG)
                ax = plt.subplot(111, polar=True); ax.set_facecolor(AX_BG)
                ax.set_theta_offset(np.pi/2); ax.set_theta_direction(-1)
                ax.set_xticks(theta); ax.set_xticklabels(labels, fontsize=AXIS_FS, color=LABEL_COLOR, fontweight=600)
                ax.set_yticks([]); ax.grid(False); [s.set_visible(False) for s in ax.spines.values()]
                for i in range(10):
                    r0, r1 = np.linspace(INNER_HOLE,100,11)[i], np.linspace(INNER_HOLE,100,11)[i+1]
                    band = GRID_BAND_A if i % 2 == 0 else GRID_BAND_B
                    ax.add_artist(Wedge((0,0), r1, 0, 360, width=(r1-r0),
                                        transform=ax.transData._b, facecolor=band, edgecolor="none", zorder=0.8))
                ring_t = np.linspace(0, 2*np.pi, 361)
                for r in np.linspace(INNER_HOLE,100,11):
                    ax.plot(ring_t, np.full_like(ring_t, r), color=RING_COLOR, lw=RING_LW, zorder=0.9)
                start_idx = 2
                for i, ang in enumerate(theta):
                    vals = ticks[i][start_idx:]
                    for rr, v in zip(np.linspace(INNER_HOLE,100,11)[start_idx:], vals):
                        ax.text(ang, rr-1.8, f"{v:.1f}", ha="center", va="center",
                                fontsize=TICK_FS, color=TICK_COLOR, zorder=1.1)
                ax.add_artist(Circle((0,0), radius=INNER_HOLE-0.6, transform=ax.transData._b,
                                     color=PAGE_BG, zorder=1.2, ec="none"))
                if show_avg and AVG_r is not None:
                    Avg = np.concatenate([AVG_r, AVG_r[:1]])
                    ax.plot(theta_closed, Avg, lw=1.5, color="#94A3B8", ls="--", alpha=0.9, zorder=2.2)
                ax.plot(theta_closed, Ar, color=COL_A, lw=2.2, zorder=3); ax.fill(theta_closed, Ar, color=FILL_A, zorder=2.5)
                ax.plot(theta_closed, Br, color=COL_B, lw=2.2, zorder=3); ax.fill(theta_closed, Br, color=FILL_B, zorder=2.5)
                ax.set_rlim(0, 105)
                minsA = f"{int(pd.to_numeric(rowA.get('Minutes played',0))):,} mins" if pd.notna(rowA.get('Minutes played')) else "Minutes: N/A"
                minsB = f"{int(pd.to_numeric(rowB.get('Minutes played',0))):,} mins" if pd.notna(rowB.get('Minutes played')) else "Minutes: N/A"
                fig.text(0.12, 0.96,  f"{pA}", color=COL_A, fontsize=TITLE_FS, fontweight="bold", ha="left")
                fig.text(0.12, 0.935, f"{rowA['Team']} — {rowA['League']}", color=COL_A, fontsize=SUB_FS, ha="left")
                fig.text(0.12, 0.915, minsA, color="#374151", fontsize=10, ha="left")
                fig.text(0.88, 0.96,  f"{pB}", color=COL_B, fontsize=TITLE_FS, fontweight="bold", ha="right")
                fig.text(0.88, 0.935, f"{rowB['Team']} — {rowB['League']}", color=COL_B, fontsize=SUB_FS, ha="right")
                fig.text(0.88, 0.915, minsB, color="#374151", fontsize=10, ha="right")
                return fig

            figr = draw_radar(labels, A_r, B_r, axis_ticks, pA, "", "", pB, "", "", show_avg=show_avg, AVG_r=AVG_r)
            st.pyplot(figr, use_container_width=True)
        else:
            st.info("No players remain in radar pool after filters.")
    except Exception as e:
        st.info(f"Radar could not be drawn: {e}")

# =====================================================================
# ELITE ONE-PAGER — clean, readable, downloadable profile
# =====================================================================
st.markdown("---")
st.header("Player Profile")

if player_row.empty:
    st.info("Pick a player above.")
else:
    # ---------- metric groups ----------
    METRICS = [
        'Defensive duels per 90','Aerial duels won, %', 'Non-penalty goals per 90',
        'xG per 90','Shots per 90','Shots on target, %', 'Crosses per 90','Accurate crosses, %','Dribbles per 90',
        'Successful dribbles, %', 'Offensive duels per 90','Offensive duels won, %','Touches in box per 90',
        'Progressive runs per 90','Accelerations per 90','Passes per 90','Accurate passes, %', 'Forward passes per 90',
        'Long passes per 90','xA per 90','Smart passes per 90','Key passes per 90',
        'Passes to final third per 90','Accurate passes to final third, %',
        'Passes to penalty area per 90','Accurate passes to penalty area, %', 
        'Deep completions per 90','Progressive passes per 90','Accurate progressive passes, %'
    ]
    GROUPS = {
        "Attacking": [
            'Non-penalty goals per 90','xG per 90','Shots per 90','Shots on target, %',
            'Touches in box per 90','Dribbles per 90','Successful dribbles, %',
            'Offensive duels per 90','Offensive duels won, %','Progressive runs per 90','Accelerations per 90'
        ],
        "Passing & Creativity": [
            'Passes per 90','Accurate passes, %','Forward passes per 90','Long passes per 90',
            'xA per 90','Smart passes per 90','Key passes per 90',
            'Passes to final third per 90','Accurate passes to final third, %',
            'Passes to penalty area per 90','Accurate passes to penalty area, %',
            'Deep completions per 90','Progressive passes per 90','Accurate progressive passes, %',
            'Crosses per 90','Accurate crosses, %'
        ],
        "Defensive": [
            'Defensive duels per 90','Aerial duels won, %'
        ],
    }

    # ---------- percentiles: pool first, league fallback ----------
    def pct_for(metric: str) -> float:
        try:
            if isinstance(pool_df, pd.DataFrame) and metric in pool_df.columns:
                return percentile_in_series(ply[metric], pool_df[metric])
        except Exception:
            pass
        col = f"{metric} Percentile"
        return float(player_row[col].iloc[0]) if col in player_row and pd.notna(player_row[col].iloc[0]) else np.nan
    PCTS = {m: pct_for(m) for m in METRICS}

    # ---------- best role tag (only among first 3 roles) ----------
    best_tag = ""
    if role_scores:
        first3 = list(ROLES.keys())[:3]
        items = [(r, role_scores.get(r, np.nan)) for r in first3 if pd.notna(role_scores.get(r, np.nan))]
        if items:
            r, v = max(items, key=lambda kv: kv[1])
            best_tag = f"{r} {int(round(v))}"

    # ---------- label helpers ----------
    import re, matplotlib.pyplot as plt, io, numpy as np
    def short(s: str) -> str:
        s = re.sub(r"\s*per\s*90", "", s, flags=re.I)
        s = s.replace("Non-penalty goals", "Non-pen goals")
        s = s.replace("Passes to penalty area", "Passes to pen. area")
        s = s.replace("Passes to final third", "Passes to final 3rd")
        s = s.replace("Progressive passes", "Prog. passes").replace("Progressive runs", "Prog. runs")
        return s

    def raw(metric: str, v) -> str:
        if pd.isna(v): return "—"
        if "%" in metric: return f"{float(v):.0f}%"
        # compact but legible
        return f"{float(v):.2f}" if abs(float(v)) < 10 else f"{float(v):.1f}"

    # R->Y->G map
    def ryG(p):
        p = 0 if pd.isna(p) else max(0, min(100, float(p)))
        if p <= 50:
            t=p/50; a=(190/255,42/255,62/255); b=(244/255,209/255,102/255)
        else:
            t=(p-50)/50; a=(244/255,209/255,102/255); b=(34/255,197/255,94/255)
        return tuple(a[i] + (b[i]-a[i])*t for i in range(3))

    # ------------- CANVAS -------------
    fig = plt.figure(figsize=(14.5, 10), dpi=260)
    fig.patch.set_facecolor("#0b1220")

    # header
    head = fig.add_axes([0.04, 0.92, 0.92, 0.06]); head.axis("off")
    head.add_patch(plt.Rectangle((0,0),1,1,color="#0f172a"))
    name = str(ply.get("Player", player_name))
    head.text(0.02, 0.60, name, color="white", fontsize=22, fontweight=800)
    meta = f"{ply.get('Team','?')} — {ply.get('League','?')} • Age {int(ply['Age']) if pd.notna(ply.get('Age')) else '—'} • " \
           f"{int(ply['Minutes played']):,} mins • Pool {len(pool_df):,}" if isinstance(pool_df, pd.DataFrame) else \
           f"{ply.get('Team','?')} — {ply.get('League','?')}"
    head.text(0.02, 0.18, meta, color="#94a3b8", fontsize=10, fontweight=500)
    if best_tag:
        head.add_patch(plt.Rectangle((0.82, 0.20), 0.16, 0.60, color="#065f46"))
        head.text(0.90, 0.50, best_tag, color="#d1fae5", ha="center", va="center", fontsize=12, fontweight=800)

    # columns
    axL = fig.add_axes([0.04, 0.13, 0.44, 0.75]); axL.axis("off")
    axR = fig.add_axes([0.52, 0.13, 0.44, 0.75]); axR.axis("off")

    def draw_block(ax, title, metrics):
        ax.text(0.0, 1.02, title, color="white", fontsize=13, fontweight=800, ha="left")
        y = 0.95
        H, G = 0.06, 0.025    # bar height, gap
        W = 0.88              # track width
        for m in metrics:
            p = PCTS.get(m, np.nan)
            # track
            ax.add_patch(plt.Rectangle((0, y-H/2), W, H, color="#e5e7eb", ec="none", zorder=1))
            # fill
            if not pd.isna(p):
                ax.add_patch(plt.Rectangle((0, y-H/2), W*(p/100), H, color=ryG(p), ec="black", lw=0.4, zorder=2))
            # raw value label (small, black) at start
            ax.text(0.008, y, raw(m, ply.get(m)), color="black", fontsize=8,
                    ha="left", va="center",
                    bbox=dict(facecolor="white", edgecolor="#d1d5db", boxstyle="round,pad=0.15"))
            # metric label above left
            ax.text(0.0, y + H*0.65, short(m), color="white", fontsize=9.4, fontweight=600, ha="left")
            # percentile number at fill end
            if not pd.isna(p):
                ax.text(W*(p/100)+0.01, y, f"{int(round(p))}", color="#111827", fontsize=8, ha="left", va="center",
                        bbox=dict(facecolor="white", edgecolor="#d1d5db", pad=0.2))
            y -= (H + G)

    # left: Attacking + Defensive
    draw_block(axL, "Attacking", GROUPS["Attacking"])
    axL2 = fig.add_axes([0.04, 0.13, 0.44, 0.26]); axL2.axis("off")
    draw_block(axL2, "Defensive", GROUPS["Defensive"])

    # right: Passing & Creativity
    draw_block(axR, "Passing & Creativity", GROUPS["Passing & Creativity"])

    # chips bar under right column
    chips = fig.add_axes([0.52, 0.06, 0.44, 0.05]); chips.axis("off")
    def chip_row(x0, title, items, face):
        chips.text(x0, 0.75, title, color="white", fontsize=10, fontweight=800, ha="left")
        x = x0 + 0.12
        for t in (items or [])[:14]:
            chips.add_patch(plt.Rectangle((x, 0.25), 0.12, 0.45, facecolor=face, edgecolor="#d1d5db", lw=0.6))
            chips.text(x+0.006, 0.48, t, color="#111827", fontsize=8.6, ha="left", va="center")
            x += 0.125
    chip_row(0.00, "Style",       styles,     "#bfdbfe")
    chip_row(0.36, "Strengths",   strengths,  "#a7f3d0")
    chip_row(0.72, "Weaknesses",  weaknesses, "#fecaca")

    fig.text(0.5, 0.02, "Player Profile", color="#94a3b8", ha="center", fontsize=10)
    st.pyplot(fig, use_container_width=True)

    # download (PNG)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=320, bbox_inches="tight", facecolor=fig.get_facecolor())
    st.download_button("⬇️ Download PNG", data=buf.getvalue(),
                       file_name=f"{name.replace(' ','_')}_profile.png", mime="image/png")
    plt.close(fig)


# ----------------- (C) SIMILAR PLAYERS (adjustable pool) -----------------
st.markdown("---")
st.header("🧭 Similar players (within adjustable pool)")

# --- Feature basket declared FIRST so UI can use it ---
SIM_FEATURES = [
    'Defensive duels per 90','Aerial duels per 90','Aerial duels won, %','PAdj Interceptions',
    'xG per 90','Non-penalty goals per 90','Shots per 90','Crosses per 90','Accurate crosses, %',
    'Dribbles per 90','Successful dribbles, %','Touches in box per 90','Progressive runs per 90',
    'Accelerations per 90','Passes per 90','Accurate passes, %','xA per 90','Smart passes per 90',
    'Key passes per 90','Passes to final third per 90','Accurate passes to final third, %',
    'Passes to penalty area per 90','Accurate passes to penalty area, %','Deep completions per 90',
    'Progressive passes per 90'
]

# league strength map (supports either variable name)
LS_MAP = globals().get('LEAGUE_STRENGTHS', globals().get('league_strengths', {}))

# defaults for advanced weights (others default to 1)
DEFAULT_SIM_WEIGHTS = {f: 1 for f in SIM_FEATURES}
DEFAULT_SIM_WEIGHTS.update({
    'Passes per 90': 3,
    'Accurate passes, %': 2,
    'Dribbles per 90': 3,
    'Non-penalty goals per 90': 2,
    'Shots per 90': 2,
    'Successful dribbles, %': 2,
    'Aerial duels won, %': 2,
    'xA per 90': 2,
    'xG per 90': 2,
    'Touches in box per 90': 2,
    'Passes to penalty area per 90': 2,
    'Passes to final third per 90': 2,
    'Crosses per 90': 2,
})

# --- Build local presets safely (no reliance on _PRESETS_CF existing) ---
_leagues_from_df = df['League'].dropna().unique().tolist() if 'League' in df.columns else []
_included_from_global = list(globals().get('INCLUDED_LEAGUES', []))
_included_leagues_cf = sorted(set(_included_from_global) | set(_leagues_from_df))

_PRESET_LEAGUES_SAFE = globals().get('PRESET_LEAGUES', {})  # may be missing; that's ok
_PRESETS_SIM = {
    "All listed leagues": _included_leagues_cf,
    "T5":  sorted(list(_PRESET_LEAGUES_SAFE.get("Top 5 Europe", []))),
    "T20": sorted(list(_PRESET_LEAGUES_SAFE.get("Top 20 Europe", []))),
    "EFL": sorted(list(_PRESET_LEAGUES_SAFE.get("EFL (England 2–4)", []))),
    "Custom": None,
}
# ------------------------------------------------------------------------

with st.expander("Similarity settings", expanded=False):
    candidate_league_options = _included_leagues_cf
    default_sel = leagues_sel if 'leagues_sel' in globals() else _included_leagues_cf

    sim_preset_choices = list(_PRESETS_SIM.keys())
    sim_preset = st.selectbox(
        "Candidate league preset",
        sim_preset_choices,
        index=sim_preset_choices.index("All listed leagues"),
        key="sim_preset"
    )

    if sim_preset != "Custom":
        preset_vals = _PRESETS_SIM.get(sim_preset) or []
        preset_vals = sorted([lg for lg in preset_vals if lg in candidate_league_options])
        sim_leagues = st.multiselect(
            "Candidate leagues",
            candidate_league_options,
            default=(preset_vals if preset_vals else default_sel),
            key="sim_leagues",
            disabled=bool(preset_vals)
        )
        if preset_vals:
            st.caption(f"Preset: {sim_preset} — {len(preset_vals)} league(s)")
        else:
            st.warning("This preset has no leagues configured. Edit manually or define PRESET_LEAGUES.")
    else:
        sim_leagues = st.multiselect(
            "Candidate leagues",
            candidate_league_options,
            default=default_sel,
            key="sim_leagues",
        )

    # Base filters
    sim_min_minutes, sim_max_minutes = st.slider("Minutes played (candidates)", 0, 5000, (1000, 5000), key="sim_min")
    sim_min_age, sim_max_age = st.slider("Age (candidates)", 14, 45, (16, 40), key="sim_age")

    # Optional league quality filter (0–101), applied pre-computation
    use_strength_filter = st.toggle("Filter by league quality (0–101)", value=False, key="sim_use_strength")
    if use_strength_filter:
        sim_min_strength, sim_max_strength = st.slider("League quality (strength)", 0, 101, (0, 101), key="sim_strength")

    # Blend between percentile distance and actual-value distance
    percentile_weight = st.slider("Percentile weight", 0.0, 1.0, 0.7, 0.05, key="sim_pw")

    # Toggleable league difficulty adjustment (default on)
    apply_league_adjust = st.toggle("Apply league difficulty adjustment", value=True, key="sim_apply_ladj")
    league_weight_sim = st.slider(
        "League weight (difficulty adj.)",
        0.0, 1.0, 0.2, 0.05,
        key="sim_lw",
        disabled=not apply_league_adjust
    )

    # Always-available advanced feature weights (no toggle)
    with st.expander("Advanced feature weights (1–5)", expanded=False):
        adv_weights = {}
        for f in SIM_FEATURES:
            key = "simw_" + f.replace(" ", "_").replace("%", "pct").replace(",", "").replace(".", "_")
            # keep previous choice if present
            default_val = int(st.session_state.get(key, DEFAULT_SIM_WEIGHTS.get(f, 1)))
            adv_weights[f] = st.slider(f"Weight — {f}", 1, 5, default_val, key=key)

    top_n_sim = st.number_input("Show top N", min_value=5, max_value=200, value=50, step=5, key="sim_top")

# --- Similarity computation ---
if not player_row.empty:
    target_row_full = df[df['Player'] == player_name].head(1).iloc[0]
    target_league = target_row_full['League']

    df_candidates = df[df['League'].isin(sim_leagues)].copy()

    # optional league strength filter
    if use_strength_filter and LS_MAP:
        df_candidates['League strength'] = df_candidates['League'].map(LS_MAP).fillna(0.0)
        df_candidates = df_candidates[
            (df_candidates['League strength'] >= float(sim_min_strength)) &
            (df_candidates['League strength'] <= float(sim_max_strength))
        ]

    # --- enforce attacker position filter (RW/LW/AMF family) ---
    if 'Position' in df_candidates.columns:
        df_candidates = df_candidates[df_candidates['Position'].astype(str).apply(position_filter)]
    else:
        st.warning("No 'Position' column found; cannot apply attacker position filter.")

    # base filters
    df_candidates['Minutes played'] = pd.to_numeric(df_candidates['Minutes played'], errors='coerce')
    df_candidates['Age'] = pd.to_numeric(df_candidates['Age'], errors='coerce')

    df_candidates = df_candidates[
        df_candidates['Minutes played'].between(sim_min_minutes, sim_max_minutes) &
        df_candidates['Age'].between(sim_min_age, sim_max_age)
    ]

    # ---- De-duplicate candidates: one row per Player (keep most minutes, then stronger league) ----
    df_candidates['League strength'] = df_candidates['League'].map(LS_MAP).fillna(0.0) if LS_MAP else 0.0
    df_candidates = (
        df_candidates.sort_values(
            ['Player', 'Minutes played', 'League strength'],
            ascending=[True, False, False]
        )
        .drop_duplicates(subset=['Player'], keep='first')
    )

    # Remove the selected player from candidates
    df_candidates = df_candidates[df_candidates['Player'] != player_name]

    # Ensure features are present and numeric
    df_candidates = df_candidates.dropna(subset=SIM_FEATURES)
    for f in SIM_FEATURES:
        df_candidates[f] = pd.to_numeric(df_candidates[f], errors='coerce')
    df_candidates = df_candidates.dropna(subset=SIM_FEATURES)

    # ---- Robust target percentiles (1×F) within the target league ----
    # compute percentile rank of the target player against *its league*
    league_mask = (df['League'] == target_league)
    league_block = df.loc[league_mask, SIM_FEATURES].apply(pd.to_numeric, errors='coerce')
    league_ranks = league_block.rank(pct=True)
    # locate target row within that league
    target_mask_in_league = league_mask & (df['Player'] == player_name)
    if not target_mask_in_league.any():
        st.info("Target player not found in league block for percentile calc.")
        target_percentiles_vec = np.full(len(SIM_FEATURES), 0.5)  # safe default
    else:
        # If duplicates exist, take the first (we already pick one above, but this is extra safety)
        target_percentiles_vec = league_ranks.loc[target_mask_in_league].iloc[0].values

    if not df_candidates.empty:
        # percentile ranks for candidates (per-league robustness)
        percl = df_candidates.groupby('League')[SIM_FEATURES].rank(pct=True).values

        # standardize on candidate pool (actual values)
        scaler = StandardScaler()
        standardized_features = scaler.fit_transform(df_candidates[SIM_FEATURES])
        target_features_standardized = scaler.transform([target_row_full[SIM_FEATURES].astype(float).values])

        # weights
        weights_vec = np.array([float(adv_weights.get(f, 1)) for f in SIM_FEATURES], dtype=float)

        # distances
        percentile_distances = np.linalg.norm((percl - target_percentiles_vec) * weights_vec, axis=1)
        actual_value_distances = np.linalg.norm((standardized_features - target_features_standardized) * weights_vec, axis=1)
        combined = percentile_distances * percentile_weight + actual_value_distances * (1.0 - percentile_weight)

        # normalize -> similarity 0..100
        arr = np.asarray(combined, dtype=float).ravel()
        rng = np.ptp(arr)
        norm = (arr - arr.min()) / (rng if rng != 0 else 1.0)
        similarities = ((1.0 - norm) * 100.0).round(2)

        out = df_candidates[['Player','Team','League','Position','Age','Minutes played','Market value']].copy()
        out['League strength'] = out['League'].map(LS_MAP).fillna(0.0) if LS_MAP else 0.0
        tgt_ls = float(LS_MAP.get(target_league, 1.0)) if LS_MAP else 1.0

        # symmetric league ratio (≤1)
        eps = 1e-6
        cand_ls = np.maximum(out['League strength'].astype(float), eps)
        tgt_ls_safe = max(tgt_ls, eps)
        league_ratio = np.minimum(cand_ls / tgt_ls_safe, tgt_ls_safe / cand_ls)

        out['Similarity'] = similarities
        out['Adjusted Similarity'] = (
            out['Similarity'] * ((1 - league_weight_sim) + league_weight_sim * league_ratio)
        ) if apply_league_adjust else out['Similarity']

        out = out.sort_values('Adjusted Similarity', ascending=False).reset_index(drop=True)
        out.insert(0, 'Rank', np.arange(1, len(out) + 1))
        st.caption(f"Candidates after filters: {len(out):,}")
        st.dataframe(out.head(int(top_n_sim)), use_container_width=True)
    else:
        st.info("No candidates after similarity filters.")
else:
    st.caption("Pick a player to see similar players.")



# ---------------------------- (D) CLUB FIT — self-contained block ----------------------------
st.markdown("---")
st.header("🏟️ Club Fit Finder")

# ---------- SAFE FALLBACKS (use existing globals if present; use UPPERCASE names) ----------
# leagues list
if 'INCLUDED_LEAGUES' in globals():
    _included_leagues_cf = list(INCLUDED_LEAGUES)
else:
    _included_leagues_cf = sorted(pd.Series(df.get('League', pd.Series([]))).dropna().unique().tolist())

# presets (add Top 20 + EFL explicitly)
if 'PRESET_LEAGUES' in globals():
    _PRESETS_CF = {
        "All listed leagues": _included_leagues_cf,
        "Top 5 Europe": sorted(list(PRESET_LEAGUES.get("Top 5 Europe", []))),
        "Top 20 Europe": sorted(list(PRESET_LEAGUES.get("Top 20 Europe", []))),
        "EFL (England 2–4)": sorted(list(PRESET_LEAGUES.get("EFL (England 2–4)", []))),
        "Custom": None,
    }
else:
    _PRESETS_CF = {
        "All listed leagues": _included_leagues_cf,
        "Top 5 Europe": [],
        "Top 20 Europe": [],
        "EFL (England 2–4)": [],
        "Custom": None,
    }

# default per-metric weights (preloaded, not all 1s)
_DEFAULT_W_CF = {
    'Passes per 90': 3,
    'Accurate passes, %': 2,
    'Dribbles per 90': 3,
    'Non-penalty goals per 90': 2,
    'Shots per 90': 2,
    'Successful dribbles, %': 2,
    'Aerial duels won, %': 2,
    'xA per 90': 2,
    'xG per 90': 2,
    'Touches in box per 90': 2,
    'Passes to final third per 90': 2,
    'Passes to penalty area per 90': 2,
    'Crosses per 90': 2,
}

# league strengths (use your table; no neutral fallback unless truly unknown)
if 'LEAGUE_STRENGTHS' in globals():
    _LS_CF = dict(LEAGUE_STRENGTHS)
else:
    _LS_CF = {lg: 50.0 for lg in _included_leagues_cf}

# weight dials
DEFAULT_LEAGUE_WEIGHT = 0.5
DEFAULT_MARKET_WEIGHT = 0.2

# features (fixed)
CF_FEATURES = [
    'Defensive duels per 90','Aerial duels won, %', 'Non-penalty goals per 90',
    'xG per 90','Shots per 90','Shots on target, %', 'Crosses per 90','Accurate crosses, %','Dribbles per 90',
    'Successful dribbles, %', 'Offensive duels per 90','Offensive duels won, %','Touches in box per 90',
    'Progressive runs per 90','Accelerations per 90','Passes per 90','Accurate passes, %', 'Forward passes per 90',
    'Long passes per 90','xA per 90','Smart passes per 90','Key passes per 90',
    'Passes to final third per 90','Accurate passes to final third, %',
    'Passes to penalty area per 90','Accurate passes to penalty area, %', 
    'Deep completions per 90','Progressive passes per 90','Accurate progressive passes, %'
]

required_cols_cf = {'Player','Team','League','Age','Position','Minutes played','Market value', *CF_FEATURES}
missing_cf = [c for c in required_cols_cf if c not in df.columns]
if missing_cf:
    st.error(f"Club Fit: dataset missing required columns: {missing_cf}")
else:

    # -------------------- Controls --------------------
    with st.expander("Club-fit settings", expanded=False):
        leagues_available_cf = sorted(set(_included_leagues_cf) | set(df.get('League', pd.Series([])).dropna().unique()))

        # Target leagues (only affects target player list)
        target_leagues_cf = st.multiselect(
            "Target leagues (choose target from here)",
            leagues_available_cf,
            default=leagues_available_cf,
            key="cf_target_leagues"
        )

        # Candidate pool via preset + extras
        if 'candidate_leagues_cf' not in st.session_state:
            st.session_state.candidate_leagues_cf = list(_included_leagues_cf)

        preset_name_cf = st.selectbox("Candidate pool preset", list(_PRESETS_CF.keys()), index=0, key="cf_preset_name")
        c1a, c1b = st.columns([1,2])
        if c1a.button("Apply preset", key="cf_apply_preset"):
            if _PRESETS_CF.get(preset_name_cf) is not None:
                st.session_state.candidate_leagues_cf = list(_PRESETS_CF[preset_name_cf])

        extra_candidate_leagues_cf = c1b.multiselect(
            "Extra leagues to add",
            leagues_available_cf,
            default=[],
            key="cf_extra_leagues"
        )
        leagues_selected_cf = sorted(set(st.session_state.candidate_leagues_cf) | set(extra_candidate_leagues_cf))
        st.caption(f"Candidate pool leagues: **{len(leagues_selected_cf)}** selected.")

        # default position prefix + default target = selected player
        pos_scope_cf = st.text_input("Position startswith (club fit)", default_pos_prefix, key="cf_pos_scope")

        # Target player selector (from target leagues) default to selected player
        target_pool_cf = df[df['League'].isin(target_leagues_cf)]
        target_pool_cf = target_pool_cf[target_pool_cf['Position'].astype(str).apply(position_filter)]
        target_options_cf = sorted(target_pool_cf['Player'].dropna().unique())
        try:
            default_target_idx = target_options_cf.index(player_name)
        except Exception:
            default_target_idx = 0 if target_options_cf else 0
        target_player_cf = st.selectbox(
            "Target player",
            target_options_cf,
            index=default_target_idx if target_options_cf else 0,
            key="cf_target_player"
        )

        # Minutes / age filters for candidate pool (teams built from these players)
        max_minutes_in_data_cf = int(pd.to_numeric(df.get('Minutes played', pd.Series([0])), errors='coerce').fillna(0).max())
        slider_max_minutes_cf = int(max(1000, max_minutes_in_data_cf))
        min_minutes_cf, max_minutes_cf = st.slider(
            "Minutes filter (candidates)",
            0, slider_max_minutes_cf,
            (500, slider_max_minutes_cf),
            key="cf_minutes_slider"
        )

        age_series_cf = pd.to_numeric(df.get('Age', pd.Series([16, 45])), errors='coerce')
        age_min_data_cf = int(np.nanmin(age_series_cf)) if age_series_cf.notna().any() else 14
        age_max_data_cf = int(np.nanmax(age_series_cf)) if age_series_cf.notna().any() else 45
        # default age 16–40 (changed)
        min_age_cf, max_age_cf = st.slider(
            "Age filter (candidates)",
            age_min_data_cf, age_max_data_cf,
            (16, 40),
            key="cf_age_slider"
        )

        min_strength_cf, max_strength_cf = st.slider("League quality (strength)", 0, 101, (0, 101), key="cf_strength")

        # Weights
        league_weight_cf = st.slider("League weight", 0.0, 1.0, DEFAULT_LEAGUE_WEIGHT, 0.05, key="cf_league_w")
        market_value_weight_cf = st.slider("Market value weight", 0.0, 1.0, DEFAULT_MARKET_WEIGHT, 0.05, key="cf_market_w")
        manual_override_cf = st.number_input("Target market value override (€)", min_value=0, value=0, step=100_000, key="cf_mv_override")

        # Advanced feature weights (preloaded defaults)
        st.subheader("Advanced feature weights")
        st.caption("Unlisted features default to weight = 1.")
        weights_ui_cf = {}
        for f in CF_FEATURES:
            default_val = _DEFAULT_W_CF.get(f, 1)
            weights_ui_cf[f] = st.slider(f"• {f}", 0, 5, int(default_val), key=f"cf_w_{f}")

        top_n_cf = st.number_input("Show top N teams", 5, 100, 20, 5, key="cf_topn")

    # -------------------- Compute --------------------
    if target_player_cf and (target_player_cf in df['Player'].values):
        # Candidate player pool
        df_candidates_cf = df[df['League'].isin(leagues_selected_cf)].copy()
        df_candidates_cf = df_candidates_cf[df_candidates_cf['Position'].astype(str).apply(position_filter)]

        # Numerics + filters
        df_candidates_cf['Minutes played'] = pd.to_numeric(df_candidates_cf['Minutes played'], errors='coerce')
        df_candidates_cf['Age'] = pd.to_numeric(df_candidates_cf['Age'], errors='coerce')
        df_candidates_cf['Market value'] = pd.to_numeric(df_candidates_cf['Market value'], errors='coerce')

        df_candidates_cf = df_candidates_cf[
            df_candidates_cf['Minutes played'].between(min_minutes_cf, max_minutes_cf, inclusive='both')
        ]
        df_candidates_cf = df_candidates_cf[
            df_candidates_cf['Age'].between(min_age_cf, max_age_cf, inclusive='both')
        ]
        df_candidates_cf = df_candidates_cf.dropna(subset=CF_FEATURES)

        if df_candidates_cf.empty:
            st.info("No candidate players after filters. Widen candidate leagues or relax filters.")
        else:
            # Target (from target leagues)
            df_target_pool_cf = df[df['League'].isin(target_leagues_cf)].copy()
            df_target_pool_cf = df_target_pool_cf[df_target_pool_cf['Position'].astype(str).apply(position_filter)]

            if target_player_cf not in df_target_pool_cf['Player'].values:
                st.info("Target player not found in selected target leagues.")
            else:
                df_target_pool_cf['Market value'] = pd.to_numeric(df_target_pool_cf['Market value'], errors='coerce')
                target_row_cf = df_target_pool_cf.loc[df_target_pool_cf['Player'] == target_player_cf].iloc[0]
                target_vector_cf = target_row_cf[CF_FEATURES].values
                target_ls_cf = float(_LS_CF.get(target_row_cf['League'], 50.0))

                # Target MV (override if provided)
                target_market_value_cf = (
                    float(manual_override_cf) if manual_override_cf and manual_override_cf > 0
                    else (float(target_row_cf['Market value']) if pd.notna(target_row_cf['Market value']) and target_row_cf['Market value'] > 0
                          else 2_000_000.0)
                )

                # Team profiles (mean of players that survived candidate filters)
                club_profiles_cf = df_candidates_cf.groupby('Team')[CF_FEATURES].mean().reset_index()

                # Team league & average team MV from same pool
                team_league_cf = df_candidates_cf.groupby('Team')['League'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])
                team_market_cf = df_candidates_cf.groupby('Team')['Market value'].mean()
                club_profiles_cf['League'] = club_profiles_cf['Team'].map(team_league_cf)
                club_profiles_cf['Avg Team Market Value'] = club_profiles_cf['Team'].map(team_market_cf)
                club_profiles_cf = club_profiles_cf.dropna(subset=['Avg Team Market Value'])

                # Standardize & weighted distance
                scaler_cf = StandardScaler()
                X_team = scaler_cf.fit_transform(club_profiles_cf[CF_FEATURES])
                x_tgt = scaler_cf.transform([target_vector_cf])[0]
                weights_vec_cf = np.array([weights_ui_cf.get(f, 1) for f in CF_FEATURES], dtype=float)

                # Distance (lower = more similar)
                dist_cf = np.linalg.norm((X_team - x_tgt) * weights_vec_cf, axis=1)
                rng = float(dist_cf.max() - dist_cf.min())
                club_fit_base = (1 - (dist_cf - float(dist_cf.min())) / (rng if rng > 0 else 1.0)) * 100.0
                club_profiles_cf['Club Fit %'] = club_fit_base.round(2)

                # League strength adjustment & filter
                club_profiles_cf['League strength'] = club_profiles_cf['League'].map(_LS_CF).fillna(50.0)
                club_profiles_cf = club_profiles_cf[
                    (club_profiles_cf['League strength'] >= float(min_strength_cf)) &
                    (club_profiles_cf['League strength'] <= float(max_strength_cf))
                ]

                if club_profiles_cf.empty:
                    st.info("No teams remain after league-strength filter.")
                else:
                    # Difficulty ratio vs target league
                    ratio_cf = (club_profiles_cf['League strength'] / target_ls_cf).clip(0.5, 1.2)
                    club_profiles_cf['Adjusted Fit %'] = (
                        club_profiles_cf['Club Fit %'] * (1 - league_weight_cf) +
                        club_profiles_cf['Club Fit %'] * ratio_cf * league_weight_cf
                    )

                    # Mild penalty if destination league >> target
                    league_gap_cf = (club_profiles_cf['League strength'] - target_ls_cf).clip(lower=0)
                    penalty_cf = (1 - (league_gap_cf / 100)).clip(lower=0.7)
                    club_profiles_cf['Adjusted Fit %'] = club_profiles_cf['Adjusted Fit %'] * penalty_cf

                    # Market value fit (closer team MV to target MV gets rewarded)
                    value_fit_ratio_cf = (club_profiles_cf['Avg Team Market Value'] / target_market_value_cf).clip(0.5, 1.5)
                    value_fit_score_cf = (1 - abs(1 - value_fit_ratio_cf)) * 100.0

                    club_profiles_cf['Final Fit %'] = (
                        club_profiles_cf['Adjusted Fit %'] * (1 - market_value_weight_cf) +
                        value_fit_score_cf * market_value_weight_cf
                    )

                    # ---------------- Results ----------------
                    results_cf = club_profiles_cf[[
                        'Team','League','League strength',
                        'Club Fit %','Adjusted Fit %','Final Fit %'
                    ]].copy()

                    results_cf = results_cf.sort_values('Final Fit %', ascending=False).reset_index(drop=True)
                    results_cf.insert(0, 'Rank', np.arange(1, len(results_cf) + 1))

                    st.caption(
                        f"Target: {target_player_cf} — {target_row_cf.get('Team','Unknown')} ({target_row_cf['League']}) • "
                        f"Target MV used: €{target_market_value_cf:,.0f} • Target LS {target_ls_cf:.2f} • "
                        f"Candidates: {len(leagues_selected_cf)} leagues (preset: {preset_name_cf})"
                    )
                    st.dataframe(results_cf.head(int(top_n_cf)), use_container_width=True)

                    # Export
                    csv_cf = results_cf.to_csv(index=False).encode('utf-8')
                    st.download_button("⬇️ Download all results (CSV)", data=csv_cf,
                                       file_name="club_fit_results.csv", mime="text/csv")

                    # Debug / Repro
                    with st.expander("Debug / Repro details"):
                        st.write({
                            "preset": preset_name_cf,
                            "candidate_leagues_count": len(leagues_selected_cf),
                            "target_leagues_count": len(target_leagues_cf),
                            "league_weight": float(league_weight_cf),
                            "market_value_weight": float(market_value_weight_cf),
                            "target_market_value_used": float(target_market_value_cf),
                            "minutes_range": (int(min_minutes_cf), int(max_minutes_cf)),
                            "age_range": (int(min_age_cf), int(max_age_cf)),
                            "strength_range": (int(min_strength_cf), int(max_strength_cf)),
                            "n_teams": int(results_cf.shape[0]),
                        })




















































































































































































