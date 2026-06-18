"""
F1 Pit Stop Strategy Analyzer
Day 5 — Polished Multi-Page Streamlit App

Combines:
  Page 1 · Live Predictor   (Day 3 upgraded)
  Page 2 · Strategy Simulator (Day 4 integrated)
  Page 3 · Tire Analyst     (deep degradation explorer)
  Page 4 · Race Replay      (lap-by-lap model validation)

Run:
    streamlit run day5_app.py
"""

import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import itertools
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="F1 Strategy Analyzer",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0f0f0f;
    color: #eeeeee;
    font-family: 'Courier New', monospace;
}
[data-testid="stSidebar"] {
    background-color: #111111;
    border-right: 1px solid #222;
}
[data-testid="stSidebar"] * { color: #cccccc !important; }

/* Top banner */
.top-banner {
    background: linear-gradient(90deg, #1a0000 0%, #0f0f0f 100%);
    border-left: 5px solid #e8002d;
    padding: 18px 24px 14px 24px;
    border-radius: 0 6px 6px 0;
    margin-bottom: 20px;
}
.top-banner h1 {
    font-size: 1.7rem; font-weight: 900;
    color: #fff; letter-spacing: 3px;
    margin: 0 0 3px 0; text-transform: uppercase;
}
.top-banner p {
    color: #666; font-size: 0.72rem;
    letter-spacing: 2px; margin: 0; text-transform: uppercase;
}

/* Section label */
.sec { font-size: 0.65rem; letter-spacing: 3px; color: #e8002d;
       text-transform: uppercase; border-bottom: 1px solid #222;
       padding-bottom: 5px; margin: 18px 0 12px 0; }

/* Recommendation card */
.rec { border-radius: 6px; padding: 20px 24px; text-align: center; margin-bottom: 14px; }
.rec-stay { background:#071a07; border:2px solid #00c853; }
.rec-soon { background:#1a1600; border:2px solid #ffd700; }
.rec-now  { background:#1a0000; border:2px solid #e8002d;
            animation: pulse 1.4s ease-in-out infinite; }
@keyframes pulse {
    0%,100% { box-shadow: 0 0 0 0 rgba(232,0,45,0.5); }
    50%      { box-shadow: 0 0 0 14px rgba(232,0,45,0); }
}
.rec-lbl { font-size:2rem; font-weight:900; letter-spacing:4px;
           text-transform:uppercase; margin-bottom:5px; }
.rec-sub { font-size:0.72rem; letter-spacing:2px; color:#999; text-transform:uppercase; }

/* Metric tiles */
.mrow { display:flex; gap:10px; margin-bottom:16px; }
.mtile { flex:1; background:#161616; border:1px solid #222; border-radius:5px;
         padding:12px 14px; text-align:center; }
.mtile .v { font-size:1.5rem; font-weight:700; color:#fff; }
.mtile .l { font-size:0.6rem; letter-spacing:2px; color:#555;
            text-transform:uppercase; margin-top:2px; }

/* Strategy card */
.strat-card { background:#141414; border:1px solid #252525; border-radius:6px;
              padding:14px 18px; margin-bottom:8px; }
.strat-rank { font-size:0.65rem; color:#e8002d; letter-spacing:2px; }
.strat-label { font-size:0.85rem; color:#eee; font-weight:600; margin:4px 0; }
.strat-time { font-size:0.7rem; color:#888; }

/* Footer */
.footer { margin-top:32px; padding-top:14px; border-top:1px solid #1a1a1a;
          text-align:center; font-size:0.6rem; color:#333; letter-spacing:2px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
TOTAL_LAPS        = 57
PIT_LOSS_SECS     = 23.0
COMPOUND_BASE     = {"SOFT": 88.5,  "MEDIUM": 89.8, "HARD": 91.0}
COMPOUND_DEG      = {"SOFT": 0.12,  "MEDIUM": 0.07, "HARD": 0.04}
COMPOUND_MAX      = {"SOFT": 20,    "MEDIUM": 35,   "HARD": 50}
COMPOUND_MIN_LAPS = {"SOFT": 10,    "MEDIUM": 15,   "HARD": 18}
COMPOUND_CODE     = {"SOFT": 0,     "MEDIUM": 1,    "HARD": 2}
COMPOUND_COLORS   = {"SOFT": "#e8002d", "MEDIUM": "#ffd700", "HARD": "#ebebeb"}
COMPOUNDS         = ["SOFT", "MEDIUM", "HARD"]
DRIVERS           = ["VER","HAM","LEC","NOR","SAI","RUS","ALO","PER","STR","GAS"]

PLOTLY_BASE = dict(
    paper_bgcolor="#0f0f0f",
    plot_bgcolor="#161616",
    font=dict(family="Courier New", color="#cccccc", size=11),
    margin=dict(l=10, r=10, t=36, b=10),
)


# ─────────────────────────────────────────────────────────────
# LOAD RESOURCES
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_bundle():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    df = pd.read_csv("processed_laps.csv")
    # Add TyreLife if missing
    if "TyreLife" not in df.columns:
        tl, c = [], 0
        for _, grp in df.groupby("Driver"):
            for _, row in grp.iterrows():
                c += 1
                tl.append(c)
                if row.get("Pitted", 0) == 1:
                    c = 0
        df["TyreLife"] = tl
    return df

try:
    bundle = load_bundle()
    model  = bundle["model"]
    le     = bundle["label_encoder"]
    feats  = bundle["feature_cols"]
    df_race= load_data()
except FileNotFoundError as e:
    st.error(f"Missing file: {e}. Run day1_eda.py then day2_model.py first.")
    st.stop()


# ─────────────────────────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────────────────────────
def build_X(compound, tyre_life, lap_number, position, stint_length):
    base      = COMPOUND_BASE[compound]
    lt        = base + COMPOUND_DEG[compound] * tyre_life
    rolling   = base + COMPOUND_DEG[compound] * max(tyre_life - 1.5, 1)
    pace_loss = COMPOUND_DEG[compound] * tyre_life
    health    = max(0.0, 100 - (tyre_life / COMPOUND_MAX[compound]) * 100)
    in_win    = 1 if (15 <= lap_number <= 25) or (35 <= lap_number <= 45) else 0
    row = {
        "TyreLife": tyre_life, "TyreHealthPct": health,
        "LapTimeSec": lt, "RollingAvgLapTime": rolling,
        "LapTimeDelta": COMPOUND_DEG[compound], "DegRate": COMPOUND_DEG[compound],
        "DegAccel": 0.0, "PaceLoss": pace_loss, "StintLength": stint_length,
        "LapsRemaining": TOTAL_LAPS - lap_number, "RaceProgress": lap_number / TOTAL_LAPS,
        "CompoundCode": COMPOUND_CODE[compound], "Position": position,
        "InPitWindow": in_win,
    }
    return pd.DataFrame([row])[feats]

def predict(compound, tyre_life, lap_number, position, stint_length):
    X     = build_X(compound, tyre_life, lap_number, position, stint_length)
    proba = model.predict_proba(X)[0]
    idx   = int(np.argmax(proba))
    label = le.inverse_transform([idx])[0]
    return label, proba

def lap_time_est(compound, tyre_life, noise=False):
    t = COMPOUND_BASE[compound] + COMPOUND_DEG[compound] * tyre_life
    if noise:
        t += np.random.normal(0, 0.25)
    return t

def total_time(stints, current_lap, noise=False):
    total, lap, tl = 0.0, current_lap, 0
    for i, (c, n) in enumerate(stints):
        if i > 0:
            total += PIT_LOSS_SECS
        tl = 0
        for _ in range(n):
            if lap > TOTAL_LAPS:
                break
            tl += 1
            total += lap_time_est(c, tl, noise)
            lap += 1
    return round(total, 3)

def generate_strategies(current_lap, current_compound, current_tyre_life):
    remaining = TOTAL_LAPS - current_lap
    strats = []

    # 0-stop
    if current_tyre_life + remaining <= COMPOUND_MAX[current_compound] * 1.3:
        strats.append({"label": f"0-stop · {current_compound}({remaining}L)",
                       "stints": [(current_compound, remaining)],
                       "stops": 0, "pit_laps": []})

    # 1-stop
    for c2 in COMPOUNDS:
        if c2 == current_compound:
            continue
        for pit in range(current_lap + COMPOUND_MIN_LAPS[current_compound],
                         TOTAL_LAPS - COMPOUND_MIN_LAPS[c2] + 1):
            l1, l2 = pit - current_lap, TOTAL_LAPS - pit
            if l1 < 1 or l2 < 1:
                continue
            strats.append({
                "label":  f"1-stop · {current_compound}({l1}L)→{c2}({l2}L)",
                "stints": [(current_compound, l1), (c2, l2)],
                "stops":  1, "pit_laps": [pit],
            })

    # 2-stop
    for c2, c3 in itertools.product(COMPOUNDS, repeat=2):
        if c2 == current_compound and c3 == current_compound:
            continue
        m1, m2, m3 = (COMPOUND_MIN_LAPS[current_compound],
                      COMPOUND_MIN_LAPS[c2], COMPOUND_MIN_LAPS[c3])
        for p1 in range(current_lap + m1, TOTAL_LAPS - m2 - m3 + 1, 3):
            for p2 in range(p1 + m2, TOTAL_LAPS - m3 + 1, 3):
                l1, l2, l3 = p1 - current_lap, p2 - p1, TOTAL_LAPS - p2
                if l1 < 1 or l2 < 1 or l3 < 1:
                    continue
                strats.append({
                    "label":  f"2-stop · {current_compound}({l1}L)→{c2}({l2}L)→{c3}({l3}L)",
                    "stints": [(current_compound, l1), (c2, l2), (c3, l3)],
                    "stops":  2, "pit_laps": [p1, p2],
                })

    results = []
    for s in strats:
        times = [total_time(s["stints"], current_lap, noise=True) for _ in range(20)]
        results.append({
            "Strategy":   s["label"],
            "Stops":      s["stops"],
            "Est Time":   round(np.mean(times), 2),
            "Uncertainty":round(np.std(times), 2),
            "Pit Laps":   s["pit_laps"],
        })
    return pd.DataFrame(results).sort_values("Est Time").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# SIDEBAR — Navigation + shared inputs
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏎️ F1 STRATEGY")
    st.markdown("---")
    page = st.radio("Navigate", [
        "🟢 Live Predictor",
        "📊 Strategy Simulator",
        "🔥 Tire Analyst",
        "🔁 Race Replay",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("**RACE INPUT**")

    compound     = st.selectbox("Tire Compound",
                                ["SOFT","MEDIUM","HARD"],
                                format_func=lambda c:
                                f"{'🔴' if c=='SOFT' else '🟡' if c=='MEDIUM' else '⚪'} {c}")
    tyre_life    = st.slider("Tire Age (laps)", 1, 50, 14)
    lap_number   = st.slider("Current Lap", 1, TOTAL_LAPS, 22)
    position     = st.slider("Track Position", 1, 20, 3)
    stint_length = st.slider("Stint Length", 1, 50, 14)

    st.markdown("---")
    st.markdown("**UNDERCUT**")
    gap_ahead = st.slider("Gap to Car Ahead (s)", 0.0, 40.0, 6.5, step=0.5)

    st.markdown("---")
    st.markdown(
        f'<p style="font-size:0.6rem;color:#333;letter-spacing:1px;">'
        f'MODEL · {bundle["model_name"]}<br>'
        f'XGB F1 {bundle["xgb_f1"]} · RF F1 {bundle["rf_f1"]}</p>',
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────────────────────
# SHARED PREDICTION (used across pages)
# ─────────────────────────────────────────────────────────────
pred_label, proba = predict(compound, tyre_life, lap_number, position, stint_length)
label_order = ["STAY_OUT", "PIT_SOON", "PIT_NOW"]
label_idx   = {lbl: i for i, lbl in enumerate(le.classes_)}
ordered_p   = [proba[label_idx[l]] for l in label_order]

health_pct  = max(0.0, 100 - (tyre_life / COMPOUND_MAX[compound]) * 100)
current_lt  = lap_time_est(compound, tyre_life)

# Undercut viability
deg_rate      = COMPOUND_DEG[compound]
pace_gain     = deg_rate * tyre_life
laps_recover  = round(PIT_LOSS_SECS / pace_gain, 1) if pace_gain > 0 else 999
undercut_ok   = gap_ahead < PIT_LOSS_SECS and laps_recover < 12


# ═══════════════════════════════════════════════════════════════
# PAGE 1 — LIVE PREDICTOR
# ═══════════════════════════════════════════════════════════════
if page == "🟢 Live Predictor":

    st.markdown("""
    <div class="top-banner">
        <h1>🏎️ F1 Pit Stop <span style="color:#e8002d">Strategy</span> Analyzer</h1>
        <p>Live pit window prediction · Tire modeling · Undercut analysis</p>
    </div>""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1.2, 1.1, 1])

    # ── Recommendation ───────────────────────────────────────
    with col1:
        st.markdown('<p class="sec">Recommendation</p>', unsafe_allow_html=True)

        cfg = {
            "STAY_OUT": ("rec-stay", "#00c853", "🟢", "Tires performing. Stay out and push."),
            "PIT_SOON": ("rec-soon", "#ffd700", "🟡", "Plan your stop within the next 3 laps."),
            "PIT_NOW":  ("rec-now",  "#e8002d", "🔴", "Tires critical. Box this lap."),
        }
        cls, col, ico, msg = cfg[pred_label]
        st.markdown(f"""
        <div class="rec {cls}">
            <div class="rec-lbl" style="color:{col}">{ico} {pred_label.replace('_',' ')}</div>
            <div class="rec-sub">{msg}</div>
        </div>""", unsafe_allow_html=True)

        health_col = "#e8002d" if health_pct < 30 else "#ffd700" if health_pct < 60 else "#00c853"
        st.markdown(f"""
        <div class="mrow">
            <div class="mtile"><div class="v">{current_lt:.2f}s</div><div class="l">Lap Time</div></div>
            <div class="mtile"><div class="v" style="color:{health_col}">{health_pct:.0f}%</div><div class="l">Tire Health</div></div>
            <div class="mtile"><div class="v">{TOTAL_LAPS-lap_number}</div><div class="l">Laps Left</div></div>
        </div>""", unsafe_allow_html=True)

        # Undercut card
        uc_col = "#00c853" if undercut_ok else "#333"
        uc_lbl = "UNDERCUT VIABLE" if undercut_ok else "UNDERCUT NOT VIABLE"
        st.markdown(f"""
        <div style="background:#111;border:1.5px solid {uc_col};border-radius:6px;
                    padding:14px 16px;margin-top:4px;">
            <div style="font-size:0.75rem;font-weight:700;color:{uc_col};
                        letter-spacing:2px;text-transform:uppercase;margin-bottom:8px;">
                ⚡ {uc_lbl}
            </div>
            <table style="width:100%;font-size:0.68rem;color:#888;border-collapse:collapse;">
                <tr><td>Gap ahead</td>
                    <td style="text-align:right;color:#ccc">{gap_ahead:.1f}s</td></tr>
                <tr><td>Pace gain / lap</td>
                    <td style="text-align:right;color:#ccc">{pace_gain:.3f}s</td></tr>
                <tr><td>Laps to recover</td>
                    <td style="text-align:right;color:#ccc">{laps_recover}</td></tr>
                <tr><td>Pit loss</td>
                    <td style="text-align:right;color:#ccc">{PIT_LOSS_SECS:.0f}s</td></tr>
            </table>
        </div>""", unsafe_allow_html=True)

    # ── Confidence ───────────────────────────────────────────
    with col2:
        st.markdown('<p class="sec">Prediction Confidence</p>', unsafe_allow_html=True)
        prob_colors = {"STAY_OUT":"#00c853","PIT_SOON":"#ffd700","PIT_NOW":"#e8002d"}
        fig_p = go.Figure()
        for i, lbl in enumerate(label_order):
            fig_p.add_trace(go.Bar(
                x=[ordered_p[i]], y=[lbl], orientation="h",
                marker_color=prob_colors[lbl], marker_line_width=0,
                text=f"{ordered_p[i]*100:.1f}%", textposition="outside",
                textfont=dict(color="#eee", size=12),
            ))
        fig_p.update_layout(**PLOTLY_BASE, height=150, showlegend=False,
                            xaxis=dict(range=[0,1.15], showgrid=False,
                                       showticklabels=False, zeroline=False),
                            yaxis=dict(gridcolor="#222", zeroline=False))
        st.plotly_chart(fig_p, use_container_width=True,
                        config={"displayModeBar": False})

        # Race progress
        st.markdown('<p class="sec">Race Progress</p>', unsafe_allow_html=True)
        fig_rp = go.Figure()
        fig_rp.add_trace(go.Bar(x=[lap_number/TOTAL_LAPS], y=[""],
                                orientation="h", marker_color="#e8002d",
                                marker_line_width=0, showlegend=False))
        fig_rp.add_trace(go.Bar(x=[(TOTAL_LAPS-lap_number)/TOTAL_LAPS], y=[""],
                                orientation="h", marker_color="#222",
                                marker_line_width=0, showlegend=False))
        fig_rp.update_layout(**PLOTLY_BASE, height=50, barmode="stack",
                             showlegend=False,
                             xaxis=dict(range=[0,1],showgrid=False,
                                        showticklabels=False,zeroline=False),
                             yaxis=dict(showticklabels=False,zeroline=False),
                             margin=dict(l=0,r=0,t=4,b=4))
        st.plotly_chart(fig_rp, use_container_width=True,
                        config={"displayModeBar": False})
        st.markdown(
            f'<p style="text-align:center;font-size:0.65rem;color:#444;margin-top:-10px;">'
            f'LAP {lap_number} / {TOTAL_LAPS}</p>', unsafe_allow_html=True)

        # Lap time delta vs fresh
        fresh_lt = COMPOUND_BASE[compound]
        delta    = current_lt - fresh_lt
        st.markdown('<p class="sec">Pace Loss vs Fresh Tire</p>', unsafe_allow_html=True)
        fig_d = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=current_lt,
            delta={"reference": fresh_lt, "suffix": "s", "increasing": {"color":"#e8002d"}},
            number={"suffix": "s", "font": {"color": "#eee", "size": 28}},
            gauge={
                "axis": {"range": [fresh_lt-0.5, fresh_lt+6],
                         "tickcolor": "#444", "tickfont": {"color":"#666","size":9}},
                "bar":  {"color": COMPOUND_COLORS[compound]},
                "bgcolor": "#1a1a1a",
                "bordercolor": "#222",
                "steps": [
                    {"range": [fresh_lt-0.5, fresh_lt+2], "color":"#071a07"},
                    {"range": [fresh_lt+2,   fresh_lt+4], "color":"#1a1200"},
                    {"range": [fresh_lt+4,   fresh_lt+6], "color":"#1a0000"},
                ],
            },
        ))
        fig_d.update_layout(**PLOTLY_BASE, height=200)
        st.plotly_chart(fig_d, use_container_width=True,
                        config={"displayModeBar": False})

    # ── Deg curve ────────────────────────────────────────────
    with col3:
        st.markdown('<p class="sec">Tire Degradation Curve</p>', unsafe_allow_html=True)
        max_age = COMPOUND_MAX[compound] + 5
        ages    = list(range(1, max_age + 1))
        times_c = [lap_time_est(compound, a) for a in ages]
        cliff   = int(COMPOUND_MAX[compound] * 0.8)

        fig_deg = go.Figure()
        fig_deg.add_trace(go.Scatter(
            x=ages, y=times_c, mode="lines",
            line=dict(color=COMPOUND_COLORS[compound], width=3),
            name=compound,
            hovertemplate="Age %{x}L · %{y:.3f}s<extra></extra>",
        ))
        fig_deg.add_trace(go.Scatter(
            x=[tyre_life], y=[current_lt],
            mode="markers",
            marker=dict(color="#fff", size=11, symbol="circle",
                        line=dict(color=COMPOUND_COLORS[compound], width=3)),
            name="Now", hovertemplate=f"Now: {current_lt:.3f}s<extra></extra>",
        ))
        fig_deg.add_vrect(x0=cliff, x1=max_age,
                          fillcolor="rgba(232,0,45,0.07)", line_width=0,
                          annotation_text="⚠ CLIFF", annotation_position="top left",
                          annotation_font_color="#e8002d", annotation_font_size=9)
        fig_deg.update_layout(**PLOTLY_BASE, height=220, showlegend=False,
                              xaxis=dict(title="Tire Age (laps)", gridcolor="#222", zeroline=False),
                              yaxis=dict(title="Lap Time (s)", gridcolor="#222", zeroline=False),
                              title=dict(text=f"{compound} DEG CURVE",
                                         font=dict(size=10,color="#555"), x=0.5))
        st.plotly_chart(fig_deg, use_container_width=True,
                        config={"displayModeBar": False})

        # All compounds overlay
        st.markdown('<p class="sec">All Compounds Compared</p>', unsafe_allow_html=True)
        fig_all = go.Figure()
        for c in COMPOUNDS:
            ma   = COMPOUND_MAX[c] + 5
            ag   = list(range(1, ma + 1))
            ts   = [lap_time_est(c, a) for a in ag]
            fig_all.add_trace(go.Scatter(
                x=ag, y=ts, mode="lines",
                line=dict(color=COMPOUND_COLORS[c], width=2),
                name=c,
                hovertemplate=f"{c} %{{x}}L · %{{y:.3f}}s<extra></extra>",
            ))
        fig_all.update_layout(**PLOTLY_BASE, height=200,
                              xaxis=dict(title="Age (laps)", gridcolor="#222", zeroline=False),
                              yaxis=dict(title="Lap Time (s)", gridcolor="#222", zeroline=False),
                              legend=dict(bgcolor="#111", bordercolor="#222"))
        st.plotly_chart(fig_all, use_container_width=True,
                        config={"displayModeBar": False})

    # ── Feature vector expander ──────────────────────────────
    with st.expander("🔬 Feature Vector (what the model sees)"):
        X_show = build_X(compound, tyre_life, lap_number, position, stint_length)
        st.dataframe(X_show.T.rename(columns={0: "Value"}).style
                     .background_gradient(cmap="Reds", axis=0)
                     .format("{:.4f}"),
                     use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 2 — STRATEGY SIMULATOR
# ═══════════════════════════════════════════════════════════════
elif page == "📊 Strategy Simulator":

    st.markdown("""
    <div class="top-banner">
        <h1>📊 Strategy <span style="color:#e8002d">Simulator</span></h1>
        <p>All valid strategies ranked by estimated race time · Monte Carlo uncertainty</p>
    </div>""", unsafe_allow_html=True)

    with st.spinner("Generating and ranking strategies..."):
        strat_df = generate_strategies(lap_number, compound, tyre_life)

    fastest_time = strat_df["Est Time"].iloc[0]
    strat_df["Delta"] = (strat_df["Est Time"] - fastest_time).round(2)

    col1, col2 = st.columns([1.8, 1])

    with col1:
        st.markdown('<p class="sec">Strategy Rankings</p>', unsafe_allow_html=True)

        stop_colors = {0: "#e8002d", 1: "#ffd700", 2: "#00d2be"}
        top = strat_df.head(10).copy()

        fig_strat = go.Figure()
        for stops in [2, 1, 0]:
            subset = top[top["Stops"] == stops]
            if subset.empty:
                continue
            labels = [f"#{i+1} {r['Strategy'][:48]}" for i, r in subset.iterrows()]
            fig_strat.add_trace(go.Bar(
                x=subset["Delta"].values,
                y=labels,
                orientation="h",
                error_x=dict(type="data", array=subset["Uncertainty"].values,
                             color="#444", thickness=1.5, width=5),
                marker_color=stop_colors[stops],
                marker_line_width=0,
                name=f"{stops}-stop",
                opacity=0.85,
                hovertemplate="%{y}<br>+%{x:.2f}s vs fastest<extra></extra>",
            ))
        fig_strat.update_layout(**PLOTLY_BASE, height=420, barmode="overlay",
                                xaxis=dict(title="Time Delta vs Fastest Strategy (s)",
                                           gridcolor="#222", zeroline=True,
                                           zerolinecolor="#e8002d"),
                                yaxis=dict(gridcolor="#222", zeroline=False,
                                           tickfont=dict(size=9)),
                                legend=dict(bgcolor="#111", bordercolor="#222"),
                                title=dict(text=f"TOP 10 STRATEGIES  ·  Lap {lap_number}, "
                                               f"{compound} age {tyre_life}L",
                                           font=dict(size=11, color="#888"), x=0.5))
        st.plotly_chart(fig_strat, use_container_width=True,
                        config={"displayModeBar": False})

        # Table
        st.markdown('<p class="sec">Full Rankings Table</p>', unsafe_allow_html=True)
        display = strat_df[["Strategy","Stops","Est Time","Uncertainty","Delta","Pit Laps"]].copy()
        display.index += 1
        st.dataframe(display.head(15).style
                     .background_gradient(subset=["Delta"], cmap="Reds")
                     .format({"Est Time": "{:.2f}s", "Uncertainty": "±{:.2f}s",
                              "Delta": "+{:.2f}s"}),
                     use_container_width=True)

    with col2:
        st.markdown('<p class="sec">Best Strategy</p>', unsafe_allow_html=True)
        best = strat_df.iloc[0]
        pit_str = ", ".join(f"Lap {l}" for l in best["Pit Laps"]) if best["Pit Laps"] else "No stops"
        st.markdown(f"""
        <div style="background:#111;border:2px solid #e8002d;border-radius:6px;padding:18px;">
            <div style="font-size:0.65rem;color:#e8002d;letter-spacing:2px;
                        text-transform:uppercase;margin-bottom:8px;">🏆 OPTIMAL STRATEGY</div>
            <div style="font-size:0.88rem;color:#eee;font-weight:600;margin-bottom:12px;
                        line-height:1.5">{best['Strategy']}</div>
            <table style="width:100%;font-size:0.7rem;color:#888;border-collapse:collapse;">
                <tr><td style="padding:4px 0">Est. Race Time</td>
                    <td style="text-align:right;color:#eee">{best['Est Time']:.2f}s</td></tr>
                <tr><td style="padding:4px 0">Uncertainty</td>
                    <td style="text-align:right;color:#eee">±{best['Uncertainty']:.2f}s</td></tr>
                <tr><td style="padding:4px 0">Pit Stop(s)</td>
                    <td style="text-align:right;color:#eee">{pit_str}</td></tr>
                <tr><td style="padding:4px 0">Stop count</td>
                    <td style="text-align:right;color:#eee">{best['Stops']}</td></tr>
            </table>
        </div>""", unsafe_allow_html=True)

        # Stops distribution donut
        st.markdown('<p class="sec" style="margin-top:20px">Strategy Mix</p>',
                    unsafe_allow_html=True)
        stop_counts = strat_df["Stops"].value_counts().sort_index()
        fig_donut = go.Figure(go.Pie(
            labels=[f"{s}-stop" for s in stop_counts.index],
            values=stop_counts.values,
            hole=0.55,
            marker_colors=[stop_colors.get(s, "#888") for s in stop_counts.index],
            textfont=dict(color="#eee", size=11),
        ))
        fig_donut.update_layout(**PLOTLY_BASE, height=230, showlegend=True,
                                legend=dict(bgcolor="#111", bordercolor="#222",
                                            font=dict(color="#ccc")))
        st.plotly_chart(fig_donut, use_container_width=True,
                        config={"displayModeBar": False})

        # Time comparison scatter
        st.markdown('<p class="sec">Time vs Uncertainty</p>', unsafe_allow_html=True)
        fig_sc = go.Figure()
        for stops in [0, 1, 2]:
            sub = strat_df[strat_df["Stops"] == stops]
            if sub.empty:
                continue
            fig_sc.add_trace(go.Scatter(
                x=sub["Delta"], y=sub["Uncertainty"],
                mode="markers",
                marker=dict(color=stop_colors[stops], size=7, opacity=0.7),
                name=f"{stops}-stop",
                hovertemplate="%{text}<extra></extra>",
                text=sub["Strategy"].str[:35],
            ))
        fig_sc.update_layout(**PLOTLY_BASE, height=220,
                             xaxis=dict(title="Time Delta (s)", gridcolor="#222", zeroline=False),
                             yaxis=dict(title="Uncertainty (s)", gridcolor="#222", zeroline=False),
                             legend=dict(bgcolor="#111", bordercolor="#222"))
        st.plotly_chart(fig_sc, use_container_width=True,
                        config={"displayModeBar": False})


# ═══════════════════════════════════════════════════════════════
# PAGE 3 — TIRE ANALYST
# ═══════════════════════════════════════════════════════════════
elif page == "🔥 Tire Analyst":

    st.markdown("""
    <div class="top-banner">
        <h1>🔥 Tire <span style="color:#e8002d">Analyst</span></h1>
        <p>Deep dive into compound degradation · Crossover points · Stint optimizer</p>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="sec">Degradation Rate Comparison</p>', unsafe_allow_html=True)
        fig_deg = go.Figure()
        for c in COMPOUNDS:
            ages = list(range(1, COMPOUND_MAX[c] + 10))
            ts   = [lap_time_est(c, a) for a in ages]
            fig_deg.add_trace(go.Scatter(
                x=ages, y=ts, mode="lines",
                line=dict(color=COMPOUND_COLORS[c], width=2.5), name=c,
                hovertemplate=f"{c}: %{{y:.3f}}s at age %{{x}}<extra></extra>",
            ))
            # Cliff zone
            cliff = int(COMPOUND_MAX[c] * 0.8)
            fig_deg.add_vrect(
                x0=cliff, x1=COMPOUND_MAX[c],
                fillcolor=f"rgba{tuple(int(COMPOUND_COLORS[c].lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.06,)}",
                line_width=0,
            )
        fig_deg.update_layout(**PLOTLY_BASE, height=300,
                              xaxis=dict(title="Tire Age (laps)", gridcolor="#222", zeroline=False),
                              yaxis=dict(title="Lap Time (s)", gridcolor="#222", zeroline=False),
                              legend=dict(bgcolor="#111", bordercolor="#222"),
                              title=dict(text="ALL COMPOUNDS · Lap Time vs Tire Age",
                                         font=dict(size=10,color="#555"), x=0.5))
        st.plotly_chart(fig_deg, use_container_width=True,
                        config={"displayModeBar": False})

        # Degradation rate per lap (derivative)
        st.markdown('<p class="sec">Degradation Rate (seconds lost per lap)</p>',
                    unsafe_allow_html=True)
        fig_rate = go.Figure()
        for c in COMPOUNDS:
            fig_rate.add_trace(go.Bar(
                x=[c], y=[COMPOUND_DEG[c]],
                marker_color=COMPOUND_COLORS[c], marker_line_width=0,
                text=f"{COMPOUND_DEG[c]:.3f}s/lap", textposition="outside",
                textfont=dict(color="#eee"),
                name=c,
            ))
        fig_rate.update_layout(**PLOTLY_BASE, height=220, showlegend=False,
                               yaxis=dict(title="Deg Rate (s/lap)", gridcolor="#222",
                                          zeroline=False),
                               title=dict(text="DEGRADATION RATE BY COMPOUND",
                                          font=dict(size=10,color="#555"), x=0.5))
        st.plotly_chart(fig_rate, use_container_width=True,
                        config={"displayModeBar": False})

    with col2:
        st.markdown('<p class="sec">Compound Crossover Points</p>',
                    unsafe_allow_html=True)
        # Find where SOFT becomes slower than MEDIUM, MEDIUM vs HARD
        fig_cross = go.Figure()
        age_range  = list(range(1, 55))
        soft_times = [lap_time_est("SOFT",   a) for a in age_range]
        med_times  = [lap_time_est("MEDIUM", a) for a in age_range]
        hard_times = [lap_time_est("HARD",   a) for a in age_range]

        fig_cross.add_trace(go.Scatter(x=age_range, y=soft_times, name="SOFT",
                                       line=dict(color="#e8002d", width=2)))
        fig_cross.add_trace(go.Scatter(x=age_range, y=med_times, name="MEDIUM",
                                       line=dict(color="#ffd700", width=2)))
        fig_cross.add_trace(go.Scatter(x=age_range, y=hard_times, name="HARD",
                                       line=dict(color="#ebebeb", width=2)))

        # Mark crossover points
        for i, a in enumerate(age_range):
            if i > 0 and soft_times[i] > med_times[i] and soft_times[i-1] <= med_times[i-1]:
                fig_cross.add_vline(x=a, line_color="#e8002d", line_dash="dash",
                                    annotation_text=f"SOFT>MED @ {a}L",
                                    annotation_font_color="#e8002d", annotation_font_size=9)
            if i > 0 and med_times[i] > hard_times[i] and med_times[i-1] <= hard_times[i-1]:
                fig_cross.add_vline(x=a, line_color="#ffd700", line_dash="dash",
                                    annotation_text=f"MED>HARD @ {a}L",
                                    annotation_font_color="#ffd700", annotation_font_size=9)

        fig_cross.update_layout(**PLOTLY_BASE, height=300,
                                xaxis=dict(title="Tire Age (laps)", gridcolor="#222", zeroline=False),
                                yaxis=dict(title="Lap Time (s)", gridcolor="#222", zeroline=False),
                                legend=dict(bgcolor="#111", bordercolor="#222"),
                                title=dict(text="CROSSOVER ANALYSIS · When does a newer tire become faster?",
                                           font=dict(size=10,color="#555"), x=0.5))
        st.plotly_chart(fig_cross, use_container_width=True,
                        config={"displayModeBar": False})

        # Health meter for current selection
        st.markdown('<p class="sec">Current Tire Health Breakdown</p>',
                    unsafe_allow_html=True)
        max_life = COMPOUND_MAX[compound]
        used_pct = min(100, (tyre_life / max_life) * 100)
        cliff_pct= 80

        fig_health = go.Figure(go.Bar(
            x=[tyre_life], y=[compound],
            orientation="h",
            marker=dict(
                color=COMPOUND_COLORS[compound],
                opacity=0.85,
            ),
            text=f"{tyre_life} / {max_life} laps · {health_pct:.0f}% health",
            textposition="inside",
            textfont=dict(color="#000" if compound == "HARD" else "#fff", size=12),
        ))
        fig_health.add_vline(x=int(max_life * 0.8), line_color="#e8002d",
                             line_dash="dash", line_width=1,
                             annotation_text="cliff", annotation_font_color="#e8002d",
                             annotation_font_size=9)
        fig_health.update_layout(**PLOTLY_BASE, height=110,
                                 xaxis=dict(range=[0, max_life + 5], gridcolor="#222",
                                            zeroline=False,
                                            title=f"Tire Age (max ~{max_life}L for {compound})"),
                                 yaxis=dict(zeroline=False))
        st.plotly_chart(fig_health, use_container_width=True,
                        config={"displayModeBar": False})

        # Stint length stats table
        st.markdown('<p class="sec">Optimal Stint Windows</p>', unsafe_allow_html=True)
        stint_data = []
        for c in COMPOUNDS:
            cliff = int(COMPOUND_MAX[c] * 0.8)
            stint_data.append({
                "Compound": c,
                "Min Stint": COMPOUND_MIN_LAPS[c],
                "Optimal Window": f"L{COMPOUND_MIN_LAPS[c]}–L{cliff}",
                "Max Life":  COMPOUND_MAX[c],
                "Deg Rate":  f"{COMPOUND_DEG[c]:.3f}s/lap",
                "Base Time": f"{COMPOUND_BASE[c]:.1f}s",
            })
        st.dataframe(pd.DataFrame(stint_data).set_index("Compound"),
                     use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 4 — RACE REPLAY
# ═══════════════════════════════════════════════════════════════
elif page == "🔁 Race Replay":

    st.markdown("""
    <div class="top-banner">
        <h1>🔁 Race <span style="color:#e8002d">Replay</span></h1>
        <p>Lap-by-lap model validation against actual race decisions</p>
    </div>""", unsafe_allow_html=True)

    col_sel, _ = st.columns([1, 3])
    with col_sel:
        driver = st.selectbox("Select Driver", DRIVERS)

    # Build replay for selected driver
    driver_df = df_race[df_race["Driver"] == driver].copy().reset_index(drop=True)

    # Add StintLength
    stint_len, cnt = [], 0
    for _, row in driver_df.iterrows():
        cnt += 1
        stint_len.append(cnt)
        if row.get("Pitted", 0) == 1:
            cnt = 0
    driver_df["StintLength"] = stint_len

    # Run predictions lap by lap
    records = []
    for _, row in driver_df.iterrows():
        c  = row["Compound"]
        tl = row.get("TyreLife", row["StintLength"])
        ln = row["LapNumber"]
        sl = row["StintLength"]
        pos= row.get("Position", 5)
        X  = build_X(c, tl, ln, pos, sl)
        p  = model.predict_proba(X)[0]
        pred = le.inverse_transform([int(np.argmax(p))])[0]
        actual = row.get("PitLabel", "STAY_OUT")
        records.append({
            "Lap":        ln,
            "Compound":   c,
            "Tire Age":   tl,
            "Predicted":  pred,
            "Actual":     actual,
            "Match":      pred == actual,
            "Confidence": round(float(max(p)) * 100, 1),
        })
    replay = pd.DataFrame(records)

    accuracy   = replay["Match"].mean()
    pit_laps_r = replay[replay["Actual"] != "STAY_OUT"]
    pit_recall = pit_laps_r["Match"].mean() if len(pit_laps_r) > 0 else 0.0

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Overall Accuracy",  f"{accuracy*100:.1f}%")
    col2.metric("Pit Recall",        f"{pit_recall*100:.1f}%")
    col3.metric("Laps Replayed",     len(replay))
    col4.metric("Pit Events",        len(pit_laps_r))

    col_a, col_b = st.columns([2, 1])

    with col_a:
        st.markdown('<p class="sec">Predicted vs Actual Decision — Lap by Lap</p>',
                    unsafe_allow_html=True)
        lmap = {"STAY_OUT": 0, "PIT_SOON": 1, "PIT_NOW": 2}
        laps  = replay["Lap"].values
        pred_v= replay["Predicted"].map(lmap).values
        act_v = replay["Actual"].map(lmap).values
        ok    = replay["Match"].values

        fig_rpl = go.Figure()
        fig_rpl.add_trace(go.Scatter(
            x=laps, y=act_v, mode="lines+markers",
            line=dict(color="#444", width=1.5), marker=dict(size=4),
            name="Actual", hovertemplate="Lap %{x} · Actual: %{text}<extra></extra>",
            text=replay["Actual"].values,
        ))
        fig_rpl.add_trace(go.Scatter(
            x=laps[ok], y=pred_v[ok], mode="markers",
            marker=dict(color="#00c853", size=9, symbol="circle"),
            name="✅ Correct",
        ))
        fig_rpl.add_trace(go.Scatter(
            x=laps[~ok], y=pred_v[~ok], mode="markers",
            marker=dict(color="#e8002d", size=11, symbol="x"),
            name="❌ Wrong",
        ))
        fig_rpl.update_layout(**PLOTLY_BASE, height=260,
                              yaxis=dict(tickvals=[0,1,2],
                                         ticktext=["STAY OUT","PIT SOON","PIT NOW"],
                                         gridcolor="#222", zeroline=False),
                              xaxis=dict(title="Lap Number", gridcolor="#222", zeroline=False),
                              legend=dict(bgcolor="#111", bordercolor="#222"))
        st.plotly_chart(fig_rpl, use_container_width=True,
                        config={"displayModeBar": False})

        # Confidence over race
        st.markdown('<p class="sec">Prediction Confidence Per Lap</p>',
                    unsafe_allow_html=True)
        fig_conf = go.Figure()
        fig_conf.add_trace(go.Scatter(
            x=laps, y=replay["Confidence"].values,
            mode="lines", fill="tozeroy",
            line=dict(color="#e8002d", width=1.5),
            fillcolor="rgba(232,0,45,0.1)",
            name="Confidence",
            hovertemplate="Lap %{x} · %{y:.1f}%<extra></extra>",
        ))
        fig_conf.add_hline(y=80, line_color="#ffd700", line_dash="dash",
                           annotation_text="80%", annotation_font_color="#ffd700")
        fig_conf.update_layout(**PLOTLY_BASE, height=180,
                               yaxis=dict(title="Confidence (%)", range=[0,105],
                                          gridcolor="#222", zeroline=False),
                               xaxis=dict(title="Lap", gridcolor="#222", zeroline=False))
        st.plotly_chart(fig_conf, use_container_width=True,
                        config={"displayModeBar": False})

    with col_b:
        st.markdown('<p class="sec">Pit Events Detail</p>', unsafe_allow_html=True)
        if len(pit_laps_r) > 0:
            for _, r in pit_laps_r.iterrows():
                icon = "✅" if r["Match"] else "❌"
                bg   = "#071a07" if r["Match"] else "#1a0000"
                bd   = "#00c853" if r["Match"] else "#e8002d"
                st.markdown(f"""
                <div style="background:{bg};border:1px solid {bd};border-radius:5px;
                            padding:10px 14px;margin-bottom:6px;">
                    <div style="font-size:0.7rem;color:#888;letter-spacing:1px;">
                        {icon} LAP {int(r['Lap'])} · {r['Compound']} · Age {int(r['Tire Age'])}L
                    </div>
                    <div style="font-size:0.75rem;color:#eee;margin-top:4px;">
                        Actual: <b>{r['Actual']}</b><br>
                        Pred:   <b>{r['Predicted']}</b>
                        <span style="float:right;color:#888">{r['Confidence']:.0f}%</span>
                    </div>
                </div>""", unsafe_allow_html=True)
        else:
            st.info("No pit events found for this driver.")

        # Accuracy donut
        st.markdown('<p class="sec" style="margin-top:16px">Accuracy Breakdown</p>',
                    unsafe_allow_html=True)
        correct   = replay["Match"].sum()
        incorrect = len(replay) - correct
        fig_acc   = go.Figure(go.Pie(
            labels=["Correct", "Wrong"],
            values=[correct, incorrect],
            hole=0.6,
            marker_colors=["#00c853", "#e8002d"],
            textfont=dict(color="#eee"),
        ))
        fig_acc.update_layout(**PLOTLY_BASE, height=200, showlegend=True,
                              legend=dict(bgcolor="#111", bordercolor="#222",
                                          font=dict(color="#ccc")))
        st.plotly_chart(fig_acc, use_container_width=True,
                        config={"displayModeBar": False})

    # Full lap table
    with st.expander("📋 Full Lap-by-Lap Table"):
        st.dataframe(
            replay.style.apply(
                lambda row: ["background:#071a07" if row["Match"] else
                             "background:#1a0000"] * len(row), axis=1
            ).format({"Confidence": "{:.1f}%"}),
            use_container_width=True,
        )


# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    F1 PIT STOP STRATEGY ANALYZER · FASTF1 + XGBOOST + STREAMLIT · FAST-NUCES CS · 2025
</div>
""", unsafe_allow_html=True)
