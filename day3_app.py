"""
F1 Pit Stop Strategy Analyzer
Day 3 — Streamlit Web App

Run:
    streamlit run day3_app.py

Requires model.pkl from Day 2 in the same directory.
"""

import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="F1 Pit Strategy Analyzer",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS — F1 dark theme
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Base */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0f0f0f;
        color: #eeeeee;
        font-family: 'Courier New', monospace;
    }
    [data-testid="stSidebar"] {
        background-color: #141414;
        border-right: 1px solid #2a2a2a;
    }
    [data-testid="stSidebar"] * { color: #cccccc !important; }

    /* Header */
    .f1-header {
        background: linear-gradient(135deg, #1a0000 0%, #0f0f0f 60%);
        border-left: 4px solid #e8002d;
        padding: 20px 28px 16px 28px;
        margin-bottom: 24px;
        border-radius: 0 8px 8px 0;
    }
    .f1-header h1 {
        font-size: 2rem;
        font-weight: 900;
        color: #ffffff;
        letter-spacing: 3px;
        margin: 0 0 4px 0;
        text-transform: uppercase;
    }
    .f1-header p {
        color: #888888;
        font-size: 0.8rem;
        letter-spacing: 2px;
        margin: 0;
        text-transform: uppercase;
    }
    .red { color: #e8002d; }

    /* Recommendation card */
    .rec-card {
        border-radius: 8px;
        padding: 24px 28px;
        text-align: center;
        margin-bottom: 16px;
    }
    .rec-stay  { background: #0a2a0a; border: 2px solid #00c853; }
    .rec-soon  { background: #2a2000; border: 2px solid #ffd700; }
    .rec-now   { background: #2a0000; border: 2px solid #e8002d;
                 animation: pulse 1.5s ease-in-out infinite; }
    @keyframes pulse {
        0%, 100% { box-shadow: 0 0 0 0 rgba(232,0,45,0.4); }
        50%       { box-shadow: 0 0 0 12px rgba(232,0,45,0); }
    }
    .rec-label {
        font-size: 2.2rem;
        font-weight: 900;
        letter-spacing: 4px;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .rec-sub {
        font-size: 0.78rem;
        letter-spacing: 2px;
        color: #aaaaaa;
        text-transform: uppercase;
    }

    /* Metric tiles */
    .metric-row { display: flex; gap: 12px; margin-bottom: 20px; }
    .metric-tile {
        flex: 1;
        background: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 6px;
        padding: 14px 16px;
        text-align: center;
    }
    .metric-tile .val {
        font-size: 1.6rem;
        font-weight: 700;
        color: #ffffff;
    }
    .metric-tile .lbl {
        font-size: 0.65rem;
        letter-spacing: 2px;
        color: #666666;
        text-transform: uppercase;
        margin-top: 2px;
    }

    /* Section headers */
    .section-head {
        font-size: 0.7rem;
        letter-spacing: 3px;
        color: #e8002d;
        text-transform: uppercase;
        border-bottom: 1px solid #2a2a2a;
        padding-bottom: 6px;
        margin: 20px 0 14px 0;
    }

    /* Sidebar labels */
    .sidebar-group {
        font-size: 0.65rem;
        letter-spacing: 2px;
        color: #e8002d;
        text-transform: uppercase;
        margin: 16px 0 6px 0;
    }

    /* Probability bar */
    .prob-bar-wrap { margin: 6px 0; }
    .prob-label {
        font-size: 0.7rem;
        letter-spacing: 2px;
        color: #888;
        text-transform: uppercase;
        margin-bottom: 3px;
    }

    /* Streamlit overrides */
    .stSlider > div > div { background: #2a2a2a; }
    div[data-testid="stMetric"] { background: #1a1a1a; border-radius: 6px; padding: 10px; }
    .stSelectbox > div > div { background: #1a1a1a; border: 1px solid #2a2a2a; }
    h2, h3 { color: #eeeeee; }
    .stPlotlyChart { background: transparent; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

try:
    bundle = load_model()
    model  = bundle["model"]
    le     = bundle["label_encoder"]
    feats  = bundle["feature_cols"]
except FileNotFoundError:
    st.error("⚠️ model.pkl not found. Run day2_model.py first.")
    st.stop()


# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
COMPOUND_COLORS = {"SOFT": "#e8002d", "MEDIUM": "#ffd700", "HARD": "#ebebeb"}
COMPOUND_CODE   = {"SOFT": 0, "MEDIUM": 1, "HARD": 2}
COMPOUND_BASE   = {"SOFT": 88.5, "MEDIUM": 89.8, "HARD": 91.0}
COMPOUND_DEG    = {"SOFT": 0.12, "MEDIUM": 0.07, "HARD": 0.04}
COMPOUND_MAX    = {"SOFT": 20,   "MEDIUM": 35,   "HARD": 50}
TOTAL_LAPS      = 57
PIT_LOSS_SECS   = 23.0   # average time lost in the pit lane


# ─────────────────────────────────────────────────────────────
# HELPER — build feature vector from sidebar inputs
# ─────────────────────────────────────────────────────────────
def build_feature_vector(compound, tyre_life, lap_number,
                          position, stint_length):
    """Derives all 14 features from the 5 user inputs."""

    # Compute lap time from degradation model
    base     = COMPOUND_BASE[compound]
    deg      = COMPOUND_DEG[compound] * tyre_life
    lap_time = base + deg

    # Rolling average (approximate — assume linear progression)
    rolling  = base + COMPOUND_DEG[compound] * max(tyre_life - 1.5, 1)

    # Deltas
    lap_delta = COMPOUND_DEG[compound]   # approx one-lap increment
    deg_rate  = COMPOUND_DEG[compound]
    deg_accel = 0.0                       # steady state assumption

    # Pace loss vs best lap on this stint
    pace_loss = COMPOUND_DEG[compound] * tyre_life

    # Remaining / progress
    laps_rem  = TOTAL_LAPS - lap_number
    progress  = lap_number / TOTAL_LAPS

    # Tyre health
    health = max(0, 100 - (tyre_life / COMPOUND_MAX[compound]) * 100)

    # Pit window
    in_window = 1 if (15 <= lap_number <= 25) or (35 <= lap_number <= 45) else 0

    row = {
        "TyreLife":          tyre_life,
        "TyreHealthPct":     health,
        "LapTimeSec":        lap_time,
        "RollingAvgLapTime": rolling,
        "LapTimeDelta":      lap_delta,
        "DegRate":           deg_rate,
        "DegAccel":          deg_accel,
        "PaceLoss":          pace_loss,
        "StintLength":       stint_length,
        "LapsRemaining":     laps_rem,
        "RaceProgress":      progress,
        "CompoundCode":      COMPOUND_CODE[compound],
        "Position":          position,
        "InPitWindow":       in_window,
    }
    return pd.DataFrame([row])[feats]


# ─────────────────────────────────────────────────────────────
# UNDERCUT / OVERCUT LOGIC
# ─────────────────────────────────────────────────────────────
def undercut_analysis(gap_ahead, tyre_life, compound):
    """
    Undercut: pit early, gain track position via fresh-tire pace.
    Viable if: gap to car ahead < pit loss time AND you can make up
               the remaining gap with faster laps on new tires.

    Returns a dict with recommendation and reasoning.
    """
    deg_rate    = COMPOUND_DEG[compound]
    pace_gain   = deg_rate * tyre_life        # approx gain per lap on fresh tires
    laps_needed = PIT_LOSS_SECS / pace_gain if pace_gain > 0 else 999

    undercut_viable = gap_ahead < PIT_LOSS_SECS and laps_needed < 10

    return {
        "undercut_viable": undercut_viable,
        "gap_ahead":       gap_ahead,
        "pace_gain_per_lap": round(pace_gain, 3),
        "laps_to_recover": round(laps_needed, 1),
    }


# ─────────────────────────────────────────────────────────────
# TIRE DEGRADATION CHART
# ─────────────────────────────────────────────────────────────
def tire_deg_chart(compound, current_tyre_life):
    max_lap = COMPOUND_MAX[compound] + 5
    laps    = list(range(1, max_lap + 1))
    times   = [COMPOUND_BASE[compound] + COMPOUND_DEG[compound] * l for l in laps]

    fig = go.Figure()

    # Degradation line
    fig.add_trace(go.Scatter(
        x=laps, y=times,
        mode="lines",
        line=dict(color=COMPOUND_COLORS[compound], width=3),
        name=compound,
        hovertemplate="Lap %{x}<br>Lap Time: %{y:.3f}s<extra></extra>",
    ))

    # Current position marker
    current_time = COMPOUND_BASE[compound] + COMPOUND_DEG[compound] * current_tyre_life
    fig.add_trace(go.Scatter(
        x=[current_tyre_life], y=[current_time],
        mode="markers",
        marker=dict(color="#ffffff", size=12, symbol="circle",
                    line=dict(color=COMPOUND_COLORS[compound], width=3)),
        name="Current",
        hovertemplate=f"Current: {current_time:.3f}s<extra></extra>",
    ))

    # Cliff zone shading (last 20% of compound life)
    cliff_start = int(COMPOUND_MAX[compound] * 0.8)
    fig.add_vrect(
        x0=cliff_start, x1=max_lap,
        fillcolor="rgba(232,0,45,0.08)",
        line_width=0,
        annotation_text="⚠ CLIFF ZONE",
        annotation_position="top left",
        annotation_font_color="#e8002d",
        annotation_font_size=10,
    )

    fig.update_layout(
        paper_bgcolor="#0f0f0f",
        plot_bgcolor="#1a1a1a",
        font=dict(family="Courier New", color="#cccccc", size=11),
        margin=dict(l=10, r=10, t=30, b=10),
        height=260,
        showlegend=False,
        xaxis=dict(title="Tire Age (laps)", gridcolor="#2a2a2a",
                   zeroline=False, color="#888888"),
        yaxis=dict(title="Lap Time (s)", gridcolor="#2a2a2a",
                   zeroline=False, color="#888888"),
        title=dict(text=f"{compound} TIRE DEGRADATION CURVE",
                   font=dict(size=11, color="#888888"),
                   x=0.5),
    )
    return fig


# ─────────────────────────────────────────────────────────────
# PROBABILITY GAUGE
# ─────────────────────────────────────────────────────────────
def prob_gauge(proba, label_order):
    colors = {"STAY_OUT": "#00c853", "PIT_SOON": "#ffd700", "PIT_NOW": "#e8002d"}
    fig = go.Figure()

    for i, label in enumerate(label_order):
        fig.add_trace(go.Bar(
            x=[proba[i]],
            y=[label],
            orientation="h",
            marker_color=colors[label],
            marker_line_width=0,
            text=f"{proba[i]*100:.1f}%",
            textposition="outside",
            textfont=dict(color="#eeeeee", size=12),
            hovertemplate=f"{label}: {proba[i]*100:.1f}%<extra></extra>",
        ))

    fig.update_layout(
        paper_bgcolor="#0f0f0f",
        plot_bgcolor="#1a1a1a",
        font=dict(family="Courier New", color="#cccccc", size=11),
        margin=dict(l=10, r=60, t=10, b=10),
        height=160,
        showlegend=False,
        barmode="overlay",
        xaxis=dict(range=[0, 1.15], showgrid=False,
                   showticklabels=False, zeroline=False),
        yaxis=dict(gridcolor="#2a2a2a", zeroline=False, color="#cccccc"),
    )
    return fig


# ─────────────────────────────────────────────────────────────
# RACE PROGRESS BAR
# ─────────────────────────────────────────────────────────────
def race_progress_chart(lap_number, total=TOTAL_LAPS):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[lap_number / total],
        y=["Race"],
        orientation="h",
        marker_color="#e8002d",
        marker_line_width=0,
        hoverinfo="skip",
    ))
    fig.add_trace(go.Bar(
        x=[(total - lap_number) / total],
        y=["Race"],
        orientation="h",
        marker_color="#2a2a2a",
        marker_line_width=0,
        hoverinfo="skip",
    ))
    fig.update_layout(
        paper_bgcolor="#0f0f0f",
        plot_bgcolor="#0f0f0f",
        barmode="stack",
        margin=dict(l=0, r=0, t=0, b=0),
        height=40,
        showlegend=False,
        xaxis=dict(range=[0, 1], showgrid=False,
                   showticklabels=False, zeroline=False),
        yaxis=dict(showticklabels=False, zeroline=False),
    )
    return fig


# ─────────────────────────────────────────────────────────────
# SIDEBAR — Inputs
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏎️ RACE INPUT")
    st.markdown("---")

    st.markdown('<p class="sidebar-group">TIRE SETUP</p>', unsafe_allow_html=True)
    compound     = st.selectbox("Compound", ["SOFT", "MEDIUM", "HARD"],
                                 format_func=lambda c: f"{'🔴' if c=='SOFT' else '🟡' if c=='MEDIUM' else '⚪'} {c}")
    tyre_life    = st.slider("Tire Age (laps)", 1, 50, 12)
    stint_length = st.slider("Stint Length (laps on this set)", 1, 50, 12)

    st.markdown('<p class="sidebar-group">RACE SITUATION</p>', unsafe_allow_html=True)
    lap_number = st.slider("Current Lap", 1, TOTAL_LAPS, 18)
    position   = st.slider("Track Position", 1, 20, 4)

    st.markdown('<p class="sidebar-group">UNDERCUT ANALYSIS</p>', unsafe_allow_html=True)
    gap_ahead  = st.slider("Gap to Car Ahead (seconds)", 0.0, 40.0, 8.0, step=0.5)

    st.markdown("---")
    st.markdown(
        f'<p style="font-size:0.65rem;color:#444;letter-spacing:1px;">'
        f'MODEL: {bundle["model_name"].upper()}<br>'
        f'XGB F1: {bundle["xgb_f1"]} · RF F1: {bundle["rf_f1"]}'
        f'</p>',
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────────────────────
X_input  = build_feature_vector(compound, tyre_life, lap_number,
                                 position, stint_length)
proba    = model.predict_proba(X_input)[0]
pred_idx = int(np.argmax(proba))
pred_lbl = le.inverse_transform([pred_idx])[0]   # STAY_OUT / PIT_SOON / PIT_NOW

undercut = undercut_analysis(gap_ahead, tyre_life, compound)

# ─────────────────────────────────────────────────────────────
# LAYOUT
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="f1-header">
    <h1>🏎️ F1 Pit Stop <span class="red">Strategy</span> Analyzer</h1>
    <p>Real-time pit window prediction · Tire degradation modeling · Undercut analysis</p>
</div>
""", unsafe_allow_html=True)

# ── Row 1: Recommendation + Probabilities + Undercut ────────
col1, col2, col3 = st.columns([1.2, 1.2, 1])

with col1:
    st.markdown('<p class="section-head">Recommendation</p>', unsafe_allow_html=True)

    if pred_lbl == "STAY_OUT":
        card_cls = "rec-stay"
        emoji    = "🟢"
        color    = "#00c853"
        msg      = "Tires are performing well. Stay out and push."
    elif pred_lbl == "PIT_SOON":
        card_cls = "rec-soon"
        emoji    = "🟡"
        color    = "#ffd700"
        msg      = "Plan your pit stop within the next 3 laps."
    else:
        card_cls = "rec-now"
        emoji    = "🔴"
        color    = "#e8002d"
        msg      = "Tires critically degraded. Box this lap."

    st.markdown(f"""
    <div class="rec-card {card_cls}">
        <div class="rec-label" style="color:{color}">{emoji} {pred_lbl.replace('_',' ')}</div>
        <div class="rec-sub">{msg}</div>
    </div>
    """, unsafe_allow_html=True)

    # Key metrics
    lap_time_now = COMPOUND_BASE[compound] + COMPOUND_DEG[compound] * tyre_life
    health_pct   = max(0, 100 - (tyre_life / COMPOUND_MAX[compound]) * 100)

    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-tile">
            <div class="val">{lap_time_now:.2f}s</div>
            <div class="lbl">Lap Time</div>
        </div>
        <div class="metric-tile">
            <div class="val" style="color:{'#e8002d' if health_pct < 30 else '#ffd700' if health_pct < 60 else '#00c853'}">{health_pct:.0f}%</div>
            <div class="lbl">Tire Health</div>
        </div>
        <div class="metric-tile">
            <div class="val">{TOTAL_LAPS - lap_number}</div>
            <div class="lbl">Laps Left</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown('<p class="section-head">Prediction Confidence</p>', unsafe_allow_html=True)
    label_order = ["STAY_OUT", "PIT_SOON", "PIT_NOW"]
    label_idx   = {lbl: i for i, lbl in enumerate(le.classes_)}
    ordered_proba = [proba[label_idx[l]] for l in label_order]
    st.plotly_chart(prob_gauge(ordered_proba, label_order),
                    use_container_width=True, config={"displayModeBar": False})

    # Race progress
    st.markdown('<p class="section-head">Race Progress</p>', unsafe_allow_html=True)
    st.plotly_chart(race_progress_chart(lap_number),
                    use_container_width=True, config={"displayModeBar": False})
    st.markdown(
        f'<p style="font-size:0.7rem;color:#555;text-align:center;margin-top:-10px;">'
        f'LAP {lap_number} / {TOTAL_LAPS}</p>',
        unsafe_allow_html=True
    )

with col3:
    st.markdown('<p class="section-head">Undercut Analysis</p>', unsafe_allow_html=True)

    uc_color = "#00c853" if undercut["undercut_viable"] else "#555555"
    uc_label = "VIABLE" if undercut["undercut_viable"] else "NOT VIABLE"

    st.markdown(f"""
    <div class="rec-card" style="background:#1a1a1a;border:2px solid {uc_color};padding:16px;">
        <div style="font-size:1.1rem;font-weight:900;color:{uc_color};
                    letter-spacing:3px;text-transform:uppercase;margin-bottom:10px;">
            ⚡ UNDERCUT {uc_label}
        </div>
        <table style="width:100%;font-size:0.72rem;color:#aaa;border-collapse:collapse;">
            <tr>
                <td style="padding:4px 0;">Gap ahead</td>
                <td style="text-align:right;color:#eee;">{undercut['gap_ahead']:.1f}s</td>
            </tr>
            <tr>
                <td style="padding:4px 0;">Pace gain / lap</td>
                <td style="text-align:right;color:#eee;">{undercut['pace_gain_per_lap']:.3f}s</td>
            </tr>
            <tr>
                <td style="padding:4px 0;">Laps to recover</td>
                <td style="text-align:right;color:#eee;">{undercut['laps_to_recover']}</td>
            </tr>
            <tr>
                <td style="padding:4px 0;">Pit loss</td>
                <td style="text-align:right;color:#eee;">{PIT_LOSS_SECS:.0f}s</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    if undercut["undercut_viable"]:
        st.success(f"Pit now to jump the car ahead. You'll recover the pit loss in ~{undercut['laps_to_recover']} laps on fresh rubber.")
    else:
        st.info("Gap too large or pace gain insufficient for a clean undercut.")


# ── Row 2: Tire degradation curve ───────────────────────────
st.markdown('<p class="section-head">Tire Degradation Model</p>', unsafe_allow_html=True)
st.plotly_chart(tire_deg_chart(compound, tyre_life),
                use_container_width=True, config={"displayModeBar": False})


# ── Row 3: Feature breakdown table ──────────────────────────
with st.expander("🔬 Feature Vector (what the model sees)", expanded=False):
    st.dataframe(
        X_input.T.rename(columns={0: "Value"}).style
        .background_gradient(cmap="Reds", axis=0)
        .format("{:.4f}"),
        use_container_width=True,
    )

# ── Footer ───────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:32px;padding-top:16px;border-top:1px solid #2a2a2a;
            text-align:center;font-size:0.65rem;color:#444;letter-spacing:2px;">
    F1 PIT STOP STRATEGY ANALYZER · BUILT WITH FASTF1 + XGBOOST · FAST-NUCES CS
</div>
""", unsafe_allow_html=True)
