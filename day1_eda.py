"""
F1 Pit Stop Strategy Analyzer
Day 1 — Data Acquisition & Exploratory Data Analysis

WHEN RUNNING LOCALLY:
  pip install fastf1 pandas numpy matplotlib plotly scikit-learn xgboost streamlit

This script has two modes:
  MODE A (local): pulls real data from fastf1 API
  MODE B (fallback): generates a realistic synthetic dataset with same schema

Switch USE_REAL_DATA = True once you have fastf1 working locally.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
USE_REAL_DATA = False   # Set True on your local machine
RACE_YEAR     = 2024
RACE_NAME     = "Bahrain"
CACHE_DIR     = "./fastf1_cache"   # folder fastf1 stores cached data in

COMPOUNDS     = ["SOFT", "MEDIUM", "HARD"]
COMPOUND_BASE = {"SOFT": 88.5, "MEDIUM": 89.8, "HARD": 91.0}   # base lap times (s)
COMPOUND_DEG  = {"SOFT": 0.12, "MEDIUM": 0.07, "HARD": 0.04}   # seconds lost per lap on tire
TOTAL_LAPS    = 57
DRIVERS       = ["VER","HAM","LEC","NOR","SAI","RUS","ALO","PER","STR","GAS"]


# ─────────────────────────────────────────────────────────────
# MODE A: Real fastf1 data
# ─────────────────────────────────────────────────────────────
def load_real_data():
    import fastf1
    fastf1.Cache.enable_cache(CACHE_DIR)

    session = fastf1.get_session(RACE_YEAR, RACE_NAME, "R")
    session.load()
    laps = session.laps.copy()

    # Standardise columns to match our expected schema
    laps = laps[[
        "Driver", "LapNumber", "Compound", "TyreLife",
        "LapTime", "PitInTime", "PitOutTime", "Position", "Team"
    ]].copy()

    # Convert LapTime timedelta → seconds
    laps["LapTimeSec"] = laps["LapTime"].dt.total_seconds()

    # Binary pit flag: did the driver pit at END of this lap?
    laps["Pitted"] = laps["PitInTime"].notna().astype(int)

    laps = laps.dropna(subset=["LapTimeSec", "Compound", "TyreLife"])
    laps = laps[laps["Compound"].isin(COMPOUNDS)]
    laps = laps.reset_index(drop=True)

    print(f"✅ Loaded REAL data: {len(laps)} laps from {RACE_NAME} {RACE_YEAR}")
    return laps


# ─────────────────────────────────────────────────────────────
# MODE B: Synthetic data (same schema as fastf1 output)
# ─────────────────────────────────────────────────────────────
def generate_synthetic_data(seed=42):
    """
    Generates realistic lap-by-lap data matching the fastf1 schema.
    Tire degradation, pit stops, and randomness are all modelled.
    """
    np.random.seed(seed)
    rows = []

    # Each driver gets a random strategy: 1-stop or 2-stop
    strategies = {
        "VER": [("MEDIUM", 1, 22), ("HARD", 23, 57)],
        "HAM": [("SOFT",   1, 15), ("MEDIUM", 16, 40), ("HARD", 41, 57)],
        "LEC": [("SOFT",   1, 18), ("HARD", 19, 57)],
        "NOR": [("MEDIUM", 1, 25), ("HARD", 26, 57)],
        "SAI": [("MEDIUM", 1, 20), ("SOFT", 21, 38), ("HARD", 39, 57)],
        "RUS": [("HARD",   1, 30), ("MEDIUM", 31, 57)],
        "ALO": [("SOFT",   1, 12), ("MEDIUM", 13, 35), ("HARD", 36, 57)],
        "PER": [("MEDIUM", 1, 22), ("HARD", 23, 57)],
        "STR": [("HARD",   1, 28), ("SOFT", 29, 57)],
        "GAS": [("SOFT",   1, 16), ("MEDIUM", 17, 40), ("HARD", 41, 57)],
    }

    positions = list(range(1, 11))
    np.random.shuffle(positions)
    driver_pos = dict(zip(DRIVERS, positions))

    for driver, stints in strategies.items():
        tyre_life = 0
        for (compound, start_lap, end_lap) in stints:
            for lap in range(start_lap, end_lap + 1):
                tyre_life += 1

                # Lap time = base + degradation + some random noise
                base     = COMPOUND_BASE[compound]
                deg      = COMPOUND_DEG[compound] * tyre_life
                noise    = np.random.normal(0, 0.3)
                lap_time = base + deg + noise

                # Add traffic / safety car randomness on early laps
                if lap <= 3:
                    lap_time += np.random.uniform(0, 2.0)

                # Pitted = True on the LAST lap of a stint (except the final one)
                is_pit_lap = (lap == end_lap) and (stints.index((compound, start_lap, end_lap)) < len(stints) - 1)

                rows.append({
                    "Driver":      driver,
                    "LapNumber":   lap,
                    "Compound":    compound,
                    "TyreLife":    tyre_life,
                    "LapTimeSec":  round(lap_time, 3),
                    "Pitted":      int(is_pit_lap),
                    "Position":    driver_pos[driver],
                    "Team":        _team(driver),
                })

    df = pd.DataFrame(rows)
    print(f"✅ Generated SYNTHETIC data: {len(df)} laps across {len(DRIVERS)} drivers")
    return df


def _team(driver):
    mapping = {
        "VER": "Red Bull", "PER": "Red Bull",
        "HAM": "Mercedes", "RUS": "Mercedes",
        "LEC": "Ferrari",  "SAI": "Ferrari",
        "NOR": "McLaren",  "GAS": "Alpine",
        "ALO": "Aston Martin", "STR": "Aston Martin",
    }
    return mapping.get(driver, "Unknown")


# ─────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
def engineer_features(df):
    """Add features we'll use for ML later (Days 2-3)."""

    df = df.sort_values(["Driver", "LapNumber"]).reset_index(drop=True)

    # Rolling average lap time (last 3 laps) — smooths out outliers
    df["RollingAvgLapTime"] = (
        df.groupby("Driver")["LapTimeSec"]
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
    )

    # Lap time delta vs previous lap — captures sudden deg spikes
    df["LapTimeDelta"] = df.groupby("Driver")["LapTimeSec"].diff().fillna(0)

    # Degradation rate per compound (seconds lost per lap on that tire)
    df["DegRate"] = df.groupby(["Driver", "Compound"])["LapTimeSec"].transform(
        lambda x: x.diff().fillna(0)
    )

    # Laps remaining
    df["LapsRemaining"] = TOTAL_LAPS - df["LapNumber"]

    # Pit in NEXT 3 laps? (multi-class target we'll train on)
    def pit_within_n(group, n=3):
        pit_laps = set(group[group["Pitted"] == 1]["LapNumber"])
        results  = []
        for _, row in group.iterrows():
            future = range(row["LapNumber"] + 1, row["LapNumber"] + n + 1)
            if row["Pitted"] == 1:
                label = "PIT_NOW"
            elif any(l in pit_laps for l in future):
                label = "PIT_SOON"
            else:
                label = "STAY_OUT"
            results.append(label)
        return results

    labels = []
    for driver, grp in df.groupby("Driver"):
        labels.extend(pit_within_n(grp))
    df["PitLabel"] = labels

    return df


# ─────────────────────────────────────────────────────────────
# EDA PLOTS
# ─────────────────────────────────────────────────────────────
def plot_eda(df, save_path="day1_eda_plots.png"):
    """4-panel EDA figure."""

    # ── F1-inspired dark theme ──────────────────────────────
    plt.rcParams.update({
        "figure.facecolor":  "#0f0f0f",
        "axes.facecolor":    "#1a1a1a",
        "axes.edgecolor":    "#333333",
        "axes.labelcolor":   "#cccccc",
        "xtick.color":       "#888888",
        "ytick.color":       "#888888",
        "text.color":        "#eeeeee",
        "grid.color":        "#2a2a2a",
        "grid.linestyle":    "--",
        "font.family":       "monospace",
        "axes.titleweight":  "bold",
        "axes.titlesize":    11,
    })

    COMPOUND_COLORS = {"SOFT": "#e8002d", "MEDIUM": "#ffd700", "HARD": "#ebebeb"}

    fig = plt.figure(figsize=(16, 10), facecolor="#0f0f0f")
    fig.suptitle(
        f"F1 PIT STOP STRATEGY ANALYZER  ·  {RACE_NAME.upper()} {RACE_YEAR}  ·  EDA",
        fontsize=14, fontweight="bold", color="#e8002d",
        y=0.98, fontfamily="monospace"
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── Plot 1: Tire degradation curves per compound ───────
    ax1 = fig.add_subplot(gs[0, 0])
    for compound in COMPOUNDS:
        cdf = df[df["Compound"] == compound].copy()
        avg = cdf.groupby("TyreLife")["LapTimeSec"].mean()
        ax1.plot(avg.index, avg.values,
                 color=COMPOUND_COLORS[compound],
                 linewidth=2.5, label=compound, marker="o", markersize=3)
    ax1.set_title("TIRE DEGRADATION CURVES")
    ax1.set_xlabel("Tire Age (laps)")
    ax1.set_ylabel("Avg Lap Time (s)")
    ax1.legend(facecolor="#1a1a1a", edgecolor="#333", labelcolor="#eee")
    ax1.grid(True)

    # ── Plot 2: Lap time distribution per compound ─────────
    ax2 = fig.add_subplot(gs[0, 1])
    for compound in COMPOUNDS:
        vals = df[df["Compound"] == compound]["LapTimeSec"].dropna()
        ax2.hist(vals, bins=30, alpha=0.7,
                 color=COMPOUND_COLORS[compound], label=compound,
                 edgecolor="none")
    ax2.set_title("LAP TIME DISTRIBUTION BY COMPOUND")
    ax2.set_xlabel("Lap Time (s)")
    ax2.set_ylabel("Frequency")
    ax2.legend(facecolor="#1a1a1a", edgecolor="#333", labelcolor="#eee")
    ax2.grid(True)

    # ── Plot 3: Lap time trace for top 3 drivers ──────────
    ax3 = fig.add_subplot(gs[1, 0])
    top3 = df["Driver"].value_counts().head(3).index
    driver_colors = ["#e8002d", "#00d2be", "#fff200"]
    for drv, col in zip(top3, driver_colors):
        d = df[df["Driver"] == drv].sort_values("LapNumber")
        ax3.plot(d["LapNumber"], d["LapTimeSec"],
                 color=col, linewidth=1.5, label=drv, alpha=0.85)
        # Mark pit laps
        pits = d[d["Pitted"] == 1]
        ax3.scatter(pits["LapNumber"], pits["LapTimeSec"],
                    color=col, s=80, zorder=5, marker="v")
    ax3.set_title("LAP TIME TRACE  (▼ = pit stop)")
    ax3.set_xlabel("Lap Number")
    ax3.set_ylabel("Lap Time (s)")
    ax3.legend(facecolor="#1a1a1a", edgecolor="#333", labelcolor="#eee")
    ax3.grid(True)

    # ── Plot 4: Label class distribution ──────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    label_counts = df["PitLabel"].value_counts()
    bar_colors   = ["#e8002d", "#ffd700", "#00d2be"]
    bars = ax4.bar(label_counts.index, label_counts.values,
                   color=bar_colors[:len(label_counts)],
                   edgecolor="none", width=0.5)
    for bar, val in zip(bars, label_counts.values):
        ax4.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 20,
                 f"{val:,}", ha="center", va="bottom",
                 color="#eeeeee", fontsize=10)
    ax4.set_title("TARGET LABEL DISTRIBUTION")
    ax4.set_ylabel("Count")
    ax4.grid(True, axis="y")
    ax4.set_axisbelow(True)

    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor="#0f0f0f")
    print(f"📊 EDA plots saved → {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*55)
    print("  F1 PIT STOP STRATEGY ANALYZER — DAY 1 EDA")
    print("="*55 + "\n")

    # Step 1: Load data
    if USE_REAL_DATA:
        df = load_real_data()
        # Real fastf1 data already has LapTimeSec; add Pitted flag
        df["Pitted"] = df.get("Pitted", df.get("PitInTime", pd.Series()).notna().astype(int))
    else:
        df = generate_synthetic_data()

    print("\n── Raw Data Sample ──────────────────────────────────")
    print(df[["Driver","LapNumber","Compound","TyreLife","LapTimeSec","Pitted"]].head(15).to_string(index=False))

    # Step 2: Engineer features
    df = engineer_features(df)

    print("\n── Feature-Engineered Columns ───────────────────────")
    print(df.columns.tolist())

    print("\n── Label Distribution ───────────────────────────────")
    print(df["PitLabel"].value_counts().to_string())

    print("\n── Avg Lap Time by Compound ─────────────────────────")
    print(df.groupby("Compound")["LapTimeSec"].agg(["mean","std","min","max"]).round(3).to_string())

    print("\n── Degradation Rate by Compound ─────────────────────")
    print(df.groupby("Compound")["DegRate"].mean().round(4).to_string())

    # Step 3: EDA plots
    plot_eda(df, "day1_eda_plots.png")

    # Step 4: Save processed dataset for Day 2
    df.to_csv("processed_laps.csv", index=False)
    print("\n✅ Processed dataset saved → processed_laps.csv")
    print("   (Load this on Day 2 for feature engineering & model training)\n")

    return df


if __name__ == "__main__":
    df = main()
