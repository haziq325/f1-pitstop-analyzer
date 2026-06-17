"""
F1 Pit Stop Strategy Analyzer
Day 4 — Strategy Simulator + Real Race Validation

Two parts:
  Part A: Strategy Simulator
          Given current race state, simulates ALL valid strategies forward
          and ranks them by estimated total race time.

  Part B: Validation
          Replays a historical race (from processed_laps.csv) lap by lap,
          runs our model's recommendation each lap, and compares vs what
          the team actually did.

Run:
    python day4_simulator.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
import itertools
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# CONSTANTS  (keep in sync with day3_app.py)
# ─────────────────────────────────────────────────────────────
TOTAL_LAPS      = 57
PIT_LOSS_SECS   = 23.0        # time lost in pitlane (s)
SAFETY_CAR_PROB = 0.10        # 10% chance per lap of SC → free pit

COMPOUND_BASE = {"SOFT": 88.5,  "MEDIUM": 89.8, "HARD": 91.0}
COMPOUND_DEG  = {"SOFT": 0.12,  "MEDIUM": 0.07, "HARD": 0.04}
COMPOUND_MAX  = {"SOFT": 20,    "MEDIUM": 35,   "HARD": 50}
COMPOUND_CODE = {"SOFT": 0,     "MEDIUM": 1,    "HARD": 2}
COMPOUND_MIN_LAPS = {"SOFT": 10, "MEDIUM": 15,  "HARD": 18}  # min stint length

# Compounds that must be used (F1 rule: at least 2 different compounds)
VALID_COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]


# ─────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────
def load_model(path="model.pkl"):
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    print(f"✅ Model loaded: {bundle['model_name']}  "
          f"(XGB F1={bundle['xgb_f1']}  RF F1={bundle['rf_f1']})")
    return bundle


# ─────────────────────────────────────────────────────────────
# PART A — STRATEGY SIMULATOR
# ─────────────────────────────────────────────────────────────

def lap_time(compound, tyre_life, add_noise=False):
    """Predicted lap time for a given compound and tire age."""
    t = COMPOUND_BASE[compound] + COMPOUND_DEG[compound] * tyre_life
    if add_noise:
        t += np.random.normal(0, 0.25)
    return t


def simulate_stint(compound, start_tyre_life, laps, add_noise=False):
    """Return list of lap times for a stint."""
    times = []
    for i in range(laps):
        tl = start_tyre_life + i + 1
        times.append(lap_time(compound, tl, add_noise))
    return times


def total_strategy_time(stints, current_lap, add_noise=False):
    """
    Given a list of stints [(compound, num_laps), ...] starting from
    current_lap, compute total estimated time to finish.

    stints: e.g. [("MEDIUM", 15), ("HARD", 22)]
    current_lap: lap we're on right now
    """
    total_time  = 0.0
    lap_counter = current_lap
    tyre_life   = 0

    for i, (compound, num_laps) in enumerate(stints):
        # Pit stop cost (except for the very first stint — already racing)
        if i > 0:
            total_time += PIT_LOSS_SECS

        tyre_life = 0
        for _ in range(num_laps):
            if lap_counter > TOTAL_LAPS:
                break
            tyre_life  += 1
            total_time += lap_time(compound, tyre_life, add_noise)
            lap_counter += 1

    return round(total_time, 3)


def generate_all_strategies(current_lap, current_compound, current_tyre_life):
    """
    Enumerate all valid 1-stop and 2-stop strategies from current race state.

    Rules:
    - Must use at least 2 different compounds total (including what's already been used)
    - Each stint must meet minimum lap requirement
    - Total laps must sum to exactly remaining laps
    """
    remaining_laps = TOTAL_LAPS - current_lap
    strategies     = []

    # ── 0-stop: finish on current tires ──────────────────────
    # Only valid if we haven't exceeded compound life by too much
    life_after = current_tyre_life + remaining_laps
    if life_after <= COMPOUND_MAX[current_compound] * 1.3:  # allow 30% overuse
        strategies.append({
            "label":    "0-stop (finish current)",
            "stints":   [(current_compound, remaining_laps)],
            "num_stops": 0,
        })

    # ── 1-stop strategies ─────────────────────────────────────
    for next_compound in VALID_COMPOUNDS:
        if next_compound == current_compound:
            continue   # must use a different compound
        for pit_in in range(current_lap + COMPOUND_MIN_LAPS[current_compound],
                            TOTAL_LAPS - COMPOUND_MIN_LAPS[next_compound] + 1):
            laps_stint1 = pit_in - current_lap
            laps_stint2 = TOTAL_LAPS - pit_in
            if laps_stint1 < 1 or laps_stint2 < 1:
                continue
            strategies.append({
                "label":    f"1-stop · {current_compound}({laps_stint1}L) → {next_compound}({laps_stint2}L)",
                "stints":   [(current_compound, laps_stint1),
                             (next_compound,    laps_stint2)],
                "pit_laps": [pit_in],
                "num_stops": 1,
            })

    # ── 2-stop strategies ─────────────────────────────────────
    for c2, c3 in itertools.product(VALID_COMPOUNDS, repeat=2):
        # At least one of c2/c3 must differ from current_compound
        if c2 == current_compound and c3 == current_compound:
            continue
        min1 = COMPOUND_MIN_LAPS[current_compound]
        min2 = COMPOUND_MIN_LAPS[c2]
        min3 = COMPOUND_MIN_LAPS[c3]

        for pit1 in range(current_lap + min1,
                          TOTAL_LAPS - min2 - min3 + 1, 3):   # step 3 to reduce combos
            for pit2 in range(pit1 + min2,
                              TOTAL_LAPS - min3 + 1, 3):
                l1 = pit1 - current_lap
                l2 = pit2 - pit1
                l3 = TOTAL_LAPS - pit2
                if l1 < 1 or l2 < 1 or l3 < 1:
                    continue
                strategies.append({
                    "label":    f"2-stop · {current_compound}({l1}L)→{c2}({l2}L)→{c3}({l3}L)",
                    "stints":   [(current_compound, l1), (c2, l2), (c3, l3)],
                    "pit_laps": [pit1, pit2],
                    "num_stops": 2,
                })

    return strategies


def rank_strategies(strategies, current_lap, current_tyre_life, top_n=10):
    """Score every strategy and return top N sorted by total time."""
    results = []

    for s in strategies:
        # Skip first stint (already running it), start from lap 2 of current stint
        stints_from_now = list(s["stints"])
        # Adjust first stint: reduce by laps already done on current tires
        first_compound, first_total_laps = stints_from_now[0]
        remaining_on_current = first_total_laps  # laps left on this set

        # Monte Carlo: run 20 simulations with noise to get variance
        times = [
            total_strategy_time(stints_from_now, current_lap, add_noise=True)
            for _ in range(20)
        ]
        mean_time = np.mean(times)
        std_time  = np.std(times)

        results.append({
            "Strategy":     s["label"],
            "Stops":        s["num_stops"],
            "Total Time (s)": round(mean_time, 2),
            "Uncertainty (±s)": round(std_time, 2),
            "Pit Laps":     s.get("pit_laps", []),
        })

    df = pd.DataFrame(results)
    df = df.sort_values("Total Time (s)").reset_index(drop=True)
    df.index += 1   # rank from 1
    return df.head(top_n)


# ─────────────────────────────────────────────────────────────
# PART B — RACE VALIDATION
# ─────────────────────────────────────────────────────────────

def build_feature_vector(row, feats):
    """Build model input from a CSV row."""
    compound     = row["Compound"]
    tyre_life    = row["TyreLife"] if "TyreLife" in row else row.get("StintLength", 10)
    lap_number   = row["LapNumber"]
    position     = row.get("Position", 5)
    stint_length = row.get("StintLength", tyre_life)
    lap_time_val = row.get("LapTimeSec", COMPOUND_BASE[compound])

    rolling  = row.get("RollingAvgLapTime", lap_time_val)
    delta    = row.get("LapTimeDelta", 0.0)
    deg_rate = row.get("DegRate", COMPOUND_DEG[compound])
    pace_loss = COMPOUND_DEG[compound] * tyre_life
    health   = max(0, 100 - (tyre_life / COMPOUND_MAX[compound]) * 100)
    laps_rem = TOTAL_LAPS - lap_number
    progress = lap_number / TOTAL_LAPS
    in_window = 1 if (15 <= lap_number <= 25) or (35 <= lap_number <= 45) else 0

    record = {
        "TyreLife":           tyre_life,
        "TyreHealthPct":      health,
        "LapTimeSec":         lap_time_val,
        "RollingAvgLapTime":  rolling,
        "LapTimeDelta":       delta,
        "DegRate":            deg_rate,
        "DegAccel":           0.0,
        "PaceLoss":           pace_loss,
        "StintLength":        stint_length,
        "LapsRemaining":      laps_rem,
        "RaceProgress":       progress,
        "CompoundCode":       COMPOUND_CODE[compound],
        "Position":           position,
        "InPitWindow":        in_window,
    }
    return pd.DataFrame([record])[feats]


def validate_on_driver(df, driver, bundle):
    """
    Replay the race for one driver lap by lap.
    For each lap, run the model and compare vs actual decision.
    """
    model  = bundle["model"]
    le     = bundle["label_encoder"]
    feats  = bundle["feature_cols"]

    driver_laps = df[df["Driver"] == driver].copy()

    # Add StintLength column (resets at pit stops)
    stint_len, count = [], 0
    for _, row in driver_laps.iterrows():
        count += 1
        stint_len.append(count)
        if row.get("Pitted", 0) == 1:
            count = 0
    driver_laps["StintLength"] = stint_len

    records = []
    for _, row in driver_laps.iterrows():
        X       = build_feature_vector(row, feats)
        proba   = model.predict_proba(X)[0]
        pred    = le.inverse_transform([int(np.argmax(proba))])[0]
        actual  = row.get("PitLabel", "STAY_OUT")
        correct = (pred == actual)

        records.append({
            "Lap":        row["LapNumber"],
            "Compound":   row["Compound"],
            "TyreLife":   row.get("TyreLife", row.get("StintLength", "?")),
            "Predicted":  pred,
            "Actual":     actual,
            "Correct":    correct,
            "Confidence": round(float(max(proba)) * 100, 1),
        })

    result_df = pd.DataFrame(records)
    accuracy  = result_df["Correct"].mean()
    pit_recall = (
        result_df[result_df["Actual"] != "STAY_OUT"]["Correct"].mean()
        if len(result_df[result_df["Actual"] != "STAY_OUT"]) > 0 else 0.0
    )

    print(f"\n── {driver} Validation ────────────────────────────────")
    print(f"   Overall accuracy : {accuracy*100:.1f}%")
    print(f"   Pit recall       : {pit_recall*100:.1f}%  "
          f"(correctly flagged pit laps)")
    print(f"   Total laps       : {len(result_df)}")

    # Show pit laps specifically
    pit_laps = result_df[result_df["Actual"] != "STAY_OUT"]
    if len(pit_laps):
        print(f"\n   Pit-related laps:")
        print(pit_laps[["Lap","Compound","TyreLife","Predicted","Actual","Confidence"]].to_string(index=False))

    return result_df, accuracy


# ─────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────
COMPOUND_COLORS = {"SOFT": "#e8002d", "MEDIUM": "#ffd700", "HARD": "#ebebeb"}

def plot_day4(strategies_df, val_df, driver, save_path="day4_simulator_plots.png"):

    plt.rcParams.update({
        "figure.facecolor": "#0f0f0f",
        "axes.facecolor":   "#1a1a1a",
        "axes.edgecolor":   "#333333",
        "axes.labelcolor":  "#cccccc",
        "xtick.color":      "#888888",
        "ytick.color":      "#888888",
        "text.color":       "#eeeeee",
        "grid.color":       "#2a2a2a",
        "grid.linestyle":   "--",
        "font.family":      "monospace",
    })

    fig = plt.figure(figsize=(18, 10), facecolor="#0f0f0f")
    fig.suptitle(
        "F1 PIT STOP STRATEGY ANALYZER  ·  DAY 4  ·  SIMULATOR + VALIDATION",
        fontsize=13, fontweight="bold", color="#e8002d", y=0.98
    )

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.38)

    # ── Plot 1: Top 10 strategies by time ────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    top10  = strategies_df.head(10).copy()
    labels = [f"#{i} {r['Strategy'][:45]}" for i, r in top10.iterrows()]
    times  = top10["Total Time (s)"].values
    errs   = top10["Uncertainty (±s)"].values
    stops  = top10["Stops"].values
    colors = ["#e8002d" if s == 0 else "#ffd700" if s == 1 else "#00d2be"
              for s in stops]

    bars = ax1.barh(labels[::-1], times[::-1] - times.min(),
                    xerr=errs[::-1], color=colors[::-1],
                    edgecolor="none", alpha=0.85,
                    error_kw=dict(ecolor="#555", capsize=3))
    ax1.set_title("TOP 10 STRATEGIES BY ESTIMATED RACE TIME  (relative to fastest)",
                  color="#eeeeee")
    ax1.set_xlabel("Time Delta vs Fastest Strategy (s)")
    ax1.grid(True, axis="x")
    ax1.set_axisbelow(True)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#e8002d", label="0-stop"),
        Patch(facecolor="#ffd700", label="1-stop"),
        Patch(facecolor="#00d2be", label="2-stop"),
    ]
    ax1.legend(handles=legend_elements, facecolor="#1a1a1a",
               edgecolor="#333", labelcolor="#eee", loc="lower right")

    # ── Plot 2: Stops distribution ────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    stop_counts = strategies_df["Stops"].value_counts().sort_index()
    ax2.bar(stop_counts.index.astype(str),
            stop_counts.values,
            color=["#e8002d", "#ffd700", "#00d2be"][:len(stop_counts)],
            edgecolor="none", width=0.5, alpha=0.85)
    ax2.set_title("STRATEGY DISTRIBUTION\n(by number of stops)", color="#eeeeee")
    ax2.set_xlabel("Pit Stops")
    ax2.set_ylabel("Count")
    ax2.grid(True, axis="y")
    ax2.set_axisbelow(True)

    # ── Plot 3: Validation — prediction vs actual per lap ─────
    ax3 = fig.add_subplot(gs[1, :2])
    label_map   = {"STAY_OUT": 0, "PIT_SOON": 1, "PIT_NOW": 2}
    label_color = {"STAY_OUT": "#444444", "PIT_SOON": "#ffd700", "PIT_NOW": "#e8002d"}

    laps       = val_df["Lap"].values
    pred_vals  = val_df["Predicted"].map(label_map).values
    actual_vals= val_df["Actual"].map(label_map).values
    correct    = val_df["Correct"].values

    ax3.step(laps, actual_vals, where="mid", color="#888888",
             linewidth=1.2, label="Actual", alpha=0.7)
    ax3.scatter(laps[correct],  pred_vals[correct],
                color="#00c853", s=20, zorder=5, label="Correct prediction")
    ax3.scatter(laps[~correct], pred_vals[~correct],
                color="#e8002d", s=30, marker="x", zorder=5, label="Wrong prediction")

    ax3.set_yticks([0, 1, 2])
    ax3.set_yticklabels(["STAY OUT", "PIT SOON", "PIT NOW"], fontsize=9)
    ax3.set_xlabel("Lap Number")
    ax3.set_title(f"MODEL VALIDATION — {driver}  ·  Predicted vs Actual Decision",
                  color="#eeeeee")
    ax3.legend(facecolor="#1a1a1a", edgecolor="#333", labelcolor="#eee")
    ax3.grid(True)

    # ── Plot 4: Confidence per lap ────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    conf = val_df["Confidence"].values
    ax4.fill_between(laps, conf, alpha=0.3, color="#e8002d")
    ax4.plot(laps, conf, color="#e8002d", linewidth=1.5)
    ax4.axhline(y=80, color="#ffd700", linewidth=1, linestyle="--",
                label="80% confidence")
    ax4.set_title(f"PREDICTION CONFIDENCE\n{driver} — per lap", color="#eeeeee")
    ax4.set_xlabel("Lap Number")
    ax4.set_ylabel("Confidence (%)")
    ax4.set_ylim(0, 105)
    ax4.legend(facecolor="#1a1a1a", edgecolor="#333", labelcolor="#eee")
    ax4.grid(True)

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
    print(f"\n📊 Plots saved → {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────
# STRATEGY REPORT — printable summary
# ─────────────────────────────────────────────────────────────
def print_strategy_report(strategies_df, current_lap, current_compound, current_tyre_life):
    fastest = strategies_df.iloc[0]

    print("\n" + "="*62)
    print("  STRATEGY RECOMMENDATION REPORT")
    print("="*62)
    print(f"  Current Lap     : {current_lap} / {TOTAL_LAPS}")
    print(f"  Current Compound: {current_compound}")
    print(f"  Tyre Age        : {current_tyre_life} laps")
    print(f"  Laps Remaining  : {TOTAL_LAPS - current_lap}")
    print("="*62)

    print(f"\n  🏆 FASTEST STRATEGY: {fastest['Strategy']}")
    print(f"     Est. Time     : {fastest['Total Time (s)']}s")
    print(f"     Uncertainty   : ±{fastest['Uncertainty (±s)']}s")
    if fastest["Pit Laps"]:
        print(f"     Pit on lap(s) : {fastest['Pit Laps']}")

    print(f"\n  TOP 5 STRATEGIES:")
    print(f"  {'Rank':<5} {'Stops':<7} {'Est Time':<12} {'±':<8} {'Strategy'}")
    print(f"  {'-'*62}")
    for rank, row in strategies_df.head(5).iterrows():
        delta = row['Total Time (s)'] - fastest['Total Time (s)']
        delta_str = f"+{delta:.2f}s" if delta > 0 else "optimal"
        print(f"  {rank:<5} {row['Stops']:<7} {row['Total Time (s)']:<12} "
              f"±{row['Uncertainty (±s)']:<6} {row['Strategy'][:38]}")

    print(f"\n  Total strategies evaluated: {len(strategies_df)}")
    print("="*62)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*62)
    print("  F1 PIT STOP STRATEGY ANALYZER — DAY 4")
    print("  Strategy Simulator + Race Validation")
    print("="*62 + "\n")

    bundle = load_model("model.pkl")
    df     = pd.read_csv("processed_laps.csv")

    # ── PART A: Strategy Simulation ──────────────────────────
    print("\n── PART A: STRATEGY SIMULATION ──────────────────────────")

    # Simulate from lap 20, currently on MEDIUM, tyre age 18
    CURRENT_LAP      = 20
    CURRENT_COMPOUND = "MEDIUM"
    CURRENT_TYRE_AGE = 18

    print(f"\nScenario: Lap {CURRENT_LAP}, on {CURRENT_COMPOUND} "
          f"(age: {CURRENT_TYRE_AGE} laps)")
    print("Generating all valid strategies...", end=" ")

    strategies = generate_all_strategies(CURRENT_LAP, CURRENT_COMPOUND, CURRENT_TYRE_AGE)
    print(f"{len(strategies)} strategies found.")

    strategies_df = rank_strategies(strategies, CURRENT_LAP, CURRENT_TYRE_AGE, top_n=10)
    print_strategy_report(strategies_df, CURRENT_LAP, CURRENT_COMPOUND, CURRENT_TYRE_AGE)

    # ── PART B: Validation ────────────────────────────────────
    print("\n── PART B: RACE VALIDATION ──────────────────────────────")

    # Add TyreLife column if missing (use StintLength proxy)
    if "TyreLife" not in df.columns:
        stint_lens = []
        for _, grp in df.groupby("Driver"):
            count = 0
            for _, row in grp.iterrows():
                count += 1
                stint_lens.append(count)
                if row.get("Pitted", 0) == 1:
                    count = 0
        df["TyreLife"] = stint_lens

    # Validate on 3 drivers
    val_results = {}
    drivers_to_validate = ["VER", "HAM", "LEC"]
    for driver in drivers_to_validate:
        val_df, acc = validate_on_driver(df, driver, bundle)
        val_results[driver] = {"df": val_df, "accuracy": acc}

    # Overall summary
    print("\n── VALIDATION SUMMARY ───────────────────────────────────")
    print(f"  {'Driver':<10} {'Accuracy':>10} {'Pit Recall':>12}")
    print(f"  {'-'*35}")
    for driver, res in val_results.items():
        vdf      = res["df"]
        acc      = res["accuracy"]
        pit_laps = vdf[vdf["Actual"] != "STAY_OUT"]
        recall   = pit_laps["Correct"].mean() if len(pit_laps) > 0 else 0.0
        print(f"  {driver:<10} {acc*100:>9.1f}% {recall*100:>11.1f}%")

    # ── PLOTS ──────────────────────────────────────────────────
    # Use VER for validation plots
    plot_day4(strategies_df, val_results["VER"]["df"], "VER")

    # ── SAVE enriched data ─────────────────────────────────────
    strategies_df.to_csv("strategies_ranked.csv", index_label="Rank")
    print("\n✅ Ranked strategies saved → strategies_ranked.csv")
    print("🏁 Day 4 complete. Day 5 = UI polish + degradation visualizer.\n")


if __name__ == "__main__":
    main()
