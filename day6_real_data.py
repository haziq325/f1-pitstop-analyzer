"""
F1 Pit Stop Strategy Analyzer
Day 6 — Real FastF1 Data Pipeline + Model Retraining

Run this LOCALLY (requires internet + fastf1):
    python day6_real_data.py

What it does:
  1. Pulls real lap data from multiple 2023 + 2024 races via fastf1
  2. Cleans and engineers features (same schema as synthetic data)
  3. Retrains both models on thousands of real laps
  4. Saves new model.pkl (replaces the synthetic-trained one)
  5. Runs validation on a held-out race
  6. Prints a before/after comparison

Once this runs:
  - Replace your model.pkl with the new one
  - Re-launch day5_app.py — it auto-loads the new model
"""

import os
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

import fastf1
from sklearn.model_selection  import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble         import RandomForestClassifier
from sklearn.metrics          import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────────────────────────
# CONFIG — edit these to pull more/fewer races
# ─────────────────────────────────────────────────────────────
CACHE_DIR   = "./fastf1_cache"
TOTAL_LAPS_DEFAULT = 57

# Races to pull for TRAINING (varied circuits = better generalisation)
TRAINING_RACES = [
    (2024, "Bahrain"),       # high deg, 2-stop common
    (2024, "Saudi Arabia"),  # low deg, 1-stop
    (2024, "Australia"),     # safety car likely
    (2024, "Japan"),         # medium deg
    (2024, "China"),         # high deg sprint weekend
    (2024, "Miami"),         # mixed strategy
    (2024, "Monaco"),        # almost no pit stops
    (2024, "Canada"),        # safety car, varied
    (2024, "Spain"),         # classic 2-stop
    (2024, "Austria"),       # sprint format
    (2023, "Bahrain"),
    (2023, "Monaco"),
    (2023, "Silverstone"),
    (2023, "Hungary"),
    (2023, "Singapore"),
]

# Race to hold out for VALIDATION (not used in training)
VALIDATION_RACE = (2024, "British")

COMPOUNDS     = ["SOFT", "MEDIUM", "HARD"]
COMPOUND_BASE = {"SOFT": 88.5, "MEDIUM": 89.8, "HARD": 91.0}
COMPOUND_DEG  = {"SOFT": 0.12, "MEDIUM": 0.07, "HARD": 0.04}
COMPOUND_MAX  = {"SOFT": 20,   "MEDIUM": 35,   "HARD": 50}
COMPOUND_CODE = {"SOFT": 0, "MEDIUM": 1, "HARD": 2}

FEATURE_COLS = [
    "TyreLife", "TyreHealthPct", "LapTimeSec", "RollingAvgLapTime",
    "LapTimeDelta", "DegRate", "DegAccel", "PaceLoss", "StintLength",
    "LapsRemaining", "RaceProgress", "CompoundCode", "Position", "InPitWindow",
]
LABEL_ORDER = ["STAY_OUT", "PIT_SOON", "PIT_NOW"]


# ─────────────────────────────────────────────────────────────
# STEP 1 — LOAD ONE RACE FROM FASTF1
# ─────────────────────────────────────────────────────────────
def load_race(year, race_name):
    """
    Pull lap data for one race from fastf1.
    Returns a cleaned DataFrame or None if loading fails.
    """
    try:
        print(f"   Loading {race_name} {year}...", end=" ", flush=True)
        session = fastf1.get_session(year, race_name, "R")
        session.load(telemetry=False, weather=False, messages=False)
        laps = session.laps.copy()

        # Keep only needed columns
        needed = ["Driver", "LapNumber", "Compound", "TyreLife",
                  "LapTime", "PitInTime", "PitOutTime", "Position", "Team"]
        laps = laps[[c for c in needed if c in laps.columns]].copy()

        # Convert LapTime timedelta → seconds
        laps["LapTimeSec"] = laps["LapTime"].dt.total_seconds()

        # Pit flag
        laps["Pitted"] = laps["PitInTime"].notna().astype(int)

        # Drop nulls and invalid compounds
        laps = laps.dropna(subset=["LapTimeSec", "Compound", "TyreLife"])
        laps = laps[laps["Compound"].isin(COMPOUNDS)]
        laps = laps[laps["LapTimeSec"] > 60]   # filter out outlaps/SC laps below 60s
        laps = laps[laps["LapTimeSec"] < 200]  # filter out red flag laps

        # Race metadata
        laps["Year"]      = year
        laps["RaceName"]  = race_name
        laps["TotalLaps"] = laps["LapNumber"].max()

        laps = laps.reset_index(drop=True)
        print(f"✅ {len(laps)} laps")
        return laps

    except Exception as e:
        print(f"❌ Failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────
# STEP 2 — FEATURE ENGINEERING (same as Day 1 + Day 2)
# ─────────────────────────────────────────────────────────────
def engineer_features(df):
    """Apply all feature engineering to a cleaned laps DataFrame."""
    df = df.copy()
    df["LapNumber"] = df["LapNumber"].astype(int)   # fastf1 returns float
    df["TyreLife"]  = df["TyreLife"].astype(int)
    df = df.sort_values(["Year", "RaceName", "Driver", "LapNumber"]).reset_index(drop=True)

    grp_key = ["Year", "RaceName", "Driver"]

    # Rolling avg lap time (last 3 laps)
    df["RollingAvgLapTime"] = (
        df.groupby(grp_key)["LapTimeSec"]
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
    )

    # Lap time delta
    df["LapTimeDelta"] = df.groupby(grp_key)["LapTimeSec"].diff().fillna(0)

    # Degradation rate per compound
    df["DegRate"] = df.groupby(grp_key + ["Compound"])["LapTimeSec"].transform(
        lambda x: x.diff().fillna(0)
    )

    # Deg acceleration
    df["DegAccel"] = df.groupby(grp_key)["DegRate"].diff().fillna(0)

    # Tire health
    df["TyreHealthPct"] = df.apply(
        lambda r: max(0, 100 - (r["TyreLife"] / COMPOUND_MAX.get(r["Compound"], 35)) * 100),
        axis=1
    )

    # Pace loss vs best lap on stint
    df["PaceLoss"] = df.groupby(grp_key + ["Compound"])["LapTimeSec"].transform(
        lambda x: x - x.min()
    )

    # Stint length (resets at pit)
    def stint_counter(group):
        counts, count = [], 0
        for pitted in group["Pitted"]:
            count += 1
            counts.append(count)
            if pitted:
                count = 0
        return counts

    stint_lens = []
    for _, grp in df.groupby(grp_key):
        stint_lens.extend(stint_counter(grp))
    df["StintLength"] = stint_lens

    df["LapsRemaining"] = df["TotalLaps"] - df["LapNumber"].astype(int)
    df["RaceProgress"]  = df["LapNumber"].astype(int) / df["TotalLaps"]
    df["CompoundCode"]  = df["Compound"].map(COMPOUND_CODE)

    # Pit window flag
    df["InPitWindow"] = df.apply(
        lambda r: 1 if (r["TotalLaps"] * 0.25 <= int(r["LapNumber"]) <= r["TotalLaps"] * 0.45)
                    or (r["TotalLaps"] * 0.55 <= int(r["LapNumber"]) <= r["TotalLaps"] * 0.75)
                  else 0,
        axis=1
    )

    # Target label
    def pit_labels(group):
        pit_laps = set(group[group["Pitted"] == 1]["LapNumber"])
        labels   = []
        for _, row in group.iterrows():
            future = range(int(row["LapNumber"]) + 1, int(row["LapNumber"]) + 4)
            if row["Pitted"] == 1:
                labels.append("PIT_NOW")
            elif any(l in pit_laps for l in future):
                labels.append("PIT_SOON")
            else:
                labels.append("STAY_OUT")
        return labels

    all_labels = []
    for _, grp in df.groupby(grp_key):
        all_labels.extend(pit_labels(grp))
    df["PitLabel"] = all_labels

    return df


# ─────────────────────────────────────────────────────────────
# STEP 3 — PULL ALL TRAINING RACES
# ─────────────────────────────────────────────────────────────
def build_training_dataset():
    fastf1.Cache.enable_cache(CACHE_DIR)
    os.makedirs(CACHE_DIR, exist_ok=True)

    print("\n── Pulling Training Races ───────────────────────────────")
    all_laps = []
    failed   = []

    for year, race in TRAINING_RACES:
        laps = load_race(year, race)
        if laps is not None:
            all_laps.append(laps)
        else:
            failed.append(f"{race} {year}")

    if not all_laps:
        raise RuntimeError("No races loaded. Check your internet connection and fastf1 cache.")

    df = pd.concat(all_laps, ignore_index=True)
    print(f"\n✅ Training data: {len(df):,} laps from {len(all_laps)} races")
    if failed:
        print(f"⚠️  Failed to load: {failed}")

    print("   Engineering features...", end=" ", flush=True)
    df = engineer_features(df)
    print("done")

    print(f"   Label distribution:")
    print(f"   {df['PitLabel'].value_counts().to_dict()}")

    df.to_csv("real_processed_laps.csv", index=False)
    print("   Saved → real_processed_laps.csv")
    return df


# ─────────────────────────────────────────────────────────────
# STEP 4 — RETRAIN
# ─────────────────────────────────────────────────────────────
def retrain(df):
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    le.fit(LABEL_ORDER)

    X = df[FEATURE_COLS].dropna()
    y = le.transform(df.loc[X.index, "PitLabel"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    cw_dict = dict(zip(classes, weights))

    weight_map     = {c: len(y_train)/(len(classes)*cnt)
                      for c, cnt in zip(*np.unique(y_train, return_counts=True))}
    sample_weights = np.array([weight_map[c] for c in y_train])

    print("\n── Training Random Forest on real data ──────────────────")
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=5,
        class_weight=cw_dict, random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_f1 = f1_score(y_test, rf.predict(X_test), average="macro", zero_division=0)
    print(f"   RF macro-F1: {rf_f1:.4f}")
    print(classification_report(y_test, rf.predict(X_test),
                                target_names=le.classes_, zero_division=0))

    print("── Training XGBoost on real data ────────────────────────")
    xgb_model = xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        eval_metric="mlogloss", objective="multi:softprob",
        num_class=3, random_state=42, verbosity=0,
    )
    xgb_model.fit(X_train, y_train, sample_weight=sample_weights)
    xgb_f1 = f1_score(y_test, xgb_model.predict(X_test), average="macro", zero_division=0)
    print(f"   XGB macro-F1: {xgb_f1:.4f}")
    print(classification_report(y_test, xgb_model.predict(X_test),
                                target_names=le.classes_, zero_division=0))

    # Pick winner
    best_model = xgb_model if xgb_f1 >= rf_f1 else rf
    best_name  = "XGBoost"  if xgb_f1 >= rf_f1 else "RandomForest"

    bundle = {
        "model":         best_model,
        "model_name":    best_name,
        "label_encoder": le,
        "feature_cols":  FEATURE_COLS,
        "label_order":   LABEL_ORDER,
        "rf_f1":         round(rf_f1, 4),
        "xgb_f1":        round(xgb_f1, 4),
        "trained_on":    f"{len(df):,} real laps",
        "races":         len(TRAINING_RACES),
    }

    with open("model.pkl", "wb") as f:
        pickle.dump(bundle, f)

    print(f"\n✅ New model.pkl saved ({best_name}, F1={max(rf_f1,xgb_f1):.4f})")
    print(f"   Trained on {len(df):,} real laps across {len(TRAINING_RACES)} races")
    return bundle, X_test, y_test, le


# ─────────────────────────────────────────────────────────────
# STEP 5 — VALIDATE ON HELD-OUT RACE
# ─────────────────────────────────────────────────────────────
def validate_held_out(bundle):
    year, race = VALIDATION_RACE
    print(f"\n── Validation: {race} {year} (held-out race) ─────────────")

    laps = load_race(year, race)
    if laps is None:
        print("   Skipping — could not load validation race")
        return

    laps = engineer_features(laps)

    model = bundle["model"]
    le    = bundle["label_encoder"]

    X    = laps[FEATURE_COLS].dropna()
    y    = le.transform(laps.loc[X.index, "PitLabel"])
    pred = model.predict(X)

    f1  = f1_score(y, pred, average="macro", zero_division=0)
    acc = (pred == y).mean()

    print(f"   Accuracy : {acc*100:.1f}%")
    print(f"   Macro F1 : {f1:.4f}")
    print(classification_report(y, pred, target_names=le.classes_, zero_division=0))


# ─────────────────────────────────────────────────────────────
# STEP 6 — FEATURE IMPORTANCE PLOT
# ─────────────────────────────────────────────────────────────
def plot_feature_importance(bundle, save_path="day6_feature_importance.png"):
    plt.rcParams.update({
        "figure.facecolor": "#0f0f0f", "axes.facecolor": "#1a1a1a",
        "axes.edgecolor": "#333", "axes.labelcolor": "#ccc",
        "xtick.color": "#888", "ytick.color": "#888",
        "text.color": "#eee", "grid.color": "#2a2a2a",
        "font.family": "monospace",
    })

    model = bundle["model"]
    if not hasattr(model, "feature_importances_"):
        print("   Model has no feature_importances_ — skipping plot")
        return

    fi = pd.DataFrame({
        "Feature":    FEATURE_COLS,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor="#0f0f0f")
    fig.suptitle(
        f"F1 PIT STOP ANALYZER · DAY 6 · REAL DATA MODEL · "
        f"{bundle.get('trained_on','?')}",
        fontsize=12, fontweight="bold", color="#e8002d", y=1.01
    )

    # Feature importance
    ax = axes[0]
    colors = ["#e8002d" if v > fi["Importance"].median() else "#555555"
              for v in fi["Importance"]]
    ax.barh(fi["Feature"], fi["Importance"], color=colors, edgecolor="none")
    ax.set_title("FEATURE IMPORTANCE (Real Data Model)", color="#eee")
    ax.set_xlabel("Importance Score")
    ax.grid(True, axis="x")
    ax.set_axisbelow(True)

    # Model comparison
    ax2 = axes[1]
    models_names = ["Synthetic\nRF", "Synthetic\nXGB", "Real Data\nRF", "Real Data\nXGB"]
    # Note: synthetic scores are stored in old bundle if you kept it
    scores = [0.476, 0.495, bundle["rf_f1"], bundle["xgb_f1"]]
    bar_colors = ["#333", "#555", "#ffd700", "#e8002d"]
    bars = ax2.bar(models_names, scores, color=bar_colors, edgecolor="none", width=0.5)
    for bar, val in zip(bars, scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", color="#eee", fontsize=10)
    ax2.set_ylim(0, 1.0)
    ax2.set_title("MODEL COMPARISON\nSynthetic vs Real Data F1 Score", color="#eee")
    ax2.set_ylabel("Macro F1 Score")
    ax2.grid(True, axis="y")
    ax2.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
    print(f"\n📊 Plot saved → {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  F1 PIT STOP STRATEGY ANALYZER — DAY 6")
    print("  Real FastF1 Data Pipeline + Model Retraining")
    print("="*60)

    # Step 1-2: Pull and engineer real data
    df = build_training_dataset()

    # Step 3: Retrain
    bundle, X_test, y_test, le = retrain(df)

    # Step 4: Validate on held-out race
    validate_held_out(bundle)

    # Step 5: Feature importance plot
    plot_feature_importance(bundle)

    print("\n" + "="*60)
    print("  DAY 6 COMPLETE")
    print(f"  model.pkl updated — {bundle['trained_on']}")
    print("  Run: python -m streamlit run day5_app.py")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
