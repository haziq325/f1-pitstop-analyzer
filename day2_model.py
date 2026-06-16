"""
F1 Pit Stop Strategy Analyzer
Day 2 — Feature Engineering + Model Training

Loads processed_laps.csv from Day 1 and:
  1. Adds more features
  2. Handles class imbalance (SMOTE)
  3. Trains Random Forest + XGBoost
  4. Evaluates both models
  5. Saves the best model → model.pkl

Run:
    python day2_model.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection  import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing    import LabelEncoder, StandardScaler
from sklearn.ensemble         import RandomForestClassifier
from sklearn.metrics          import (classification_report, confusion_matrix,
                                      ConfusionMatrixDisplay)
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import pickle

# ─────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────
def load_data(path="processed_laps.csv"):
    df = pd.read_csv(path)
    print(f"✅ Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# ─────────────────────────────────────────────────────────────
# 2. EXTRA FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
def add_features(df):
    """
    Day 1 gave us the basics. Day 2 adds richer features
    that the model needs to make smarter decisions.
    """

    df = df.copy()

    # ── Tire health score (0–100, 100 = brand new) ──────────
    # Normalise TyreLife within each compound (older = lower score)
    max_life = {"SOFT": 20, "MEDIUM": 35, "HARD": 50}
    df["TyreHealthPct"] = df.apply(
        lambda r: max(0, 100 - (r["TyreLife"] / max_life.get(r["Compound"], 40)) * 100),
        axis=1
    )

    # ── Pace loss vs fresh tire baseline ────────────────────
    # How much slower is this lap vs the driver's best lap on this stint?
    df["PaceLoss"] = df.groupby(["Driver", "Compound"])["LapTimeSec"].transform(
        lambda x: x - x.min()
    )

    # ── Stint length so far ──────────────────────────────────
    # Resets to 0 each time the driver pits
    def stint_counter(group):
        counts, count = [], 0
        for pitted in group["Pitted"]:
            count += 1
            counts.append(count)
            if pitted:
                count = 0
        return counts

    stint_lens = []
    for _, grp in df.groupby("Driver"):
        stint_lens.extend(stint_counter(grp))
    df["StintLength"] = stint_lens

    # ── Race progress (0–1) ──────────────────────────────────
    df["RaceProgress"] = df["LapNumber"] / df["LapNumber"].max()

    # ── Compound encoded as integer ──────────────────────────
    # SOFT=0  MEDIUM=1  HARD=2  (ordered by speed)
    compound_map = {"SOFT": 0, "MEDIUM": 1, "HARD": 2}
    df["CompoundCode"] = df["Compound"].map(compound_map)

    # ── Acceleration of degradation ──────────────────────────
    # Second derivative of lap time — spikes signal cliff-edge tire wear
    df["DegAccel"] = df.groupby("Driver")["DegRate"].diff().fillna(0)

    # ── Is this a strategic lap? (common pit windows) ────────
    # Teams often pit between laps 15–25 and 35–45 in a 57-lap race
    df["InPitWindow"] = df["LapNumber"].apply(
        lambda l: 1 if (15 <= l <= 25) or (35 <= l <= 45) else 0
    )

    print(f"✅ Features added. Total features: {df.shape[1]} columns")
    return df


# ─────────────────────────────────────────────────────────────
# 3. PREPARE X AND y
# ─────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "TyreLife",         # raw tire age
    "TyreHealthPct",    # normalised tire health 0–100
    "LapTimeSec",       # current lap time
    "RollingAvgLapTime",# smoothed lap time
    "LapTimeDelta",     # change vs previous lap
    "DegRate",          # degradation per lap
    "DegAccel",         # is degradation speeding up?
    "PaceLoss",         # vs best lap on this stint
    "StintLength",      # laps on current set of tyres
    "LapsRemaining",    # laps left in race
    "RaceProgress",     # 0–1 race completion
    "CompoundCode",     # 0=SOFT 1=MEDIUM 2=HARD
    "Position",         # track position
    "InPitWindow",      # are we in a typical pit window?
]

LABEL_ORDER = ["STAY_OUT", "PIT_SOON", "PIT_NOW"]   # 0 → 1 → 2 urgency

def prepare_Xy(df):
    le = LabelEncoder()
    le.fit(LABEL_ORDER)

    X = df[FEATURE_COLS].copy()
    y = le.transform(df["PitLabel"])

    print(f"\n── Feature matrix shape: {X.shape}")
    print(f"── Label mapping: { dict(zip(le.classes_, le.transform(le.classes_))) }")
    print(f"── Class distribution: { dict(zip(*np.unique(y, return_counts=True))) }\n")

    return X, y, le


# ─────────────────────────────────────────────────────────────
# 4. HANDLE CLASS IMBALANCE
# ─────────────────────────────────────────────────────────────
def get_class_weights(y):
    """
    Instead of SMOTE (which needs imbalanced-learn),
    we use sklearn's compute_class_weight.
    Passes directly into both RandomForest and XGBoost.
    """
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    cw_dict = dict(zip(classes, weights))
    print(f"── Class weights (balanced): {cw_dict}\n")
    return cw_dict


# ─────────────────────────────────────────────────────────────
# 5. TRAIN MODELS
# ─────────────────────────────────────────────────────────────
def train_random_forest(X_train, y_train, class_weights):
    print("🌲 Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,       # 200 trees
        max_depth=8,            # prevent overfitting
        min_samples_leaf=3,
        class_weight=class_weights,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    print("   ✅ Random Forest trained")
    return rf


def train_xgboost(X_train, y_train, y_full):
    print("⚡ Training XGBoost...")

    # Compute per-sample weights to handle class imbalance
    classes, counts = np.unique(y_train, return_counts=True)
    weight_map      = {c: len(y_train) / (len(classes) * cnt)
                       for c, cnt in zip(classes, counts)}
    sample_weights  = np.array([weight_map[c] for c in y_train])

    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,            # shallower → less overfit on small data
        learning_rate=0.1,      # higher LR to learn faster on small set
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=1,
        gamma=0,
        eval_metric="mlogloss",
        objective="multi:softprob",
        num_class=3,
        random_state=42,
        verbosity=0,
    )
    xgb_model.fit(X_train, y_train, sample_weight=sample_weights)
    print("   ✅ XGBoost trained\n")
    return xgb_model


# ─────────────────────────────────────────────────────────────
# 6. EVALUATE
# ─────────────────────────────────────────────────────────────
def evaluate(model, X_test, y_test, le, model_name):
    y_pred = model.predict(X_test)

    print(f"\n{'='*50}")
    print(f"  {model_name} — EVALUATION REPORT")
    print(f"{'='*50}")
    print(classification_report(
        y_test, y_pred,
        target_names=le.classes_,
        zero_division=0
    ))

    # Cross-validation score
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_test, y_test, cv=cv, scoring="f1_macro")
    print(f"  5-Fold CV F1 (macro): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    return y_pred


def feature_importance_df(model, model_name):
    if hasattr(model, "feature_importances_"):
        fi = pd.DataFrame({
            "Feature":    FEATURE_COLS,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False)
        print(f"\n── Top features ({model_name}) ──────────────────────")
        print(fi.to_string(index=False))
        return fi
    return None


# ─────────────────────────────────────────────────────────────
# 7. PLOTS
# ─────────────────────────────────────────────────────────────
def plot_results(rf, xgb_model, X_test, y_test,
                 y_pred_rf, y_pred_xgb, le, save_path="day2_model_plots.png"):

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

    class_names = le.classes_
    fig = plt.figure(figsize=(18, 10), facecolor="#0f0f0f")
    fig.suptitle(
        "F1 PIT STOP STRATEGY ANALYZER  ·  DAY 2  ·  MODEL EVALUATION",
        fontsize=13, fontweight="bold", color="#e8002d", y=0.98
    )

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.4)

    # ── Confusion Matrix: Random Forest ─────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    disp  = ConfusionMatrixDisplay(cm_rf, display_labels=class_names)
    disp.plot(ax=ax1, colorbar=False, cmap="Reds")
    ax1.set_title("CONFUSION MATRIX\nRandom Forest", color="#eeeeee")
    ax1.xaxis.label.set_color("#cccccc")
    ax1.yaxis.label.set_color("#cccccc")
    for text in ax1.texts:
        text.set_color("#eeeeee")

    # ── Confusion Matrix: XGBoost ────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    cm_xgb = confusion_matrix(y_test, y_pred_xgb)
    disp2  = ConfusionMatrixDisplay(cm_xgb, display_labels=class_names)
    disp2.plot(ax=ax2, colorbar=False, cmap="Blues")
    ax2.set_title("CONFUSION MATRIX\nXGBoost", color="#eeeeee")
    ax2.xaxis.label.set_color("#cccccc")
    ax2.yaxis.label.set_color("#cccccc")
    for text in ax2.texts:
        text.set_color("#eeeeee")

    # ── Feature Importance: XGBoost ─────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    fi = pd.DataFrame({
        "Feature":    FEATURE_COLS,
        "Importance": xgb_model.feature_importances_
    }).sort_values("Importance").tail(10)
    bars = ax3.barh(fi["Feature"], fi["Importance"],
                    color="#e8002d", edgecolor="none", alpha=0.85)
    ax3.set_title("FEATURE IMPORTANCE\nXGBoost (Top 10)", color="#eeeeee")
    ax3.set_xlabel("Importance")
    ax3.grid(True, axis="x")

    # ── Feature Importance: Random Forest ───────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    fi_rf = pd.DataFrame({
        "Feature":    FEATURE_COLS,
        "Importance": rf.feature_importances_
    }).sort_values("Importance").tail(10)
    ax4.barh(fi_rf["Feature"], fi_rf["Importance"],
             color="#ffd700", edgecolor="none", alpha=0.85)
    ax4.set_title("FEATURE IMPORTANCE\nRandom Forest (Top 10)", color="#eeeeee")
    ax4.set_xlabel("Importance")
    ax4.grid(True, axis="x")

    # ── Probability calibration plot (XGBoost) ───────────────
    ax5 = fig.add_subplot(gs[1, 1])
    proba = xgb_model.predict_proba(X_test)
    colors = ["#ebebeb", "#ffd700", "#e8002d"]
    for i, (label, col) in enumerate(zip(class_names, colors)):
        ax5.hist(proba[:, i], bins=20, alpha=0.7,
                 color=col, label=label, edgecolor="none")
    ax5.set_title("PREDICTION CONFIDENCE\nXGBoost Probability Distribution", color="#eeeeee")
    ax5.set_xlabel("Predicted Probability")
    ax5.set_ylabel("Count")
    ax5.legend(facecolor="#1a1a1a", edgecolor="#333", labelcolor="#eee")
    ax5.grid(True)

    # ── Model comparison bar chart ───────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    from sklearn.metrics import f1_score, accuracy_score
    metrics = {
        "Accuracy":     [accuracy_score(y_test, y_pred_rf),
                         accuracy_score(y_test, y_pred_xgb)],
        "F1 (macro)":   [f1_score(y_test, y_pred_rf, average="macro", zero_division=0),
                         f1_score(y_test, y_pred_xgb, average="macro", zero_division=0)],
        "F1 (weighted)":[f1_score(y_test, y_pred_rf, average="weighted", zero_division=0),
                         f1_score(y_test, y_pred_xgb, average="weighted", zero_division=0)],
    }
    x     = np.arange(len(metrics))
    width = 0.3
    ax6.bar(x - width/2, [v[0] for v in metrics.values()],
            width, label="Random Forest", color="#ffd700", alpha=0.85)
    ax6.bar(x + width/2, [v[1] for v in metrics.values()],
            width, label="XGBoost", color="#e8002d", alpha=0.85)
    ax6.set_xticks(x)
    ax6.set_xticklabels(list(metrics.keys()), fontsize=9)
    ax6.set_ylim(0, 1.1)
    ax6.set_title("MODEL COMPARISON", color="#eeeeee")
    ax6.legend(facecolor="#1a1a1a", edgecolor="#333", labelcolor="#eee")
    ax6.grid(True, axis="y")
    ax6.set_axisbelow(True)

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
    print(f"\n📊 Model evaluation plots saved → {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────
# 8. SAVE BEST MODEL
# ─────────────────────────────────────────────────────────────
def save_model(rf, xgb_model, le, X_test, y_test, scaler=None):
    from sklearn.metrics import f1_score

    rf_f1  = f1_score(y_test, rf.predict(X_test),      average="macro", zero_division=0)
    xgb_f1 = f1_score(y_test, xgb_model.predict(X_test), average="macro", zero_division=0)

    best_model      = xgb_model if xgb_f1 >= rf_f1 else rf
    best_model_name = "XGBoost"  if xgb_f1 >= rf_f1 else "RandomForest"

    bundle = {
        "model":        best_model,
        "model_name":   best_model_name,
        "label_encoder":le,
        "feature_cols": FEATURE_COLS,
        "label_order":  LABEL_ORDER,
        "rf_f1":        round(rf_f1, 4),
        "xgb_f1":       round(xgb_f1, 4),
    }

    with open("model.pkl", "wb") as f:
        pickle.dump(bundle, f)

    print(f"\n✅ Best model saved → model.pkl")
    print(f"   RF  macro-F1 : {rf_f1:.4f}")
    print(f"   XGB macro-F1 : {xgb_f1:.4f}")
    print(f"   Saved model  : {best_model_name}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*55)
    print("  F1 PIT STOP STRATEGY ANALYZER — DAY 2 MODEL")
    print("="*55 + "\n")

    # Step 1 — Load
    df = load_data("processed_laps.csv")

    # Step 2 — Add features
    df = add_features(df)

    # Step 3 — Prepare X, y
    X, y, le = prepare_Xy(df)

    # Step 4 — Train/test split (stratified to preserve class ratios)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"── Train size: {len(X_train)}   Test size: {len(X_test)}")

    # Step 5 — Handle imbalance via class weights
    class_weights = get_class_weights(y_train)

    # Step 6 — Train both models
    rf        = train_random_forest(X_train, y_train, class_weights)
    xgb_model = train_xgboost(X_train, y_train, y)

    # Step 7 — Evaluate
    y_pred_rf  = evaluate(rf,        X_test, y_test, le, "Random Forest")
    y_pred_xgb = evaluate(xgb_model, X_test, y_test, le, "XGBoost")

    # Step 8 — Feature importance
    feature_importance_df(rf,        "Random Forest")
    feature_importance_df(xgb_model, "XGBoost")

    # Step 9 — Plots
    plot_results(rf, xgb_model, X_test, y_test,
                 y_pred_rf, y_pred_xgb, le)

    # Step 10 — Save best model
    save_model(rf, xgb_model, le, X_test, y_test)

    print("\n🏁 Day 2 complete. model.pkl is ready for Day 3 (Streamlit app).\n")


if __name__ == "__main__":
    main()
