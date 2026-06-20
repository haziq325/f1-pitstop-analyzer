"""
F1 Pit Stop Strategy Analyzer
Day 7 — Final Project Verification

Run this to confirm all files are present and the full pipeline works.
    python day7_verify.py
"""

import os
import sys
import pickle
import pandas as pd

REQUIRED_FILES = [
    "day1_eda.py",
    "day2_model.py",
    "day3_app.py",
    "day4_simulator.py",
    "day5_app.py",
    "day6_real_data.py",
    "model.pkl",
    "processed_laps.csv",
    "requirements.txt",
    "README.md",
    "LICENSE",
]

GENERATED_FILES = [
    "model.pkl",
    "processed_laps.csv",
    "strategies_ranked.csv",
]

print("\n" + "="*55)
print("  F1 PIT STOP STRATEGY ANALYZER — DAY 7 VERIFY")
print("="*55)

# ── Check all files exist ─────────────────────────────────
print("\n── File Checklist ───────────────────────────────────────")
all_ok = True
for f in REQUIRED_FILES:
    exists = os.path.isfile(f)
    status = "✅" if exists else "❌ MISSING"
    size   = f"({os.path.getsize(f)/1024:.0f} KB)" if exists else ""
    print(f"  {status}  {f} {size}")
    if not exists:
        all_ok = False

# ── Validate model.pkl ────────────────────────────────────
print("\n── Model Bundle ─────────────────────────────────────────")
try:
    with open("model.pkl", "rb") as f:
        bundle = pickle.load(f)
    print(f"  ✅ Model      : {bundle['model_name']}")
    print(f"  ✅ XGB F1     : {bundle['xgb_f1']}")
    print(f"  ✅ RF F1      : {bundle['rf_f1']}")
    print(f"  ✅ Features   : {len(bundle['feature_cols'])}")
    print(f"  ✅ Labels     : {bundle['label_order']}")
    trained_on = bundle.get('trained_on', 'synthetic data')
    print(f"  ✅ Trained on : {trained_on}")
except Exception as e:
    print(f"  ❌ model.pkl error: {e}")
    all_ok = False

# ── Validate processed_laps.csv ───────────────────────────
print("\n── Dataset ──────────────────────────────────────────────")
try:
    df = pd.read_csv("processed_laps.csv")
    print(f"  ✅ Rows       : {len(df):,}")
    print(f"  ✅ Drivers    : {sorted(df['Driver'].unique())}")
    print(f"  ✅ Compounds  : {list(df['Compound'].unique())}")
    print(f"  ✅ Columns    : {len(df.columns)}")
    print(f"  ✅ Labels     : {df['PitLabel'].value_counts().to_dict()}")
except Exception as e:
    print(f"  ❌ CSV error: {e}")
    all_ok = False

# ── Test a prediction ─────────────────────────────────────
print("\n── Prediction Test ──────────────────────────────────────")
try:
    import numpy as np

    model  = bundle["model"]
    le     = bundle["label_encoder"]
    feats  = bundle["feature_cols"]

    COMPOUND_BASE = {"SOFT": 88.5, "MEDIUM": 89.8, "HARD": 91.0}
    COMPOUND_DEG  = {"SOFT": 0.12, "MEDIUM": 0.07, "HARD": 0.04}
    COMPOUND_MAX  = {"SOFT": 20,   "MEDIUM": 35,   "HARD": 50}
    COMPOUND_CODE = {"SOFT": 0,    "MEDIUM": 1,    "HARD": 2}

    tests = [
        ("SOFT",   3,  5, "early race fresh tires"),
        ("MEDIUM", 25, 30, "worn mid-race tires"),
        ("HARD",   10, 50, "late race on hard"),
    ]

    for compound, tyre_life, lap, desc in tests:
        lt       = COMPOUND_BASE[compound] + COMPOUND_DEG[compound] * tyre_life
        health   = max(0, 100 - (tyre_life / COMPOUND_MAX[compound]) * 100)
        row = {
            "TyreLife": tyre_life, "TyreHealthPct": health,
            "LapTimeSec": lt, "RollingAvgLapTime": lt,
            "LapTimeDelta": COMPOUND_DEG[compound], "DegRate": COMPOUND_DEG[compound],
            "DegAccel": 0.0, "PaceLoss": COMPOUND_DEG[compound] * tyre_life,
            "StintLength": tyre_life, "LapsRemaining": 57 - lap,
            "RaceProgress": lap / 57, "CompoundCode": COMPOUND_CODE[compound],
            "Position": 3, "InPitWindow": 1 if 15 <= lap <= 25 else 0,
        }
        X     = pd.DataFrame([row])[feats]
        proba = model.predict_proba(X)[0]
        pred  = le.inverse_transform([int(np.argmax(proba))])[0]
        conf  = max(proba) * 100
        print(f"  ✅ {desc:35s} → {pred:10s} ({conf:.0f}% conf)")

except Exception as e:
    print(f"  ❌ Prediction error: {e}")
    all_ok = False

# ── Line count summary ────────────────────────────────────
print("\n── Codebase Summary ─────────────────────────────────────")
scripts = [
    "day1_eda.py", "day2_model.py", "day3_app.py",
    "day4_simulator.py", "day5_app.py", "day6_real_data.py",
]
total = 0
for s in scripts:
    if os.path.isfile(s):
        with open(s, encoding="utf-8", errors="ignore") as f:
            lines = len(f.readlines())
        total += lines
        print(f"  {s:30s} {lines:>5} lines")
print(f"  {'TOTAL':30s} {total:>5} lines")

# ── Final result ──────────────────────────────────────────
print("\n" + "="*55)
if all_ok:
    print("  ✅ ALL CHECKS PASSED — project is ready")
    print("\n  To launch the app:")
    print("    python -m streamlit run day5_app.py")
    print("\n  To retrain on real data:")
    print("    mkdir fastf1_cache")
    print("    python day6_real_data.py")
else:
    print("  ❌ Some checks failed — see above")
print("="*55 + "\n")
