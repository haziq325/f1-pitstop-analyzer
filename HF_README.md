---
title: F1 Pit Stop Strategy Analyzer
emoji: 🏎️
colorFrom: red
colorTo: gray
sdk: streamlit
sdk_version: 1.32.0
app_file: day5_app.py
pinned: true
license: mit
short_description: ML-powered F1 pit window predictor + strategy simulator
---

# 🏎️ F1 Pit Stop Strategy Analyzer

A machine learning tool that predicts the **optimal pit stop window** for Formula 1 drivers.

### Pages
- **Live Predictor** — real-time pit recommendation with confidence scores
- **Strategy Simulator** — ranks all valid strategies by estimated race time
- **Tire Analyst** — compound degradation curves and crossover analysis
- **Race Replay** — lap-by-lap model validation against real decisions

### How to use
Adjust the sidebar sliders (compound, tire age, lap number, track position) and the app instantly updates all four pages.

Built with `fastf1` · `XGBoost` · `Streamlit` · `Plotly`
