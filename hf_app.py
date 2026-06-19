"""
Hugging Face Spaces entrypoint.
HF Spaces looks for app.py by default — this just points to day5_app.py.
"""
# This file intentionally imports and re-exports day5_app
# HF Spaces runs: streamlit run app.py
exec(open("day5_app.py").read())
