# Cryptographic Protocol Risk & Lean 4 Verification Workbench

Interactive Streamlit application accompanying the MSc dissertation **"Risk Analysis and Verification of Cryptographic Protocols"** by **Raj Guru Suwalka** (supervised by **Dr. Edon Kelmendi**, Queen Mary University of London, 2025–26).

- **Live Web App:** `https://cryptoshield-web-tool.streamlit.app/`
- **Main Research & Proofs Repository:** `https://github.com/rgsraj77/crypto-protocol-verification`

## What's Inside the App
- **Interactive Protocol Risk Simulator:** Load protocol templates from the 100-protocol dataset or configure cryptographic primitives and historical flags to inspect severity probabilities, attack-vector predictions (MITM, Replay, Downgrade, Token-theft), composite vulnerability scores, and hardening recommendations.
- **100-Protocol Explorer & Rankings:** Filter, search, and visualise all 100 protocols ranked by composite vulnerability score alongside their Lean 4 verification status.
- **ML Sensitivity & Feature Drivers:** Compare all-features vs. design-only risk predictions and inspect top feature drivers.
- **Lean 4 Formal Proof Inspector:** Browse machine-checked Dolev–Yao, Paulson inductive, and safety proof extracts in Lean 4.

## Repository Files
- `streamlit_crypto_verifier_app.py` — Main Streamlit application
- `requirements.txt` — Python dependencies
- `final_dv1.csv` — Curated 100-protocol dataset
- `protocol_ranking.csv` — Ranked protocol scores and Lean 4 status
