# Green Energy Sustainability Dashboard (Streamlit)

This repo contains a synthetic (but realistic) set of datasets and a Streamlit dashboard for sustainability analytics:
- generation, curtailment, downtime, revenue
- avoided CO₂ vs baseline (grid/coal/gas)
- weather drivers

## Run locally
```bash
pip install -r requirements.txt
python generate_datasets.py
streamlit run app.py
