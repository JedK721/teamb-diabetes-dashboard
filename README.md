# Diabetes screening from survey answers (Team B, DIRISA Datathon 2026)

A Streamlit dashboard that flags people at risk of diabetes or prediabetes from 13 survey answers, and explains each result with LIME.

- **Model:** LightGBM without heavy alcohol use and with monotonic rules (the final model of the team notebook), threshold set for 78% recall.
- **Test results (50,623 people):** recall 78.1%, precision 31.2%, ROC-AUC 0.825, PR-AUC 0.428.
- **Data:** CDC Behavioral Risk Factor Surveillance System (BRFSS) 2015, via A. Teboul, *Diabetes Health Indicators Dataset* (Kaggle).

A learning project, **not a medical tool**.

Run locally:
```
pip install -r requirements.txt
streamlit run dashboard/app.py
```
