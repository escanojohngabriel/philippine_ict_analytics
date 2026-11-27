# Philippine ICT Indicators Analytics

A Streamlit dashboard built from the supplied Colab workflow and datasets.

## Files
- `app.py` — Streamlit application
- `core_ict_indicators_raw_for_descriptive_analytics.csv` — raw dataset for descriptive analysis
- `core_ict_indicators_cleaned_for_ml.csv` — structured dataset for supervised learning
- `requirements.txt` — Python dependencies

## Run locally
1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Create and activate a virtual environment (recommended).
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Start the dashboard:
   ```bash
   streamlit run app.py
   ```

The two CSV files are loaded automatically when placed beside `app.py`. You can also upload replacement CSV files in the left sidebar.

## Analytical methods
- **Descriptive:** sector comparisons, distribution histogram, top-five-plus-others establishment share, scatter plot with trendline, and summary statistics. The Philippines aggregate row is excluded from sector-level comparisons.
- **Predictive:** simple linear regression of the unweighted mean sector internet-access rate across 2010, 2013, 2015, 2017, 2019, and 2021, extrapolated to 2022–2026.
- **Supervised:** binary target derived from the median 2021 internet-access value; Logistic Regression, Gaussian Naive Bayes, Decision Tree, and Random Forest compared using stratified cross-validation. Fold-local imputation, feature selection, and scaling are used to reduce data leakage.

## Notes
The dataset has a small number of sector-level records. Treat predictive and classification outputs as exploratory/academic analysis rather than production forecasts or validated operational models.
