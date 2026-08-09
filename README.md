# Universal Business Analytics & ML Studio

A dataset-agnostic starter kit for a 24-hour datathon. Plug in **any** tabular
business dataset (retail, supply chain, demand forecasting, risk, churn,
finance, etc.) and get a working end-to-end pipeline -- data profiling, EDA,
model training/comparison, predictions, business insights, and exportable
reports -- with **zero hardcoded column names**.

## Why this exists

The dataset for the datathon is only released on the day itself. This kit is
built so that when it drops, you spend your time on modeling and insight,
not on plumbing: upload the file, confirm the auto-detected target column,
and the rest of the pipeline adapts automatically.

## Project Structure

```
project/
├── app.py                     # Streamlit entry point (page router)
├── main.py                    # CLI pipeline runner (no UI)
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/                   # Drop the datathon dataset here (optional)
│   └── processed/             # Cleaned/engineered data snapshots
│
├── notebooks/                 # Scratch analysis notebooks
│
├── models/
│   ├── train.py                # Model registry + training
│   ├── predict.py               # Inference on new data + artifact bundling
│   ├── evaluate.py              # Classification/regression metrics
│   └── compare_models.py        # Multi-model leaderboard + ranking
│
├── preprocessing/
│   ├── clean.py                # Missing values, outliers, duplicates
│   ├── encode.py                # Categorical encoding (one-hot / label)
│   ├── scale.py                 # Numerical scaling
│   └── split.py                  # Train/test split (random or chronological)
│
├── dashboard/
│   ├── pages/                  # One module per Streamlit page
│   ├── components/              # Sidebar, uploader, KPI cards, download buttons
│   └── charts.py                 # Reusable Plotly chart builders
│
├── utils/
│   ├── helpers.py               # Column-type/target/problem-type auto-detection
│   ├── config.py                  # Paths, constants, session-state keys
│   └── logger.py                   # Centralized logging
│
└── outputs/
    ├── models/                  # Saved model artifact bundles (.joblib)
    ├── reports/                  # Logs, leaderboards, PDF reports
    └── figures/                   # Exported chart images (optional)
```

## Getting Started

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (typically `http://localhost:8501`).

### Environment variables

The AI Assistant, executive summaries, recommendations, and prediction
explanations use an LLM provider and need an API key. Everything else
(upload, preprocessing, AutoML, predictions, dashboards) works fully without
one.

```bash
cp .env.example .env
```

Then edit `.env` and set:

| Variable          | Required                       | Get a key at                         |
|-------------------|---------------------------------|---------------------------------------|
| `GROQ_API_KEY`    | Yes, for AI features (default provider) | https://console.groq.com/keys        |
| `GEMINI_API_KEY`  | Only if you switch the active provider to Gemini in `services/llm_service.py` | https://aistudio.google.com/apikey  |

`.env` is git-ignored -- never commit real keys. If a key is ever exposed
(e.g. pushed by mistake), rotate it immediately from the provider's console.

## Using it with your dataset

1. Go to **Dataset Explorer** and upload a CSV or Excel file.
2. Confirm (or override) the auto-suggested **target column**, **date
   column**, and **problem type** (classification / regression / time series).
3. Explore the data on the **EDA** page -- histograms, boxplots, correlation
   heatmap, missing-value chart, and categorical breakdowns are generated
   automatically for whatever columns exist.
4. On **Model Training**, pick a problem type and one or more models
   (Logistic/Linear Regression, Decision Tree, Random Forest, XGBoost,
   LightGBM, CatBoost), then **Train Model(s)** or **Compare Models**.
5. **Save Best Model** persists a single artifact bundle (`outputs/models/best_model.joblib`)
   containing the fitted cleaner, encoder, scaler, and model together, so
   predictions are always preprocessed identically to training.
6. Use **Predictions** to upload a new CSV/Excel file and download the
   scored output.
7. **Business Insights**, **Dashboard Analytics**, and **Reports** turn the
   trained model and predictions into KPIs, feature importance, filterable
   charts, and CSV/Excel/PDF exports.

### CLI alternative

For a quick end-to-end sanity check outside the UI:

```bash
python main.py --data data/raw/your_dataset.csv --target your_target_column
```

Omit `--target` to let the pipeline auto-detect it, and `--problem-type` to
let it auto-infer classification vs. regression from the target's values.

## How the "generic" pipeline works

- **Column-type detection** (`utils/helpers.py`) classifies every column into
  numerical, categorical, datetime, or identifier-like based on dtype and
  cardinality -- never by name.
- **Target detection** looks for common business-target keywords (target,
  label, churn, risk, demand, sales, ...) and falls back to the last column.
- **Problem-type inference** checks whether the target is numeric with high
  cardinality (regression) or categorical/low-cardinality (classification).
- **Preprocessing** (`preprocessing/`) is fit once on the training split and
  reused identically for the test split and any later prediction data.
- **Model artifact bundles** (`models/predict.py`) package the fitted
  cleaner + encoder + scaler + model + metadata into one joblib file so a
  single object is all that's needed to score new data correctly.

## Extending

- Add a new model: register it in `models/train.py`'s classification/regression
  builder functions and add its display name to `utils/config.py`.
- Add a new chart: add a function to `dashboard/charts.py` and call it from
  the relevant page in `dashboard/pages/`.
- Add a new page: create `dashboard/pages/your_page.py` with a `render()`
  function, then register it in `app.py`'s `_PAGE_ROUTER`.
