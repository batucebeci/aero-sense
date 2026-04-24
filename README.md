# Aero-Sense — Sensor Fusion Fault Diagnosis

A FastAPI web app that classifies faults from multi-sensor data, with the
pieces a production-quality ML pipeline usually has around the model:
feature engineering, anomaly detection, RUL regression, drift monitoring,
SHAP explanations, calibration, per-class threshold tuning, a versioned
model registry, an alarm log with webhooks, batch prediction, and a live
WebSocket-streaming demo.

It ships with **four real public datasets** (NASA CMAPSS FD001/FD002/FD004
and UCI AI4I 2020) plus six bundled synthetic samples covering common
edge cases (imbalanced fleets, rare critical events, RUL-style
degradation, noisy environments).

## Running it

```bash
git clone <repo-url>
cd aero-sense
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt

uvicorn app.main:app --reload --port 8000
```

Open `http://127.0.0.1:8000` in your browser. Swagger lives at `/docs`,
Prometheus metrics at `/metrics`.

> **Port 8000 in use on Windows?** Try a different one:
> `uvicorn app.main:app --reload --port 5000` and open `:5000`.

Two optional environment variables:

- `API_KEY` — when set, all mutating endpoints require an `X-API-Key`
  header.
- `ALARM_WEBHOOK_URL` — when set, critical/high alarms are POSTed there
  as JSON.

Docker works the same way:

```bash
docker build -t aero-sense .
docker run -p 8000:8000 aero-sense
```

## Getting real data

The synthetic samples are checked in and work out of the box. For the
real benchmark datasets (~45 MB combined, not in git):

**NASA CMAPSS** — download from the PHM datasets mirror:

```bash
mkdir -p data/cmapss
curl -L "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip" -o /tmp/cmapss.zip
unzip /tmp/cmapss.zip -d /tmp/cmapss
unzip "/tmp/cmapss/6. Turbofan Engine Degradation Simulation Data Set/CMAPSSData.zip" -d data/cmapss
```

**UCI AI4I 2020:**

```bash
mkdir -p data/ai4i
curl -L "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv" -o data/ai4i/ai4i2020.csv
```

Then regenerate the sample CSVs:

```bash
python -m src.sample_datasets
```

The new samples appear automatically on the Dataset page.

## Suggested flow

1. **Dataset** — load one of the bundled samples (Quickstart demo is a
   good starting point) or upload your own CSV.
2. **Features / Compare / Trace** — explore the data. The trace page
   highlights fault windows on the time axis.
3. **Train** — fits Random Forest, SVM, KNN, Logistic Regression,
   Decision Tree, LightGBM and XGBoost in one go. The best model (macro
   F1) is registered.
4. **Tuning** — Optuna TPE search to push a specific model.
5. **Evaluation** — ROC, PR, reliability diagram. Apply isotonic or
   Platt calibration if probabilities look off.
6. **Threshold** — per-class threshold tuning. For imbalanced data, the
   recall strategy raises rare-fault recall at the cost of precision;
   the F1 strategy maximises macro F1 per class.
7. **Predict / Anomaly / RUL** — predictions on the held-out set, an
   Isolation Forest layer for unknown faults, and a regressor for
   remaining useful life.
8. **Explain / Counterfactual** — SHAP global + per-sample + waterfall,
   plus a coordinate-descent counterfactual search.
9. **Drift** — PSI and KS against a reference snapshot.
10. **Live** — WebSocket streaming demo with live predictions.
11. **Report** — download CSV, HTML, or PDF.

## What's in the repo

```
app/          FastAPI: routes, charts, services, state, observability,
              auth. 21 route modules covering 20 UI pages plus the JSON
              REST API and WebSocket.
src/          Data and ML core. Generators, features, fusion, training,
              tuning, evaluation curves, anomaly, RUL, drift,
              explainability, calibration, counterfactual, threshold
              tuning, reports, alarms, webhook, run logger, model
              registry, CLI.
templates/    Jinja2 pages and partials.
static/       One CSS file (no JS build step).
data/samples/ Six synthetic sample CSVs. Real datasets are gitignored
              and downloaded separately (see above).
tests/        85 pytest tests across both layers.
```

The CLI lives in `src/__main__.py`:

```bash
python -m src generate --samples-per-class 200 --output data/synth.csv
python -m src train    --data data/synth.csv --output models/all.pkl
python -m src predict  --data data/synth.csv --model models/all.pkl --output preds.csv
python -m src anomaly  --data data/synth.csv --contamination 0.08
python -m src rul      --data data/synth.csv --model-kind random_forest
```

## Benchmarks (real data)

Random Forest on a 80/20 split, fault-classification task, after the full
feature pipeline:

| Dataset                       | Rows    | Systems | Macro F1 |
| ----------------------------- | -------:| -------:| --------:|
| CMAPSS FD001                  | 20,631  |     100 |    0.95  |
| CMAPSS FD002 (multi-condition)| 53,759  |     260 |    0.89  |
| CMAPSS FD004 (multi-mode)     | 61,249  |     249 |    0.91  |
| UCI AI4I 2020 (imbalanced)    | 10,000  |       3 |    0.42  |

RUL regression on CMAPSS FD001: MAE 18.1 cycles, R² 0.848 — squarely in
published literature range.

AI4I macro F1 is structurally low because of severe class imbalance
(96.7% Normal). Open the **Threshold** page after training and apply
either strategy to bring the rare-fault recall up.

## Tests and lint

```bash
pytest                          # 85 tests
ruff check src app tests
python scripts_smoke_test.py    # end-to-end pipeline run, no UI
```

CI runs the same on Python 3.11 and 3.12; see `.github/workflows/ci.yml`.

## Things it does not do

- No MQTT or Kafka ingestion. The Live page generates synthetic data and
  streams it over a WebSocket.
- State is in memory. A multi-user deployment would need a session store
  (Redis, a database, or similar).
- No deep-learning models. Sticking with scikit-learn, LightGBM and
  XGBoost keeps training fast and the dependency footprint small.
- No ONNX export.

## Stack

Python, FastAPI, Jinja2, Plotly, scikit-learn, LightGBM, XGBoost, SHAP,
Optuna, reportlab, structlog, Prometheus client, slowapi, pytest, ruff,
Docker.
