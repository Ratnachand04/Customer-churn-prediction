# Customer Churn Prediction: Training + Free Streamlit Deployment

This project now supports:
- sequential training of all 19 models,
- saving reusable model artifacts,
- a Streamlit app that loads trained models (instead of retraining on each click),
- a database layer that works locally and after deployment.

## Updated Project Structure

```
.
├── app.py                        # Streamlit UI (prediction + comparison)
├── requirements.txt
├── .env.example
├── scripts/
│   └── train_all_models.py       # Sequential training pipeline
├── artifacts/                    # Generated after training
│   ├── preprocessor.joblib
│   ├── leaderboard.csv
│   ├── leaderboard.json
│   ├── *.joblib                  # sklearn/other model artifacts
│   └── *.keras                   # deep learning model artifacts
├── data/
│   ├── raw/
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── sample_submission.csv
│   ├── history/
│   │   └── prediction_history.json
│   └── db/
│       └── churn_app.db          # local SQLite DB (auto-created)
└── src/
    ├── preprocessing.py
    ├── models.py
    └── database.py
```

## 1. Setup Environment

From project root:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 2. Train All Models One by One

The training pipeline runs all models in sequence and saves each one.

```bash
python scripts/train_all_models.py
```

Optional flags:

```bash
python scripts/train_all_models.py --sample-size 50000
python scripts/train_all_models.py --sample-size 50000 --skip-existing
```

What this does:
1. Loads and preprocesses `data/raw/train.csv`
2. Saves preprocessing artifacts in `artifacts/preprocessor.joblib`
3. Trains each model in `MODEL_REGISTRY` sequentially
4. Saves each model artifact in `artifacts/`
5. Stores performance in:
   - `artifacts/leaderboard.csv`
   - `artifacts/leaderboard.json`
6. Logs training runs into the DB table `training_runs`

## 3. Run Streamlit Locally

```bash
streamlit run app.py
```

App behavior:
- Uses saved model artifacts when available
- Falls back to runtime training only if a specific artifact is missing
- Logs prediction calls to `prediction_logs` table
- Stores prediction history in `data/history/prediction_history.json`
- Shows combined DB + JSON history in Streamlit `History` page
- Supports filters by model, date range, prediction class, and source

## 4. Database Setup (Local + Deployment)

### Local

No setup required.

If `DATABASE_URL` is not provided, app uses:

`sqlite:///data/db/churn_app.db`

### Production (Recommended for free deployment)

Use a free Postgres provider:
- Neon (free tier)
- Supabase (free tier)

Set environment variable:

`DATABASE_URL=postgresql+psycopg2://USER:PASSWORD@HOST:5432/DBNAME?sslmode=require`

Tables are auto-created by the app:
- `prediction_logs`
- `training_runs`

## 5. Free Deployment on Streamlit Community Cloud

1. Push this project to GitHub.
2. Go to Streamlit Community Cloud and create a new app.
3. Select repo and set main file path: `app.py`.
4. Add secrets in Streamlit:
   - `DATABASE_URL` (Neon/Supabase connection string)
5. Deploy.

Important deployment note:
- Streamlit Cloud file storage is ephemeral.
- Persistent data must be in external DB (Neon/Supabase).
- Prefer committing `artifacts/` to the repo if size permits, so deployed app can load models directly.
- JSON history file is useful for local/dev history and export, but should not be treated as durable cloud storage.

## 6. Recommended Workflow

1. Train locally with `scripts/train_all_models.py`
2. Validate leaderboard and predictions
3. Push code + artifacts to GitHub
4. Configure `DATABASE_URL` secret in Streamlit Cloud
5. Deploy and verify logs in DB

## 7. Troubleshooting

- If app says no leaderboard found:
  - run `python scripts/train_all_models.py` first.
- If a model fails during training:
  - verify optional dependencies (TensorFlow/XGBoost/LightGBM) are installed.
- If DB insert fails in cloud:
  - confirm `DATABASE_URL` is valid and SSL parameters are included.
