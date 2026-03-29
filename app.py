"""Customer Churn Prediction – Premium Streamlit Application."""
import sys, os
import json
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from src.preprocessing import (load_and_prepare_data, encode_user_input,
                                CATEGORY_OPTIONS, FEATURE_COLS, CATEGORICAL_COLS,
                                load_preprocessing_artifacts)
from src.models import MODEL_REGISTRY, train_model, predict_single, load_model_artifact
from src.database import init_db, log_prediction, fetch_recent_predictions, fetch_prediction_history
from src.history_store import append_prediction_history, read_prediction_history

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & CUSTOM CSS
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="ChurnGuard AI", page_icon="🛡️", layout="wide",
                   initial_sidebar_state="expanded")

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif; }

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 40%, #24243e 100%);
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e1e3f 0%, #0f0c29 100%);
    border-right: 1px solid rgba(255,255,255,0.05);
}
section[data-testid="stSidebar"] .stRadio label {
    color: #e0e0ff !important;
    font-weight: 500;
    padding: 8px 0;
}

/* Glass card */
.glass-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 24px;
    margin: 10px 0;
    backdrop-filter: blur(12px);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.glass-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(99,102,241,0.15);
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(139,92,246,0.1));
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 16px;
    padding: 20px;
    text-align: center;
}
.metric-value { font-size: 2.2rem; font-weight: 800; color: #a78bfa; }
.metric-label { font-size: 0.85rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; }

/* Best badge */
.best-badge {
    display: inline-block;
    background: linear-gradient(135deg, #10b981, #059669);
    color: white;
    padding: 4px 16px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 700;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(16,185,129,0.4); }
    50% { box-shadow: 0 0 0 10px rgba(16,185,129,0); }
}

/* Prediction result */
.pred-churn {
    background: linear-gradient(135deg, rgba(239,68,68,0.2), rgba(220,38,38,0.1));
    border: 1px solid rgba(239,68,68,0.35);
    border-radius: 16px; padding: 20px; text-align: center;
}
.pred-no-churn {
    background: linear-gradient(135deg, rgba(16,185,129,0.2), rgba(5,150,105,0.1));
    border: 1px solid rgba(16,185,129,0.35);
    border-radius: 16px; padding: 20px; text-align: center;
}

/* Hero */
.hero-title {
    font-size: 2.8rem; font-weight: 800;
    background: linear-gradient(135deg, #818cf8, #a78bfa, #c084fc);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}
.hero-sub { color: #94a3b8; font-size: 1.1rem; margin-top: 4px; }

/* Category badges */
.cat-badge {
    display: inline-block; padding: 3px 12px; border-radius: 12px;
    font-size: 0.7rem; font-weight: 600; margin-right: 6px;
}
.cat-supervised { background: rgba(99,102,241,0.2); color: #818cf8; }
.cat-deep { background: rgba(236,72,153,0.2); color: #f472b6; }
.cat-unsupervised { background: rgba(245,158,11,0.2); color: #fbbf24; }
.cat-semi { background: rgba(16,185,129,0.2); color: #34d399; }

h1, h2, h3, h4 { color: #e2e8f0 !important; }
p, span, label { color: #cbd5e1 !important; }

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
LEADERBOARD_PATH = ARTIFACTS_DIR / "leaderboard.csv"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.joblib"


@st.cache_data
def load_leaderboard():
    if LEADERBOARD_PATH.exists():
        return pd.read_csv(LEADERBOARD_PATH)
    return pd.DataFrame()


def metric_map_from_leaderboard(df):
    if df.empty:
        return {}
    metric_cols = ["Accuracy", "Precision", "Recall", "F1 Score"]
    out = {}
    for _, row in df.iterrows():
        out[row["Model"]] = {m: float(row[m]) for m in metric_cols}
    return out


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<p style="font-size:1.6rem;font-weight:800;color:#a78bfa;">🛡️ ChurnGuard AI</p>',
                unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.8rem;color:#64748b;">Intelligent Churn Prediction</p>',
                unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("Navigation", ["🏠 Home", "🔮 Predict", "📊 Compare Models", "🗂️ History", "ℹ️ About"],
                    label_visibility="collapsed")
    st.markdown("---")
    st.markdown('<p style="font-size:0.7rem;color:#475569;">Built with ❤️ using Streamlit</p>',
                unsafe_allow_html=True)

init_db()

# Load data once
X_train, X_test, y_train, y_test = None, None, None, None
raw_df = pd.DataFrame()
data_ready = True
data_error = None

try:
    X_train, X_test, y_train, y_test, label_encoders, scaler, raw_df = load_and_prepare_data()
except Exception as exc:
    data_ready = False
    data_error = str(exc)
    label_encoders, scaler = None, None

if PREPROCESSOR_PATH.exists():
    try:
        label_encoders, scaler = load_preprocessing_artifacts(PREPROCESSOR_PATH)
    except Exception as exc:
        st.warning(f"Could not load preprocessor artifact. Using in-memory preprocessors. ({exc})")

inference_ready = label_encoders is not None and scaler is not None

leaderboard_df = load_leaderboard()
saved_metrics = metric_map_from_leaderboard(leaderboard_df)

# Global Category CSS map
cat_css = {"Supervised": "cat-supervised", "Deep Learning": "cat-deep",
           "Unsupervised": "cat-unsupervised", "Semi-Supervised": "cat-semi"}


# ══════════════════════════════════════════════════════════════════════════════
# 🏠  HOME PAGE
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown('<p class="hero-title">ChurnGuard AI</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">Predict customer churn with 19 Machine Learning, Deep Learning & Neural Network models</p>',
                unsafe_allow_html=True)
    st.markdown("")

    if data_error:
        st.info(
            "Training dataset is not available in deployment. "
            "App is running in artifact mode. "
            "Predictions can still run using saved artifacts."
        )

    # Quick stats
    c1, c2, c3, c4 = st.columns(4)
    total = len(raw_df)
    churned = int((raw_df['Churn'] == 'Yes').sum()) if 'Churn' in raw_df.columns else 0
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{total:,}</div>'
                    f'<div class="metric-label">Customers</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{churned:,}</div>'
                    f'<div class="metric-label">Churned</div></div>', unsafe_allow_html=True)
    with c3:
        rate = round(churned/total*100, 1) if total > 0 else 0
        st.markdown(f'<div class="metric-card"><div class="metric-value">{rate}%</div>'
                    f'<div class="metric-label">Churn Rate</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><div class="metric-value">19</div>'
                    f'<div class="metric-label">Models Available</div></div>', unsafe_allow_html=True)

    recent_preds = fetch_recent_predictions(limit=5)
    if recent_preds:
        st.markdown("### 🗂️ Recent Predictions")
        st.dataframe(pd.DataFrame(recent_preds), use_container_width=True)

    if not raw_df.empty and 'Churn' in raw_df.columns and 'tenure' in raw_df.columns:
        st.markdown("### 📈 Data Overview")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            churn_counts = raw_df['Churn'].value_counts()
            fig = go.Figure(go.Pie(
                labels=['Retained', 'Churned'], values=[churn_counts.get(0,0), churn_counts.get(1,0)],
                hole=0.6, marker=dict(colors=['#6366f1', '#ec4899']),
                textfont=dict(color='white', size=14)
            ))
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                              font=dict(color='#e2e8f0'), height=350,
                              title=dict(text="Churn Distribution", font=dict(size=16, color='#e2e8f0')),
                              legend=dict(font=dict(color='#e2e8f0')))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            fig2 = go.Figure()
            for churn_val, color, name in [(0, '#6366f1', 'Retained'), (1, '#ec4899', 'Churned')]:
                subset = raw_df[raw_df['Churn'] == churn_val]['tenure']
                fig2.add_trace(go.Histogram(x=subset, nbinsx=30, name=name,
                                             marker_color=color, opacity=0.7))
            fig2.update_layout(barmode='overlay', paper_bgcolor='rgba(0,0,0,0)',
                               plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#e2e8f0'),
                               height=350, title=dict(text="Tenure Distribution by Churn",
                               font=dict(size=16, color='#e2e8f0')),
                               xaxis=dict(title="Tenure (months)", gridcolor='rgba(255,255,255,0.05)'),
                               yaxis=dict(title="Count", gridcolor='rgba(255,255,255,0.05)'),
                               legend=dict(font=dict(color='#e2e8f0')))
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # Available Models
    st.markdown("### 🤖 Available Models")
    cats = {"Supervised": [], "Deep Learning": [], "Unsupervised": [], "Semi-Supervised": []}
    for name, info in MODEL_REGISTRY.items():
        cats[info["category"]].append(f'{info["icon"]} {name}')
    cols = st.columns(4)
    for i, (cat, items) in enumerate(cats.items()):
        with cols[i]:
            st.markdown(f'<div class="glass-card"><span class="cat-badge {cat_css[cat]}">{cat}</span>'
                        f'<br><br>{"<br>".join(items)}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# 🔮  PREDICT PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict":
    st.markdown('<p class="hero-title">🔮 Predict Churn</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">Enter customer details and select models to predict</p>',
                unsafe_allow_html=True)

    # ── Model Selection ─────────────────────────────────────────────────────
    st.markdown("### Select Models")
    cat_groups = {}
    for n, info in MODEL_REGISTRY.items():
        cat_groups.setdefault(info["category"], []).append(n)

    selected_models = []
    cols = st.columns(4)
    cat_order = ["Supervised", "Deep Learning", "Unsupervised", "Semi-Supervised"]
    for i, cat in enumerate(cat_order):
        with cols[i]:
            st.markdown(f'<span class="cat-badge {cat_css[cat]}">{cat}</span>', unsafe_allow_html=True)
            for m in cat_groups.get(cat, []):
                icon = MODEL_REGISTRY[m]["icon"]
                if st.checkbox(f"{icon} {m}", key=f"chk_{m}",
                               value=(m in ["Random Forest", "XGBoost", "Neural Network"])):
                    selected_models.append(m)

    st.markdown("---")

    # ── Customer Input Form ──────────────────────────────────────────────────
    st.markdown("### 👤 Customer Details")
    user_data = {}
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        user_data['gender'] = st.selectbox("Gender", CATEGORY_OPTIONS['gender'])
        user_data['SeniorCitizen'] = st.selectbox("Senior Citizen", [0, 1])
        user_data['Partner'] = st.selectbox("Partner", CATEGORY_OPTIONS['Partner'])
        user_data['Dependents'] = st.selectbox("Dependents", CATEGORY_OPTIONS['Dependents'])
        user_data['tenure'] = st.slider("Tenure (months)", 0, 72, 12)
    with c2:
        user_data['PhoneService'] = st.selectbox("Phone Service", CATEGORY_OPTIONS['PhoneService'])
        user_data['MultipleLines'] = st.selectbox("Multiple Lines", CATEGORY_OPTIONS['MultipleLines'])
        user_data['InternetService'] = st.selectbox("Internet Service", CATEGORY_OPTIONS['InternetService'])
        user_data['OnlineSecurity'] = st.selectbox("Online Security", CATEGORY_OPTIONS['OnlineSecurity'])
        user_data['OnlineBackup'] = st.selectbox("Online Backup", CATEGORY_OPTIONS['OnlineBackup'])
    with c3:
        user_data['DeviceProtection'] = st.selectbox("Device Protection", CATEGORY_OPTIONS['DeviceProtection'])
        user_data['TechSupport'] = st.selectbox("Tech Support", CATEGORY_OPTIONS['TechSupport'])
        user_data['StreamingTV'] = st.selectbox("Streaming TV", CATEGORY_OPTIONS['StreamingTV'])
        user_data['StreamingMovies'] = st.selectbox("Streaming Movies", CATEGORY_OPTIONS['StreamingMovies'])
    with c4:
        user_data['Contract'] = st.selectbox("Contract", CATEGORY_OPTIONS['Contract'])
        user_data['PaperlessBilling'] = st.selectbox("Paperless Billing", CATEGORY_OPTIONS['PaperlessBilling'])
        user_data['PaymentMethod'] = st.selectbox("Payment Method", CATEGORY_OPTIONS['PaymentMethod'])
        user_data['MonthlyCharges'] = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0, 0.5)
        user_data['TotalCharges'] = st.slider("Total Charges ($)", 18.0, 9000.0, 1500.0, 10.0)

    st.markdown("---")

    # ── Run Predictions ──────────────────────────────────────────────────────
    if st.button("🚀 Run Prediction", use_container_width=True, type="primary"):
        if not inference_ready:
            st.error(
                "Preprocessor not available. Add artifacts/preprocessor.joblib and model artifacts, "
                "or include data/raw/train.csv to build preprocessing on startup."
            )
            st.stop()

        if not selected_models:
            st.warning("⚠️ Please select at least one model.")
        else:
            X_input = encode_user_input(user_data, label_encoders, scaler)
            results = {}
            progress = st.progress(0, text="Loading models and predicting...")

            for idx, mname in enumerate(selected_models):
                progress.progress((idx + 1) / len(selected_models),
                                  text=f"Running {mname}...")

                model = load_model_artifact(mname, ARTIFACTS_DIR)
                metrics = saved_metrics.get(mname, {})

                if model is None:
                    if not data_ready:
                        continue
                    model, metrics, _ = train_model(mname, X_train, X_test, y_train, y_test)

                if model is not None and "error" not in metrics:
                    if not metrics:
                        metrics = {"Accuracy": 0.0, "Precision": 0.0, "Recall": 0.0, "F1 Score": 0.0}
                    pred, conf = predict_single(model, mname, X_input)
                    results[mname] = {"prediction": pred, "confidence": conf, "metrics": metrics}
                    try:
                        log_prediction(mname, pred, conf, user_data)
                    except Exception:
                        pass
                    append_prediction_history(mname, pred, conf, user_data)

            progress.empty()

            if not results:
                st.error("No models could be trained. Check dependencies.")
            else:
                # Find best model by F1
                best_name = max(results, key=lambda k: results[k]["metrics"].get("F1 Score", 0.0))

                # ── Best Model Highlight ─────────────────────────────────
                st.markdown("### 🏆 Best Model Result")
                best = results[best_name]
                pred_class = "pred-churn" if best["prediction"] == 1 else "pred-no-churn"
                pred_text = "⚠️ WILL CHURN" if best["prediction"] == 1 else "✅ WILL NOT CHURN"
                pred_color = "#ef4444" if best["prediction"] == 1 else "#10b981"

                bc1, bc2, bc3 = st.columns([2, 1, 1])
                with bc1:
                    st.markdown(f'<div class="{pred_class}">'
                                f'<span class="best-badge">BEST — {best_name}</span><br><br>'
                                f'<span style="font-size:2rem;font-weight:800;color:{pred_color}">'
                                f'{pred_text}</span><br>'
                                f'<span style="color:#94a3b8">Confidence: {best["confidence"]*100:.1f}%</span>'
                                f'</div>', unsafe_allow_html=True)
                with bc2:
                    st.markdown(f'<div class="metric-card"><div class="metric-value">'
                                f'{best["metrics"]["Accuracy"]}%</div>'
                                f'<div class="metric-label">Accuracy</div></div>', unsafe_allow_html=True)
                with bc3:
                    st.markdown(f'<div class="metric-card"><div class="metric-value">'
                                f'{best["metrics"]["F1 Score"]}%</div>'
                                f'<div class="metric-label">F1 Score</div></div>', unsafe_allow_html=True)

                # ── All Model Results ────────────────────────────────────
                st.markdown("### 📋 All Model Results")
                for mname, res in sorted(results.items(),
                                          key=lambda x: x[1]["metrics"].get("F1 Score", 0.0), reverse=True):
                    info = MODEL_REGISTRY[mname]
                    cat = info["category"]
                    is_best = mname == best_name
                    icon = info["icon"]
                    pred_emoji = "⚠️ Churn" if res["prediction"] == 1 else "✅ Retained"

                    with st.expander(f'{icon} {mname} — {pred_emoji} '
                                     f'{"🏆" if is_best else ""} '
                                     f'(F1: {res["metrics"].get("F1 Score", 0.0)}%)', expanded=is_best):
                        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
                        mc1.metric("Prediction", pred_emoji)
                        mc2.metric("Accuracy", f'{res["metrics"].get("Accuracy", 0.0)}%')
                        mc3.metric("Precision", f'{res["metrics"].get("Precision", 0.0)}%')
                        mc4.metric("Recall", f'{res["metrics"].get("Recall", 0.0)}%')
                        mc5.metric("F1 Score", f'{res["metrics"].get("F1 Score", 0.0)}%')
                        st.caption(f'Category: {cat} · Confidence: {res["confidence"]*100:.1f}%')

                # ── Comparison Chart ─────────────────────────────────────
                st.markdown("### 📊 Model Comparison")
                model_names = list(results.keys())
                metrics_list = ["Accuracy", "Precision", "Recall", "F1 Score"]
                colors = ['#6366f1', '#ec4899', '#f59e0b', '#10b981']

                fig = go.Figure()
                for mi, metric in enumerate(metrics_list):
                    vals = [results[m]["metrics"].get(metric, 0.0) for m in model_names]
                    fig.add_trace(go.Bar(name=metric, x=model_names, y=vals,
                                         marker_color=colors[mi], opacity=0.85))
                fig.update_layout(barmode='group', paper_bgcolor='rgba(0,0,0,0)',
                                  plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#e2e8f0'),
                                  height=450,
                                  xaxis=dict(tickangle=-45, gridcolor='rgba(255,255,255,0.05)'),
                                  yaxis=dict(title="Score (%)", gridcolor='rgba(255,255,255,0.05)',
                                             range=[0, 105]),
                                  legend=dict(font=dict(color='#e2e8f0'), orientation='h',
                                              yanchor='bottom', y=1.02, xanchor='right', x=1))
                st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# 📊  COMPARE MODELS PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Compare Models":
    st.markdown('<p class="hero-title">📊 Model Comparison</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">View saved training metrics from the latest training run</p>',
                unsafe_allow_html=True)

    cat_filter = st.multiselect("Filter by Category",
                                 ["Supervised", "Deep Learning", "Unsupervised", "Semi-Supervised"],
                                 default=["Supervised", "Deep Learning", "Unsupervised", "Semi-Supervised"])
    models_to_run = [n for n, info in MODEL_REGISTRY.items() if info["category"] in cat_filter]

    if st.button("📂 Load Saved Leaderboard", use_container_width=True, type="primary"):
        all_results = {}
        if not leaderboard_df.empty:
            filtered = leaderboard_df[leaderboard_df["Model"].isin(models_to_run)]
            for _, row in filtered.iterrows():
                all_results[row["Model"]] = {
                    "Accuracy": float(row["Accuracy"]),
                    "Precision": float(row["Precision"]),
                    "Recall": float(row["Recall"]),
                    "F1 Score": float(row["F1 Score"]),
                }
        else:
            st.info("No saved leaderboard found. Train models first using scripts/train_all_models.py")

        if all_results:
            # Results table
            df_results = pd.DataFrame(all_results).T
            df_results.index.name = "Model"
            df_results = df_results.sort_values("F1 Score", ascending=False)

            best = df_results.index[0]
            st.markdown(f'### 🏆 Best Model: <span class="best-badge">{best}</span>',
                        unsafe_allow_html=True)

            st.markdown("### 📋 Results Table")
            st.dataframe(df_results.style.highlight_max(axis=0, color='rgba(99,102,241,0.3)')
                         .format("{:.2f}%"), use_container_width=True)

            # Bar chart
            fig = go.Figure()
            colors = ['#6366f1', '#ec4899', '#f59e0b', '#10b981']
            for mi, metric in enumerate(["Accuracy", "Precision", "Recall", "F1 Score"]):
                fig.add_trace(go.Bar(name=metric, x=df_results.index,
                                      y=df_results[metric], marker_color=colors[mi], opacity=0.85))
            fig.update_layout(barmode='group', paper_bgcolor='rgba(0,0,0,0)',
                              plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#e2e8f0'),
                              height=500,
                              xaxis=dict(tickangle=-45, gridcolor='rgba(255,255,255,0.05)'),
                              yaxis=dict(title="Score (%)", gridcolor='rgba(255,255,255,0.05)',
                                         range=[0, 105]),
                              legend=dict(font=dict(color='#e2e8f0'), orientation='h',
                                          yanchor='bottom', y=1.02, xanchor='right', x=1))
            st.plotly_chart(fig, use_container_width=True)

            # Radar chart for top 5
            st.markdown("### 🕸️ Radar Chart — Top 5 Models")
            top5 = df_results.head(5)
            radar_fig = go.Figure()
            radar_colors = ['#6366f1', '#ec4899', '#f59e0b', '#10b981', '#8b5cf6']
            metrics_radar = ["Accuracy", "Precision", "Recall", "F1 Score"]
            for i, (mname, row) in enumerate(top5.iterrows()):
                vals = [row[m] for m in metrics_radar] + [row[metrics_radar[0]]]
                radar_fig.add_trace(go.Scatterpolar(
                    r=vals, theta=metrics_radar + [metrics_radar[0]],
                    fill='toself', name=mname,
                    line=dict(color=radar_colors[i % len(radar_colors)]),
                    opacity=0.7
                ))
            radar_fig.update_layout(
                polar=dict(bgcolor='rgba(0,0,0,0)',
                           radialaxis=dict(visible=True, range=[0, 105],
                                           gridcolor='rgba(255,255,255,0.1)',
                                           color='#94a3b8'),
                           angularaxis=dict(gridcolor='rgba(255,255,255,0.1)',
                                            color='#e2e8f0')),
                paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#e2e8f0'), height=500,
                legend=dict(font=dict(color='#e2e8f0'))
            )
            st.plotly_chart(radar_fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# 🗂️  HISTORY PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🗂️ History":
    st.markdown('<p class="hero-title">🗂️ Prediction History</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">Combined history from Database + JSON store</p>',
                unsafe_allow_html=True)

    hist_limit = st.slider("Records to load", 20, 1000, 200, 20)

    json_rows_raw = read_prediction_history(limit=hist_limit)
    db_rows_raw = []
    try:
        db_rows_raw = fetch_prediction_history(limit=hist_limit)
    except Exception:
        db_rows_raw = []

    json_rows = []
    for row in json_rows_raw:
        json_rows.append(
            {
                "source": "JSON",
                "timestamp": row.get("timestamp"),
                "model_name": row.get("model_name"),
                "prediction": int(row.get("prediction", 0)),
                "confidence": row.get("confidence"),
                "input": row.get("input"),
            }
        )

    db_rows = []
    for row in db_rows_raw:
        timestamp_val = row.get("created_at")
        db_rows.append(
            {
                "source": "DB",
                "timestamp": timestamp_val.isoformat() if hasattr(timestamp_val, "isoformat") else str(timestamp_val),
                "model_name": row.get("model_name"),
                "prediction": int(row.get("prediction", 0)),
                "confidence": row.get("confidence"),
                "input": row.get("input_payload"),
            }
        )

    merged_rows = db_rows + json_rows

    if not merged_rows:
        st.info("No prediction history found yet.")
    else:
        df_hist = pd.DataFrame(merged_rows)
        df_hist["timestamp_dt"] = pd.to_datetime(df_hist["timestamp"], errors="coerce")
        df_hist["prediction_label"] = df_hist["prediction"].map({1: "Churn", 0: "No Churn"}).fillna("Unknown")

        f1, f2, f3 = st.columns(3)
        with f1:
            model_options = ["All"] + sorted(df_hist["model_name"].dropna().unique().tolist())
            model_filter = st.selectbox("Filter by Model", model_options)
        with f2:
            pred_filter = st.selectbox("Filter by Prediction", ["All", "Churn", "No Churn"])
        with f3:
            source_filter = st.selectbox("Filter by Source", ["All", "DB", "JSON"])

        valid_ts = df_hist["timestamp_dt"].dropna()
        date_range = None
        if not valid_ts.empty:
            min_date = valid_ts.dt.date.min()
            max_date = valid_ts.dt.date.max()
            date_range = st.date_input("Filter by Date Range", value=(min_date, max_date))
        else:
            st.info("Date filter disabled because history timestamps are unavailable.")

        filtered = df_hist.copy()
        if model_filter != "All":
            filtered = filtered[filtered["model_name"] == model_filter]
        if pred_filter != "All":
            filtered = filtered[filtered["prediction_label"] == pred_filter]
        if source_filter != "All":
            filtered = filtered[filtered["source"] == source_filter]

        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
            filtered = filtered[
                filtered["timestamp_dt"].dt.date.between(start_date, end_date, inclusive="both")
            ]

        filtered = filtered.sort_values("timestamp_dt", ascending=False)
        display_cols = ["source", "timestamp", "model_name", "prediction_label", "confidence", "input"]

        st.caption(f"Showing {len(filtered)} records after filtering")
        st.dataframe(filtered[display_cols], use_container_width=True)

        export_json = filtered[display_cols].to_dict(orient="records")
        json_blob = json.dumps(export_json, indent=2, default=str)
        st.download_button(
            "Download Filtered History JSON",
            data=json_blob,
            file_name="combined_prediction_history_export.json",
            mime="application/json",
            use_container_width=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# ℹ️  ABOUT PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.markdown('<p class="hero-title">ℹ️ About ChurnGuard AI</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
    <h3>🎯 What is Customer Churn?</h3>
    <p>Customer churn refers to when customers stop doing business with a company.
    Predicting churn helps businesses take proactive measures to retain customers.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
    <h3>🤖 Models Used</h3>
    <table style="width:100%; color:#cbd5e1; border-collapse:collapse;">
    <tr style="border-bottom:1px solid rgba(255,255,255,0.1)">
        <th style="padding:8px;text-align:left;color:#a78bfa;">Category</th>
        <th style="padding:8px;text-align:left;color:#a78bfa;">Models</th>
        <th style="padding:8px;text-align:left;color:#a78bfa;">Count</th>
    </tr>
    <tr style="border-bottom:1px solid rgba(255,255,255,0.05)">
        <td style="padding:8px;">📈 Supervised</td>
        <td style="padding:8px;">Logistic Regression, Decision Tree, Random Forest, Gradient Boosting,
        AdaBoost, XGBoost, LightGBM, SVM, KNN, Naive Bayes, Extra Trees</td>
        <td style="padding:8px;">11</td>
    </tr>
    <tr style="border-bottom:1px solid rgba(255,255,255,0.05)">
        <td style="padding:8px;">🧠 Deep Learning</td>
        <td style="padding:8px;">Neural Network (3-layer), Deep Neural Network (5-layer)</td>
        <td style="padding:8px;">2</td>
    </tr>
    <tr style="border-bottom:1px solid rgba(255,255,255,0.05)">
        <td style="padding:8px;">🔵 Unsupervised</td>
        <td style="padding:8px;">K-Means, DBSCAN, Isolation Forest</td>
        <td style="padding:8px;">3</td>
    </tr>
    <tr>
        <td style="padding:8px;">🏷️ Semi-Supervised</td>
        <td style="padding:8px;">Label Propagation, Label Spreading, Self-Training</td>
        <td style="padding:8px;">3</td>
    </tr>
    </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
    <h3>📊 Features Used (19)</h3>
    <p>gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines,
    InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
    StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
    MonthlyCharges, TotalCharges</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
    <h3>⚙️ How It Works</h3>
    <ol style="color:#cbd5e1;">
    <li>Data is loaded from <code>train.csv</code> and sampled (50K rows) for performance</li>
    <li>Categorical features are label-encoded, numerical features are standardized</li>
    <li>Selected models are trained on 80% data, evaluated on 20%</li>
    <li>Your input is encoded the same way and passed through each model</li>
    <li>The best model (by F1 Score) is highlighted with its prediction</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)
