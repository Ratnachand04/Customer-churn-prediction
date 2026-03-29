"""ML/DL model definitions and training for Customer Churn Prediction."""
import numpy as np
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               AdaBoostClassifier, ExtraTreesClassifier, IsolationForest)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans, DBSCAN
from sklearn.semi_supervised import LabelPropagation, LabelSpreading, SelfTrainingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    HAS_TF = True
except ImportError:
    HAS_TF = False

# ── Model Registry ──────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    # Supervised
    "Logistic Regression": {"category": "Supervised", "icon": "📈"},
    "Decision Tree": {"category": "Supervised", "icon": "🌳"},
    "Random Forest": {"category": "Supervised", "icon": "🌲"},
    "Gradient Boosting": {"category": "Supervised", "icon": "🚀"},
    "AdaBoost": {"category": "Supervised", "icon": "⚡"},
    "XGBoost": {"category": "Supervised", "icon": "🔥"},
    "LightGBM": {"category": "Supervised", "icon": "💡"},
    "SVM": {"category": "Supervised", "icon": "🎯"},
    "KNN": {"category": "Supervised", "icon": "👥"},
    "Naive Bayes": {"category": "Supervised", "icon": "📊"},
    "Extra Trees": {"category": "Supervised", "icon": "🌿"},
    # Deep Learning
    "Neural Network": {"category": "Deep Learning", "icon": "🧠"},
    "Deep Neural Network": {"category": "Deep Learning", "icon": "🤖"},
    # Unsupervised
    "K-Means": {"category": "Unsupervised", "icon": "🔵"},
    "DBSCAN": {"category": "Unsupervised", "icon": "🟣"},
    "Isolation Forest": {"category": "Unsupervised", "icon": "🌐"},
    # Semi-Supervised
    "Label Propagation": {"category": "Semi-Supervised", "icon": "🏷️"},
    "Label Spreading": {"category": "Semi-Supervised", "icon": "📡"},
    "Self-Training": {"category": "Semi-Supervised", "icon": "🔄"},
}


def _build_keras_model(input_dim, deep=False):
    """Build a Keras Sequential model."""
    from tensorflow import keras
    model = keras.Sequential()
    model.add(keras.layers.Dense(128, activation='relu', input_dim=input_dim))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.2))
    if deep:
        model.add(keras.layers.Dense(64, activation='relu'))
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(32, activation='relu'))
        model.add(keras.layers.Dropout(0.1))
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def _get_sklearn_model(name):
    """Return an sklearn-compatible model instance."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42, algorithm='SAMME'),
        "SVM": SVC(probability=True, kernel='rbf', random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "Naive Bayes": GaussianNB(),
        "Extra Trees": ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(n_estimators=100, use_label_encoder=False,
                                           eval_metric='logloss', random_state=42,
                                           verbosity=0)
    if HAS_LGBM:
        models["LightGBM"] = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    return models.get(name)


def _train_unsupervised(name, X_train, y_train, X_test, y_test):
    """Train unsupervised model and return metrics + model."""
    if name == "K-Means":
        model = KMeans(n_clusters=2, random_state=42, n_init=10)
        model.fit(X_train)
        preds_train = model.predict(X_train)
        # Map clusters to labels by majority vote
        from scipy.stats import mode as sp_mode
        mapping = {}
        for c in [0, 1]:
            mask = preds_train == c
            if mask.sum() > 0:
                mapping[c] = int(np.round(np.mean(y_train[mask])))
            else:
                mapping[c] = 0
        model._mapping = mapping
        y_pred = np.array([mapping.get(p, 0) for p in model.predict(X_test)])
    elif name == "DBSCAN":
        model = DBSCAN(eps=1.5, min_samples=5)
        labels = model.fit_predict(X_train)
        # Simple: noise = churn
        core_label = 0 if np.mean(y_train[labels != -1]) < 0.5 else 1
        noise_label = 1 - core_label
        test_preds = []
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=5).fit(X_train)
        dists, idxs = nn.kneighbors(X_test)
        for i in range(len(X_test)):
            neighbor_labels = labels[idxs[i]]
            if np.any(neighbor_labels == -1):
                test_preds.append(noise_label)
            else:
                test_preds.append(core_label)
        y_pred = np.array(test_preds)
        model._mapping = {"core": core_label, "noise": noise_label}
        model._nn = nn
        model._train_labels = labels
    elif name == "Isolation Forest":
        model = IsolationForest(n_estimators=100, contamination=0.25, random_state=42)
        model.fit(X_train)
        raw = model.predict(X_test)
        # -1 = outlier = churn(1), 1 = normal = no churn(0)
        y_pred = np.where(raw == -1, 1, 0)
    else:
        return None, {}

    metrics = _calc_metrics(y_test, y_pred)
    return model, metrics, y_pred


def _train_semi_supervised(name, X_train, y_train, X_test, y_test):
    """Train semi-supervised model (mask 70% of labels as -1)."""
    rng = np.random.RandomState(42)
    y_semi = y_train.copy()
    mask = rng.rand(len(y_semi)) < 0.7
    y_semi[mask] = -1

    if name == "Label Propagation":
        model = LabelPropagation(kernel='knn', n_neighbors=7, max_iter=100)
    elif name == "Label Spreading":
        model = LabelSpreading(kernel='knn', n_neighbors=7, max_iter=100)
    elif name == "Self-Training":
        base = LogisticRegression(max_iter=1000, random_state=42)
        model = SelfTrainingClassifier(base, threshold=0.75)
    else:
        return None, {}

    model.fit(X_train, y_semi)
    y_pred = model.predict(X_test)
    metrics = _calc_metrics(y_test, y_pred)
    return model, metrics, y_pred


def _calc_metrics(y_true, y_pred):
    return {
        "Accuracy": round(accuracy_score(y_true, y_pred) * 100, 2),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0) * 100, 2),
        "Recall": round(recall_score(y_true, y_pred, zero_division=0) * 100, 2),
        "F1 Score": round(f1_score(y_true, y_pred, zero_division=0) * 100, 2),
    }


def train_model(name, _X_train, _X_test, _y_train, _y_test):
    """Train a single model by name. Returns (model, metrics_dict, y_pred)."""
    info = MODEL_REGISTRY[name]
    cat = info["category"]

    if cat == "Unsupervised":
        return _train_unsupervised(name, _X_train, _y_train, _X_test, _y_test)

    if cat == "Semi-Supervised":
        return _train_semi_supervised(name, _X_train, _y_train, _X_test, _y_test)

    if cat == "Deep Learning":
        if not HAS_TF:
            return None, {"error": "TensorFlow not installed"}, None
        deep = (name == "Deep Neural Network")
        model = _build_keras_model(_X_train.shape[1], deep=deep)
        model.fit(_X_train, _y_train, epochs=20, batch_size=256, verbose=0,
                  validation_split=0.1)
        y_prob = model.predict(_X_test, verbose=0).flatten()
        y_pred = (y_prob >= 0.5).astype(int)
        metrics = _calc_metrics(_y_test, y_pred)
        return model, metrics, y_pred

    # Supervised sklearn
    model = _get_sklearn_model(name)
    if model is None:
        return None, {"error": f"{name} not available"}, None
    model.fit(_X_train, _y_train)
    y_pred = model.predict(_X_test)
    metrics = _calc_metrics(_y_test, y_pred)
    return model, metrics, y_pred


def _safe_name(name):
    return name.lower().replace("-", "_").replace(" ", "_")


def save_model_artifact(model, name, artifacts_dir):
    """Save trained model to disk using a format based on model category."""
    category = MODEL_REGISTRY[name]["category"]
    out_dir = Path(artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe_name(name)

    if category == "Deep Learning":
        model_path = out_dir / f"{stem}.keras"
        model.save(model_path)
    else:
        model_path = out_dir / f"{stem}.joblib"
        joblib.dump(model, model_path)
    return str(model_path)


def load_model_artifact(name, artifacts_dir):
    """Load a previously trained model from disk."""
    category = MODEL_REGISTRY[name]["category"]
    stem = _safe_name(name)
    base = Path(artifacts_dir)

    if category == "Deep Learning":
        model_path = base / f"{stem}.keras"
        if not model_path.exists() or not HAS_TF:
            return None
        from tensorflow import keras
        return keras.models.load_model(model_path)

    model_path = base / f"{stem}.joblib"
    if not model_path.exists():
        return None
    return joblib.load(model_path)


def predict_single(model, name, X_input):
    """Predict on a single encoded input. Returns (prediction, confidence)."""
    info = MODEL_REGISTRY[name]
    cat = info["category"]

    if cat == "Deep Learning" and HAS_TF:
        prob = model.predict(X_input, verbose=0).flatten()[0]
        return int(prob >= 0.5), float(max(prob, 1 - prob))

    if cat == "Unsupervised":
        if name == "K-Means":
            cluster = model.predict(X_input)[0]
            pred = int(getattr(model, "_mapping", {}).get(cluster, 0))
            return pred, 0.60
        elif name == "Isolation Forest":
            raw = model.predict(X_input)[0]
            pred = 1 if raw == -1 else 0
            score = abs(model.decision_function(X_input)[0])
            return pred, min(float(0.5 + score), 0.99)
        elif name == "DBSCAN":
            dists, idxs = model._nn.kneighbors(X_input)
            neighbor_labels = model._train_labels[idxs[0]]
            pred = model._mapping["noise"] if np.any(neighbor_labels == -1) else model._mapping["core"]
            return pred, 0.55
        return 0, 0.5

    # Sklearn supervised / semi-supervised
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X_input)[0]
        pred = int(np.argmax(proba))
        conf = float(np.max(proba))
        return pred, conf
    else:
        pred = int(model.predict(X_input)[0])
        return pred, 0.70
