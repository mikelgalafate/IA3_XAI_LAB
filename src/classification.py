import numpy as np
import pandas as pd
import shap
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score
)
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance

two_class_models = [
    "Linear SVM",
    "RBF SVM",
    "Nearest Neighbors",
    "Decision Tree",
    "Random Forest",
    "MLP",
    "AdaBoost",
    "Naive Bayes"
]


def is_multiclass(y):
    num_classes = len(y.unique())
    if num_classes == 2:
        return False
    elif num_classes > 2:
        return True
    else:
        return None


def get_model_list(multiclass):
    if multiclass:
        return ["OneVsOne", "OneVsRest"]
    elif not multiclass:
        return two_class_models
    else:
        raise Exception


def compute_classification_metrics(y_train, y_test, y_pred):
    """
    Calcula métricas principales para clasificación:
    - Accuracy
    - MAE
    - MSE
    - RMSE
    - Precision
    - Recall
    - Sensibilidad (igual que recall)

    Devuelve un diccionario con todas las métricas.
    """

    # --- Accuracy ---
    acc = accuracy_score(y_test, y_pred)

    # --- Codificación numérica para métricas tipo error ---
    le = LabelEncoder()
    y_all = pd.concat(
        [pd.Series(y_train), pd.Series(y_test)],
        axis=0
    ).astype(str)

    le.fit(y_all)

    # --- Precision / Recall ---
    unique_classes = pd.Series(y_test).unique()
    avg = "binary" if len(unique_classes) == 2 else "macro"

    prec = precision_score(y_test, y_pred, average=avg, zero_division=0)
    recall = recall_score(y_test, y_pred, average=avg, zero_division=0)

    f1 = 2 * prec * recall / (prec + recall)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": recall,
        "f1-score": f1,
        "labels": list(le.classes_)  # útil para la matriz de confusión
    }


MODEL_PARAM_SPECS = {
    # Binario (two_class_models)
    "Linear SVM": {
        "C": {"type": "float", "min": 0.01, "max": 10.0, "default": 1.0, "step": 0.01},
    },
    "RBF SVM": {
        "C": {"type": "float", "min": 0.01, "max": 10.0, "default": 1.0, "step": 0.01},
        "gamma": {"type": "select", "options": ["scale", "auto"], "default": "scale"},
    },
    "Nearest Neighbors": {
        "n_neighbors": {"type": "int", "min": 1, "max": 25, "default": 5, "step": 1},
        "weights": {"type": "select", "options": ["uniform", "distance"], "default": "uniform"},
    },
    "Decision Tree": {
        "max_depth": {"type": "int", "min": 1, "max": 30, "default": 4, "step": 1},
        "criterion": {"type": "select", "options": ["gini", "entropy", "log_loss"], "default": "gini"},
    },
    "Random Forest": {
        "n_estimators": {"type": "int", "min": 10, "max": 500, "default": 200, "step": 10},
        "max_depth": {"type": "int", "min": 1, "max": 30, "default": 10, "step": 1},
        "max_features": {"type": "select", "options": ["sqrt", "log2", None], "default": "sqrt"},
    },
    "MLP": {
        "hidden_layer_sizes": {"type": "select", "options": [(50,), (100,), (50, 50), (100, 50)], "default": (100,)},
        "alpha": {"type": "float", "min": 0.00001, "max": 0.1, "default": 0.0001, "step": 0.00001},
        "max_iter": {"type": "int", "min": 200, "max": 2000, "default": 1000, "step": 50},
    },
    "AdaBoost": {
        "n_estimators": {"type": "int", "min": 10, "max": 500, "default": 100, "step": 10},
        "learning_rate": {"type": "float", "min": 0.01, "max": 2.0, "default": 1.0, "step": 0.01},
    },
    "Naive Bayes": {
        # GaussianNB casi no tiene hiperparámetros "útiles" para UI básica
        "var_smoothing": {"type": "float", "min": 1e-12, "max": 1e-6, "default": 1e-9, "step": 1e-12},
    },
}

def get_param_specs(model_name: str):
    return MODEL_PARAM_SPECS.get(model_name, {})


def get_default_params(model_name: str) -> dict:
    specs = get_param_specs(model_name)
    return {p: meta["default"] for p, meta in specs.items()}


def create_classifier(multiclass, ovo=True, estimator="SVM", classifier="Linear SVM", params=None):
    """
    Crea y devuelve un clasificador scikit-learn.
    - multiclass: bool
    - ovo: bool (True -> OneVsOne, False -> OneVsRest) si multiclass=True
    - estimator: str ("SVM" o "RandomForest") para multiclass
    - classifier: str (nombre del modelo) para binario
    - params: dict con hiperparámetros del modelo (se aplican al estimador correspondiente)
    """
    params = params or {}

    if multiclass:
        # Estimador base para OVO/OVR
        if estimator == "SVM":
            base_estimator = SVC(**params)
        elif estimator == "RandomForest":
            base_estimator = RandomForestClassifier(**params)
        else:
            # fallback razonable
            base_estimator = create_classifier(multiclass=False,
                                               classifier=estimator,
                                               **params)

        # Wrapper multiclass
        if ovo:
            clf = OneVsOneClassifier(base_estimator)
        else:
            clf = OneVsRestClassifier(base_estimator)

    else:
        # Modelos binarios
        if classifier == "Linear SVM":
            # params típicos: C
            clf = SVC(kernel="linear", probability=True, **params)

        elif classifier == "RBF SVM":
            # params típicos: C, gamma
            clf = SVC(kernel="rbf", **params)

        elif classifier == "Nearest Neighbors":
            # params típicos: n_neighbors, weights
            clf = KNeighborsClassifier(**params)

        elif classifier == "Decision Tree":
            # params típicos: max_depth, criterion, min_samples_split...
            clf = DecisionTreeClassifier(**params)

        elif classifier == "Random Forest":
            # params típicos: n_estimators, max_depth, max_features...
            clf = RandomForestClassifier(**params)

        elif classifier == "MLP":
            # params típicos: hidden_layer_sizes, alpha, max_iter...
            clf = MLPClassifier(**params)

        elif classifier == "AdaBoost":
            # params típicos: n_estimators, learning_rate...
            clf = AdaBoostClassifier(**params)

        elif classifier == "Naive Bayes":
            # params típicos: var_smoothing
            clf = GaussianNB(**params)

        else:
            # fallback (por si llega un nombre inesperado)
            clf = SVC(kernel="linear", **params)

    return clf


def render_hyperparams_ui(model_name: str, key_prefix: str = "hp") -> dict:
    specs = get_param_specs(model_name)
    defaults = get_default_params(model_name)

    params = {}
    if not specs:
        st.info("Este modelo no tiene hiperparámetros configurados en la UI.")
        return params

    st.markdown("**Hiperparámetros**")

    for p, meta in specs.items():
        k = f"{key_prefix}_{model_name}_{p}"

        if meta["type"] == "int":
            params[p] = st.number_input(
                p, min_value=int(meta["min"]), max_value=int(meta["max"]),
                value=int(st.session_state.get(k, defaults[p])),
                step=int(meta.get("step", 1)),
                key=k
            )
        elif meta["type"] == "float":
            params[p] = st.number_input(
                p, min_value=float(meta["min"]), max_value=float(meta["max"]),
                value=float(st.session_state.get(k, defaults[p])),
                step=float(meta.get("step", 0.01)),
                format="%.6f",
                key=k
            )
        elif meta["type"] == "select":
            options = meta["options"]
            default_val = st.session_state.get(k, defaults[p])
            # index seguro
            idx = options.index(default_val) if default_val in options else 0
            params[p] = st.selectbox(p, options=options, index=idx, key=k)
        else:
            st.warning(f"Tipo de hiperparámetro no soportado: {meta['type']}")

    return params

def _unwrap_estimator(clf):
    """
    Si clf es OneVsRest / OneVsOne, devuelve info del/los estimadores internos.
    """
    if hasattr(clf, "estimators_") and isinstance(getattr(clf, "estimators_"), (list, tuple)):
        return clf.estimators_
    if hasattr(clf, "estimator"):
        return [clf.estimator]
    return [clf]


def get_model_feature_importance(clf, feature_names):
    """
    Devuelve (values, feature_names) usando feature_importances_ si existe.
    En wrappers multiclass agrega (media) las importancias de los estimadores internos.
    Si no existe, devuelve None.
    """
    estimators = _unwrap_estimator(clf)
    importances = []

    for est in estimators:
        if hasattr(est, "feature_importances_"):
            importances.append(np.asarray(est.feature_importances_, dtype=float))

    if not importances:
        return None

    imp = np.mean(np.vstack(importances), axis=0)
    return imp, list(feature_names)


def get_model_coefficients_importance(clf, feature_names):
    """
    Devuelve (values, feature_names) usando coef_ si existe.
    Agrega como media del valor absoluto (útil para multiclass).
    Si no existe, devuelve None.
    """
    estimators = _unwrap_estimator(clf)
    coefs = []

    for est in estimators:
        if hasattr(est, "coef_"):
            c = np.asarray(est.coef_, dtype=float)
            # puede ser (n_classes, n_features) o (1, n_features)
            c = np.mean(np.abs(c), axis=0)
            coefs.append(c)

    if not coefs:
        return None

    coef_imp = np.mean(np.vstack(coefs), axis=0)
    return coef_imp, list(feature_names)


def get_permutation_importance(clf, X_test, y_test, feature_names, scoring, n_repeats=10, random_state=42):
    """
    Permutation importance (model-agnostic).
    Devuelve (values, feature_names) con importances_mean.
    """
    r = permutation_importance(
        clf, X_test, y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring=scoring
    )
    return r.importances_mean, list(feature_names)


def shapley_values(clf, X_train, X_test):
    if hasattr(clf, "predict_proba"):
        predict_func = clf.predict_proba
    else:
        predict_func = clf.decision_function
    # Crear un explainer de SHAP para el modelo
    if isinstance(clf, RandomForestClassifier):
        explainer = shap.TreeExplainer(clf)  # Para modelos basados en árboles
    elif isinstance(clf, SVC):
        explainer = shap.KernelExplainer(predict_func, X_train)  # Para modelos no basados en árboles
    else:
        try:
            explainer = shap.Explainer(clf, X_train)  # Intenta con el Explainer genérico
        except:
            explainer = shap.Explainer(lambda x: predict_func(x), X_train)

    return explainer(X_test)
