from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import streamlit as st
import shap


def create_regressor(model_name, params=None):
    params = params or {}

    if model_name == "Linear Regression":
        return LinearRegression(**params)
    elif model_name == "Random Forest Regressor":
        return RandomForestRegressor(**params)
    elif model_name == "SVR":
        return SVR(**params)
    elif model_name == "Decision Tree Regressor":
        return DecisionTreeRegressor(**params)
    elif model_name == "MLP Regressor":
        return MLPRegressor(**params)
    return LinearRegression()


REGRESSION_PARAM_SPECS = {
    "Linear Regression": {},
    "Random Forest Regressor": {
        "n_estimators": {"type": "int", "min": 10, "max": 500, "default": 100, "step": 10},
        "max_depth": {"type": "int", "min": 1, "max": 30, "default": 10, "step": 1},
    },
    "SVR": {
        "C": {"type": "float", "min": 0.01, "max": 10.0, "default": 1.0, "step": 0.01},
        "kernel": {"type": "select", "options": ["linear", "rbf", "poly"], "default": "rbf"},
    },
    "Decision Tree Regressor": {
        "max_depth": {"type": "int", "min": 1, "max": 30, "default": 5, "step": 1},
        "criterion": {"type": "select", "options": ["squared_error", "absolute_error"], "default": "squared_error"},
    },
    "MLP Regressor": {
        "hidden_layer_sizes": {"type": "select", "options": [(50,), (100,), (50, 50)], "default": (100,)},
        "max_iter": {"type": "int", "min": 200, "max": 2000, "default": 1000, "step": 50},
    }
}


# Añade estos a tu MODEL_PARAM_SPECS
def get_model_list_reg():
    return REGRESSION_PARAM_SPECS.keys()


def get_param_specs_reg(model_name: str):
    return REGRESSION_PARAM_SPECS.get(model_name, {})


def get_default_params_reg(model_name: str) -> dict:
    specs = get_param_specs_reg(model_name)
    return {p: meta["default"] for p, meta in specs.items()}


def render_hyperparams_ui_reg(model_name: str, key_prefix: str = "hp") -> dict:
    specs = get_param_specs_reg(model_name)
    defaults = get_default_params_reg(model_name)

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


def compute_regression_metrics(y_test, y_pred):
    """
    Calcula métricas principales para regresión:
    - R² (Coeficiente de determinación)
    - MAE (Error absoluto medio)
    - MSE (Error cuadrático medio)
    - RMSE (Raíz del error cuadrático medio)
    - MAPE (Error porcentual absoluto medio)

    Devuelve un diccionario con todas las métricas.
    """

    # --- R² (El equivalente al "Accuracy" en términos de bondad de ajuste) ---
    r2 = r2_score(y_test, y_pred)

    # --- MAE (Error promedio en las mismas unidades que el target) ---
    mae = mean_absolute_error(y_test, y_pred)

    # --- MSE y RMSE (Penalizan más los errores grandes) ---
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    # --- MAPE (Error en porcentaje, muy intuitivo) ---
    # Evitamos división por cero si hay valores 0 en y_test
    y_test_array = np.array(y_test)
    mask = y_test_array != 0
    mape = np.mean(np.abs((y_test_array[mask] - y_pred[mask]) / y_test_array[mask])) * 100

    return {
        "r2": r2,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "mape": mape
    }

def shapley_values_reg(model, X_train, X_test):
    if isinstance(model, (RandomForestRegressor, DecisionTreeRegressor)):
        explainer = shap.TreeExplainer(model)

    else:
        try:
            explainer = shap.Explainer(model, X_train)
        except Exception:
            explainer = shap.KernelExplainer(model.predict, X_train)

    return explainer(X_test)
