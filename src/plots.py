import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.inspection import PartialDependenceDisplay
from sklearn.inspection import partial_dependence
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def fig_univar_hist_box(x: pd.Series, col: str, bins="auto"):
    """
    Devuelve una figura con histograma + boxplot para una serie numérica.
    """
    fig, axes = plt.subplots(
        1, 2,
        figsize=(12, 4),
        gridspec_kw={"width_ratios": [3, 1]},
        constrained_layout=True  # mejor que tight_layout en muchos casos
    )

    axes[0].hist(x, bins=bins, edgecolor="black", linewidth=1.0)
    axes[0].set_title(f"Histograma: {col}")
    axes[0].set_xlabel(col)
    axes[0].set_ylabel("Frecuencia")

    axes[1].boxplot(x, vert=True)
    axes[1].set_title(f"Boxplot: {col}")
    axes[1].set_xticks([])

    return fig


def fig_bivar_scatter(tmp, feature: str, target: str):
    """
    Devuelve una figura con histograma + boxplot para una serie numérica.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(tmp[feature], tmp[target])
    ax.set_title(f"{feature} vs {target}")
    ax.set_xlabel(feature)
    ax.set_ylabel(target)
    plt.tight_layout()
    return fig


def fig_matrix_corr(n: int, corr_plot, method: str, st):
    # Ajuste de tamaño según número de variables (para que no se vea enano)
    fig_w = min(16, 1.2 + 0.6 * n)
    fig_h = min(12, 1.2 + 0.6 * n)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(corr_plot, vmin=-1, vmax=1, cmap="Blues")  # tonos azules

    # Ticks
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr_plot.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr_plot.index)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"Correlación ({method})")

    # Anotaciones por celda (si el tamaño no es enorme)
    # (más de ~25 columnas suele ser ilegible, así que lo limitamos)
    if n <= 25:
        for i in range(n):
            for j in range(n):
                val = corr_plot.iloc[i, j]
                if pd.notna(val):
                    ax.text(j, i, format(val, ".2f"), ha="center", va="center", fontsize=8)
    else:
        st.info("Hay muchas variables (>25). Se muestran sin anotaciones para mejorar legibilidad.")

    ax.set_title(f"Matriz de correlación ({method})")
    plt.tight_layout()
    
    return fig


# def class_balance_hist(y):
#     fig, ax = plt.subplots(figsize=(7, 2))
#     ax.hist(y, bins="auto", edgecolor="black", linewidth=1.0)

#     return fig

def class_balance_hist(y):
    """
    Barplot con el conteo de cada clase (más intuitivo que histograma para categóricas).
    Mantengo el nombre para no tocar imports en app.py.
    """
    counts = pd.Series(y).value_counts(dropna=False)

    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.bar(counts.index.astype(str), counts.values, edgecolor="black", linewidth=1.0)

    ax.set_title("Conteo por clase")
    ax.set_xlabel("Clase")
    ax.set_ylabel("Frecuencia")

    # Etiquetas encima de las barras (opcional, pero útil)
    for i, v in enumerate(counts.values):
        ax.text(i, v, str(int(v)), ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    return fig


def fig_confusion_matrix(y_true, y_pred, labels=None, normalize=None, title="Matriz de confusión"):
    """
    normalize: None | 'true' | 'pred' | 'all'
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)

    fig, ax = plt.subplots(figsize=(7, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format=".2f" if normalize else "d")
    ax.set_title(title)
    plt.tight_layout()
    return fig


def fig_global_importance_bar(values, feature_names, title="Importancia global", top_n=20):
    """
    Barplot horizontal con las importancias (ordenadas). Muestra top_n.
    """
    import numpy as np

    values = np.asarray(values, dtype=float)
    feature_names = np.asarray(feature_names, dtype=str)

    # ordenar por valor absoluto (muy útil para coeficientes)
    order = np.argsort(np.abs(values))[::-1]
    order = order[: min(top_n, len(order))]

    vals = values[order]
    names = feature_names[order]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(range(len(vals))[::-1], vals)  # invertido para que el más importante arriba
    ax.set_yticks(range(len(vals))[::-1])
    ax.set_yticklabels(names)
    ax.set_title(title)
    ax.set_xlabel("Importancia")
    plt.tight_layout()
    return fig


def fig_pdp_ice(estimator, X, features, kind="average", target=None):
    """
    kind: 'average' (PDP) | 'individual' (ICE)
    features: list[str] o list[int] o list[tuple]
    target: para multiclass (índice de clase) o None
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    PartialDependenceDisplay.from_estimator(
        estimator,
        X,
        features=features,
        kind=kind,
        target=target,
        ax=ax
    )
    plt.tight_layout()
    return fig


def pdp_plot(clf, x, feature_idx, classes, multiclass):
    results = partial_dependence(clf, x, features=[feature_idx])
    feature_values = results['values'][0]
    pdp_values = results['average']

    fig, ax = plt.subplots(figsize=(7, 2.8))
    for i, line in enumerate(pdp_values[classes]):
        label = f"Clase {clf.classes_[classes[i]]}" if multiclass else "Clase positiva"
        ax.plot(feature_values, line, label=label)
    fig.legend(loc='center left', bbox_to_anchor=(1, .75))

    return fig


def lime_plot(clf, x_train, x_test, features, instance=None):
    explainer = LimeTabularExplainer(x_train, feature_names=features, class_names=clf.classes_,
                                     discretize_continuous=False)
    # Explicar la predicción de la clase en una muestra de X_test
    # Elegimos una instancia aleatoria del conjunto de test
    if instance is None:
        instance = np.random.randint(0, x_test.shape[0])
    sample = np.array(x_test)[instance]  # Reshape para la instancia individual
    if hasattr(clf, "predict_proba"):
        predict_func = clf.predict_proba
    else:
        predict_func = clf.decision_function

    # Explicación de LIME para la
    explanation = explainer.explain_instance(sample, predict_func, num_features=len(features),
                                             top_labels=1)

    # Extraer las explicaciones de las características ordenadas de mayor a menor
    feature_importance = explanation.as_list(explanation.top_labels[0])[::-1]

    # Graficar las características más importantes de la clase específica
    features, importances = zip(*feature_importance)
    # Crear una figura para el gráfico
    plt.figure(figsize=(8, 6))
    fig, ax = plt.subplots(figsize=(7, 2.8))
    plt.xticks(rotation=45)
    ax.barh(features, np.array(importances), label=f'Clase {clf.classes_[explanation.top_labels[0]]}')

    # Personalizar gráfico
    ax.set_ylabel('Características')
    ax.set_xlabel('Valor LIME')
    ax.set_title(f"Explicaciones LIME instancia {instance}")
    ax.legend()
    return fig


def shapley_importance(clf, shap_values, features):
    # Crear una figura para la importancia global de las características
    plt.figure(figsize=(8, 6))
    fig, ax = plt.subplots(figsize=(7, 2.8))

    importances = np.zeros((len(features), len(clf.classes_)))
    acumulado = np.zeros(len(features))

    # Iterar sobre cada etiqueta (clase) y calcular la importancia de Shapley
    for i, class_name in enumerate(clf.classes_):
        print(i)
        print(shap_values)
        # Obtener valores absolutos medios de SHAP para la clase actual
        shap_importance = np.abs(shap_values[:, :, i].values).mean(axis=0)
        importances[:, i] = shap_importance

    for i in range(len(clf.classes_)):
        ax.barh(features, importances[:, i], left=acumulado, label=f'Clase {clf.classes_[i]}')
        acumulado += importances[:, i]  # Acumular para la siguiente clase

    plt.ylabel('Índice de la Característica')
    plt.xlabel('Importancia SHAP Media')
    plt.title('Importancia SHAP por Característica y Clase')
    plt.legend()

    return fig


def shapley_summary(clf, shap_values, features, class_name, X_test):
    class_idx = np.where(clf.classes_ == class_name)[0][0]
    fig = plt.figure()
    shap.summary_plot(shap_values[:, :, class_idx],
                      X_test,
                      feature_names=[f'Feature {feat}' for feat in features],
                      show=False)
    plt.title(f"Shapley summary for class {class_name}")

    return fig


def shapley_dependence(clf, shap_values, features, feature_name, class_name, X_test):
    feature_idx = np.where(features == feature_name)[0][0]
    class_idx = np.where(clf.classes_ == class_name)[0][0]

    feature_name = features[feature_idx]

    fig, ax = plt.subplots()
    shap.dependence_plot(feature_name,
                         shap_values.values[:, :, class_idx],
                         X_test,
                         interaction_index=None,
                         show=False,
                         ax=ax)
    plt.title(f"Shapley dependence for class {class_name}, feature {feature_name}")

    return fig
