## Comando para ejecutar:
# python -m pip install -r requirements.txt
# python -m streamlit run app.py
from src.io import *
from src.plots import *
from src.classification import *
from src.regression import *
import os


def footer(file_path):
    st.markdown("---")
    imagenes = st.columns([3,4,5,5,6,3])
    path = os.path.join(file_path, "img", "footer")
    for img_col, img_path in zip(imagenes, os.listdir(path)):
        img_col.image(os.path.join(path, img_path))

    st.markdown("---")


def stop(file_path):
    footer(file_path)
    st.stop()


#################################################################
################# PARAMETRIZACIÓN ###############################
#################################################################

# Logo de la aplicación web + Título
file_path = os.path.dirname(os.path.abspath(__file__))
icon_path = os.path.join(file_path, "img", "icon.jpeg")

st.set_page_config(page_title="IA3 XAI LAB", page_icon=f"{file_path}/img/icon.jpeg", layout="wide")

col1, col2 = st.columns([1, 10], vertical_alignment="bottom")
with col1:
    st.image(f"{file_path}/img/icon.jpeg", width=100)
with col2:
    st.title("IA3 XAI LAB",)

#################################################################
################# CARGA DE DATOS ################################
#################################################################
st.write("Sube un archivo **CSV** o **Excel** para empezar.")

uploaded = st.file_uploader("📁 Subir archivo", type=["csv", "xlsx", "xls"],
                            on_change=st.session_state.clear)

if uploaded is None:
    st.info("Sube un archivo para continuar.")
    footer(file_path)
    st.stop()

file_type = detect_file_type(uploaded.name)

# Opciones de lectura (en el cuerpo)
st.subheader("Configuración de lectura")

if file_type == "csv":
    c1, c2, c3 = st.columns(3)
    sep = c1.text_input("Separador", value=",", help="Ej: ,  ;  \\t")
    decimal = c2.text_input("Decimal", value=".", help="Ej: .  o  ,")
    encoding = c3.selectbox("Encoding", options=[None, "utf-8", "latin-1", "cp1252"])
    sheet = None
elif file_type == "excel":
    # listar hojas
    xls = pd.ExcelFile(uploaded)
    sheet = st.selectbox("Selecciona hoja", options=xls.sheet_names)
    sep = decimal = encoding = None
else:
    st.error("Formato no soportado. Usa CSV o Excel.")
    footer(file_path)
    st.stop()

# Cargar datos
try:
    if file_type == "csv":
        df = read_csv(uploaded, sep=sep if sep else ",", decimal=decimal if decimal else ".", encoding=encoding)
    else:
        df = read_excel(uploaded, sheet_name=sheet)
except Exception as e:
    st.error("No se pudo leer el archivo. Revisa las opciones.")
    st.exception(e)
    footer(file_path)
    st.stop()

if df.empty:
    st.warning("El archivo se cargó pero está vacío.")
    footer(file_path)
    st.stop()

problem_type = st.radio("Selecciona el tipo de problema",
                        ["Clasificacion", "Regresion"],
                        horizontal=True,
                        key="problem_type",
                        index=None)

while problem_type is None:
    st.info("Selecciona el tipo de problema para continuar")
    footer(file_path)
    st.stop()

datos, model_tab = st.tabs(["Datos", problem_type])

with datos:
    try:
        #################################################################
        ################# RESUME DATASET ################################
        #################################################################

        st.subheader("Resumen del dataset")

        left, right = st.columns([1, 3])  # izquierda más ancha

        with left:
            st.metric("**Filas**", f"{df.shape[0]:,}".replace(",", "."))
            st.metric("**Columnas**", f"{df.shape[1]:,}".replace(",", "."))
            st.metric("**Nulos (total)**", f"{int(df.isna().sum().sum()):,}".replace(",", "."))

        with right:
            st.markdown("**Cabecera del dataset**")
            st.dataframe(df.head(10), use_container_width=True)

        st.subheader("Estadística descriptiva")
        left, right = st.columns([1, 2])

        with left:
            st.markdown("**Variables del dataset**")
            vars_df = df.dtypes.reset_index().rename(columns={"index": "Variable", 0: "Tipo"})
            vars_df["Tipo"] = vars_df["Tipo"].astype(str)
            st.dataframe(vars_df, height=200, use_container_width=True)


        with right:
            st.markdown("**Descripciónde variables del dataset**")
            desc = df.describe(include="all")
            st.dataframe(desc, use_container_width=True)

        #################################################################
        ################# ANÁLISIS UNIVARIANTE ##########################
        #################################################################

        st.subheader("Análisis univariante")

        # --- Selector de variable ---
        col_sel_1, col_sel_2 = st.columns([3, 1])

        with col_sel_1:
            col = st.selectbox("Selecciona una variable", options=list(df.columns))

        with col_sel_2:
            bins_opt = st.selectbox("Bins (histograma)", options=["auto", 10, 20, 30, 50, 100], index=0)

        # --- Preparar serie numérica ---
        s = df[col]

        # Si no es numérica, intentamos convertirla
        if not pd.api.types.is_numeric_dtype(s):
            s_num = pd.to_numeric(s, errors="coerce")
        else:
            s_num = s

        x = s_num.dropna()

        # --- Si no hay datos numéricos válidos, avisar ---
        if x.empty:
            st.warning(f"La columna **{col}** no tiene valores numéricos válidos (todo NaN o no convertible).")
        else:
            # --- Layout: métricas arriba, plots abajo ---
            st.markdown(f"### Resultados para: **{col}**")

            # Métricas descriptivas (tipo describe) en formato tabla
            desc = x.describe()
            st.dataframe(desc.to_frame().T, use_container_width=True)

            # --- Histograma + Boxplot (matplotlib) ---
            fig = fig_univar_hist_box(x, col, bins=bins_opt)
            st.pyplot(fig, clear_figure=True)

        #################################################################
        ################# ANÁLISIS BIVARIANTE ###########################
        #################################################################

        st.subheader("Análisis bivariante")
        # --- Selección de variables ---
        b1, b2 = st.columns([1, 1])

        with b1:
            feature = st.selectbox("Feature (X)", options=list(df.columns), key="bi_feature")

        with b2:
            # por defecto intenta seleccionar otra diferente
            default_target_idx = 0 if feature != df.columns[0] else (1 if len(df.columns) > 1 else 0)
            target = st.selectbox("Target (Y)", options=list(df.columns), index=default_target_idx, key="bi_target")

        # --- Preparar datos (forzar a numérico sin tocar df original) ---
        x = pd.to_numeric(df[feature], errors="coerce")
        y = pd.to_numeric(df[target], errors="coerce")

        tmp = pd.DataFrame({feature: x, target: y}).dropna()

        if tmp.empty:
            st.warning("No hay datos numéricos válidos para graficar (todo NaN tras conversión).")
        else:
            # Métricas rápidas
            m1, m2, m3 = st.columns(3)
            m1.metric("Puntos válidos", f"{len(tmp):,}".replace(",", "."))
            m2.metric("Correlación de Pearson", f"{tmp[feature].corr(tmp[target], method='pearson'):.4f}")
            m3.metric("Correlación de Spearman ρ", f"{tmp[feature].corr(tmp[target], method='spearman'):.4f}")

            # --- Scatter plot ---
            fig = fig_bivar_scatter(tmp, feature, target)
            col_left, col_center, col_right = st.columns([1, 2, 1])
            with col_center:
                st.pyplot(fig, clear_figure=True)

        #################################################################
        ################# MATRIZ CORRELACIÓN ############################
        #################################################################

        st.subheader("Matriz de correlación")

        # --- Controles ---
        method = st.selectbox("Método", ["pearson", "spearman", "kendall"], index=0, key="corr_method")

        # --- Seleccionar numéricas ---
        num_df = df.select_dtypes(include=["number"])

        if num_df.shape[1] < 2:
            st.warning("Necesitas al menos **2 columnas numéricas** para calcular una matriz de correlación.")
        else:
            corr = num_df.corr(method=method)

            # Opcional: filtrar por mínimo absoluto (para resaltar relaciones)
            corr_plot = corr

            # Mostrar tabla (útil para copiar/inspeccionar)
            with st.expander("Ver tabla de correlaciones"):
                st.dataframe(corr, use_container_width=True)

            # --- Heatmap ---
            n = corr_plot.shape[0]
            fig = fig_matrix_corr(n, corr_plot, method, st)

            col_left, col_center, col_right = st.columns([1, 2, 1])
            with col_center:
                st.pyplot(fig, clear_figure=True)

    finally:
        footer(file_path)

#############################################
######## CLASIFICACIÓN ######################
#############################################

if problem_type == "Clasificacion":
    with model_tab:
        try:
            # =========================
            # Estado de sesión
            # =========================
            if "trained" not in st.session_state:
                st.session_state.trained = False
                st.session_state.clf = None
                st.session_state.pred = None
                st.session_state.metrics = None
                st.session_state.X_test = None
                st.session_state.y_test = None
                st.session_state.X_train = None
                st.session_state.y_train = None
                st.session_state.model_signature = None  # para saber con qué config se entrenó

            # =========================
            # FORM: Configuración + Entrenamiento
            # =========================

            st.subheader("Análisis sobre variable objetivo")
            df_sample = df.copy()

            target_select, target_histogram = st.columns([1, 2])

            with target_select:
                target_column = st.selectbox(
                    "Target column",
                    options=df_sample.columns,
                    index=len(df_sample.columns) - 1,
                    key="target_select"
                )
                st.markdown("Clases")
                st.dataframe(df_sample[target_column].unique(), use_container_width=True)

            X = df_sample.loc[:, df_sample.columns != target_column]
            y = df_sample[target_column]
            features = X.columns

            # Mostrar balance de clases (barplot)
            with target_histogram:
                fig = class_balance_hist(y)
                st.pyplot(fig, clear_figure=True)

            # Partición train-test
            st.subheader("Preparación del modelo")
            train_data_col, test_data_col, test_size_select = st.columns(3)

            with test_size_select:
                test_size = st.number_input(
                    label="Test size",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.2,
                    format="%.2f",
                    key="size_select"
                )

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(test_size), random_state=42)

            with train_data_col:
                st.metric("**Datos para entrenamiento**", f"{X_train.shape[0]:,}".replace(",", "."))
            with test_data_col:
                st.metric("**Datos para test**", f"{X_test.shape[0]:,}".replace(",", "."))

            if is_multiclass(y_train) is None:
                st.error("No hay datos suficientes de cada clase para entrenar")
                train_clicked = st.form_submit_button("Entrenar modelo", type="primary", disabled=True)
                st.stop()

            if len(pd.Series(y_train).unique()) != len(pd.Series(y).unique()):
                st.warning(
                    "El porcentaje de datos en train es muy bajo. El conjunto no contiene ejemplos de todas las clases;"
                    " algunas clases no se aprenderán."
                )

            multiclass = is_multiclass(y)
            estimator = None

            if multiclass is None:
                st.error("No hay suficientes clases")
                train_clicked = st.form_submit_button("Entrenar modelo", type="primary", disabled=True)
                st.stop()

            elif multiclass:
                estimator_select, model_type_select = st.columns([2, 1])

                with model_type_select:
                    model_type = st.selectbox("Estrategia", get_model_list(multiclass), index=0, key="model")

                # Solo mostrar estimador si es OVO/OVR
                if model_type in ("OneVsOne", "OneVsRest"):
                    with estimator_select:
                        estimator = st.selectbox("Estimador", get_model_list(False), index=0, key="estimator")
            else:
                model_type = st.selectbox("Modelo", get_model_list(multiclass), index=0, key="model")

            # Parametrización del modelo (UI dinámica)
            params = render_hyperparams_ui(model_type if not multiclass else estimator,
                                           key_prefix="hp_bin" if not multiclass else "hp_multi")

            # =========================
            # Entrenamiento (solo cuando se pulsa el submit)
            # =========================
            if st.button("Entrenar modelo", type="primary"):
                with st.spinner("Entrenando modelo..."):
                    clf = create_classifier(
                        multiclass=multiclass,
                        ovo=(model_type == "OneVsOne"),
                        estimator=st.session_state.estimator if multiclass else None,  # si luego añades RF, cambia aquí
                        classifier=model_type,
                        params=params
                    )

                    clf.fit(X_train, y_train)
                    pred = clf.predict(X_test)

                    metrics = compute_classification_metrics(y_train, y_test, pred)

                    # Guardar en sesión para que NO se pierda al tocar XAI
                    st.session_state.trained = True
                    st.session_state.clf = clf
                    st.session_state.pred = pred
                    st.session_state.metrics = metrics
                    st.session_state.X_train = X_train
                    st.session_state.y_train = y_train
                    st.session_state.X_test = X_test
                    st.session_state.y_test = y_test

                    # Firma de configuración (útil para debug)
                    st.session_state.model_signature = {
                        "multiclass": multiclass,
                        "model_type": model_type,
                        "estimator": estimator,
                        "params": params,
                        "test_size": float(test_size),
                        "target": target_column,
                    }

                st.success("Entrenamiento finalizado ✅")

            # =========================
            # Resultados + XAI (se muestran si ya hay entrenamiento guardado)
            # =========================
            if not st.session_state.trained:
                st.info("Configura el modelo y pulsa **Entrenar modelo** para ver resultados y explicabilidad.")
                footer(file_path)
                st.stop()

            # Recuperar del estado (IMPORTANTE: aquí ya NO se vuelve a entrenar)
            clf = st.session_state.clf
            pred = st.session_state.pred
            metrics = st.session_state.metrics
            X_train = st.session_state.X_train
            y_train = st.session_state.y_train
            X_test = st.session_state.X_test
            y_test = st.session_state.y_test

            # =========================
            # Resultados: Precisión del modelo
            # =========================
            with st.expander("PRECISIÓN DEL MODELO", expanded=False):
                # --- FILA 2: métricas + matriz confusión ---
                left, _, right, _ = st.columns([1, 3, 4, 2])

                with left:
                    st.markdown("### Métricas")
                    st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
                    st.metric("Precisión", f"{metrics['precision']:.4f}")
                    st.metric("Recall", f"{metrics['recall']:.4f}")
                    st.metric("F1-score", f"{metrics['f1-score']:.4f}")

                with right:
                    fig_cm = fig_confusion_matrix(
                        y_test.astype(str),
                        pd.Series(pred).astype(str),
                        labels=metrics["labels"]
                    )
                    st.pyplot(fig_cm, clear_figure=True)

            with st.expander("EXPLICABILIDAD LOCAL", expanded=False):
                st.subheader("LIME")
                checker, selector = st.columns([1, 6])
                with checker:
                    st.write(".")
                    random = st.toggle("Instancia random", value=True,
                                       on_change=lambda: setattr(st.session_state,
                                                                 "instance_select",
                                                                 None if random is False else 0))
                with selector:
                    instance = st.number_input(label="Instance",
                                               min_value=0,
                                               max_value=len(X_test) - 1,
                                               key="instance_select",
                                               disabled=random)
                fig = lime_plot(clf, X_train, X_test, features=features, instance=instance)
                _, plot, _ = st.columns([1, 6, 1])
                with plot:
                    st.pyplot(fig, clear_figure=True)
                plt.close("all")

            with st.expander("EXPLICABILIDAD GLOBAL", expanded=False):
                st.subheader("Partial Dependence Plot")
                # Seleccionar la característica para la que calcular PDP
                feature_idx = st.segmented_control("Variable para PDP",
                                                   features,
                                                   selection_mode="single",
                                                   default=None)
                # Seleccionar las clases para las que calcular PDP
                if multiclass:
                    seleccion = st.segmented_control("Clase objetivo", clf.classes_, selection_mode="multi")
                    if len(seleccion) == 0:
                        class_select = None
                    else:
                        class_select = [list(clf.classes_).index(op) for op in seleccion]
                        class_select.sort()
                else:
                    class_select = [0]

                if feature_idx is not None and class_select is not None:
                    try:
                        fig = pdp_plot(clf, X_train, feature_idx, class_select, multiclass)
                    except Exception as e:
                        st.exception(e)

                    _, plot, _ = st.columns([1, 6, 1])
                    with plot:
                        st.pyplot(fig, clear_figure=True)

                st.subheader("SHAP values")
                if not hasattr(st.session_state, "shap_values"):
                    st.session_state.shap_values = None
                if st.session_state.shap_values is None:
                    st.info("El cálculo de los valores SHAP puede llevar mucho rato si el conjunto es muy grande.")

                if st.button(
                        "Calcular valores SHAP" if st.session_state.shap_values is None else "Recalcular valores SHAP"):
                    with st.spinner("Calculando valores SHAP. Si el conjunto es muy grande, podría llevar un rato..."):
                        st.session_state.shap_values = shapley_values(clf, X_train, X_test)
                if st.session_state.shap_values is not None:
                    st.subheader("Shapley Importance")
                    fig = shapley_importance(st.session_state.shap_values, classes=clf.classes_, features=features)

                    _, plot, _ = st.columns([1, 6, 1])
                    with plot:
                        st.pyplot(fig, clear_figure=True)

                    st.subheader("Shapley Summary")
                    class_name_shap_sum = st.segmented_control("Clase para el análisis",
                                                               clf.classes_,
                                                               selection_mode="single",
                                                               default=clf.classes_[0],
                                                               key="class_shap_sum")
                    if class_name_shap_sum is not None:
                        fig = shapley_summary(clf,
                                              st.session_state.shap_values,
                                              features,
                                              class_name_shap_sum,
                                              X_test)
                        _, plot, _ = st.columns([1, 6, 1])
                        with plot:
                            st.pyplot(fig, clear_figure=True)

                    st.subheader("Shapley Dependence")
                    feature_name_shap_dep = st.segmented_control("Variable para shapley dependence",
                                                                 features,
                                                                 selection_mode="single",
                                                                 default=df_sample.columns[0])
                    class_name_shap_dep = st.segmented_control("Clase para el análisis",
                                                               clf.classes_,
                                                               selection_mode="single",
                                                               default=clf.classes_[0],
                                                               key="class_shap_dep")

                    if feature_name_shap_dep is not None and class_name_shap_dep is not None:
                        fig = shapley_dependence(clf,
                                                 st.session_state.shap_values,
                                                 features,
                                                 feature_name_shap_dep,
                                                 class_name_shap_dep,
                                                 X_test)

                        _, plot, _ = st.columns([1, 6, 1])
                        with plot:
                            st.pyplot(fig, clear_figure=True)

            # --- EXPLICABILIDAD GLOBAL ---
            with st.expander("IMPORTANCIA DE VARIABLES", expanded=False):
                st.markdown("Selecciona qué explicación global quieres ver:")

                # OJO: muchos modelos de sklearn requieren X numérico
                X_train_num = X_train.select_dtypes(include=["number"])
                X_test_num = X_test.select_dtypes(include=["number"])

                if X_train_num.shape[1] == 0:
                    st.warning("No hay variables numéricas en X. La explicabilidad global requiere features numéricas.")
                else:
                    if X_train_num.shape[1] != X_train.shape[1]:
                        st.info("Se usará solo la parte numérica de X para la explicabilidad global (hay columnas "
                                "no numéricas).")

                    feature_names = X_train_num.columns.tolist()
                    n_features = len(feature_names)

                    opt = st.radio(
                        "Método",
                        ["Importancia del modelo", "Importancia por permutación", "Coeficientes del modelo"],
                        horizontal=True,
                        key="xai_global_method"
                    )

                    # Slider Top N robusto
                    if n_features == 1:
                        top_n = 1
                        st.info("Solo hay 1 variable numérica disponible; se mostrará esa variable.")
                    else:
                        top_n = st.slider(
                            "Top N variables",
                            min_value=1,
                            max_value=min(50, n_features),
                            value=min(20, n_features),
                            key="xai_top_n"
                        )

                    if opt == "Importancia del modelo":
                        out = get_model_feature_importance(clf, feature_names)
                        if out is None:
                            st.warning("Este modelo no expone `feature_importances_` "
                                       "(solo disponible en árboles/ensembles).")
                        else:
                            values, names = out
                            fig = fig_global_importance_bar(
                                values, names,
                                title="Importancia del modelo (feature_importances_)",
                                top_n=top_n
                            )
                            _, plot, _ = st.columns([1, 6, 1])
                            with plot:
                                st.pyplot(fig, clear_figure=True)

                    elif opt == "Importancia por permutación":
                        n_rep = st.slider(
                            "n_repeats (permutación)",
                            min_value=3,
                            max_value=30,
                            value=10,
                            key="xai_perm_repeats"
                        )
                        values, names = get_permutation_importance(clf, X_test_num, y_test, feature_names, "accuracy",
                                                                   n_repeats=n_rep, random_state=42)
                        fig = fig_global_importance_bar(
                            values, names,
                            title="Importancia por permutación (mean)",
                            top_n=top_n
                        )
                        _, plot, _ = st.columns([1, 6, 1])
                        with plot:
                            st.pyplot(fig, clear_figure=True)

                    else:  # Coeficientes
                        out = get_model_coefficients_importance(clf, feature_names)
                        if out is None:
                            st.warning("Este modelo no expone `coef_` (típico de modelos lineales).")
                        else:
                            values, names = out
                            fig = fig_global_importance_bar(
                                values, names,
                                title="Coeficientes (media |coef|)",
                                top_n=top_n
                            )
                            _, plot, _ = st.columns([1, 6, 1])
                            with plot:
                                st.pyplot(fig, clear_figure=True)
        finally:
            footer(file_path)

#############################################
########### REGRESIÓN #######################
#############################################

elif problem_type == "Regresion":
    with model_tab:
        try:
            if "trained" not in st.session_state:
                st.session_state.trained = False
                st.session_state.clf = None
                st.session_state.pred = None
                st.session_state.metrics = None
                st.session_state.X_test = None
                st.session_state.y_test = None
                st.session_state.X_train = None
                st.session_state.y_train = None
                st.session_state.model_signature = None
            # =========================
            # FORM: Configuración + Entrenamiento
            # =========================
            st.subheader("Análisis sobre variable objetivo")

            df_sample = df.copy()

            target_select, target_histogram = st.columns([1, 2])

            with target_select:
                target_column = st.selectbox(
                    "Target column",
                    options=df_sample.columns,
                    index=len(df_sample.columns) - 1,
                    key="target_select"
                )

            X = df_sample.loc[:, df_sample.columns != target_column]
            y = df_sample[target_column]
            features = X.columns

            # Mostrar balance de clases (barplot)
            with target_histogram:
                fig = distribution_hist(y)
                st.pyplot(fig, clear_figure=True)

            # Partición train-test
            st.subheader("Preparación del modelo")
            train_data_col, test_data_col, test_size_select = st.columns(3)

            with test_size_select:
                test_size = st.number_input(
                    label="Test size",
                    min_value=0.01,
                    max_value=0.99,
                    value=0.2,
                    format="%.2f",
                    key="size_select_reg",
                    on_change=lambda: setattr(st.session_state,
                                              "trained",
                                              False)
                )

            # Realizamos la partición
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(test_size), random_state=42)

            with train_data_col:
                st.metric("**Datos para entrenamiento**", f"{X_train.shape[0]:,}".replace(",", "."))
                st.caption(f"Rango Target: [{y_train.min():.2f}, {y_train.max():.2f}]")

            with test_data_col:
                st.metric("**Datos para test**", f"{X_test.shape[0]:,}".replace(",", "."))
                st.caption(f"Media Target: {y_test.mean():.2f}")

            # Validaciones para Regresión
            if len(y_train) < 5:
                st.error("⚠️ No hay datos suficientes para entrenar un modelo de regresión.")
                st.stop()

            # Verificación de nulos (Crucial en regresión)
            if y.isnull().any():
                st.warning("⚠️ El target contiene valores nulos. Scikit-learn fallará al entrenar.")

            # Selección del tipo de modelo
            model_type = st.selectbox("Modelo", get_model_list_reg(), index=0, key="model_reg",
                                      on_change=lambda: setattr(st.session_state,
                                                                "trained",
                                                                False))
            # Parametrización del modelo (UI dinámica)
            params = render_hyperparams_ui_reg(model_type, key_prefix="hp_reg")

            # =========================
            # Entrenamiento (solo cuando se pulsa el submit)
            # =========================
            if st.button("Entrenar modelo", type="primary"):
                with st.spinner("Entrenando modelo..."):
                    try:
                        clf = create_regressor(model_name=model_type, params=params)
                    except Exception as e:
                        st.exception(e)
                        st.stop()
                    clf.fit(X_train, y_train)
                    pred = clf.predict(X_test)

                    metrics = compute_regression_metrics(y_test, pred)

                    # Guardar en sesión para que NO se pierda al tocar XAI
                    st.session_state.trained = True
                    st.session_state.clf = clf
                    st.session_state.pred = pred
                    st.session_state.metrics = metrics
                    st.session_state.X_train = X_train
                    st.session_state.y_train = y_train
                    st.session_state.X_test = X_test
                    st.session_state.y_test = y_test

                    # Firma de configuración (útil para debug)
                    st.session_state.model_signature = {
                        "model_type": model_type,
                        "params": params,
                        "test_size": float(test_size),
                        "target": target_column,
                    }

                st.success("Entrenamiento finalizado ✅")

            if not st.session_state.trained:
                st.info("Configura el modelo y pulsa **Entrenar modelo** para ver resultados y explicabilidad.")
                st.stop()

            clf = st.session_state.clf
            pred = st.session_state.pred
            metrics = st.session_state.metrics
            X_train = st.session_state.X_train
            y_train = st.session_state.y_train
            X_test = st.session_state.X_test
            y_test = st.session_state.y_test

            # =========================
            # Resultados: Precisión del modelo
            # =========================
            if st.session_state.trained:
                with st.expander("PRECISIÓN DEL MODELO", expanded=False):
                    # --- FILA 2: métricas + gráfico de dispersión ---
                    left, _, right, _ = st.columns([1, 3, 4, 2])

                    with left:
                        st.markdown("### Métricas")
                        # Mostramos las métricas calculadas por compute_regression_metrics
                        st.metric("R² (Bondad de ajuste)", f"{metrics['r2']:.4f}")
                        st.metric("MAE (Error medio)", f"{metrics['mae']:.4f}")
                        st.metric("RMSE", f"{metrics['rmse']:.4f}")
                        st.metric("MAPE (Error %)", f"{metrics['mape']:.2f}%")

                    with right:
                        # El equivalente a la matriz de confusión: Gráfico Real vs Predicho
                        fig_reg = fig_regression_results(y_test, pred)
                        st.pyplot(fig_reg, clear_figure=True)

                with st.expander("EXPLICABILIDAD LOCAL", expanded=False):
                    st.subheader("LIME")
                    checker, selector = st.columns([1, 6])
                    with checker:
                        st.write(".")
                        random = st.toggle("Instancia random", value=True,
                                           on_change=lambda: setattr(st.session_state,
                                                                     "instance_select",
                                                                     None if not random else 0))
                    with selector:
                        instance = st.number_input(label="Instance",
                                                   min_value=0,
                                                   max_value=len(X_test) - 1,
                                                   key="instance_select",
                                                   disabled=random)
                    fig = lime_plot_reg(clf, X_train, X_test, features=features, instance=instance)
                    _, plot, _ = st.columns([1, 6, 1])
                    with plot:
                        st.pyplot(fig, clear_figure=True)
                    plt.close("all")

                with st.expander("EXPLICABILIDAD GLOBAL", expanded=False):
                    st.subheader("Partial Dependence Plot")
                    # Seleccionar la característica para la que calcular PDP
                    feature_idx = st.segmented_control("Variable para PDP",
                                                       features,
                                                       selection_mode="single",
                                                       default=None)

                    if feature_idx is not None:
                        try:
                            fig = pdp_plot_reg(clf, X_train, feature_idx)
                        except Exception as e:
                            st.exception(e)
                            st.stop()

                        _, plot, _ = st.columns([1, 6, 1])
                        with plot:
                            st.pyplot(fig, clear_figure=True)

                    st.subheader("SHAP values")
                    if not hasattr(st.session_state, "shap_values"):
                        st.session_state.shap_values = None
                    if st.session_state.shap_values is None:
                        st.info("El cálculo de los valores SHAP puede llevar mucho rato si el conjunto es muy grande.")

                    if st.button(
                            "Calcular valores SHAP" if st.session_state.shap_values is None else "Recalcular valores SHAP"):
                        with st.spinner("Calculando valores SHAP. Si el conjunto es muy grande,"
                                        " podría llevar un rato..."):
                            st.session_state.shap_values = shapley_values_reg(clf, X_train, X_test)
                    if st.session_state.shap_values is not None:
                        st.subheader("Shapley Importance")
                        fig = shapley_importance_reg(st.session_state.shap_values, features=features)
                        _, plot, _ = st.columns([1, 6, 1])
                        with plot:
                            st.pyplot(fig, clear_figure=True)

                        st.subheader("Shapley Summary")
                        fig = shapley_summary_reg(st.session_state.shap_values, features, X_test)
                        _, plot, _ = st.columns([1, 6, 1])
                        with plot:
                            st.pyplot(fig, clear_figure=True)

                with st.expander("IMPORTANCIA DE VARIABLES", expanded=False):
                    st.markdown("Selecciona qué explicación global quieres ver:")

                    # OJO: muchos modelos de sklearn requieren X numérico
                    X_train_num = X_train.select_dtypes(include=["number"])
                    X_test_num = X_test.select_dtypes(include=["number"])

                    if X_train_num.shape[1] == 0:
                        st.warning("No hay variables numéricas en X. La explicabilidad global requiere features"
                                   " numéricas.")
                    else:
                        if X_train_num.shape[1] != X_train.shape[1]:
                            st.info("Se usará solo la parte numérica de X para la explicabilidad global (hay columnas"
                                    " no numéricas).")

                        feature_names = X_train_num.columns.tolist()
                        n_features = len(feature_names)

                        opt = st.radio(
                            "Método",
                            ["Importancia del modelo", "Importancia por permutación", "Coeficientes del modelo"],
                            horizontal=True,
                            key="xai_global_method"
                        )

                        # Slider Top N robusto
                        if n_features == 1:
                            top_n = 1
                            st.info("Solo hay 1 variable numérica disponible; se mostrará esa variable.")
                        else:
                            top_n = st.slider(
                                "Top N variables",
                                min_value=1,
                                max_value=min(50, n_features),
                                value=min(20, n_features),
                                key="xai_top_n"
                            )

                        if opt == "Importancia del modelo":
                            out = get_model_feature_importance(clf, feature_names)
                            if out is None:
                                st.warning("Este modelo no expone `feature_importances_` "
                                           "(solo disponible en árboles/ensembles).")
                            else:
                                values, names = out
                                fig = fig_global_importance_bar(
                                    values, names,
                                    title="Importancia del modelo (feature_importances_)",
                                    top_n=top_n
                                )
                                _, plot, _ = st.columns([1, 6, 1])
                                with plot:
                                    st.pyplot(fig, clear_figure=True)

                        elif opt == "Importancia por permutación":
                            n_rep = st.slider(
                                "n_repeats (permutación)",
                                min_value=3,
                                max_value=30,
                                value=10,
                                key="xai_perm_repeats"
                            )
                            values, names = get_permutation_importance(clf, X_test_num, y_test, feature_names, "r2",
                                                                       n_repeats=n_rep, random_state=42)
                            fig = fig_global_importance_bar(
                                values, names,
                                title="Importancia por permutación (mean)",
                                top_n=top_n
                            )
                            _, plot, _ = st.columns([1, 6, 1])
                            with plot:
                                st.pyplot(fig, clear_figure=True)

                        else:  # Coeficientes
                            out = get_model_coefficients_importance(clf, feature_names)
                            if out is None:
                                st.warning("Este modelo no expone `coef_` (típico de modelos lineales).")
                            else:
                                values, names = out
                                fig = fig_global_importance_bar(
                                    values, names,
                                    title="Coeficientes (media |coef|)",
                                    top_n=top_n
                                )
                                _, plot, _ = st.columns([1, 6, 1])
                                with plot:
                                    st.pyplot(fig, clear_figure=True)

        finally:
            footer(file_path)
