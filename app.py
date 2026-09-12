"""
Dashboard de KPIs de Negocio con Machine Learning
Dataset: Amazon Sale Report

4 KPIs, cada uno con su propio modelo de ML:
  1. Ingresos por Categoría        -> Regresión Lineal
  2. Tasa de Cancelación de Pedidos -> Árbol de Decisión
  3. Ticket Promedio por Pedido     -> K-Means (Clustering)
  4. Participación de Ingresos      -> Clustering Jerárquico
"""

import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

sns.set_style("whitegrid")

st.set_page_config(page_title="KPIs Amazon Sale Report", layout="wide")

# ---------------------------------------------------------------------------
# 1. Carga de datos
# ---------------------------------------------------------------------------

@st.cache_data
def cargar_datos(archivo):
    firma = archivo.read(2)
    archivo.seek(0)

    if firma == b"PK":  # csv que en realidad es un zip disfrazado
        with zipfile.ZipFile(archivo) as zf:
            nombre_interno = zf.namelist()[0]
            with zf.open(nombre_interno) as f:
                df_raw = pd.read_csv(f, low_memory=False)
    elif archivo.name.lower().endswith((".xlsx", ".xls")):
        df_raw = pd.read_excel(archivo)
    else:
        df_raw = pd.read_csv(archivo, low_memory=False)

    return df_raw


@st.cache_data
def limpiar_datos(df_raw):
    columnas_sin_contenido = [
        "New", "PendingS", "currency", "ship-country", "fulfilled-by", "index",
    ]
    columnas_a_eliminar = [c for c in columnas_sin_contenido if c in df_raw.columns]

    df = df_raw.drop(columns=columnas_a_eliminar)
    df = df.drop_duplicates()

    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
    df["Qty"] = pd.to_numeric(df["Qty"], errors="coerce")
    df["Category"] = df["Category"].fillna("Desconocido")
    df["Status"] = df["Status"].fillna("Desconocido")
    df["Fulfilment"] = df["Fulfilment"].fillna("Desconocido")
    df["ship-state"] = df["ship-state"].fillna("Desconocido")

    df["Es_Cancelado"] = df["Status"].apply(lambda x: 1 if "Cancelled" in str(x) else 0)

    return df, columnas_a_eliminar


st.title("📊 Dashboard de KPIs de Negocio con Machine Learning")
st.caption("Dataset: Amazon Sale Report")

archivo = st.sidebar.file_uploader("Sube tu archivo (.csv o .xlsx)", type=["csv", "xlsx", "xls"])

if archivo is None:
    st.info("👈 Sube el archivo Amazon Sale Report para comenzar.")
    st.stop()

df_raw = cargar_datos(archivo)
df, columnas_eliminadas = limpiar_datos(df_raw)

with st.sidebar:
    st.markdown("### Resumen del dataset")
    st.write(f"Filas originales: **{df_raw.shape[0]:,}**")
    st.write(f"Filas tras limpieza: **{df.shape[0]:,}**")
    st.write(f"Columnas eliminadas: {', '.join(columnas_eliminadas) if columnas_eliminadas else 'ninguna'}")

with st.expander("Ver muestra de los datos limpios"):
    st.dataframe(df.head(20))

tab1, tab2, tab3, tab4 = st.tabs([
    "1️⃣ Ingresos por Categoría",
    "2️⃣ Tasa de Cancelación",
    "3️⃣ Ticket Promedio",
    "4️⃣ Participación de Ingresos",
])

# ---------------------------------------------------------------------------
# KPI 1: Ingresos por Categoría -> Regresión Lineal
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("KPI 1: Ingresos por Categoría de Producto")
    st.markdown("**Modelo:** Regresión Lineal &nbsp;|&nbsp; **Objetivo:** `Amount` &nbsp;|&nbsp; **Predictoras:** `Category`, `Qty`, `Status`")

    df_kpi1 = df.dropna(subset=["Amount", "Qty"]).copy()

    le_cat = LabelEncoder()
    le_status = LabelEncoder()
    df_kpi1["Category_code"] = le_cat.fit_transform(df_kpi1["Category"])
    df_kpi1["Status_code"] = le_status.fit_transform(df_kpi1["Status"])

    X = df_kpi1[["Category_code", "Qty", "Status_code"]]
    y = df_kpi1["Amount"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    modelo_reg = LinearRegression()
    modelo_reg.fit(X_train, y_train)
    y_pred = modelo_reg.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    col1, col2 = st.columns(2)
    col1.metric("R² Score", f"{r2:.4f}")
    col2.metric("RMSE", f"${rmse:,.2f}")

    resumen_categoria = (
        df_kpi1.groupby("Category")["Amount"]
        .agg(Ingresos_Totales="sum", Ingreso_Promedio="mean", N_Pedidos="count")
        .reset_index()
        .sort_values("Ingresos_Totales", ascending=False)
    )
    st.dataframe(resumen_categoria, use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(data=resumen_categoria, x="Category", y="Ingresos_Totales", hue="Category",
                palette="Blues_d", legend=False, ax=ax)
    ax.set_ylabel("Ingresos Totales ($)")
    ax.set_title("Ingresos Totales por Categoría")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

# ---------------------------------------------------------------------------
# KPI 2: Tasa de Cancelación -> Árbol de Decisión
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("KPI 2: Tasa de Cancelación de Pedidos")
    st.markdown(
        "**Modelo:** Árbol de Decisión (Clasificación) &nbsp;|&nbsp; "
        "**Objetivo:** `Es_Cancelado` &nbsp;|&nbsp; "
        "**Predictoras:** `Category`, `Fulfilment`, `ship-state`"
    )
    st.caption("Se excluyen `Qty`/`Amount` porque quedan en 0/nulo DESPUÉS de una cancelación (fuga de datos).")

    df_kpi2 = df.copy()
    for col in ["Category", "Fulfilment", "ship-state"]:
        le = LabelEncoder()
        df_kpi2[col + "_code"] = le.fit_transform(df_kpi2[col].astype(str))

    features = ["Category_code", "Fulfilment_code", "ship-state_code"]
    X = df_kpi2[features]
    y = df_kpi2["Es_Cancelado"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    profundidad = st.slider("Profundidad máxima del árbol", 2, 6, 3)

    arbol = DecisionTreeClassifier(max_depth=profundidad, random_state=42, class_weight="balanced")
    arbol.fit(X_train, y_train)
    y_pred = arbol.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    st.metric("Precisión del modelo (Accuracy)", f"{acc*100:.2f}%")

    reporte = classification_report(
        y_test, y_pred, target_names=["No Cancelado", "Cancelado"], output_dict=True
    )
    st.dataframe(pd.DataFrame(reporte).transpose(), use_container_width=True)

    importancias = pd.DataFrame({
        "Variable": features,
        "Importancia": arbol.feature_importances_,
    }).sort_values("Importancia", ascending=False)
    st.dataframe(importancias, use_container_width=True)

    fig, ax = plt.subplots(figsize=(14, 6))
    plot_tree(arbol, feature_names=features, class_names=["No Cancelado", "Cancelado"],
              filled=True, fontsize=8, ax=ax)
    ax.set_title("Árbol de Decisión: Predicción de Cancelación")
    st.pyplot(fig)

# ---------------------------------------------------------------------------
# KPI 3: Ticket Promedio por Pedido -> K-Means
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("KPI 3: Ticket Promedio por Pedido")
    st.markdown("**Modelo:** K-Means (Clustering) &nbsp;|&nbsp; **Variables:** `Qty`, `Amount` agregados por `Order ID`")

    df_kpi3 = df.dropna(subset=["Amount"]).copy()

    pedidos = df_kpi3.groupby("Order ID").agg(
        Amount=("Amount", "sum"),
        Qty=("Qty", "sum"),
        Category=("Category", lambda x: x.mode()[0] if not x.mode().empty else "Desconocido"),
    ).reset_index()

    n_clusters = st.slider("Número de clusters (segmentos de valor)", 2, 5, 3)

    scaler = StandardScaler()
    escaladas = scaler.fit_transform(pedidos[["Qty", "Amount"]])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    pedidos["Cluster"] = kmeans.fit_predict(escaladas)

    resumen_clusters = pedidos.groupby("Cluster").agg(
        Ticket_Promedio=("Amount", "mean"),
        Cantidad_Promedio=("Qty", "mean"),
        Total_Pedidos=("Order ID", "count"),
    ).reset_index().sort_values("Ticket_Promedio")

    etiquetas = ["Bajo valor", "Valor medio", "Alto valor", "Muy alto valor", "Premium"]
    resumen_clusters["Etiqueta"] = etiquetas[:len(resumen_clusters)]

    st.dataframe(resumen_clusters, use_container_width=True)

    muestra = pedidos.sample(min(3000, len(pedidos)), random_state=42)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.scatterplot(data=muestra, x="Qty", y="Amount", hue="Cluster", palette="viridis", alpha=0.6, ax=ax)
    ax.set_title("Segmentación del Ticket de Compra por Pedido (Amount vs Qty)")
    st.pyplot(fig)

# ---------------------------------------------------------------------------
# KPI 4: Participación de Ingresos -> Clustering Jerárquico
# ---------------------------------------------------------------------------
with tab4:
    st.subheader("KPI 4: Participación de Ingresos por Categoría")
    st.markdown("**Modelo:** Clustering Jerárquico &nbsp;|&nbsp; **Variables:** `Category`, `Amount`, `Qty`, `Status`")

    resumen_cat = df.groupby("Category").agg(
        Total_Ingresos=("Amount", "sum"),
        Ticket_Promedio=("Amount", "mean"),
        Total_Unidades=("Qty", "sum"),
        Total_Pedidos=("Order ID", "nunique"),
        Tasa_Cancelacion=("Es_Cancelado", "mean"),
    ).reset_index()

    resumen_cat["Participacion_%"] = (
        resumen_cat["Total_Ingresos"] / resumen_cat["Total_Ingresos"].sum() * 100
    ).round(2)

    features_cat = resumen_cat[["Total_Ingresos", "Ticket_Promedio", "Total_Unidades", "Total_Pedidos"]]
    scaler = StandardScaler()
    escaladas_cat = scaler.fit_transform(features_cat)

    linked = linkage(escaladas_cat, method="ward")

    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(linked, labels=resumen_cat["Category"].values, orientation="top",
               distance_sort="descending", show_leaf_counts=True, ax=ax)
    ax.set_title("Dendrograma: Categorías de Alta, Media y Baja Contribución")
    ax.set_ylabel("Distancia Euclídea")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    n_niveles = st.slider("Número de niveles de contribución", 2, 4, 3)

    agglo = AgglomerativeClustering(n_clusters=n_niveles)
    resumen_cat["Cluster"] = agglo.fit_predict(escaladas_cat)

    orden = resumen_cat.groupby("Cluster")["Total_Ingresos"].mean().sort_values().index.tolist()
    niveles = ["Baja contribución", "Media contribución", "Alta contribución", "Muy alta contribución"]
    mapa_niveles = {cl: niveles[i] for i, cl in enumerate(orden)}
    resumen_cat["Nivel"] = resumen_cat["Cluster"].map(mapa_niveles)

    st.dataframe(
        resumen_cat.sort_values("Total_Ingresos", ascending=False)[
            ["Category", "Total_Ingresos", "Participacion_%", "Tasa_Cancelacion", "Nivel"]
        ],
        use_container_width=True,
    )
