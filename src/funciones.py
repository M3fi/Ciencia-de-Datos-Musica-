"""
funciones.py
============
Toda la lógica del proyecto de clasificación de géneros musicales.

Los cuadernillos solo orquestan: importan estas funciones, las ejecutan y
muestran/interpretan resultados. Organización del módulo:

    0. Configuración y rutas
    1. Carga y diagnóstico de calidad
    2. Limpieza (operaciones deterministas, sin parámetros aprendidos)
    3. Pruebas estadísticas para el EDA
    4. Visualizaciones del EDA
    5. Desbalanceo de clases
    6. Ingeniería de características (transformadores sklearn)
    7. Preprocesamiento: ColumnTransformer + Pipeline (imblearn)
    8. Modelos, validación cruzada y control de overfitting
    9. Optimización de hiperparámetros (Randomized vs. Optuna/TPE)
   10. Probabilidad condicional y Teorema de Bayes
   11. Evaluación final, calibración y serialización

Regla anti-fuga (data leakage)
------------------------------
* Antes del split solo se aplican operaciones *sin estado* (no aprenden nada
  de los datos): eliminar filas vacías, convertir centinelas a NaN, quitar
  duplicados, eliminar identificadores.
* Todo lo que *aprende parámetros* (imputación, codificación, escalado,
  SMOTE) vive dentro del Pipeline y se ajusta solo con los folds de
  entrenamiento.
"""
from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    cross_validate,
    learning_curve,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
    TargetEncoder,
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# =============================================================================
# 0. CONFIGURACIÓN Y RUTAS
# =============================================================================
SEMILLA: int = 42
OBJETIVO: str = "music_genre"

COLUMNAS_ID: list[str] = ["instance_id", "obtained_date", "track_name"]
COLUMNA_ARTISTA: str = "artist_name"
COLUMNAS_CATEGORICAS: list[str] = ["key", "mode"]
COLUMNAS_NUMERICAS: list[str] = [
    "popularity", "acousticness", "danceability", "duration_ms", "energy",
    "instrumentalness", "liveness", "loudness", "speechiness", "tempo", "valence",
]
# Valores que el dataset usa para disfrazar faltantes.
CENTINELAS: dict[str, list[Any]] = {
    "duration_ms": [-1.0],
    "tempo": ["?"],
    "artist_name": ["empty_field"],
}

ESCALADO_METRICAS: dict[str, str] = {
    "f1_macro": "f1_macro",
    "roc_auc_ovr": "roc_auc_ovr",
    "neg_log_loss": "neg_log_loss",
}

sns.set_theme(style="whitegrid", context="notebook")


@dataclass(frozen=True)
class Rutas:
    """Rutas del repositorio resueltas a partir de su carpeta raíz."""

    raiz: Path

    @property
    def raw(self) -> Path:
        return self.raiz / "datos" / "raw"

    @property
    def processed(self) -> Path:
        return self.raiz / "datos" / "processed"

    @property
    def model_data(self) -> Path:
        return self.raiz / "datos" / "model"

    @property
    def modelo(self) -> Path:
        return self.raiz / "modelo"

    def crear(self) -> "Rutas":
        for p in (self.raw, self.processed, self.model_data, self.modelo):
            p.mkdir(parents=True, exist_ok=True)
        return self


def obtener_rutas(desde: Path | str | None = None) -> Rutas:
    """Busca la raíz del repo (la carpeta que contiene ``src/funciones.py``)
    subiendo desde ``desde`` (por defecto, el directorio actual)."""
    actual = Path(desde or Path.cwd()).resolve()
    for candidato in (actual, *actual.parents):
        if (candidato / "src" / "funciones.py").exists():
            return Rutas(candidato).crear()
    raise FileNotFoundError("No se encontró la raíz del repositorio (src/funciones.py).")


# =============================================================================
# 1. CARGA Y DIAGNÓSTICO DE CALIDAD
# =============================================================================
def cargar_raw(ruta: Path | str) -> pd.DataFrame:
    """Lee el CSV original sin alterar ningún valor (``tempo`` queda como
    texto porque contiene el centinela ``'?'``)."""
    return pd.read_csv(ruta)


def diagnosticar_calidad(df: pd.DataFrame,
                         centinelas: Mapping[str, Sequence[Any]] = CENTINELAS) -> pd.DataFrame:
    """Tabla de calidad por columna: nulos explícitos, centinelas encubiertos,
    faltante real total, cardinalidad y tipo.

    ``isnull()`` solo detecta 5 filas; la tabla muestra que el faltante real es
    ~20 veces mayor porque está codificado como -1, '?' o 'empty_field'.
    """
    n = len(df)
    filas = []
    for col in df.columns:
        nulos = int(df[col].isna().sum())
        cent = int(df[col].isin(centinelas.get(col, [])).sum()) if col in centinelas else 0
        filas.append({
            "columna": col,
            "dtype": str(df[col].dtype),
            "nulos_explicitos": nulos,
            "centinelas": cent,
            "faltante_real": nulos + cent,
            "pct_faltante_real": round(100 * (nulos + cent) / n, 2),
            "n_unicos": int(df[col].nunique(dropna=True)),
        })
    return pd.DataFrame(filas).set_index("columna").sort_values("faltante_real", ascending=False)


def resumen_duplicados(df: pd.DataFrame, claves: Sequence[str] = ("track_name", "artist_name"),
                       objetivo: str = OBJETIVO) -> dict[str, Any]:
    """Cuantifica duplicados exactos y duplicados por canción (track+artista),
    distinguiendo los que tienen **etiquetas contradictorias**.

    Se excluyen artistas faltantes: con artista desconocido, dos títulos iguales
    ("Intro", "Home") no son necesariamente la misma canción.
    """
    claves = list(claves)
    base = df.dropna(subset=claves)
    dup_mask = base.duplicated(claves, keep=False)
    grupos = base[dup_mask].groupby(claves)[objetivo]
    combinaciones = grupos.apply(lambda s: " + ".join(sorted(set(s)))).value_counts()
    return {
        "duplicados_exactos": int(df.duplicated().sum()),
        "duplicados_exactos_sin_id": int(df.drop(columns=["instance_id"], errors="ignore").duplicated().sum()),
        "filas_duplicadas_por_cancion": int(base.duplicated(claves).sum()),
        "grupos_duplicados": int(grupos.ngroups),
        "grupos_con_generos_distintos": int((grupos.nunique() > 1).sum()),
        "combinaciones_mas_frecuentes": combinaciones.head(10),
        "generos_afectados": base[dup_mask][objetivo].value_counts(),
    }


# =============================================================================
# 2. LIMPIEZA (determinista, sin parámetros aprendidos -> segura antes del split)
# =============================================================================
def reemplazar_centinelas(df: pd.DataFrame,
                          centinelas: Mapping[str, Sequence[Any]] = CENTINELAS) -> pd.DataFrame:
    """Convierte los centinelas en NaN y fija tipos. Es una regla fija (no
    aprende nada), por eso puede aplicarse a todo el dataset y también a
    registros nuevos en producción."""
    out = df.copy()
    for col, valores in centinelas.items():
        if col in out.columns:
            out[col] = out[col].mask(out[col].isin(valores))
    if "tempo" in out.columns:
        out["tempo"] = pd.to_numeric(out["tempo"], errors="coerce")
    return out


def deduplicar_canciones(df: pd.DataFrame, claves: Sequence[str] = ("track_name", "artist_name"),
                         semilla: int = SEMILLA) -> pd.DataFrame:
    """Deja una sola fila por canción (track+artista conocido).

    Justificación: la misma canción aparece con géneros distintos (p. ej.
    Hip-Hop y Rap) y con vectores acústicos idénticos. Si se conservan las dos:
      1. una copia puede caer en train y otra en test -> el test deja de medir
         generalización (el modelo ya "vio" ese vector), y
      2. se introduce ruido de etiqueta irreducible (mismo x, distinto y).
    Se elige la etiqueta conservada **al azar con semilla**, no con
    ``keep='first'``: el CSV está ordenado por género y ``'first'`` favorecería
    sistemáticamente a los géneros que aparecen primero en el archivo.
    """
    claves = list(claves)
    con_clave = df.dropna(subset=claves)
    sin_clave = df[df[claves].isna().any(axis=1)]
    con_clave = con_clave.sample(frac=1.0, random_state=semilla).drop_duplicates(claves, keep="first")
    return pd.concat([con_clave, sin_clave]).sort_index()


def limpiar_dataset(df_raw: pd.DataFrame, semilla: int = SEMILLA) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pipeline de limpieza completo. Devuelve el dataset limpio y una
    bitácora con las filas que se pierden en cada paso (trazabilidad).

    Pasos: (1) filas 100 % nulas, (2) centinelas -> NaN, (3) duplicados
    exactos, (4) duplicados por canción con etiqueta contradictoria.
    Los faltantes de ``duration_ms``/``tempo``/``artist_name`` NO se imputan
    aquí: la imputación aprende parámetros y va dentro del Pipeline.
    """
    bitacora: list[dict[str, Any]] = [{"paso": "0. raw", "filas": len(df_raw)}]

    df = df_raw.dropna(how="all")
    bitacora.append({"paso": "1. sin filas 100% nulas", "filas": len(df)})

    df = reemplazar_centinelas(df)
    bitacora.append({"paso": "2. centinelas -> NaN", "filas": len(df)})

    df = df.drop_duplicates()
    sin_id = [c for c in df.columns if c != "instance_id"]
    df = df.drop_duplicates(subset=sin_id)
    bitacora.append({"paso": "3. sin duplicados exactos", "filas": len(df)})

    df = deduplicar_canciones(df, semilla=semilla)
    bitacora.append({"paso": "4. una fila por canción", "filas": len(df)})

    df = df.reset_index(drop=True)
    log = pd.DataFrame(bitacora)
    log["eliminadas"] = (-log["filas"].diff()).fillna(0).astype(int)
    return df, log


def eliminar_identificadores(df: pd.DataFrame, columnas: Sequence[str] = COLUMNAS_ID) -> pd.DataFrame:
    """Quita columnas sin poder predictivo generalizable:
    * ``instance_id``: clave técnica, no es una propiedad de la canción.
    * ``obtained_date``: fecha de scraping (4 valores, ~90 % '4-Apr'), sin
      relación causal con el género; usarla aprendería artefactos de la
      recolección.
    * ``track_name``: texto casi único por fila (cardinalidad ≈ n); como
      categoría solo memoriza. Requeriría NLP y no es objetivo del proyecto.
    """
    return df.drop(columns=[c for c in columnas if c in df.columns])


def preparar_registros_nuevos(df_nuevo: pd.DataFrame) -> pd.DataFrame:
    """Aplica a datos de producción las mismas reglas *sin estado* que se
    aplicaron antes del split (centinelas e identificadores). El resto lo
    resuelve el Pipeline serializado."""
    return eliminar_identificadores(reemplazar_centinelas(df_nuevo))


# =============================================================================
# 3. PRUEBAS ESTADÍSTICAS
# =============================================================================
def prueba_mcar_por_genero(df: pd.DataFrame, columnas: Sequence[str],
                           objetivo: str = OBJETIVO) -> pd.DataFrame:
    """¿El faltante depende del género? Prueba χ² de independencia entre el
    indicador de faltante y la clase.

    H0: P(faltante | género) = P(faltante) para todo género (compatible con
    MCAR respecto a la clase). Un p-valor alto implica que eliminar o imputar
    no distorsiona la distribución de clases y que imputar *por género*
    no es necesario (además de ser imposible en producción).
    """
    filas = []
    for col in columnas:
        tabla = pd.crosstab(df[col].isna(), df[objetivo])
        if tabla.shape[0] < 2:
            continue
        chi2, p, gl, _ = stats.chi2_contingency(tabla)
        filas.append({"variable": col, "pct_faltante": round(100 * df[col].isna().mean(), 2),
                      "chi2": round(chi2, 2), "gl": gl, "p_valor": p,
                      "cramer_v": round(cramer_v(df[col].isna(), df[objetivo]), 4)})
    return pd.DataFrame(filas).set_index("variable")


def prueba_mar_por_variables(df: pd.DataFrame, col_faltante: str,
                             columnas: Sequence[str]) -> pd.DataFrame:
    """¿El faltante de ``col_faltante`` depende de otras variables
    observadas (MAR)? Mann-Whitney U entre filas con y sin faltante, con
    tamaño de efecto r = 1 - 2U/(n1·n2) (correlación biserial de rangos)."""
    mask = df[col_faltante].isna()
    filas = []
    for col in columnas:
        if col == col_faltante:
            continue
        a, b = df.loc[mask, col].dropna(), df.loc[~mask, col].dropna()
        if len(a) < 10 or len(b) < 10:
            continue
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        filas.append({"variable": col, "mediana_faltante": a.median(), "mediana_observado": b.median(),
                      "p_valor": p, "efecto_r": round(1 - 2 * u / (len(a) * len(b)), 4)})
    return pd.DataFrame(filas).set_index("variable").sort_values("efecto_r", key=np.abs, ascending=False)


def cramer_v(x: pd.Series, y: pd.Series) -> float:
    """V de Cramér con corrección de sesgo de Bergsma (2013). 0 = independencia,
    1 = asociación perfecta entre dos variables categóricas."""
    tabla = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(tabla, correction=False)[0]
    n = tabla.to_numpy().sum()
    r, k = tabla.shape
    phi2 = chi2 / n
    phi2c = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    den = min(kc - 1, rc - 1)
    return float(np.sqrt(phi2c / den)) if den > 0 else 0.0


def kruskal_por_genero(df: pd.DataFrame, columnas: Sequence[str],
                       objetivo: str = OBJETIVO) -> pd.DataFrame:
    """Kruskal-Wallis por variable vs. género + tamaño de efecto
    η²_H = (H - k + 1)/(n - k).

    Se usa Kruskal (no ANOVA) porque varias variables son no normales,
    acotadas en [0, 1] y con inflación de ceros (instrumentalness,
    speechiness). Con n ≈ 45 000 todo p-valor será ~0, por eso el ranking se
    hace con el tamaño de efecto: η² > 0.14 = efecto grande (Cohen).
    """
    filas = []
    for col in columnas:
        sub = df[[col, objetivo]].dropna()
        grupos = [g[col].to_numpy() for _, g in sub.groupby(objetivo)]
        h, p = stats.kruskal(*grupos)
        n, k = len(sub), len(grupos)
        filas.append({"variable": col, "H": round(h, 1), "p_valor": p,
                      "eta2_H": round((h - k + 1) / (n - k), 4)})
    out = pd.DataFrame(filas).set_index("variable").sort_values("eta2_H", ascending=False)
    out["magnitud"] = pd.cut(out["eta2_H"], [-np.inf, 0.01, 0.06, 0.14, np.inf],
                             labels=["despreciable", "pequeño", "mediano", "grande"])
    return out


def informacion_mutua(X: pd.DataFrame, y: pd.Series, semilla: int = SEMILLA) -> pd.Series:
    """Información mutua I(X_j; Y) en nats: captura dependencias no lineales
    que la correlación ignora. Solo sobre filas completas (análisis
    descriptivo, no alimenta al modelo)."""
    from sklearn.feature_selection import mutual_info_classif
    datos = pd.concat([X, y], axis=1).dropna()
    mi = mutual_info_classif(datos[X.columns], datos[y.name], random_state=semilla)
    return pd.Series(mi, index=X.columns, name="MI").sort_values(ascending=False)


# =============================================================================
# 4. VISUALIZACIONES DEL EDA
# =============================================================================
def _orden_generos(df: pd.DataFrame, col: str, objetivo: str = OBJETIVO) -> list[str]:
    return df.groupby(objetivo)[col].median().sort_values().index.tolist()


def paleta_generos(generos: Iterable[str]) -> dict[str, tuple]:
    generos = sorted(generos)
    return dict(zip(generos, sns.color_palette("tab10", len(generos))))


def grafico_faltantes(df_raw: pd.DataFrame, centinelas: Mapping[str, Sequence[Any]] = CENTINELAS,
                      objetivo: str = OBJETIVO) -> plt.Figure:
    """Barras de faltante 'aparente' (isnull) vs. 'real' (con centinelas) y
    mapa de calor del % de faltante real por género."""
    diag = diagnosticar_calidad(df_raw, centinelas)
    diag = diag[diag["faltante_real"] > 0]
    limpio = reemplazar_centinelas(df_raw.dropna(how="all"), centinelas)
    pct = limpio.groupby(objetivo)[list(centinelas)].apply(lambda g: g.isna().mean() * 100)

    fig, ax = plt.subplots(1, 2, figsize=(15, 4.8), gridspec_kw={"width_ratios": [1, 1.3]})
    diag[["nulos_explicitos", "faltante_real"]].plot.barh(ax=ax[0], color=["#bbbbbb", "#c0392b"])
    ax[0].set_title("Faltante aparente (isnull) vs. real (con centinelas)")
    ax[0].set_xlabel("Filas")
    sns.heatmap(pct.T, annot=True, fmt=".1f", cmap="Reds", ax=ax[1], cbar_kws={"label": "% faltante"})
    ax[1].set_title("% de faltante real por género (uniforme: no depende de la clase)")
    fig.tight_layout()
    return fig


def grafico_raincloud(df: pd.DataFrame, columnas: Sequence[str], objetivo: str = OBJETIVO,
                      n_cols: int = 2, muestra: int = 4000, semilla: int = SEMILLA) -> plt.Figure:
    """Raincloud plot (media violín + caja + puntos) por género. Muestra
    simultáneamente forma de la distribución, mediana/IQR y densidad real de
    observaciones, que un boxplot solo ocultaría (p. ej. bimodalidad)."""
    n_filas = int(np.ceil(len(columnas) / n_cols))
    fig, axes = plt.subplots(n_filas, n_cols, figsize=(8 * n_cols, 4.6 * n_filas), squeeze=False)
    pal = paleta_generos(df[objetivo].dropna().unique())
    sub = df.sample(min(muestra, len(df)), random_state=semilla)
    for ax, col in zip(axes.ravel(), columnas):
        orden = _orden_generos(df, col, objetivo)
        sns.violinplot(data=df, x=col, y=objetivo, hue=objetivo, order=orden, palette=pal, split=False,
                       inner=None, cut=0, density_norm="width", linewidth=0, ax=ax, legend=False)
        for coll in ax.collections:  # recorta la mitad inferior de cada violín -> "nube"
            for path in coll.get_paths():
                v = path.vertices
                centro = np.mean([v[:, 1].min(), v[:, 1].max()])
                v[:, 1] = np.clip(v[:, 1], -np.inf, centro)
        sns.boxplot(data=df, x=col, y=objetivo, order=orden, width=0.15, showfliers=False,
                    boxprops={"facecolor": "white", "zorder": 3}, ax=ax)
        sns.stripplot(data=sub, x=col, y=objetivo, hue=objetivo, order=orden, palette=pal, size=1.4,
                      alpha=0.35, jitter=0.12, ax=ax, legend=False,
                      native_scale=True)
        for coll in ax.collections[-len(orden):]:
            coll.set_offsets(coll.get_offsets() + np.array([0, 0.28]))
        ax.set_title(f"{col} por género (ordenado por mediana)")
        ax.set_ylabel("")
    for ax in axes.ravel()[len(columnas):]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def grafico_ridge(df: pd.DataFrame, columna: str, objetivo: str = OBJETIVO,
                  solapamiento: float = 0.65) -> plt.Figure:
    """Ridge plot (joyplot): densidades KDE apiladas por género. Permite ver
    de un vistazo qué géneros son separables en una variable y cuáles se
    solapan (p. ej. Rap y Hip-Hop casi idénticos)."""
    orden = _orden_generos(df, columna, objetivo)
    pal = paleta_generos(orden)
    datos = df[[columna, objetivo]].dropna()
    xs = np.linspace(datos[columna].quantile(0.005), datos[columna].quantile(0.995), 400)
    fig, ax = plt.subplots(figsize=(10, 0.75 * len(orden) + 1.5))
    for i, g in enumerate(orden):
        valores = datos.loc[datos[objetivo] == g, columna].to_numpy()
        dens = stats.gaussian_kde(valores)(xs)
        dens = dens / dens.max()
        base = i * (1 - solapamiento)
        ax.fill_between(xs, base, base + dens, color=pal[g], alpha=0.75, zorder=len(orden) - i)
        ax.plot(xs, base + dens, color="black", lw=0.6, zorder=len(orden) - i)
        ax.text(xs[0], base + 0.08, g, ha="right", va="bottom", fontsize=10)
    ax.set_yticks([])
    ax.set_xlabel(columna)
    ax.set_title(f"Ridge plot de {columna} por género")
    ax.margins(x=0.12)
    fig.tight_layout()
    return fig


def grafico_correlacion_jerarquica(df: pd.DataFrame, columnas: Sequence[str],
                                   metodo: Literal["spearman", "pearson"] = "spearman") -> sns.matrix.ClusterGrid:
    """Mapa de calor de correlación de Spearman reordenado por clustering
    jerárquico (distancia 1-|ρ|, enlace promedio). Agrupa bloques de
    variables redundantes (energy–loudness–acousticness) y guía la creación
    de interacciones y el diagnóstico de multicolinealidad."""
    corr = df[list(columnas)].corr(method=metodo)
    dist = squareform(1 - corr.abs().to_numpy(), checks=False)
    enlace = hierarchy.linkage(dist, method="average")
    g = sns.clustermap(corr, row_linkage=enlace, col_linkage=enlace, cmap="RdBu_r", vmin=-1, vmax=1,
                       annot=True, fmt=".2f", annot_kws={"size": 8}, figsize=(11, 10),
                       cbar_kws={"label": f"ρ de {metodo}"})
    g.fig.suptitle("Correlación jerárquica (distancia = 1 - |ρ|)", y=1.02)
    return g


def grafico_bivariado_objetivo(df: pd.DataFrame, columnas: Sequence[str] = ("popularity", "acousticness", "speechiness"),
                               objetivo: str = OBJETIVO) -> plt.Figure:
    """Medias por género con IC 95 % bootstrap para las variables clave del
    análisis bivariado. Las barras de error permiten afirmar diferencias,
    no solo verlas."""
    fig, axes = plt.subplots(1, len(columnas), figsize=(6 * len(columnas), 5), sharey=True)
    for ax, col in zip(np.atleast_1d(axes), columnas):
        orden = df.groupby(objetivo)[col].mean().sort_values().index
        sns.pointplot(data=df, x=col, y=objetivo, order=orden, errorbar=("ci", 95), linestyle="none",
                      color="#2c3e50", ax=ax)
        ax.axvline(df[col].mean(), ls="--", c="#c0392b", lw=1, label="media global")
        ax.set_title(f"{col}: media ± IC95%")
        ax.set_ylabel("")
        ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def grafico_multivariado(df: pd.DataFrame, x: str = "acousticness", y: str = "speechiness",
                         color: str = "popularity", objetivo: str = OBJETIVO,
                         muestra: int = 600, semilla: int = SEMILLA) -> plt.Figure:
    """Facetas por género de x vs. y coloreadas por una tercera variable:
    interacción conjunta de tres variables + la clase."""
    n = min(muestra, int(df[objetivo].value_counts().min()))
    sub = df.groupby(objetivo).sample(n=n, random_state=semilla)
    generos = sorted(sub[objetivo].unique())
    fig, axes = plt.subplots(2, int(np.ceil(len(generos) / 2)), figsize=(20, 8), sharex=True, sharey=True)
    norm = plt.Normalize(df[color].quantile(0.02), df[color].quantile(0.98))
    for ax, g in zip(axes.ravel(), generos):
        d = sub[sub[objetivo] == g]
        sc = ax.scatter(d[x], d[y], c=d[color], cmap="viridis", norm=norm, s=8, alpha=0.7)
        ax.set_title(g)
        ax.set_yscale("log")
    for ax in axes.ravel()[len(generos):]:
        ax.axis("off")
    fig.supxlabel(x)
    fig.supylabel(f"{y} (escala log)")
    fig.colorbar(sc, ax=axes, label=color, shrink=0.8)
    fig.suptitle(f"Interacción {x} × {y} × {color} por género")
    return fig


def proyeccion_2d_3d(df: pd.DataFrame, columnas: Sequence[str], objetivo: str = OBJETIVO,
                     metodo: Literal["pca", "umap"] = "pca", muestra: int = 8000,
                     semilla: int = SEMILLA) -> tuple[plt.Figure, pd.Series | None]:
    """Proyección PCA (lineal) o UMAP (no lineal) en 2D y 3D para evaluar
    separabilidad de clases. Las variables se estandarizan antes (PCA
    maximiza varianza: sin escalar, duration_ms dominaría por su escala).

    Es un análisis **descriptivo**: el escalado se ajusta sobre la muestra
    del EDA y no se reutiliza en el modelo, por lo que no hay fuga.
    Devuelve la figura y, para PCA, la varianza explicada acumulada.
    """
    sub = df[list(columnas) + [objetivo]].dropna()
    sub = sub.sample(min(muestra, len(sub)), random_state=semilla)
    Z = StandardScaler().fit_transform(sub[list(columnas)])
    var_exp = None
    if metodo == "pca":
        pca = PCA(n_components=min(len(columnas), 10), random_state=semilla).fit(Z)
        emb = pca.transform(Z)[:, :3]
        var_exp = pd.Series(np.cumsum(pca.explained_variance_ratio_),
                            index=[f"PC{i + 1}" for i in range(pca.n_components_)], name="varianza_acumulada")
        etiquetas = [f"PC{i + 1} ({pca.explained_variance_ratio_[i]:.0%})" for i in range(3)]
    else:
        import umap  # import diferido: dependencia opcional
        emb = umap.UMAP(n_components=3, n_neighbors=30, min_dist=0.1, random_state=semilla).fit_transform(Z)
        etiquetas = ["UMAP-1", "UMAP-2", "UMAP-3"]

    pal = paleta_generos(sub[objetivo].unique())
    colores = sub[objetivo].map(pal).tolist()
    fig = plt.figure(figsize=(17, 7))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(emb[:, 0], emb[:, 1], c=colores, s=4, alpha=0.5)
    ax1.set_xlabel(etiquetas[0]); ax1.set_ylabel(etiquetas[1])
    ax1.set_title(f"{metodo.upper()} 2D")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.scatter(emb[:, 0], emb[:, 1], emb[:, 2], c=colores, s=3, alpha=0.45)
    ax2.set_xlabel(etiquetas[0]); ax2.set_ylabel(etiquetas[1]); ax2.set_zlabel(etiquetas[2])
    ax2.set_title(f"{metodo.upper()} 3D")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=g) for g, c in pal.items()]
    fig.legend(handles=handles, loc="center right", title="Género")
    fig.tight_layout(rect=(0, 0, 0.9, 1))
    return fig, var_exp


# =============================================================================
# 5. DESBALANCEO DE CLASES
# =============================================================================
def metricas_desbalanceo(y: pd.Series) -> dict[str, float]:
    """Indicadores de desbalanceo:
    * IR (imbalance ratio) = n_max / n_min.
    * Balance de Shannon = H(y)/log(K) ∈ [0, 1]; 1 = perfectamente balanceado.
    """
    conteo = y.value_counts()
    p = conteo / conteo.sum()
    return {"n_clases": int(len(conteo)), "IR": round(conteo.max() / conteo.min(), 3),
            "balance_shannon": round(float(-(p * np.log(p)).sum() / np.log(len(conteo))), 4),
            "clase_mayor": conteo.idxmax(), "clase_menor": conteo.idxmin()}


def comparar_distribuciones(escenarios: Mapping[str, pd.Series]) -> pd.DataFrame:
    """Tabla de conteos por clase para varios escenarios + métricas de desbalanceo."""
    tabla = pd.DataFrame({k: v.value_counts() for k, v in escenarios.items()}).fillna(0).astype(int)
    met = pd.DataFrame({k: metricas_desbalanceo(v) for k, v in escenarios.items()})
    return pd.concat([tabla, met])


def agrupar_subgeneros(y: pd.Series, mapa: Mapping[str, str] | None = None) -> pd.Series:
    """Agrupa géneros musicalmente indistinguibles. Por defecto Rap + Hip-Hop
    -> 'Urbano' (son los que más comparten canciones). Duplica el tamaño de
    una clase y genera desbalanceo IR ≈ 2."""
    mapa = mapa or {"Rap": "Urbano", "Hip-Hop": "Urbano"}
    return y.replace(mapa)


def simular_desbalanceo(X: pd.DataFrame, y: pd.Series, fracciones: Mapping[str, float],
                        semilla: int = SEMILLA) -> tuple[pd.DataFrame, pd.Series]:
    """Submuestrea clases para simular un escenario realista de desbalanceo
    (p. ej. un catálogo con poca música clásica). Úsese solo sobre train."""
    rng = np.random.default_rng(semilla)
    idx = []
    for clase, grupo in y.groupby(y):
        frac = fracciones.get(clase, 1.0)
        n = max(1, int(round(len(grupo) * frac)))
        idx.extend(rng.choice(grupo.index.to_numpy(), size=n, replace=False))
    idx = np.sort(np.array(idx))
    return X.loc[idx], y.loc[idx]


# =============================================================================
# 6. INGENIERÍA DE CARACTERÍSTICAS
# =============================================================================
class IngenieriaMusical(BaseEstimator, TransformerMixin):
    """Crea variables con sentido musical. **Sin estado**: ``fit`` no aprende
    nada, así que no puede filtrar información del test.

    Se ubica *después* de la imputación dentro del Pipeline para que las
    variables derivadas de ``tempo`` y ``duration_ms`` se calculen con
    valores ya imputados (coherentes entre sí), en vez de imputar por
    separado la variable original y su derivada.

    Variables creadas:
    log_duration      log(duration_ms): la duración es log-normal con cola
                      de piezas de >60 min (clásica); el log estabiliza la
                      varianza y reduce la influencia de atípicos.
    energia_acustica  log((energy+ε)/(acousticness+ε)): contraste
                      eléctrico vs. acústico (Rock/Electrónica vs.
                      Clásica/Jazz) en una sola dimensión aditiva.
    pulso_bailable    (tempo/120)·danceability: un tempo alto solo "empuja"
                      a bailar si el ritmo es regular. Captura la
                      interacción que un modelo lineal no ve por separado.
    intensidad        energy·(loudness+60)/60: energía percibida
                      ponderada por sonoridad (normalizada a ~[0, 1]).
    vocalidad         (1-instrumentalness)·speechiness: presencia de voz
                      hablada (Rap/Hip-Hop) vs. instrumental.
    log_speechiness,  log1p(100·x): variables con inflación de ceros y
    log_liveness,     sesgo fuerte; la transformación separa los valores
    log_instrumental  bajos donde está la información discriminante.
    """

    EPS: float = 1e-3

    def __init__(self, eliminar_originales: bool = True) -> None:
        self.eliminar_originales = eliminar_originales

    def fit(self, X: pd.DataFrame, y: Any = None) -> "IngenieriaMusical":
        self.feature_names_in_ = np.asarray(X.columns if hasattr(X, "columns") else
                                            [f"x{i}" for i in range(X.shape[1])], dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X, columns=self.feature_names_in_) if not hasattr(X, "columns") else X.copy()
        e = self.EPS
        X["log_duration"] = np.log(X["duration_ms"].clip(lower=1_000))
        X["energia_acustica"] = np.log((X["energy"] + e) / (X["acousticness"] + e))
        X["pulso_bailable"] = (X["tempo"] / 120.0) * X["danceability"]
        X["intensidad"] = X["energy"] * (X["loudness"] + 60.0) / 60.0
        X["vocalidad"] = (1 - X["instrumentalness"]) * X["speechiness"]
        X["log_speechiness"] = np.log1p(100 * X["speechiness"])
        X["log_liveness"] = np.log1p(100 * X["liveness"])
        X["log_instrumental"] = np.log1p(100 * X["instrumentalness"])
        if self.eliminar_originales:
            X = X.drop(columns=["duration_ms", "speechiness", "liveness", "instrumentalness"])
        return X

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:
        base = list(input_features if input_features is not None else self.feature_names_in_)
        nuevas = ["log_duration", "energia_acustica", "pulso_bailable", "intensidad", "vocalidad",
                  "log_speechiness", "log_liveness", "log_instrumental"]
        if self.eliminar_originales:
            base = [c for c in base if c not in ("duration_ms", "speechiness", "liveness", "instrumentalness")]
        return np.asarray(base + nuevas, dtype=object)


class CodificadorFrecuencia(BaseEstimator, TransformerMixin):
    """Frequency encoding: reemplaza cada artista por log(1 + nº de canciones
    en train). Aprende las frecuencias solo en ``fit`` (train); un artista
    no visto recibe 0. No usa la etiqueta, por lo que no puede filtrar y.
    Alternativa a Target Encoding cuando se teme sobreajuste a artistas."""

    def fit(self, X: pd.DataFrame, y: Any = None) -> "CodificadorFrecuencia":
        X = pd.DataFrame(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.frecuencias_ = {c: X[c].value_counts() for c in X.columns}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X, columns=self.feature_names_in_)
        out = pd.DataFrame(index=X.index)
        for c in X.columns:
            out[f"{c}_freq"] = np.log1p(X[c].map(self.frecuencias_[c]).astype(float).fillna(0.0))
        return out

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:
        return np.asarray([f"{c}_freq" for c in self.feature_names_in_], dtype=object)


def separar_X_y(df: pd.DataFrame, objetivo: str = OBJETIVO) -> tuple[pd.DataFrame, pd.Series]:
    """Elimina identificadores y separa predictores de la variable objetivo."""
    df = eliminar_identificadores(df)
    return df.drop(columns=[objetivo]), df[objetivo]


def dividir_train_test(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2,
                       semilla: int = SEMILLA) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split estratificado 80/20. Se hace ANTES de cualquier transformación
    con estado; la estratificación preserva las proporciones de clase
    (importante tras la deduplicación, que deja Rap/Hip-Hop con menos filas)."""
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=semilla)


def guardar_conjuntos(rutas: Rutas, X_train: pd.DataFrame, X_test: pd.DataFrame,
                      y_train: pd.Series, y_test: pd.Series) -> None:
    """Persiste los cuatro conjuntos en ``datos/model`` (CSV, legibles y
    versionables)."""
    X_train.to_csv(rutas.model_data / "X_train.csv", index=False)
    X_test.to_csv(rutas.model_data / "X_test.csv", index=False)
    y_train.to_frame().to_csv(rutas.model_data / "y_train.csv", index=False)
    y_test.to_frame().to_csv(rutas.model_data / "y_test.csv", index=False)


def cargar_conjuntos(rutas: Rutas) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Carga X_train, X_test, y_train, y_test desde ``datos/model``."""
    tipos = {c: str for c in COLUMNAS_CATEGORICAS + [COLUMNA_ARTISTA]}  # NaN, no pd.NA
    X_train = pd.read_csv(rutas.model_data / "X_train.csv", dtype=tipos)
    X_test = pd.read_csv(rutas.model_data / "X_test.csv", dtype=tipos)
    y_train = pd.read_csv(rutas.model_data / "y_train.csv")[OBJETIVO]
    y_test = pd.read_csv(rutas.model_data / "y_test.csv")[OBJETIVO]
    return X_train, X_test, y_train, y_test


# =============================================================================
# 7. PREPROCESAMIENTO: ColumnTransformer + Pipeline
# =============================================================================
CodificacionArtista = Literal["target", "frecuencia", "ninguna"]


def evaluar_imputadores(X: pd.DataFrame, columnas_objetivo: Sequence[str] = ("tempo", "duration_ms"),
                        columnas: Sequence[str] = COLUMNAS_NUMERICAS, frac_mascara: float = 0.10,
                        muestra: int = 12_000, semilla: int = SEMILLA) -> pd.DataFrame:
    """Experimento de enmascaramiento para elegir imputador **con datos de
    train**: sobre filas completas se oculta un ``frac_mascara`` de los
    valores de ``columnas_objetivo``, se imputan y se mide NRMSE = RMSE/σ
    (NRMSE = 1 equivale a imputar la media). Compara mediana, KNN e
    IterativeImputer (regresión bayesiana encadenada, MICE)."""
    rng = np.random.default_rng(semilla)
    completo = X[list(columnas)].dropna()
    completo = completo.sample(min(muestra, len(completo)), random_state=semilla)
    enmascarado = completo.copy()
    mascaras = {}
    for c in columnas_objetivo:
        m = rng.random(len(completo)) < frac_mascara
        enmascarado.loc[m, c] = np.nan
        mascaras[c] = m

    escalador = StandardScaler().fit(enmascarado)  # ignora NaN al ajustar
    Z = pd.DataFrame(escalador.transform(enmascarado), columns=enmascarado.columns)
    imputadores: dict[str, Any] = {
        "Mediana global": SimpleImputer(strategy="median"),
        "KNN (k=15)": KNNImputer(n_neighbors=15, weights="distance"),
        "Iterative (BayesianRidge)": IterativeImputer(estimator=BayesianRidge(), max_iter=15,
                                                      random_state=semilla),
    }
    filas = []
    for nombre, imp in imputadores.items():
        t0 = time.perf_counter()
        Zi = pd.DataFrame(imp.fit_transform(Z), columns=Z.columns)
        rec = pd.DataFrame(escalador.inverse_transform(Zi), columns=Z.columns)
        dur = time.perf_counter() - t0
        for c in columnas_objetivo:
            m = mascaras[c]
            real, pred = completo[c].to_numpy()[m], rec[c].to_numpy()[m]
            rmse = float(np.sqrt(np.mean((real - pred) ** 2)))
            filas.append({"imputador": nombre, "variable": c, "RMSE": round(rmse, 2),
                          "NRMSE": round(rmse / completo[c].std(), 4), "segundos": round(dur, 2)})
    return pd.DataFrame(filas).pivot(index="imputador", columns="variable", values=["NRMSE", "RMSE", "segundos"])


def construir_preprocesador(codificacion_artista: CodificacionArtista = "target",
                            imputador: Literal["iterative", "knn", "mediana"] = "iterative",
                            escalador: Literal["standard", "robust"] = "standard",
                            semilla: int = SEMILLA) -> ColumnTransformer:
    """ColumnTransformer con tres ramas:

    * Numérica: imputación (+ indicador de faltante) -> ingeniería musical ->
      escalado. ``add_indicator=True`` deja que el modelo aprenda si el
      *hecho* de faltar es informativo, sin asumirlo.
    * Categórica (key, mode): moda -> OneHotEncoder. Son nominales (la
      tonalidad A# no es "mayor" que D), un ordinal impondría un orden falso.
    * Artista (6 800+ niveles): OneHot crearía miles de columnas dispersas.
      - ``target``: TargetEncoder multiclase de sklearn: codifica P(género|artista)
        con *shrinkage* bayesiano hacia la prior y **cross-fitting interno**
        en ``fit_transform`` (cada fila se codifica con folds que no la
        contienen) -> sin fuga de y.
      - ``frecuencia``: popularidad del artista en el catálogo (no usa y).
      - ``ninguna``: excluye al artista (mide generalización a artistas nuevos).
    """
    imputadores = {
        "iterative": IterativeImputer(estimator=BayesianRidge(), max_iter=15, random_state=semilla,
                                      add_indicator=True),
        "knn": KNNImputer(n_neighbors=15, weights="distance", add_indicator=True),
        "mediana": SimpleImputer(strategy="median", add_indicator=True),
    }
    escaladores = {"standard": StandardScaler(), "robust": RobustScaler()}

    rama_num = ImbPipeline([
        ("imputar", imputadores[imputador]),
        ("ingenieria", IngenieriaMusical()),
        ("escalar", escaladores[escalador]),
    ])
    rama_cat = ImbPipeline([
        ("imputar", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    transformadores: list[tuple[str, Any, list[str]]] = [
        ("num", rama_num, COLUMNAS_NUMERICAS),
        ("cat", rama_cat, COLUMNAS_CATEGORICAS),
    ]
    if codificacion_artista == "target":
        transformadores.append(("artista", ImbPipeline([
            ("imputar", SimpleImputer(strategy="constant", fill_value="__desconocido__")),
            ("target", TargetEncoder(target_type="multiclass", smooth="auto", cv=5)),
        ]), [COLUMNA_ARTISTA]))
    elif codificacion_artista == "frecuencia":
        transformadores.append(("artista", ImbPipeline([
            ("imputar", SimpleImputer(strategy="constant", fill_value="__desconocido__")),
            ("freq", CodificadorFrecuencia()),
            ("escalar", StandardScaler()),
        ]), [COLUMNA_ARTISTA]))

    ct = ColumnTransformer(transformadores, remainder="drop", verbose_feature_names_out=True)
    return ct.set_output(transform="pandas")


def construir_pipeline(modelo: BaseEstimator, codificacion_artista: CodificacionArtista = "target",
                       balanceo: Literal["ninguno", "smote"] = "ninguno",
                       imputador: Literal["iterative", "knn", "mediana"] = "iterative",
                       escalador: Literal["standard", "robust"] = "standard",
                       semilla: int = SEMILLA) -> ImbPipeline:
    """Pipeline de imblearn: preprocesador -> [SMOTE] -> clasificador.

    Se usa ``imblearn.pipeline.Pipeline`` porque SMOTE debe ejecutarse **solo
    durante fit** (en cada fold de entrenamiento) y nunca en predict: las
    muestras sintéticas jamás llegan a validación/test, lo que evita estimar
    métricas optimistas.
    """
    pasos: list[tuple[str, Any]] = [("prep", construir_preprocesador(codificacion_artista, imputador,
                                                                       escalador, semilla))]
    if balanceo == "smote":
        pasos.append(("smote", SMOTE(k_neighbors=5, random_state=semilla)))
    pasos.append(("clf", modelo))
    return ImbPipeline(pasos)


def nombres_salida(pipeline: ImbPipeline, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Ajusta solo el preprocesador sobre (X, y) y devuelve la matriz
    transformada (DataFrame) para inspeccionar las columnas generadas."""
    prep = clone(pipeline.named_steps["prep"])
    return prep.fit_transform(X, y)


# =============================================================================
# 8. MODELOS, VALIDACIÓN CRUZADA Y OVERFITTING
# =============================================================================
def codificar_objetivo(y_train: pd.Series, *otros: pd.Series) -> tuple[LabelEncoder, list[np.ndarray]]:
    """LabelEncoder ajustado en y_train (XGBoost/LightGBM requieren enteros
    0..K-1). El mapeo es una biyección fija clase->entero, no una
    estadística de los datos, así que no constituye fuga."""
    le = LabelEncoder().fit(y_train)
    return le, [le.transform(y_train), *[le.transform(o) for o in otros]]


def obtener_modelos(semilla: int = SEMILLA, n_jobs: int = -1) -> dict[str, BaseEstimator]:
    """Catálogo de modelos a comparar (uno por familia):

    * Dummy (prior): piso absoluto. F1-macro ≈ 1/K·(algo), log-loss = H(y).
    * Regresión logística multinomial: baseline lineal interpretable;
      mide cuánto aportan las no linealidades.
    * GaussianNB calibrado: probabilístico generativo; asume independencia
      condicional (falsa aquí: energy~loudness), por eso se calibra con
      Platt (sigmoide) para corregir probabilidades extremas.
    * Random Forest: bagging de árboles; reduce varianza, robusto a escalas
      y a interacciones no lineales.
    * LightGBM / XGBoost: boosting por gradiente; reduce sesgo secuencialmente
      y suele dominar en datos tabulares heterogéneos.
    * MLP: modelo basado en márgenes/distancias en un espacio aprendido.
      Se prefiere a SVC-RBF porque SVC es O(n²)-O(n³) en memoria/tiempo con
      ~36 000 filas × 10 clases (OvO = 45 clasificadores) y necesita Platt
      interno con 5-CV extra para dar probabilidades.

    Descartados: KNN (maldición de la dimensionalidad con OneHot+TargetEnc,
    predicción O(n)); árboles individuales (alta varianza); modelos
    secuenciales/CNN (no hay señal de audio cruda, solo descriptores).
    """
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    return {
        "Dummy (prior)": DummyClassifier(strategy="prior"),
        "Regresión Logística": LogisticRegression(C=1.0, max_iter=3000),
        "GaussianNB calibrado": CalibratedClassifierCV(GaussianNB(), method="sigmoid", cv=3),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3, max_features="sqrt",
                                                n_jobs=n_jobs, random_state=semilla),
        "LightGBM": LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.8,
                                   subsample_freq=1, colsample_bytree=0.8, random_state=semilla,
                                   n_jobs=n_jobs, verbose=-1),
        "XGBoost": XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.8,
                                 colsample_bytree=0.8, tree_method="hist", eval_metric="mlogloss",
                                 random_state=semilla, n_jobs=n_jobs),
        "MLP": MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3, batch_size=256,
                             learning_rate_init=1e-3, early_stopping=True, n_iter_no_change=10,
                             max_iter=300, random_state=semilla),
    }


def cv_estratificada(n_splits: int = 5, semilla: int = SEMILLA) -> StratifiedKFold:
    """StratifiedKFold barajado: cada fold mantiene la proporción de clases,
    lo que reduce la varianza del estimador de F1-macro."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=semilla)


def evaluar_modelos_cv(modelos: Mapping[str, BaseEstimator], X: pd.DataFrame, y: np.ndarray,
                       cv: Any = None, groups: np.ndarray | None = None,
                       metricas: Mapping[str, str] = ESCALADO_METRICAS,
                       **kwargs_pipeline: Any) -> pd.DataFrame:
    """Validación cruzada de cada modelo dentro del mismo Pipeline,
    registrando métricas de **train y validación** para medir overfitting.

    Métricas:
    * F1-macro: media no ponderada del F1 por clase; trata igual a todas
      las clases (no se esconde una clase mal predicha detrás de otras).
    * ROC-AUC OvR: capacidad de *ordenar* (ranking) cada clase contra el
      resto, independiente del umbral.
    * Log-loss: -1/n Σ log p̂(y_i|x_i); penaliza probabilidades confiadas y
      erróneas -> mide calibración, clave para la sección bayesiana.
    Brecha = train - validación: > ~0.05 en F1 sugiere sobreajuste.
    """
    cv = cv or cv_estratificada()
    filas = []
    for nombre, modelo in modelos.items():
        pipe = construir_pipeline(clone(modelo), **kwargs_pipeline)
        t0 = time.perf_counter()
        res = cross_validate(pipe, X, y, cv=cv, groups=groups, scoring=dict(metricas),
                             return_train_score=True, n_jobs=1, error_score="raise")
        fila: dict[str, Any] = {"modelo": nombre, "tiempo_total_s": round(time.perf_counter() - t0, 1)}
        for m in metricas:
            signo = -1 if m.startswith("neg_") else 1
            nom = m.replace("neg_", "")
            tr, va = signo * res[f"train_{m}"], signo * res[f"test_{m}"]
            fila[f"{nom}_train"] = tr.mean()
            fila[f"{nom}_val"] = va.mean()
            fila[f"{nom}_val_std"] = va.std()
            fila[f"{nom}_brecha"] = tr.mean() - va.mean()
        filas.append(fila)
        print(f"✔ {nombre:<24} F1-macro val = {fila['f1_macro_val']:.4f}  ({fila['tiempo_total_s']} s)")
    return pd.DataFrame(filas).set_index("modelo").sort_values("f1_macro_val", ascending=False)


def grafico_train_vs_val(resultados: pd.DataFrame,
                         metricas: Sequence[str] = ("f1_macro", "roc_auc_ovr", "log_loss")) -> plt.Figure:
    """Barras train vs. validación por modelo y métrica (con desviación
    estándar entre folds). Visualiza sesgo (ambas bajas) y varianza (brecha)."""
    fig, axes = plt.subplots(1, len(metricas), figsize=(6.5 * len(metricas), 5))
    for ax, m in zip(np.atleast_1d(axes), metricas):
        d = resultados[[f"{m}_train", f"{m}_val"]].rename(columns=lambda c: c.split("_")[-1])
        d.plot.barh(ax=ax, xerr={"val": resultados[f"{m}_val_std"]}, color=["#95a5a6", "#2980b9"])
        ax.set_title(m + (" (menor es mejor)" if m == "log_loss" else ""))
        ax.set_ylabel("")
    fig.tight_layout()
    return fig


def comparar_balanceo(modelo: BaseEstimator, X: pd.DataFrame, y: pd.Series,
                      cv: Any = None, **kwargs_pipeline: Any) -> pd.DataFrame:
    """Compara sin balanceo, ``class_weight='balanced'`` y SMOTE en el mismo
    escenario. ``class_weight`` re-pondera la pérdida con w_k = n/(K·n_k)
    (equivale a un cambio de prior, sin datos nuevos); SMOTE interpola
    x_new = x_i + λ(x_vecino - x_i), λ~U(0,1), solo en folds de train."""
    cv = cv or cv_estratificada()
    pesado = clone(modelo).set_params(class_weight="balanced")
    variantes = {
        "Sin balanceo": (clone(modelo), "ninguno"),
        "class_weight='balanced'": (pesado, "ninguno"),
        "SMOTE (en pipeline)": (clone(modelo), "smote"),
    }
    filas = []
    for nombre, (mod, bal) in variantes.items():
        pipe = construir_pipeline(mod, balanceo=bal, **kwargs_pipeline)
        res = cross_validate(pipe, X, y, cv=cv, scoring={"f1_macro": "f1_macro",
                                                          "balanced_accuracy": "balanced_accuracy",
                                                          "neg_log_loss": "neg_log_loss"}, n_jobs=1)
        filas.append({"estrategia": nombre, "f1_macro": res["test_f1_macro"].mean(),
                      "balanced_accuracy": res["test_balanced_accuracy"].mean(),
                      "log_loss": -res["test_neg_log_loss"].mean(),
                      "tiempo_fit_s": res["fit_time"].mean()})
    return pd.DataFrame(filas).set_index("estrategia")


def f1_por_clase_cv(pipeline: ImbPipeline, X: pd.DataFrame, y: np.ndarray, clases: Sequence[str],
                    cv: Any = None) -> pd.Series:
    """F1 por clase con predicciones out-of-fold (útil para ver qué clase
    sufre con el desbalanceo)."""
    pred = cross_val_predict(pipeline, X, y, cv=cv or cv_estratificada(), n_jobs=1)
    return pd.Series(f1_score(y, pred, average=None), index=clases, name="F1")


def grafico_curvas_aprendizaje(modelos: Mapping[str, ImbPipeline], X: pd.DataFrame, y: np.ndarray,
                               cv: Any = None, scoring: str = "f1_macro",
                               tamanos: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 1.0)) -> plt.Figure:
    """Curvas de aprendizaje: score de train y validación vs. tamaño del
    conjunto de entrenamiento.
    * Curvas que convergen a un valor bajo -> sesgo alto (subajuste).
    * Brecha grande y persistente -> varianza alta (sobreajuste).
    * Validación aún creciendo al 100 % -> más datos ayudarían.
    """
    cv = cv or cv_estratificada(3)
    fig, axes = plt.subplots(1, len(modelos), figsize=(6 * len(modelos), 4.5), sharey=True)
    for ax, (nombre, pipe) in zip(np.atleast_1d(axes), modelos.items()):
        n, tr, va = learning_curve(pipe, X, y, cv=cv, scoring=scoring, train_sizes=list(tamanos),
                                   n_jobs=1, shuffle=True, random_state=SEMILLA)
        for s, etiqueta, color in ((tr, "train", "#95a5a6"), (va, "validación", "#2980b9")):
            ax.plot(n, s.mean(1), "o-", color=color, label=etiqueta)
            ax.fill_between(n, s.mean(1) - s.std(1), s.mean(1) + s.std(1), color=color, alpha=0.2)
        ax.set_title(nombre)
        ax.set_xlabel("Filas de entrenamiento")
        ax.legend()
    np.atleast_1d(axes)[0].set_ylabel(scoring)
    fig.tight_layout()
    return fig


def grupos_por_artista(X: pd.DataFrame) -> np.ndarray:
    """Etiqueta de grupo = artista; artistas faltantes reciben un grupo único
    por fila (no se puede asumir que sean el mismo artista)."""
    art = X[COLUMNA_ARTISTA].astype(object)
    faltantes = art.isna()
    art = art.where(~faltantes, [f"__na_{i}" for i in range(len(art))])
    return LabelEncoder().fit_transform(art.astype(str))


def evaluar_dependencia_artista(modelo: BaseEstimator, X: pd.DataFrame, y: np.ndarray,
                                n_splits: int = 5) -> pd.DataFrame:
    """¿Cuánto del rendimiento viene de *memorizar artistas*? Compara:
    (a) StratifiedKFold (artistas compartidos entre folds, escenario de
        catálogo conocido) vs.
    (b) StratifiedGroupKFold por artista (artistas nunca vistos),
    con y sin la variable artista. Define qué métrica reportar al negocio
    según si el modelo clasificará catálogo existente o lanzamientos nuevos."""
    escenarios = {
        "Artistas conocidos · con artista (target)": (cv_estratificada(n_splits), "target"),
        "Artistas conocidos · sin artista": (cv_estratificada(n_splits), "ninguna"),
        "Artistas nuevos · con artista (target)": (StratifiedGroupKFold(n_splits, shuffle=True,
                                                                        random_state=SEMILLA), "target"),
        "Artistas nuevos · sin artista": (StratifiedGroupKFold(n_splits, shuffle=True,
                                                               random_state=SEMILLA), "ninguna"),
    }
    grupos = grupos_por_artista(X)
    filas = []
    for nombre, (cv, cod) in escenarios.items():
        pipe = construir_pipeline(clone(modelo), codificacion_artista=cod)
        g = grupos if isinstance(cv, StratifiedGroupKFold) else None
        s = cross_val_score(pipe, X, y, cv=cv, groups=g, scoring="f1_macro", n_jobs=1)
        filas.append({"escenario": nombre, "f1_macro": s.mean(), "std": s.std()})
    return pd.DataFrame(filas).set_index("escenario")


# =============================================================================
# 9. OPTIMIZACIÓN DE HIPERPARÁMETROS
# =============================================================================
@dataclass
class ResultadoBusqueda:
    """Resultado homogéneo de una estrategia de búsqueda."""

    estrategia: str
    mejores_parametros: dict[str, Any]
    mejor_score: float
    historial: list[float] = field(default_factory=list)
    segundos: float = 0.0

    def resumen(self) -> dict[str, Any]:
        return {"estrategia": self.estrategia, "mejor_f1_macro_cv": round(self.mejor_score, 4),
                "evaluaciones": len(self.historial), "segundos": round(self.segundos, 1)}


# Espacio de búsqueda de LightGBM (prefijo clf__ = paso del pipeline).
ESPACIO_LGBM: dict[str, tuple] = {
    "clf__n_estimators": ("int", 200, 1200),
    "clf__learning_rate": ("log", 0.01, 0.2),
    "clf__num_leaves": ("int", 15, 255),
    "clf__max_depth": ("int", 3, 12),
    "clf__min_child_samples": ("int", 10, 200),
    "clf__subsample": ("float", 0.5, 1.0),
    "clf__colsample_bytree": ("float", 0.4, 1.0),
    "clf__reg_lambda": ("log", 1e-3, 10.0),
}
SIGNIFICADO_HIPERPARAMETROS: dict[str, str] = {
    "n_estimators": "Nº de árboles M en F_M(x)=Σ η·f_m(x). Más árboles -> menor sesgo; con η bajo se necesita M alto.",
    "learning_rate": "Shrinkage η: cada árbol corrige solo una fracción η del gradiente. η↓ = frontera más suave y menos varianza, pero requiere más M.",
    "num_leaves": "Hojas por árbol = nº de regiones en que cada árbol parte el espacio. Controla la complejidad de las interacciones (≈ 2^profundidad).",
    "max_depth": "Profundidad máxima: orden máximo de interacción entre variables que un árbol puede modelar. Limita varianza.",
    "min_child_samples": "Mínimo de filas por hoja: regulariza fronteras con pocas observaciones (evita 'islas' de sobreajuste).",
    "subsample": "Fracción de filas por árbol (boosting estocástico): decorrelaciona árboles y reduce varianza.",
    "colsample_bytree": "Fracción de columnas por árbol: obliga a usar variables alternativas (ej. no depender solo de artista).",
    "reg_lambda": "Penalización L2 sobre los pesos de las hojas: w* = -G/(H+λ). λ↑ encoge las predicciones hacia 0, suaviza la frontera.",
    "C": "Inverso de la regularización L2 (LogReg/SVC): C↑ = menor penalización, frontera más ajustada a train.",
    "var_smoothing": "GaussianNB: suma ε·max(σ²) a todas las varianzas; evita densidades degeneradas y suaviza fronteras cuadráticas.",
}


def _muestrear_aleatorio(espacio: Mapping[str, tuple], n: int, semilla: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(semilla)
    muestras = []
    for _ in range(n):
        p: dict[str, Any] = {}
        for k, (tipo, lo, hi) in espacio.items():
            if tipo == "int":
                p[k] = int(rng.integers(lo, hi + 1))
            elif tipo == "log":
                p[k] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
            else:
                p[k] = float(rng.uniform(lo, hi))
        muestras.append(p)
    return muestras


def busqueda_aleatoria(pipeline: ImbPipeline, X: pd.DataFrame, y: np.ndarray,
                       espacio: Mapping[str, tuple] = ESPACIO_LGBM, n_iter: int = 30,
                       cv: Any = None, semilla: int = SEMILLA) -> ResultadoBusqueda:
    """RandomizedSearchCV: muestrea configuraciones i.i.d. del espacio.
    Mejor que una rejilla en espacios de >3 dimensiones (Bergstra & Bengio,
    2012): con n evaluaciones explora n valores distintos de cada
    hiperparámetro, mientras que la rejilla solo explora n^(1/d)."""
    cv = cv or cv_estratificada(3)
    candidatos = [{k: [v] for k, v in p.items()} for p in _muestrear_aleatorio(espacio, n_iter, semilla)]
    t0 = time.perf_counter()
    bus = RandomizedSearchCV(pipeline, candidatos, n_iter=n_iter, scoring="f1_macro", cv=cv,
                             random_state=semilla, n_jobs=1, refit=False)
    bus.fit(X, y)
    hist = bus.cv_results_["mean_test_score"].tolist()
    return ResultadoBusqueda("Random Search", dict(bus.cv_results_["params"][int(np.argmax(hist))]),
                             float(np.max(hist)), hist, time.perf_counter() - t0)


def busqueda_rejilla(pipeline: ImbPipeline, X: pd.DataFrame, y: np.ndarray,
                     rejilla: Mapping[str, Sequence[Any]], cv: Any = None) -> ResultadoBusqueda:
    """GridSearchCV exhaustivo: adecuado solo para espacios pequeños y de
    baja dimensión (ej. var_smoothing de GaussianNB o C de LogReg)."""
    from sklearn.model_selection import GridSearchCV
    t0 = time.perf_counter()
    bus = GridSearchCV(pipeline, dict(rejilla), scoring="f1_macro", cv=cv or cv_estratificada(3),
                       n_jobs=1, refit=False).fit(X, y)
    hist = bus.cv_results_["mean_test_score"].tolist()
    return ResultadoBusqueda("Grid Search", dict(bus.best_params_), float(bus.best_score_), hist,
                             time.perf_counter() - t0)


def busqueda_bayesiana(pipeline: ImbPipeline, X: pd.DataFrame, y: np.ndarray,
                       espacio: Mapping[str, tuple] = ESPACIO_LGBM, n_trials: int = 30,
                       cv: Any = None, semilla: int = SEMILLA) -> tuple[ResultadoBusqueda, Any]:
    """Optimización bayesiana con Optuna (TPE).

    TPE modela p(θ | y) con dos densidades: l(θ) para las configuraciones
    del mejor γ-cuantil y g(θ) para el resto, y propone el θ que maximiza
    l(θ)/g(θ), que es proporcional a la Mejora Esperada (EI). Así cada
    evaluación *aprende* de las anteriores: concentra el presupuesto en
    regiones prometedoras (explotación) sin dejar de explorar. La búsqueda
    aleatoria, en cambio, ignora la historia. Se agrega un pruner
    (MedianPruner) que corta trials cuyos primeros folds van por debajo
    de la mediana -> ahorro de cómputo.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cv = cv or cv_estratificada(3)
    splits = list(cv.split(X, y))

    def objetivo(trial: optuna.Trial) -> float:
        params: dict[str, Any] = {}
        for k, (tipo, lo, hi) in espacio.items():
            if tipo == "int":
                params[k] = trial.suggest_int(k, lo, hi)
            elif tipo == "log":
                params[k] = trial.suggest_float(k, lo, hi, log=True)
            else:
                params[k] = trial.suggest_float(k, lo, hi)
        pipe = clone(pipeline).set_params(**params)
        scores = []
        for i, (tr, va) in enumerate(splits):
            pipe.fit(X.iloc[tr], y[tr])
            scores.append(f1_score(y[va], pipe.predict(X.iloc[va]), average="macro"))
            trial.report(float(np.mean(scores)), i)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    estudio = optuna.create_study(direction="maximize",
                                  sampler=optuna.samplers.TPESampler(seed=semilla, n_startup_trials=8),
                                  pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=0))
    t0 = time.perf_counter()
    estudio.optimize(objetivo, n_trials=n_trials, show_progress_bar=False)
    hist = [t.value if t.state == optuna.trial.TrialState.COMPLETE else np.nan for t in estudio.trials]
    return (ResultadoBusqueda("Optuna (TPE)", dict(estudio.best_params), float(estudio.best_value), hist,
                              time.perf_counter() - t0), estudio)


def grafico_convergencia(resultados: Sequence[ResultadoBusqueda]) -> plt.Figure:
    """Mejor score acumulado vs. nº de evaluaciones: la estrategia cuya
    curva sube antes es más eficiente en cómputo."""
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for r in resultados:
        h = pd.Series(r.historial, dtype=float)
        ax.plot(np.arange(1, len(h) + 1), h.ffill().cummax(), "o-", ms=3,
                label=f"{r.estrategia} (máx {r.mejor_score:.4f}, {r.segundos:.0f}s)")
        ax.scatter(np.arange(1, len(h) + 1), h, s=10, alpha=0.35)
    ax.set_xlabel("Evaluación (trial)")
    ax.set_ylabel("F1-macro CV")
    ax.set_title("Convergencia: mejor score acumulado")
    ax.legend()
    fig.tight_layout()
    return fig


def tabla_significado(parametros: Mapping[str, Any]) -> pd.DataFrame:
    """Une los valores óptimos con su significado matemático."""
    filas = []
    for k, v in parametros.items():
        nombre = k.split("__")[-1]
        filas.append({"hiperparámetro": nombre, "valor": v,
                      "significado": SIGNIFICADO_HIPERPARAMETROS.get(nombre, "")})
    return pd.DataFrame(filas).set_index("hiperparámetro")


def guardar_json(datos: Mapping[str, Any], ruta: Path) -> None:
    """Guarda un dict como JSON legible (convierte tipos numpy)."""
    def _conv(o: Any) -> Any:
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    ruta.write_text(json.dumps(datos, indent=2, ensure_ascii=False, default=_conv), encoding="utf-8")


def cargar_json(ruta: Path) -> dict[str, Any]:
    return json.loads(Path(ruta).read_text(encoding="utf-8"))


# =============================================================================
# 10. PROBABILIDAD CONDICIONAL Y TEOREMA DE BAYES
# =============================================================================
def probabilidad_condicional(df: pd.DataFrame, evento_A: pd.Series, evento_B: pd.Series,
                             nombre_A: str = "A", nombre_B: str = "B",
                             prior_beta: tuple[float, float] = (1.0, 1.0),
                             nivel: float = 0.95) -> dict[str, Any]:
    """Calcula P(A|B) de dos formas y cuantifica la incertidumbre.

    1. Directa (frecuentista): P(A|B) = |A∩B| / |B|.
    2. Por el Teorema de Bayes:  P(A|B) = P(B|A)·P(A) / P(B).
       Ambas coinciden por construcción; mostrar la segunda descompone la
       respuesta en *verosimilitud* P(B|A), *prior* P(A) y *evidencia* P(B).
    3. Bayesiana: con prior Beta(α, β) y k éxitos en n ensayos, la
       posterior de θ = P(A|B) es Beta(α+k, β+n-k) (conjugación
       Beta-Binomial). Se reporta la media posterior y el intervalo de
       credibilidad: "dado lo observado, θ está en [l, u] con prob. 95 %".
    También se reporta el *lift* = P(A|B)/P(A): cuántas veces B multiplica
    la probabilidad base de A.
    """
    A = evento_A.fillna(False).astype(bool).to_numpy()
    B = evento_B.fillna(False).astype(bool).to_numpy()
    n, nB, nA, nAB = len(df), int(B.sum()), int(A.sum()), int((A & B).sum())
    pA, pB = nA / n, nB / n
    pB_dado_A = nAB / nA if nA else np.nan
    pA_dado_B = nAB / nB if nB else np.nan
    a, b = prior_beta[0] + nAB, prior_beta[1] + nB - nAB
    lo, hi = stats.beta.ppf([(1 - nivel) / 2, (1 + nivel) / 2], a, b)
    return {
        "pregunta": f"P({nombre_A} | {nombre_B})",
        "n_B": nB, "n_A∩B": nAB,
        "P(A) prior": pA, "P(B) evidencia": pB, "P(B|A) verosimilitud": pB_dado_A,
        "P(A|B) directa": pA_dado_B,
        "P(A|B) vía Bayes": (pB_dado_A * pA / pB) if pB else np.nan,
        "media posterior Beta": a / (a + b),
        f"IC credibilidad {int(nivel * 100)}%": (round(float(lo), 4), round(float(hi), 4)),
        "lift P(A|B)/P(A)": pA_dado_B / pA if pA else np.nan,
    }


def tabla_bayes(df: pd.DataFrame, preguntas: Mapping[str, tuple[Callable[[pd.DataFrame], pd.Series],
                                                                 Callable[[pd.DataFrame], pd.Series]]]) -> pd.DataFrame:
    """Evalúa varias preguntas {nombre: (f_A, f_B)} con ``probabilidad_condicional``."""
    filas = []
    for nombre, (fA, fB) in preguntas.items():
        r = probabilidad_condicional(df, fA(df), fB(df), *nombre.split(" | "))
        filas.append(r)
    return pd.DataFrame(filas).set_index("pregunta")


def probabilidades_oof(pipeline: ImbPipeline, X: pd.DataFrame, y: np.ndarray, cv: Any = None,
                       calibrar: bool = True) -> np.ndarray:
    """Probabilidades out-of-fold P̂(género | x) sobre train, opcionalmente
    calibradas (isotónica, CV interno). Out-of-fold = cada fila se predice
    con un modelo que no la vio -> estimación honesta sin tocar el test."""
    est = CalibratedClassifierCV(pipeline, method="isotonic", cv=3) if calibrar else pipeline
    return cross_val_predict(est, X, y, cv=cv or cv_estratificada(5), method="predict_proba", n_jobs=1)


def comparar_empirico_vs_modelo(df: pd.DataFrame, proba: np.ndarray, clases: Sequence[str],
                                condiciones: Mapping[str, tuple[str, Callable[[pd.DataFrame], pd.Series]]],
                                objetivo: str = OBJETIVO) -> pd.DataFrame:
    """Para cada condición B y clase A: compara la frecuencia empírica
    P(A|B) con el promedio de las probabilidades del modelo
    E[P̂(A|x) | x∈B]. Si el modelo está bien calibrado ambas coinciden
    (propiedad de calibración: E[P̂ | P̂=p] = p)."""
    filas = []
    idx = {c: i for i, c in enumerate(clases)}
    for nombre, (clase, fB) in condiciones.items():
        B = fB(df).fillna(False).to_numpy(bool)
        filas.append({"condición": nombre, "clase": clase, "n": int(B.sum()),
                      "P(A|B) empírica": float((df.loc[B, objetivo] == clase).mean()),
                      "E[P̂(A|x)] modelo": float(proba[B, idx[clase]].mean()),
                      "P(A) prior": float((df[objetivo] == clase).mean())})
    out = pd.DataFrame(filas).set_index("condición")
    out["error_calibración"] = out["E[P̂(A|x)] modelo"] - out["P(A|B) empírica"]
    return out


def actualizacion_bayesiana(proba_modelo: np.ndarray, prior_entrenamiento: np.ndarray,
                            prior_nuevo: np.ndarray) -> np.ndarray:
    """Ajuste de prior (label shift) con el Teorema de Bayes:

        P_nuevo(k|x) ∝ P_modelo(k|x) · π_nuevo(k) / π_train(k)

    Si en producción cambia la mezcla de géneros (ej. una radio con 40 %
    de Clásica), se corrigen las probabilidades sin reentrenar, asumiendo
    que P(x|k) no cambia."""
    w = proba_modelo * (np.asarray(prior_nuevo) / np.asarray(prior_entrenamiento))
    return w / w.sum(axis=1, keepdims=True)


def error_calibracion_esperado(y: np.ndarray, proba: np.ndarray, n_bins: int = 15) -> float:
    """ECE top-label: Σ_b (|B_b|/n)·|acc(B_b) - conf(B_b)|."""
    conf, pred = proba.max(1), proba.argmax(1)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(ece)


def grafico_confiabilidad(y: np.ndarray, probas: Mapping[str, np.ndarray], clases: Sequence[str],
                          clases_mostrar: Sequence[str] | None = None, n_bins: int = 10) -> plt.Figure:
    """Diagrama de confiabilidad one-vs-rest: frecuencia observada vs.
    probabilidad predicha. La diagonal = calibración perfecta."""
    clases_mostrar = list(clases_mostrar or clases[:4])
    fig, axes = plt.subplots(1, len(clases_mostrar), figsize=(4.6 * len(clases_mostrar), 4.4), sharey=True)
    for ax, c in zip(np.atleast_1d(axes), clases_mostrar):
        k = list(clases).index(c)
        for nombre, p in probas.items():
            fo, pm = calibration_curve((y == k).astype(int), p[:, k], n_bins=n_bins, strategy="quantile")
            ax.plot(pm, fo, "o-", ms=3, label=nombre)
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_title(c)
        ax.set_xlabel("P̂ predicha")
    np.atleast_1d(axes)[0].set_ylabel("Frecuencia observada")
    np.atleast_1d(axes)[0].legend()
    fig.tight_layout()
    return fig


# =============================================================================
# 11. EVALUACIÓN FINAL, CALIBRACIÓN Y SERIALIZACIÓN
# =============================================================================
def metricas_clasificacion(y: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    """F1-macro, accuracy, ROC-AUC OvR (macro), log-loss, Brier multiclase y ECE."""
    pred = proba.argmax(1)
    Y = np.eye(proba.shape[1])[y]
    return {"f1_macro": f1_score(y, pred, average="macro"),
            "accuracy": float((pred == y).mean()),
            "roc_auc_ovr": roc_auc_score(y, proba, multi_class="ovr", average="macro"),
            "log_loss": log_loss(y, proba, labels=np.arange(proba.shape[1])),
            "brier": float(np.mean(np.sum((proba - Y) ** 2, axis=1))),
            "ece": error_calibracion_esperado(y, proba)}


def evaluar_en_test(modelos: Mapping[str, Any], X_train: pd.DataFrame, y_train: np.ndarray,
                    X_test: pd.DataFrame, y_test: np.ndarray) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Métricas de train y test para cada modelo ya ajustado (permite
    comprobar la brecha final de sobreajuste)."""
    filas, probas = [], {}
    for nombre, m in modelos.items():
        p_tr, p_te = m.predict_proba(X_train), m.predict_proba(X_test)
        probas[nombre] = p_te
        for conjunto, (yy, pp) in {"train": (y_train, p_tr), "test": (y_test, p_te)}.items():
            filas.append({"modelo": nombre, "conjunto": conjunto, **metricas_clasificacion(yy, pp)})
    return pd.DataFrame(filas).set_index(["modelo", "conjunto"]).round(4), probas


def reporte_clasificacion(y: np.ndarray, pred: np.ndarray, clases: Sequence[str]) -> pd.DataFrame:
    """classification_report como DataFrame."""
    return pd.DataFrame(classification_report(y, pred, target_names=list(clases), output_dict=True)).T.round(4)


def grafico_matriz_confusion(y: np.ndarray, pred: np.ndarray, clases: Sequence[str]) -> plt.Figure:
    """Matriz de confusión normalizada por fila (recall por clase): muestra
    qué géneros se confunden entre sí (Rap↔Hip-Hop, Rock↔Alternative)."""
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay.from_predictions(y, pred, display_labels=list(clases), normalize="true",
                                            values_format=".2f", cmap="Blues", ax=ax, colorbar=False,
                                            xticks_rotation=45)
    ax.set_title("Matriz de confusión normalizada (test)")
    fig.tight_layout()
    return fig


def importancia_permutacion(modelo: Any, X: pd.DataFrame, y: np.ndarray, n_repeticiones: int = 5,
                            muestra: int = 4000, semilla: int = SEMILLA) -> pd.DataFrame:
    """Importancia por permutación sobre las columnas *originales* (antes del
    pipeline): caída de F1-macro al barajar cada variable. Agnóstica al
    modelo y medida en test, no en train (evita premiar sobreajuste)."""
    idx = X.sample(min(muestra, len(X)), random_state=semilla).index
    pos = X.index.get_indexer(idx)
    r = permutation_importance(modelo, X.loc[idx], y[pos], scoring="f1_macro", n_repeats=n_repeticiones,
                               random_state=semilla, n_jobs=1)
    return (pd.DataFrame({"importancia_media": r.importances_mean, "std": r.importances_std}, index=X.columns)
            .sort_values("importancia_media", ascending=False))


def serializar_modelo(modelo: Any, codificador: LabelEncoder, metricas: Mapping[str, Any],
                      parametros: Mapping[str, Any], rutas: Rutas,
                      nombre: str = "modelo_final") -> dict[str, Path]:
    """Guarda el pipeline completo (preprocesamiento + modelo) con joblib y
    un JSON de metadatos (versiones, clases, métricas, hiperparámetros)
    para trazabilidad y reproducibilidad."""
    import joblib
    import sklearn
    import lightgbm
    ruta_modelo = rutas.modelo / f"{nombre}.joblib"
    ruta_meta = rutas.modelo / f"{nombre}_metadatos.json"
    joblib.dump({"pipeline": modelo, "label_encoder": codificador}, ruta_modelo, compress=3)
    guardar_json({
        "fecha": time.strftime("%Y-%m-%d %H:%M:%S"),
        "clases": codificador.classes_.tolist(),
        "columnas_entrada": COLUMNAS_NUMERICAS + COLUMNAS_CATEGORICAS + [COLUMNA_ARTISTA],
        "metricas_test": dict(metricas),
        "hiperparametros": dict(parametros),
        "versiones": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                      "lightgbm": lightgbm.__version__, "pandas": pd.__version__, "numpy": np.__version__},
    }, ruta_meta)
    return {"modelo": ruta_modelo, "metadatos": ruta_meta}


def cargar_modelo(ruta: Path | str) -> tuple[Any, LabelEncoder]:
    """Carga el pipeline serializado y su LabelEncoder."""
    import joblib
    obj = joblib.load(ruta)
    return obj["pipeline"], obj["label_encoder"]


def predecir_genero(modelo: Any, codificador: LabelEncoder, df_nuevo: pd.DataFrame,
                    top_k: int = 3) -> pd.DataFrame:
    """Predice el género de registros nuevos *crudos* (con centinelas
    incluidos) y devuelve las top-k clases con su probabilidad."""
    X = preparar_registros_nuevos(df_nuevo)
    proba = modelo.predict_proba(X)
    orden = np.argsort(-proba, axis=1)[:, :top_k]
    filas = []
    for i in range(len(X)):
        fila: dict[str, Any] = {}
        for j, k in enumerate(orden[i], start=1):
            fila[f"top{j}"] = codificador.classes_[k]
            fila[f"p{j}"] = round(float(proba[i, k]), 4)
        filas.append(fila)
    return pd.DataFrame(filas, index=df_nuevo.index)
