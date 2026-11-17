# Clasificación multiclase de géneros musicales

Proyecto reproducible para predecir `music_genre` (10 clases) a partir de los descriptores de audio de Spotify en `music_genre.csv` (50 005 filas × 18 columnas).

## Flujo de datos

```text
datos/raw/music_genre.csv  (intacto)
        │
        ▼  01_eda_y_limpieza.ipynb ─────────────── reglas SIN ESTADO (seguras antes del split)
        │   • 5 filas 100 % nulas → fuera
        │   • centinelas → NaN: duration_ms=-1 (4 939), tempo='?' (4 980), artist='empty_field' (2 489)
        │   • 3 802 duplicados track+artista con género contradictorio → 1 fila/canción (etiqueta al azar con semilla)
        │   • diagnóstico MCAR (χ², Mann-Whitney), EDA uni/bi/multivariado, PCA/UMAP
        ▼
datos/processed/music_genre_limpio.csv      (46 198 filas, NaN sin imputar)
        │
        ▼  02_ingenieria_y_preprocesamiento.ipynb
        │   • elimina instance_id, obtained_date, track_name
        │   • SPLIT ESTRATIFICADO 80/20  ◄── antes de cualquier paso que aprenda parámetros
        │   • solo con train: elige imputador (enmascaramiento), valida variables nuevas, estrategia de balanceo
        │   • define el Pipeline (ver abajo)
        ├──► datos/processed/music_genre_procesado.csv (limpio + variables musicales, para negocio)
        ▼
datos/model/X_train.csv · X_test.csv · y_train.csv · y_test.csv   (limpios, SIN transformar)
        │
        ▼  03_evaluacion_y_optimizacion.ipynb   (solo train, StratifiedKFold)
        │   • 7 modelos: Dummy, LogReg, GaussianNB calibrado, RF, LightGBM, XGBoost, MLP
        │   • brecha train/val, curvas de aprendizaje, dependencia del artista (GroupKFold)
        │   • Grid (NB, LogReg) · Random vs Optuna-TPE (LightGBM)
        │   • Bayes: P(A|B) con IC Beta, calibración OOF, P(real|predicho), cambio de prior
        ├──► modelo/mejores_parametros.json · modelo/comparacion_modelos_cv.csv
        ▼
modelo/entrenamiento_final.ipynb   (ÚNICO uso del test)
        • ajuste en todo train · LightGBM vs. LightGBM calibrado (isotónica)
        • métricas de test, matriz de confusión, confiabilidad, importancia por permutación
        └──► modelo/modelo_final.joblib · modelo/modelo_final_metadatos.json
```

### Pipeline (se ajusta dentro de cada fold → sin fuga)

```text
ImbPipeline
├── prep: ColumnTransformer
│   ├── num     IterativeImputer(BayesianRidge, +indicador) → IngenieriaMusical → StandardScaler
│   ├── cat     moda → OneHotEncoder(key, mode)
│   └── artista '__desconocido__' → TargetEncoder(multiclase, cross-fitting)
├── [smote]  SMOTE solo en fit (opcional, para escenarios desbalanceados)
└── clf      LightGBM (u otro)
```

## Resultados (presupuesto completo)

| Etapa | Resultado |
|---|---|
| CV 5 folds, valores por defecto (F1-macro) | XGBoost 0.822 · LightGBM 0.820 · RF 0.817 · MLP 0.815 · LogReg 0.814 · GNB 0.782 · Dummy 0.020 |
| Búsqueda LightGBM (30 trials, 3 folds) | Optuna-TPE 0.8194 en 17 min · Random 0.8178 en 44 min |
| Calibración isotónica (log-loss OOF) | 0.391 vs. 0.371 sin calibrar → **no se calibra** |
| **Test (modelo final)** | **F1-macro 0.830 · accuracy 0.841 · ROC-AUC OvR 0.988 · log-loss 0.356 · ECE 0.017** |
| Artistas nuevos (StratifiedGroupKFold) | F1-macro ≈ 0.595 con el modelo sin artista (`modelo_artistas_nuevos.joblib`) |

Los errores se concentran en Rap↔Hip-Hop y Rock↔Alternative, géneros que comparten canciones en el dataset original.

## Estructura

```text
proyecto-clasificacion-musical/
├── datos/{raw,processed,model}/
├── cuadernillos/01_… 02_… 03_….ipynb
├── modelo/entrenamiento_final.ipynb
├── src/funciones.py      # toda la lógica: limpieza, estadística, gráficos, pipelines, métricas, Bayes
├── requirements.txt
└── README.md
```

## Ejecución

```bash
pip install -r requirements.txt
```

Ejecuta los cuadernillos en orden: 01 → 02 → 03 → `modelo/entrenamiento_final.ipynb`. Cada uno lee lo que escribió el anterior.

### En Google Colab

1. Sube la carpeta `proyecto-clasificacion-musical` **completa** (con `src/`, `datos/` y `requirements.txt`) a la raíz de *Mi unidad* en Google Drive.
2. Abre un cuadernillo desde Drive con *Abrir con → Google Colaboratory*.
3. Ejecuta las celdas en orden. La primera monta Drive, entra a la carpeta del proyecto e instala las dependencias. Si subiste la carpeta a otro lugar, cambia `RUTA_PROYECTO` en esa celda.
4. Sigue el orden 01 → 02 → 03 → final. Cada cuadernillo guarda sus salidas en Drive, y el siguiente las lee desde ahí.

Para iterar rápido (submuestras y menos trials de búsqueda), define la variable de entorno `MODO_RAPIDO=1` antes de abrir Jupyter.
