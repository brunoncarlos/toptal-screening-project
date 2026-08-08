# Toptal Screening Project – Fraud Detection

Este proyecto implementa un pipeline completo de detección de fraude basado en:
- Ingeniería de características temporal y comportamental
- Embeddings de sesiones con Sentence-Transformers (MiniLM-L6-v2)
- Pipeline de features con scikit-learn
- Modelo final XGBoost
- Optimización de umbral para maximizar F1
- Reproducibilidad mediante un script de predicción (`predictions_code.py`)

---

## 📁 Estructura del proyecto

TOTAL SCREENING PROJECT/
│
├── data/
│   ├── domain_cache.pkl
│   ├── timezone_lookup.pkl
│   └── verify.zip
│
├── models/
│   ├── model_selected.pkl
│   ├── fe_pipeline.pkl
│   └── threshold_model_selected.json
│
├── notebook/
│   └── training_notebook.ipynb
│
├── output/
│   └── result.csv
|
├── src/
│   └── __init__.py
│   ├── behavioral_features.py
│   ├── temporal_features.py
│   ├── compute_session_embeddings.py
│   ├── pipeline.py
│
├── predictions_code.py
├── requirements.txt
└── README.md

---

## 🧠 Model Selection & Temporal Drift

La selección del modelo no se basa únicamente en la métrica de validación.  
El comportamiento de los usuarios cambia con el tiempo, y un modelo con alto recall en validación puede deteriorarse cuando se evalúa en sesiones futuras.

**El drift temporal mide cuánto cae el desempeño del modelo al pasar de validación a test.**

El objetivo no es el modelo con el mayor recall en validación, sino el que **mantiene recall y precisión estables** entre ventanas temporales.

Los modelos estables generalizan mejor y son más seguros para desplegar.

---

## ⚙️ Predicción final

El archivo `predictions_code.py` ejecuta el pipeline completo:

- carga el modelo serializado  
- aplica el pipeline de features  
- genera los embeddings de sesión  
- calcula los scores  
- aplica el umbral óptimo  
- produce un archivo `result.csv` con las predicciones finales

El archivo `result.csv` se guarda en la carpeta `output/` si existe.

---

## 🧩 Componentes principales

### **Feature Engineering**
- Features temporales (sesiones, timestamps, timezone)
- Features comportamentales (frecuencias, patrones de navegación)
- Embeddings de sesiones con MiniLM

### **Pipeline**
- Normalización y ensamblado de features
- Carga de embeddings desde cache
- Aplicación del modelo XGBoost
- Umbral optimizado para maximizar F1

### **Modelo**
- Entrenado en notebook
- Serializado en `models/`
- Compatible con Python 3.10

---

🧪 Experiment Tracking (Lightweight)
Para asegurar reproducibilidad y trazabilidad, implementé un sistema de experiment tracking lightweight basado en pandas.
Cada experimento registra:

configuración del modelo (hiperparámetros, arquitectura)

técnicas de undersampling

métricas clave (precision, recall, F1)

drift detectado

threshold seleccionado

features utilizadas

timestamp del experimento

Esto permite:

comparar modelos de forma consistente

identificar configuraciones ganadoras

auditar resultados

reproducir cualquier experimento

seleccionar el modelo final con evidencia cuantitativa

df["model_index"] = i
df["undersampling"] = str(res["undersampling"])
df["config"] = str(res["model"])
df["recall"] = res["recall"]
df["precision"] = res["precision"]
df["drift"] = res["drift"]
df["threshold"] = res["threshold"]


⭐ Feature Importance
Durante el análisis identifiqué cuatro grupos de features críticos para predecir si un usuario es Joe:

1. Temporal Features
timezone

local_hour

day_of_week

session_time
Estas variables capturan patrones de comportamiento diarios y geográficos.

2. Behavioral Features
número de sitios únicos

diversidad (entropy)

repetición de dominios

longitud de la sesión
Estas variables capturan el estilo de navegación del usuario.

3. Environment Features
browser

OS

locale

location
Estas variables ayudan a distinguir contextos de navegación.

4. Embedding Features (NLP)
Usé SentenceTransformer MiniLM-L6-v2 para convertir listas de sitios en vectores semánticos.
Estos embeddings fueron los más importantes para el modelo finalgit add Dockerfile
git commit -m "update Dockerfile"

## ✔️ Estado final

El pipeline es completamente reproducible, el modelo está serializado, el drift está analizado y el script de predicción genera el archivo final sin dependencias externas.

