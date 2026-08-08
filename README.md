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

## ✔️ Estado final

El pipeline es completamente reproducible, el modelo está serializado, el drift está analizado y el script de predicción genera el archivo final sin dependencias externas.

