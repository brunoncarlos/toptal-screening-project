**Toptal Screening Project – User Session Classification (Identify Joe)**

This project is part of the Toptal technical screening process.
The objective is to analyze user session data and build a machine‑learning model capable of identifying a specific user: user_id = 0 (codename: Joe).

The project demonstrates:

- data understanding and EDA

- temporal and behavioral feature engineering

- semantic embeddings using Sentence‑Transformers

- disciplined experiment tracking

- model development and temporal‑drift evaluation

- production‑ready pipeline design

- reproducibility through a standalone prediction script 

The final deliverable is a fully reproducible pipeline that generates result.csv containing predictions for the verification dataset.

📁 Project Structure
Código
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
├── src/
│   ├── __init__.py
│   ├── behavioral_features.py
│   ├── temporal_features.py
│   ├── compute_session_embeddings.py
│   ├── pipeline.py
│
├── predictions_code.py
├── run_standalone.py
├── sitecustomize.py
├── requirements.txt
└── README.md

🧪 Data Understanding & EDA Summary
The dataset is clean, diverse, and rich in behavioral structure.
The classification problem is highly imbalanced, with Joe representing only a small fraction of all sessions.

Key opportunities identified during EDA:

- Local‑time normalization reveals true temporal habits hidden in GMT.

- Domain‑level aggregation captures browsing preferences and concentration patterns.

- Behavioral intensity metrics quantify engagement and navigation style.

- Environment attributes (browser, OS, locale, location) reflect stable user preferences.

These insights form a strong foundation for building a robust pipeline capable of distinguishing Joe from other users.

🕒 Temporal Split (Avoiding Leakage)
The goal is to identify Joe’s sessions explicitly, not treat them as anomalies.
Training, validation, and test sets contain sessions from all users.

To avoid temporal leakage:

- older sessions → training

- middle‑period sessions → validation

- most recent sessions → test

This setup reflects real‑world deployment, where the model must generalize to future behavior and consistently identify Joe even when his activity appears normal.

🧩 Feature Engineering
1. Temporal Features
Session timestamps are provided in GMT.
Using timezone_lookup.pkl, each timestamp is converted to the user’s local timezone.

Extracted features:

- local_hour

- local_day_of_week

- is_local_weekend

- time_bin (morning, afternoon, evening, night)

- is_night_session

These features reveal true behavioral patterns that were hidden in GMT.

2. Behavioral Features
Domain‑level behavior captures how users browse, not just when.

Extracted:

- number of sites

- total duration

- average duration per site

- duration variance

- entropy (diversity)

These features encode:

- habits

- intensity

- diversity

- preferences

- engagement

- navigation style

3. Environment Features
Stable user context:

- browser

- operating system

- locale

- location

- gender

These variables often correlate with long‑term browsing patterns.

4. Embedding Features (NLP)
Joe has a behavioral signature:

- the types of sites he visits

- the semantic meaning of those domains

- the sequence of domains

- the “topic flow” of his sessions

Using SentenceTransformer MiniLM‑L6‑v2, each session is embedded into a semantic vector.

Embeddings allow the model to understand:

- topic similarity

- domain semantics

- session clustering

- behavioral consistency

This adds massive predictive power beyond raw counts.

⚖️ Handling Class Imbalance
Joe represents ~0.5% of the dataset.
Training on raw distribution would cause the model to ignore Joe entirely.

Applied controlled undersampling ratios:

- 1:5

- 1:9

These preserve data integrity while allowing the model to learn meaningful patterns.

🧪 Experiment Tracking (Lightweight)
A lightweight experiment‑tracking system was implemented using pandas.
Each experiment logs:

- model configuration

- undersampling strategy

- precision, recall, F1

- temporal drift

- selected threshold

This enables:

- consistent model comparison

- reproducibility

- selection of the most stable model

Example fields:

python
df["model_index"] = i
df["undersampling"] = str(res["undersampling"])
df["config"] = str(res["model"])
df["recall"] = res["recall"]
df["precision"] = res["precision"]
df["drift"] = res["drift"]
df["threshold"] = res["threshold"]

📉 Model Selection & Temporal Drift
Behavioral data changes over time.
A model with high validation recall may collapse when evaluated on future sessions.

Temporal drift measures how much performance deteriorates from validation → test.

The final model is selected based on:

- stability across time windows

- consistent recall and precision

- robustness to behavioral changes

The chosen model is XGBoost, which demonstrated the best temporal stability.

🔥 Feature Importance (XGBoost — Temporal & Behavioral)
Using XGBoost’s gain metric on the first 13 named features:

Feature	Importance
temporal__time_bin_night	32.73
temporal__is_night_session	30.76
temporal__local_hour	24.13
temporal__time_bin_evening	8.21
temporal__time_bin_afternoon	6.75
behavioral__total_duration	3.26
behavioral__duration_variance	3.10
behavioral__num_sites	2.84
behavioral__avg_duration_per_site	2.00
temporal__local_day_of_week	1.95
behavioral__site_entropy	1.62
temporal__time_bin_morning	1.59
temporal__is_local_weekend	0.43


Interpretation
- Temporal features dominate the model: Joe’s strongest behavioral signature is nighttime browsing, with clear peaks in late evening and night hours.

- Behavioral features contribute meaningfully but are less dominant.
Joe’s browsing behavior is characterized by longer sessions, variable durations, and lower diversity.

📈 Model Performance on Unseen Test Data
To ensure that the model generalizes beyond the training distribution, a fully unseen temporal test window was used for final evaluation. This dataset represents future user behavior and was never used during training or hyperparameter tuning.
The final XGBoost model was evaluated on the most recent sessions, which represent unseen future data. This test window is the most important one, as it reflects real‑world deployment conditions.

Detected Joe: 557

Class 0 (Not Joe)
- Precision: 0.9994
- Recall:    0.9812
- F1-score:  0.9902
- Support:   23878

Class 1 (Joe)
- Precision: 0.1939
- Recall:    0.8852
- F1-score:  0.3181
- Support:   122

Overall Accuracy:      0.9807

Interpretation
- The model successfully identifies 88.5% of Joe’s sessions in the unseen test dataset.

- Precision for Joe is naturally low due to extreme class imbalance (Joe ≈ 0.5% of all sessions).

- Non‑Joe performance remains extremely high, ensuring minimal disruption to normal users.

These results demonstrate that the model can reliably detect Joe’s sessions even when his behavior appears normal and even when evaluated on future data not seen during training. As expected in behavioral modeling, precision decreases over time as user patterns evolve, but the model maintains strong recall and temporal stability, ensuring consistent identification of Joe across changing conditions.


⚙️ Final Prediction Pipeline
"run_standalone.py" executes the full production‑ready prediction pipeline:

- Loads the serialized model and feature‑engineering pipeline

- Computes temporal, behavioral and environment features from raw session data

- Generates semantic session embeddings using Sentence‑Transformers

- Applies the final XGBoost classifier

- Applies the optimal decision threshold for binary classification

- Writes result.csv containing one prediction per line
(0 = Joe, 1 = Not Joe)

The script is fully standalone, reproducible, and does not depend on notebooks or Docker. It can be executed in any clean Python environment using only requirements.txt.

🐳 Running the Project - Local Execution (recommended)
1. Install dependencies: pip install -r requirements.txt

2. Run the standalone script: python run_standalone.py

3. Output: result.csv (This file contains one prediction per line:
0 = Joe, 1 = Not Joe)

🚀 Next Steps & Possible Improvements
Although the pipeline is complete and production‑ready, several enhancements could further improve robustness and long‑term maintainability:

1. SHAP‑based interpretability  
Provide deeper insight into how temporal features, behavioral patterns, environment features and semantic embeddings influence model decisions.

2. Online drift monitoring  
Continuously track changes in Joe’s behavior over time and trigger automatic retraining when drift is detected.

3. Domain clustering  
Group domains into semantic categories to reduce embedding noise and improve generalization.

54. Real‑time inference  
Deploy the model behind an API for live session scoring and continuous monitoring

✔️ Final Status
- The pipeline is fully reproducible, the model is serialized, temporal drift is analyzed, and the standalone script generates the final predictions without external dependencies.

- This project demonstrates a complete, production‑ready workflow for identifying Joe’s sessions.

⭐ Conclusion
This screening project delivers a complete end‑to‑end workflow for user‑session classification — from EDA and feature engineering to model development, temporal‑drift evaluation, and production preparation.
The final model is not only accurate but also temporally stable, supported by careful undersampling, semantic embeddings, and a robust feature‑engineering pipeline.

The standalone prediction script ensures full reproducibility, allowing results to be generated consistently across different machines without relying on notebooks or containerized environments.

Overall, the solution meets all challenge requirements and provides a strong foundation for future enhancements such as drift monitoring, and real‑time inference.