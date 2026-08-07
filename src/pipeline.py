# 1. Imports
import json
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer

# 2. Import your feature modules
from temporal_features import resolve_timezone, add_local_time_features
from behavioral_features import extract_behavioral_features

# 3. Temporal transformer
def temporal_transformer_func(df):
    df = df.copy()

    # 3.1 Resolve timezone
    df["timezone"] = df["location"].apply(resolve_timezone)

    # 3.2 Add local time features
    df = add_local_time_features(df)

    # 3.3 Return only temporal columns
    return df

temporal_transformer = FunctionTransformer(
    temporal_transformer_func,
    validate=False
)

# 4. Behavioral transformer
def behavioral_transformer_func(col):
    feats = []

    for sites in col.iloc[:, 0]:

        # 4.1 List input
        if isinstance(sites, list):
            feats.append(extract_behavioral_features(sites))
            continue

        # 4.2 JSON-like string input
        if isinstance(sites, str):
            try:
                sites_parsed = json.loads(sites.replace("'", '"'))
                if isinstance(sites_parsed, list):
                    feats.append(extract_behavioral_features(sites_parsed))
                    continue
            except Exception:
                pass

        # 4.3 Fallback
        feats.append(extract_behavioral_features([]))

    return pd.DataFrame(feats)

behavioral_transformer = FunctionTransformer(
    behavioral_transformer_func,
    validate=False
)

# 5. Environment transformer
def freq_encode(col):
    if isinstance(col, pd.DataFrame):
        col = col.iloc[:, 0]

    s = pd.Series(col)
    freqs = s.value_counts(dropna=False).to_dict()
    encoded = s.map(freqs).fillna(0)

    return pd.DataFrame(encoded)

env_ohe_cols = ["browser", "os", "gender"]
env_freq_cols = ["locale", "location"]

env_preprocessor = ColumnTransformer(
    transformers=[
        ("env_ohe", OneHotEncoder(handle_unknown="ignore"), env_ohe_cols),
        ("locale_freq", FunctionTransformer(freq_encode), ["locale"]),
        ("location_freq", FunctionTransformer(freq_encode), ["location"]),
    ],
    remainder="drop"
)

# 6. Identity transformer (for user_id and raw sites)
def identity_transform(x):
    return x

id_transformer = FunctionTransformer(identity_transform, validate=False)

# 7. Final unified pipeline
fe_pipeline_all = ColumnTransformer(
    transformers=[
# The original pipeline expected a 'user_id' column during training.
# In production this column does NOT exist, but the pipeline structure must remain identical.
# I pass column index 0 as a dummy placeholder to keep the transformer slot aligned.
# The id_transformer does nothing, so this placeholder does not change any features or predictions.
        ("user_id", id_transformer, 0),
        ("sites_raw", id_transformer, ["sites"]),
        ("temporal", temporal_transformer, ["date", "time", "location"]),
        ("behavioral", behavioral_transformer, ["sites"]),
        ("env", env_preprocessor, env_ohe_cols + env_freq_cols),
    ],
    remainder="drop"
)
