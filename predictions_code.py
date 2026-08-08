import sys
import json
import requests
import pickle
import importlib
import zipfile
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os

import warnings
# Ignorar el UserWarning específico de XGBoost al cargar modelos serializados
warnings.filterwarnings(
    "ignore",
    message=r".*If you are loading a serialized model.*",
    category=UserWarning,
)

# Ignorar FutureWarning (por ejemplo el de huggingface_hub)
warnings.filterwarnings("ignore", category=FutureWarning)

# Opcional: ignorar todos los warnings de una librería concreta
import logging
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)


############################################################
# 1. Download all required files from GitHub
############################################################

def download_from_github(path, save_as=None):
    """Download small files directly from GitHub."""
    url = f"https://raw.githubusercontent.com/brunoncarlos/toptal-screening-project/main/{path}"
    r = requests.get(url)
    if r.status_code != 200:
        raise Exception(f"Error downloading {path}: {r.status_code}")
    save_as = save_as or path
    with open(save_as, "wb") as f:
        f.write(r.content)
    print(f"✔ Downloaded: {save_as}")

def download_from_github_lfs(path, save_as=None):
    """Download large LFS files (domain cache)."""
    url = f"https://media.githubusercontent.com/media/brunoncarlos/toptal-screening-project/main/{path}"
    r = requests.get(url, stream=True)
    if r.status_code != 200:
        raise Exception(f"Error downloading {path}: {r.status_code}")
    save_as = save_as or path
    with open(save_as, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024*1024):
            if chunk:
                f.write(chunk)
    print(f"✔ Downloaded (LFS): {save_as}")

# Create folders
os.makedirs("src", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("output", exist_ok=True)

# Download source code
download_from_github("src/__init__.py")
download_from_github("src/temporal_features.py")
download_from_github("src/behavioral_features.py")
download_from_github("src/pipeline.py")
download_from_github("src/compute_session_embeddings.py")

# Download models
download_from_github("models/fe_pipeline.pkl")
download_from_github("models/model_selected.pkl")
download_from_github("models/threshold_model_selected.json")

# Download data
download_from_github("data/timezone_lookup.pkl")
download_from_github("data/verify.zip")
download_from_github_lfs("data/domain_cache.pkl")

############################################################
# 2. Load all required objects BEFORE importing modules
############################################################

with open("data/timezone_lookup.pkl", "rb") as f:
    timezone_lookup = pickle.load(f)

with open("models/fe_pipeline.pkl", "rb") as f:
    fe_pipeline = pickle.load(f)

with open("data/domain_cache.pkl", "rb") as f:
    domain_cache = pickle.load(f)

with open("models/model_selected.pkl", "rb") as f:
    model_selected = pickle.load(f)

with open("models/threshold_model_selected.json", "r") as f:
    threshold = json.load(f)["threshold"]

# Make objects available globally
import __main__
__main__.timezone_lookup = timezone_lookup
__main__.domain_cache = domain_cache

############################################################
# 3. Import project modules
############################################################

import src.pipeline as pipeline
import src.temporal_features as temporal_features
import src.behavioral_features as behavioral_features
import src.compute_session_embeddings as compute_session_embeddings

# Reload to ensure fresh state
importlib.reload(pipeline)
importlib.reload(temporal_features)
importlib.reload(behavioral_features)
importlib.reload(compute_session_embeddings)

############################################################
# 4. Register modules and functions
############################################################

sys.modules['pipeline'] = pipeline
sys.modules['temporal_features'] = temporal_features
sys.modules['behavioral_features'] = behavioral_features
sys.modules['compute_session_embeddings'] = compute_session_embeddings

# Register pipeline functions globally
__main__.temporal_transformer_func = pipeline.temporal_transformer_func
__main__.behavioral_transformer_func = pipeline.behavioral_transformer_func
__main__.freq_encode = pipeline.freq_encode
__main__.identity_transform = pipeline.identity_transform

# Inject timezone lookup and domain cache into modules
temporal_features.timezone_lookup = timezone_lookup
compute_session_embeddings.DOMAIN_CACHE = domain_cache

############################################################
# 5. Load verification dataset
############################################################


output_dir = os.path.join(os.getcwd(), "catchjoe")
with zipfile.ZipFile("data/verify.zip", "r") as z:
    z.extractall(output_dir)

def load_json(path):
    """Load JSON file with UTF-8 handling."""
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)

verify_data = load_json(os.path.join(output_dir, "verify.json"))
verify_df = pd.json_normalize(verify_data)
del verify_data

# Insert dummy user_id (required by pipeline)
verify_df.insert(0, "user_id", 0)

############################################################
# 6. Apply feature engineering pipeline
############################################################

X_pred = fe_pipeline.transform(verify_df)

############################################################
# 7. Compute session embeddings (full feature set)
############################################################

# Load embedding model
st_model = SentenceTransformer("all-MiniLM-L6-v2")
compute_session_embeddings.model = st_model

# Build full embedding-based features
X_pred = compute_session_embeddings.build_full_features(
    X_pred,
    user_col=0,
    sites_col=1
)

############################################################
# 8. Predict using final model + threshold
############################################################

probas = model_selected.predict_proba(X_pred)[:, 1]
preds = (probas >= threshold).astype(int)

# Convert to challenge format:
# 0 = Joe, 1 = Not Joe
labels = np.where(preds == 1, 0, 1)

############################################################
# 9. Save result.csv (final output)
############################################################

result_df = pd.DataFrame({"label": labels})
result_df.to_csv("output/result.csv", index=False)

print("✔ result.csv created successfully!")

def main():
    return labels