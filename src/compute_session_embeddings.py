import numpy as np
import json
import pickle
from joblib import Parallel, delayed

# 2. Global cache for domain embeddings
DOMAIN_CACHE = {}

def normalize_domains(domains):
    """
    Normaliza la columna 'sites' para que siempre sea una lista de strings.
    Acepta:
    - lista de strings
    - lista de dicts con clave 'site'
    - string JSON con lista
    """
    if isinstance(domains, list) and all(isinstance(x, str) for x in domains):
        return domains

    if isinstance(domains, list) and all(isinstance(x, dict) for x in domains):
        return [d.get("site", "") for d in domains]

    if isinstance(domains, str):
        try:
            parsed = json.loads(domains.replace("'", '"'))
            if isinstance(parsed, list):
                if all(isinstance(x, dict) for x in parsed):
                    return [d.get("site", "") for d in parsed]
                if all(isinstance(x, str) for x in parsed):
                    return parsed
        except:
            pass

    return []

def embed_domain(domain):
    """
    Devuelve el embedding del dominio usando cache.
    Si no existe, lo calcula con model.encode().
    """
    if domain not in DOMAIN_CACHE:
        DOMAIN_CACHE[domain] = model.encode([domain], show_progress_bar=False)[0]
    return DOMAIN_CACHE[domain]

def embed_session(sites):
    """
    Embedding de una sesión:
    - mean embedding
    - std embedding
    - count de dominios
    """
    clean_domains = normalize_domains(sites)

    if len(clean_domains) == 0:
        dummy_vec = embed_domain("dummy.com")
        mean_vec = np.zeros_like(dummy_vec)
        std_vec  = np.zeros_like(dummy_vec)
        count    = np.array([0])
        return np.concatenate([mean_vec, std_vec, count])

    vectors = np.vstack([embed_domain(d) for d in clean_domains])
    mean_vec = vectors.mean(axis=0)
    std_vec  = vectors.std(axis=0)
    count    = np.array([len(clean_domains)])

    return np.concatenate([mean_vec, std_vec, count])

def compute_session_embeddings(X, sites_col=1, log_every=10000, parallel=False, n_jobs=2):
    """
    Calcula embeddings para todas las sesiones.
    """
    sites_series = X[:, sites_col]

    if not parallel:
        embeddings = []
        for i, sites in enumerate(sites_series):
            emb = embed_session(sites)
            embeddings.append(emb)
            if (i + 1) % log_every == 0:
                print(f"[embed] Procesadas {i+1:,} sesiones...")
        return np.vstack(embeddings)

    def wrapper(i, sites):
        emb = embed_session(sites)
        if (i + 1) % log_every == 0:
            print(f"[embed] Procesadas {i+1:,} sesiones...")
        return emb

    embeddings = Parallel(n_jobs=n_jobs)(
        delayed(wrapper)(i, sites) for i, sites in enumerate(sites_series)
    )
    return np.vstack(embeddings)

def build_user_features(X, session_embs, user_col=0):
    """
    Construye:
    - embedding promedio por usuario
    - distancia de cada sesión al embedding del usuario
    """
    user_ids = X[:, user_col]

    user_embeddings = {
        uid: session_embs[user_ids == uid].mean(axis=0)
        for uid in np.unique(user_ids)
    }

    dists = np.array([
        np.linalg.norm(sess - user_embeddings[uid])
        for uid, sess in zip(user_ids, session_embs)
    ])[:, None]

    X_no_user = np.delete(X, user_col, axis=1)
    X_no_user = np.delete(X_no_user, user_col, axis=1)

    return np.hstack([X_no_user, session_embs, dists])

def build_full_features(X, user_col=0, sites_col=1, log_every=10000, parallel=False, n_jobs=2):
    """
    Construye todas las features basadas en embeddings.
    """
    session_embs = compute_session_embeddings(
        X,
        sites_col=sites_col,
        log_every=log_every,
        parallel=parallel,
        n_jobs=n_jobs
    )

    return build_user_features(X, session_embs, user_col=user_col)
