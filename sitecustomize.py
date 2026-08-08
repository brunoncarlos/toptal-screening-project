# sitecustomize.py — runs at Python startup
import sys
try:
    import src.temporal_features as _src_temporal_features
    import src.behavioral_features as _src_behavioral_features
    import src.compute_session_embeddings as _src_compute_session_embeddings
    import src.pipeline as _src_pipeline

    sys.modules.setdefault('temporal_features', _src_temporal_features)
    sys.modules.setdefault('behavioral_features', _src_behavioral_features)
    sys.modules.setdefault('compute_session_embeddings', _src_compute_session_embeddings)
    sys.modules.setdefault('pipeline', _src_pipeline)
except Exception:
    pass
