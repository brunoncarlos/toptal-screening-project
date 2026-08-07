# 1. Imports
import math
import numpy as np

# 2. Volume Features
def compute_num_sites(sites):
    """
    Number of sites visited in the session.
    High → exploratory user
    Low → focused user
    """
    return len(sites)

# 3. Duration Features
def compute_total_duration(sites):
    """
    Total browsing duration in the session.
    High → deep engagement
    Low → quick check-ins
    """
    return sum(s["length"] for s in sites)

# 4. Diversity Features (Entropy)
def compute_site_entropy(sites):
    """
    Shannon entropy of time distribution across sites.
    High entropy → diverse browsing
    Low entropy → focused browsing
    """
    durations = [s["length"] for s in sites]
    total = sum(durations)

    if total == 0:
        return 0.0

    probs = [d / total for d in durations]
    return -sum(p * math.log(p) for p in probs if p > 0)

# 5. Variance of time spent across sites
def compute_duration_variance(sites, avg_duration):
    """
    Variance of time spent across sites.
    """
    durations = [s["length"] for s in sites]
    if len(durations) <= 1:
        return 0.0
    return sum((d - avg_duration)**2 for d in durations) / len(durations)

# 6. Unified Feature Extractor
def extract_behavioral_features(sites):
    """
    Unified behavioral feature extractor.
    Includes:
    - Volume (num_sites)
    - Duration (total_duration, avg_duration_per_site)
    - Diversity (site_entropy)
    - Variability (duration_variance)
    """
    num_sites = compute_num_sites(sites)
    total_duration = compute_total_duration(sites)
    avg_duration = total_duration / num_sites if num_sites > 0 else 0
    entropy = compute_site_entropy(sites)
    duration_variance = compute_duration_variance(sites, avg_duration)

    return {
        "num_sites": num_sites,
        "total_duration": total_duration,
        "avg_duration_per_site": avg_duration,
        "site_entropy": entropy,
        "duration_variance": duration_variance
    }
