from typing import Dict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def similarity_score(resume_text: str, job_text: str) -> Dict[str, float]:
    """Return cosine similarity (0..1) between resume and job text."""
    vec = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2))
    X = vec.fit_transform([resume_text, job_text]).toarray()
    v1, v2 = X[0], X[1]
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) or 1.0
    sim = float(np.dot(v1, v2) / denom)
    return {"similarity": round(sim, 3)}