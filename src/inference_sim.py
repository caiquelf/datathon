from pathlib import Path
import joblib
from scipy import sparse
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

ART = Path(__file__).resolve().parents[1] / "artifacts"

@st.cache_resource
def _load():
    vec = joblib.load(ART/"tfidf.joblib")
    Vj  = sparse.load_npz(ART/"V_jobs.npz")
    Vc  = sparse.load_npz(ART/"V_cands.npz")
    dj  = pd.read_pickle(ART/"jobs_meta.pkl")
    dc  = pd.read_pickle(ART/"cands_meta.pkl")
    return vec, Vj, Vc, dj, dc

def rank(job_id: str, topk: int = 100) -> pd.DataFrame:
    _, Vj, Vc, dj, dc = _load()
    idx = dj.index[dj["job_id"] == job_id]
    if len(idx) == 0:
        return pd.DataFrame(columns=["candidate_id","score"])
    jvec   = Vj[idx[0]]
    scores = cosine_similarity(Vc, jvec).ravel()
    order  = np.argsort(-scores)[:topk]
    return pd.DataFrame({"candidate_id": dc.iloc[order]["candidate_id"].values,
                         "score": scores[order]})
