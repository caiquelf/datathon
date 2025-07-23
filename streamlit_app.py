import streamlit as st

st.set_page_config(page_title='Decision AI', layout='wide')
st.title('Decision AI – Plataforma de Recrutamento Inteligente')

@st.cache_resource
def load_artifacts():
    from pathlib import Path
    import joblib, pandas as pd
    from scipy import sparse
    ART = Path("artifacts")
    vec = joblib.load(ART/"tfidf.joblib")
    Vj  = sparse.load_npz(ART/"V_jobs.npz")
    Vc  = sparse.load_npz(ART/"V_cands.npz")
    dj  = pd.read_pickle(ART/"jobs_meta.pkl")
    dc  = pd.read_pickle(ART/"cands_meta.pkl")
    return vec, Vj, Vc, dj, dc
st.markdown("""Use o menu à esquerda:
1. **Ranking** – Similaridade vaga ↔ candidato.  
2. **Clusters** – Personas de candidatos (KMeans).  
""")
