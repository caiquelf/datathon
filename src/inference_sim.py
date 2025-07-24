from pathlib import Path
from scipy import sparse
import pandas as pd
import joblib, lzma, pickle
import numpy as np

ART = Path(__file__).resolve().parents[1] / "artifacts"

from pathlib import Path
from scipy import sparse
import pandas as pd
import joblib, lzma, pickle
import numpy as np

ART = Path(__file__).resolve().parents[1] / "artifacts"


def _load():
    vec = joblib.load(ART / "tfidf.joblib")
    Vj  = sparse.load_npz(ART / "V_jobs.npz")

    # tenta full, depois SVD
    Vc_path_full = ART / "V_cands.npz"
    Vc_path_comp = ART / "V_cands_compressed.npz"
    Vc_path_svd  = ART / "V_cands_svd.npz"

    comps = None
    if Vc_path_full.exists():
        Vc = sparse.load_npz(Vc_path_full)
    elif Vc_path_comp.exists():
        Vc = sparse.load_npz(Vc_path_comp)
    elif Vc_path_svd.exists():
        dat  = np.load(Vc_path_svd)
        Vc   = sparse.csr_matrix(dat["Vc"].astype("float32"))
        comps = dat["comps"].astype("float32")  # (k, F)
    else:
        raise FileNotFoundError("Nenhum V_cands* encontrado.")

    dj = pd.read_pickle(ART / "jobs_meta.pkl")

    cand_pkl = ART / "cands_meta.pkl"
    if cand_pkl.exists():
        dc = pd.read_pickle(cand_pkl)
    else:
        with lzma.open(ART / "cands_meta_light.pkl.xz", "rb") as f:
            dc = pickle.load(f)

    return vec, Vj, Vc, dj, dc, comps


def rank(jid, topk=100, vec=None, Vj=None, Vc=None, dj=None, comps=None):
    jidx = dj.index[dj["job_id"] == jid][0]
    vj = Vj[jidx]  


    if comps is not None and Vc.shape[1] != vj.shape[1]:
        vj_dense = vj.astype("float32").toarray()  # (1, F)
        vj_red   = vj_dense @ comps.T              # (1, k)
        sims = (Vc @ vj_red.T).ravel()
    else:
        sims = (Vc @ vj.T).ravel()


    sims = np.asarray(sims).ravel()
    best_idx = sims.argsort()[::-1][:topk]
    return pd.DataFrame({
        "candidate_id": dj.index if "candidate_id" in dj.columns else best_idx,
        "score": sims[best_idx]
    }).reset_index(drop=True)