import json
import re
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import Normalizer

ART = Path(__file__).resolve().parents[1] / "artifacts"


STOP = {
    "de", "da", "do", "das", "dos", "pra", "pro", "em", "para", "por", "com",
    "sem", "e", "ou", "na", "no", "nas", "nos", "ao", "aos",
    "nivel", "profissional", "conhecimentos", "tecnicos", "certificacoes",
    "outras", "area", "atuacao", "remuneracao", "curriculo", "cv", "experiencia",
    "anos", "ultimos", "meses"
}

CAND_KEYS_KEEP = [
    "titulo_profissional", "area_atuacao",
    "conhecimentos_tecnicos", "certificacoes", "outras_certificacoes",
    "cargo_atual", "projeto_atual"
]

TOP_TERMS = 15

TRY_K = [4, 6, 8, 10]


# ----------------------- Helpers -----------------------
def clean(txt: str) -> str:
    txt = str(txt).lower()
    txt = re.sub(r"\d+", " ", txt)
    txt = re.sub(r"[^\w\s]", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return " ".join(t for t in txt.split() if t not in STOP)


def extract_signal(txt: str) -> str:
    """Remove chaves/caracteres e deixa só conteúdo bruto."""
    return re.sub(r"[{}\"\[\]]", " ", str(txt))


def build_clean_series(dc: pd.DataFrame) -> pd.Series:
    """Mantém apenas valores dos campos relevantes do dump `text_cand`."""
    out = []
    for raw in dc["text_cand"].fillna(""):
        parts = [p for p in re.split(r",\s*'", raw) if ":" in p]
        keep_values = []
        for p in parts:
            k, v = p.split(":", 1)
            k = k.lower()
            if any(k1 in k for k1 in CAND_KEYS_KEEP):
                keep_values.append(v)
        if not keep_values:  # fallback
            keep_values = [raw]
        out.append(extract_signal(" ".join(keep_values)))
    return pd.Series(out, index=dc.index)


# ----------------------- Main -----------------------
def main():

    cand_pkl = ART / "cands_meta.pkl"
    if cand_pkl.exists():
        dc = pd.read_pickle(cand_pkl)
    else:
        import lzma, pickle
        with lzma.open(ART / "cands_meta_light.pkl.xz", "rb") as f:
            dc = pickle.load(f)

    texts = build_clean_series(dc).map(clean)


    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=120,       
        max_df=0.35,
        sublinear_tf=True
    )
    Vc = vec.fit_transform(texts)
    joblib.dump(vec, ART / "tfidf_cluster.joblib")

    svd = TruncatedSVD(n_components=300, random_state=42)
    Vc_red = svd.fit_transform(Vc)
    Vc_red = Normalizer(copy=False).fit_transform(Vc_red)


    best_score, best_k, best_labels, best_km = -1, None, None, None
    for k in TRY_K:
        km = KMeans(n_clusters=k, n_init="auto", random_state=42)
        lbl = km.fit_predict(Vc_red)
        sc = silhouette_score(Vc_red, lbl, sample_size=2000, random_state=42)
        if sc > best_score:
            best_score, best_k, best_labels, best_km = sc, k, lbl, km

    print(f"Melhor k={best_k} | silhouette={best_score:.3f}")

    out = dc[["candidate_id", "text_cand"]].copy()
    out["cluster"] = best_labels
    out.to_csv(ART / "clusters_assignments.csv", index=False)

    joblib.dump(
        {"k": best_k, "silhouette": best_score},
        ART / "clusters_meta.joblib"
    )

    rows = []
    for c in range(best_k):
        idx = np.flatnonzero(best_labels == c)
        if idx.size == 0:
            rows.append(sparse.csr_matrix((1, Vc.shape[1])))
            continue
        avg = Vc[idx].sum(axis=0) / idx.size
        rows.append(sparse.csr_matrix(avg))
    means = sparse.vstack(rows)

    feats = np.array(vec.get_feature_names_out())
    tops = {}
    for c in range(best_k):
        col = means[c].toarray().ravel()
        top_idx = col.argsort()[-TOP_TERMS:][::-1]
        tops[c] = feats[top_idx].tolist()

    with open(ART / "clusters_keywords.json", "w", encoding="utf-8") as f:
        json.dump(tops, f, ensure_ascii=False, indent=2)

    labels = {c: " / ".join(tops[c][:3]) for c in tops}
    with open(ART / "clusters_labels.json", "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    print("Clusters salvos.")


if __name__ == "__main__":
    main()
