import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ART = Path(__file__).resolve().parents[1] / "artifacts"

st.title("Clusters – Personas de candidatos")

# ----------------- Load -----------------
try:
 
    parq = ART / "clusters_assignments.parquet"
    csv  = ART / "clusters_assignments.csv"
    if parq.exists():
        df = pd.read_parquet(parq)
    else:
        df = pd.read_csv(csv)

    meta = joblib.load(ART / "clusters_meta.joblib")

    with open(ART / "clusters_keywords.json", encoding="utf-8") as f:
        kw = json.load(f)


    labels_path = ART / "clusters_labels.json"
    if labels_path.exists():
        with open(labels_path, encoding="utf-8") as f:
            labels = json.load(f)
    else:
        labels = {int(k): " / ".join(v[:3]) for k, v in kw.items()}

    df["cluster"] = df["cluster"].astype(int)
    df["cluster_name"] = df["cluster"].map(labels).fillna(
        df["cluster"].map(lambda c: " / ".join(kw.get(str(c), [])[:3]))
    )

except FileNotFoundError:
    st.warning("Clusters ainda não gerados. Rode: `python -m src.clustering_unsup`")
    st.stop()

st.metric("Silhouette", f"{meta['silhouette']:.3f}")

sizes = df["cluster"].value_counts().sort_index()
st.bar_chart(sizes)

st.subheader("Palavras‑chave por cluster")
kw_table = pd.DataFrame({
    "cluster": list(kw.keys()),
    "keywords": [", ".join(v) for v in kw.values()]
})
st.dataframe(kw_table, use_container_width=True, height=300)
st.subheader("Amostras por cluster")
for c in sizes.index:
    st.markdown(f"**Cluster {c} – {sizes[c]} candidatos**")
    sample = df[df["cluster"] == c].sample(min(5, sizes[c]), random_state=42)
    st.dataframe(sample[["candidate_id", "cluster_name", "text_cand"]],
                 use_container_width=True, height=180)

st.download_button("Baixar clusters.csv",
                   df.to_csv(index=False).encode("utf-8"),
                   "clusters.csv")
