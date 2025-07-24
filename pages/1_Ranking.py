import streamlit as st
import pandas as pd

from src.inference_sim import rank, _load
from src.utils_display import (
    get_job_fields, get_cv_series, snippet,
    get_name_series, get_name_fallback
)

st.title("Ranking – Similaridade TF‑IDF")


out = _load()
if len(out) == 6:
    vec, Vj, Vc, dj, dc, comps = out
else:                    # fallback
    vec, Vj, Vc, dj, dc = out
    comps = None

dj["job_id"]       = dj["job_id"].astype(str)
dc["candidate_id"] = dc["candidate_id"].astype(str)

def label_row(r):
    titulo, cliente, *_ = get_job_fields(r)
    return f"{r['job_id']} - {titulo or ''} - {cliente or ''}".strip(" -")

choice = st.selectbox("Selecione a vaga", dj.apply(label_row, axis=1))
jid = choice.split(" - ", 1)[0]
topk = st.slider("Top K", 10, 500, 100)

df_rank = rank(jid, topk=topk, vec=vec, Vj=Vj, Vc=Vc, dj=dj, comps=comps).astype({"candidate_id": str})

if "nome" in dc.columns:
    name_map = dc.set_index("candidate_id")["nome"].fillna("").to_dict()
else:
    ns = get_name_series(dc)
    if ns.eq("").all():
        ns = get_name_fallback(dc)
    name_map = ns.to_dict()


cvs = get_cv_series(dc)
cvs.index = dc["candidate_id"]

df_rank["nome"]       = df_rank["candidate_id"].map(name_map).replace("", None).fillna(df_rank["candidate_id"])
df_rank["cv_snippet"] = df_rank["candidate_id"].map(cvs).fillna("").apply(lambda x: snippet(x, 45))


df_rank = df_rank.sort_values("score", ascending=False).reset_index(drop=True)
df_rank["rank"] = df_rank.index + 1
df_rank["pct"]  = (1 - (df_rank["rank"] - 1) / max(len(df_rank)-1, 1)) * 100
df_rank["score"] = df_rank["score"].round(4)
df_rank["pct"]   = df_rank["pct"].round(1)

row_job = dj.loc[dj["job_id"] == jid].iloc[0]
titulo, cliente, area, princ, comp = get_job_fields(row_job)
st.subheader(f"Vaga: {area or titulo} ({jid})")
with st.expander("Descrição da vaga (clique para ver)"):
    if princ:
        st.markdown("**Principais atividades**")
        st.write(princ)
    if comp:
        st.markdown("**Competências técnicas e comportamentais**")
        st.write(comp)

cols_show = ["candidate_id", "nome", "score", "pct", "cv_snippet"]
st.dataframe(df_rank[cols_show], use_container_width=True, height=600)

st.download_button(
    "Baixar CSV",
    df_rank[cols_show].to_csv(index=False).encode("utf-8"),
    f"ranking_{jid}.csv"
)

st.caption("Score = similaridade cosseno (TF‑IDF). Pct = posição percentual dentro do Top K.")
