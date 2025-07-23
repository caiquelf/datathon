import streamlit as st
import pandas as pd
from src.data_loader import load_applicants
import json
from src.inference_sim import rank, _load
from src.utils_display import (get_name_series, get_cv_series, snippet,
                               get_job_fields, get_name_fallback)

from src.utils_display import get_job_fields

st.title("🏆 Ranking – Similaridade TF‑IDF")

# ===== Artefatos =====
vec, Vj, Vc, dj, dc = _load()
dj["job_id"]       = dj["job_id"].astype(str)
dc["candidate_id"] = dc["candidate_id"].astype(str)

# ===== Opções do select: id + título + cliente =====
def label_row(r):
    titulo, cliente, *_ = get_job_fields(r)
    return f"{r['job_id']} - {titulo or ''} - {cliente or ''}".strip(" -")

labels = dj.apply(label_row, axis=1)
choice = st.selectbox("Selecione a vaga", labels, key="vaga_sel")
jid = choice.split(" - ", 1)[0]
topk = st.slider("Top K", 10, 500, 100, key="topk_sel")

# ===== Ranking =====
df_rank = rank(jid, topk=topk).astype({"candidate_id": str})

# ---- mapa de nomes direto do JSON bruto ----
apps_raw = load_applicants()              # dict {cid: {...}}
def pick_name(blob):
    if not isinstance(blob, dict):
        try:
            blob = json.loads(blob)
        except Exception:
            return ""
    # tenta caminhos usuais
    for k1 in ("infos_basicas","informacoes_basicas","informacoes_pessoais","dados_pessoais"):
        d = blob.get(k1)
        if isinstance(d, dict):
            if d.get("nome"):
                return str(d["nome"])
        elif isinstance(d, list):
            for it in d:
                if isinstance(it, dict) and it.get("nome"):
                    return str(it["nome"])
    # fallback direto
    return str(blob.get("nome",""))

name_map = {str(cid): pick_name(data) for cid, data in apps_raw.items()}

# ---- CV/snippet (continua usando os artefatos) ----
from src.utils_display import snippet, get_cv_series, get_name_fallback, get_name_series
cvs = get_cv_series(dc)
cvs.index = dc["candidate_id"].astype(str)

df_rank["nome"]       = df_rank["candidate_id"].map(name_map).replace("", None)
# se ainda vier vazio, usa fallback do meta
if df_rank["nome"].isna().all():
    names_meta = get_name_series(dc)
    names_meta.index = dc["candidate_id"].astype(str)
    df_rank["nome"] = df_rank["candidate_id"].map(names_meta)

df_rank["nome"] = df_rank["nome"].fillna(df_rank["candidate_id"])

df_rank["cv_snippet"] = df_rank["candidate_id"].map(cvs).fillna("").apply(lambda x: snippet(x, 45))

# ---- ordenar + rank/pct ----
df_rank = df_rank.sort_values("score", ascending=False).reset_index(drop=True)
df_rank["rank"] = df_rank.index + 1
if len(df_rank) > 1:
    df_rank["pct"] = (1 - (df_rank["rank"] - 1) / (len(df_rank) - 1)) * 100
else:
    df_rank["pct"] = 100.0

df_rank["score"] = df_rank["score"].round(4)
df_rank["pct"]   = df_rank["pct"].round(1)

cols_show = ["rank","candidate_id","nome","score","pct","cv_snippet"]


# nomes / cvs
names_main = get_name_series(dc)
if names_main.eq("").all():
    names_main = get_name_fallback(dc)
cvs = get_cv_series(dc)

names_main.index = dc["candidate_id"]
cvs.index        = dc["candidate_id"]

# ordena por score e calcula posição e porcentagem
df_rank = df_rank.sort_values("score", ascending=False).reset_index(drop=True)
df_rank["rank"] = df_rank.index + 1
if len(df_rank) > 1:
    df_rank["pct"] = (1 - (df_rank["rank"] - 1) / (len(df_rank) - 1)) * 100
else:
    df_rank["pct"] = 100.0
df_rank["pct"]   = df_rank["pct"].round(1)
df_rank["score"] = df_rank["score"].round(4)



# ===== Header da vaga com os campos que você pediu =====
row_job = dj.loc[dj["job_id"] == jid].iloc[0]
titulo, cliente, area, princ, comp = get_job_fields(row_job)

st.subheader(f"Vaga: {area or titulo} ({jid})")

with st.expander("Descrição da vaga (clique para ver)", expanded=False):
    if princ:
        st.markdown("**Principais atividades**")
        st.write(princ)
    if comp:
        st.markdown("**Competências técnicas e comportamentais**")
        st.write(comp)

# ===== Tabela =====
cols_show = ["candidate_id","nome","score","pct","cv_snippet"]
st.dataframe(df_rank[cols_show], use_container_width=True, height=600)

st.download_button("Baixar CSV",
                   df_rank[cols_show].to_csv(index=False).encode("utf-8"),
                   f"ranking_{jid}.csv")

st.caption("Score = similaridade cosseno (TF‑IDF). Pct = posição percentual dentro do Top K.")
