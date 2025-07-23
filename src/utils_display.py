import json, re
import pandas as pd

def find_col(df, kws):
    for c in df.columns:
        if any(k in c.lower() for k in kws):
            return c
    return None

def _parse_json_like(x):
    if isinstance(x, dict):
        return x
    if isinstance(x, str) and x.strip().startswith("{") and x.strip().endswith("}"):
        try:
            return json.loads(x)
        except Exception:
            return None
    return None

def get_name_series(dc: pd.DataFrame) -> pd.Series:
    """
    Retorna Series com o nome do candidato.
    Olha:
      - colunas diretas (nome/name)
      - dicts em 'infos_basicas', 'informacoes_basicas', 'informacoes_pessoais' (lista ou dict)
      - fallback titulo_profissional / cargo_atual
    """
    import pandas as pd


    for col in ["nome", "name", "full_name"]:
        if col in dc.columns:
            return dc[col].astype(str)

    blk_cols = [c for c in dc.columns if any(k in c.lower()
                 for k in ["infos_basicas", "informacoes_basicas", "informacoes_pessoais", "dados_pessoais"])]
    if blk_cols:
        def extract(v):
            d = _parse_json_like(v)
            if isinstance(d, dict):
                for k in ("nome","name","full_name"):
                    if d.get(k):
                        return d[k]
                for sub in d.values():
                    dd = _parse_json_like(sub)
                    if isinstance(dd, dict):
                        for k in ("nome","name","full_name"):
                            if dd.get(k):
                                return dd[k]
                    elif isinstance(dd, list):
                        for item in dd:
                            ii = _parse_json_like(item)
                            if isinstance(ii, dict):
                                for k in ("nome","name","full_name"):
                                    if ii.get(k):
                                        return ii[k]
            return None

        s = pd.Series([None]*len(dc), index=dc.index)
        for col in blk_cols:
            s = s.fillna(dc[col].apply(extract))
        if s.notna().any():
            return s.fillna("")

    for col in ["titulo_profissional", "cargo_atual", "area_atuacao"]:
        if col in dc.columns:
            return dc[col].astype(str)

    return pd.Series([""]*len(dc), index=dc.index)


    blk_cols = [c for c in dc.columns if any(k in c.lower() for k in
                  ["infos_basicas", "informacoes_basicas", "informacoes_pessoais", "dados_pessoais"])]
    if blk_cols:
        def extract(v):
            d = _parse_json_like(v)
            if isinstance(d, dict):
                for k in ("nome", "name", "full_name"):
                    if d.get(k):
                        return d[k]
            return None
        s = pd.Series([None]*len(dc), index=dc.index)
        for col in blk_cols:
            s = s.fillna(dc[col].apply(extract))
        if s.notna().any():
            return s.fillna("")

    for col in ["titulo_profissional", "cargo_atual", "area_atuacao"]:
        if col in dc.columns:
            return dc[col].astype(str)

    return pd.Series([""]*len(dc), index=dc.index)


def get_cv_series(dc: pd.DataFrame) -> pd.Series:
    for c in ["cv_pt","cv_en","cv","curriculo"]:
        if c in dc.columns:
            return dc[c].astype(str)
    for blk in ["informacoes_profissionais","resumo_profissional"]:
        if blk in dc.columns:
            return dc[blk].astype(str)
    return pd.Series([""]*len(dc), index=dc.index)

def snippet(txt, n_words=40):
    w = re.sub(r"\s+", " ", str(txt)).split()
    return " ".join(w[:n_words]) + ("..." if len(w) > n_words else "")

def clean_jsonish(text):
    return re.sub(r"'[^']+':", "", str(text))

def job_title(row: pd.Series):
    for k in ["titulo","title","nome_da_vaga","perfil_vaga","principais_atividades"]:
        if k in row.index and pd.notna(row[k]) and str(row[k]).strip():
            return snippet(row[k], 15)

    return snippet(clean_jsonish(row["text_job"]), 12)

def pick_from_row(row, keys):
    """Procura em colunas e dentro de dicts por uma chave da lista."""
    for col in row.index:
        v = row[col]
        if isinstance(v, str):
            for k in keys:
                if k in col.lower():
                    return v
        d = _parse_json_like(v)
        if isinstance(d, dict):
            for k in keys:
                for dk, dv in d.items():
                    if k in dk.lower() and dv:
                        return dv
    return ""

def get_job_fields(row):
    titulo  = pick_from_row(row, ["titulo_vaga","titulo","title"])
    cliente = pick_from_row(row, ["cliente","solicitante_cliente"])
    area    = pick_from_row(row, ["areas_atuacao","area_atuacao"])
    princ   = pick_from_row(row, ["principais_atividades"])
    comp    = pick_from_row(row, ["competencia_tecnicas_e_comportamentais","competencias_tecnicas"])
    return titulo, cliente, area, princ, comp

def get_name_fallback(dc: pd.DataFrame):

    for col in ["nome","name","titulo_profissional","cargo_atual","area_atuacao"]:
        if col in dc.columns:
            return dc[col].astype(str)
    return pd.Series([""]*len(dc), index=dc.index)