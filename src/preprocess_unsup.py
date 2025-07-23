import pandas as pd
from typing import Union, Dict, List
from src.data_loader import load_jobs, load_applicants

JOB_TEXT_KEYS  = ("perfil_vaga","descri","descrição","requisito","perfil","respons")
JOB_TITLE_KEYS = ("titulo","title","perfil_vaga","nome")

CAND_TEXT_KEYS = ("cv_pt","cv_en","cv","resumo","experi","informacoes_profissionais",
                  "informacoes_basicas","conhec","skill","curriculo")
CAND_NAME_KEYS = ("nome","name","titulo_profissional","cargo_atual")

def _df_jobs(jobs: Union[Dict, List]) -> pd.DataFrame:
    if isinstance(jobs, dict):
        df = pd.DataFrame(jobs).T.reset_index().rename(columns={"index":"job_id"})
    else:
        df = pd.DataFrame(jobs)
        if "job_id" not in df.columns: df["job_id"] = df.index.astype(str)
    return df

def _df_apps(apps: Union[Dict, List]) -> pd.DataFrame:
    if isinstance(apps, dict):
        df = pd.DataFrame(apps).T.reset_index().rename(columns={"index":"candidate_id"})
    else:
        df = pd.DataFrame(apps)
        if "candidate_id" not in df.columns: df["candidate_id"] = df.index.astype(str)
    return df

def build_jobs_and_candidates():
    jobs = load_jobs()
    apps = load_applicants()

    df_jobs = _df_jobs(jobs)
    df_apps = _df_apps(apps)

    jcols = [c for c in df_jobs.columns if any(k in c.lower() for k in JOB_TEXT_KEYS)]
    ccols = [c for c in df_apps.columns if any(k in c.lower() for k in CAND_TEXT_KEYS)]

    df_jobs["text_job"]  = df_jobs[jcols].astype(str).agg("\n".join, axis=1) if jcols else ""
    df_apps["text_cand"] = df_apps[ccols].astype(str).agg("\n".join, axis=1)  if ccols else ""


    jtitle = [c for c in df_jobs.columns if any(k in c.lower() for k in JOB_TITLE_KEYS)]
    ncols  = [c for c in df_apps.columns if any(k in c.lower() for k in CAND_NAME_KEYS)]
    cvcols = [c for c in df_apps.columns if any(k in c.lower() for k in ("cv_pt","cv_en","cv","curriculo"))]

    jobs_meta = df_jobs[["job_id","text_job"] + jtitle].copy()
    cands_meta = df_apps[["candidate_id","text_cand"] + ncols + cvcols].copy()

    return jobs_meta, cands_meta
