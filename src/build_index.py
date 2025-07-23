import joblib
from pathlib import Path
from scipy import sparse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from src.preprocess_unsup import build_jobs_and_candidates

ART = Path(__file__).resolve().parents[1] / "artifacts"
ART.mkdir(parents=True, exist_ok=True)

def main():
    df_jobs, df_apps = build_jobs_and_candidates()
    texts = pd.concat([df_jobs["text_job"], df_apps["text_cand"]], ignore_index=True)

    vec = TfidfVectorizer(max_features=60000, ngram_range=(1,2), min_df=2)
    vec.fit(texts)

    V_jobs  = vec.transform(df_jobs["text_job"])
    V_cands = vec.transform(df_apps["text_cand"])

    joblib.dump(vec, ART/"tfidf.joblib")
    sparse.save_npz(ART/"V_jobs.npz", V_jobs)
    sparse.save_npz(ART/"V_cands.npz", V_cands)

    df_jobs.to_pickle(ART/"jobs_meta.pkl")
    df_apps.to_pickle(ART/"cands_meta.pkl")

    print("OK: artefatos salvos em artifacts/")

if __name__ == "__main__":
    main()
