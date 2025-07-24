from pathlib import Path
import pandas as pd
import joblib, lzma, pickle
from scipy import sparse
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer

# raiz do projeto (decisaoApp)
ROOT = Path(__file__).resolve().parents[1]
ART  = ROOT / "artifacts"


# --- 1) cands_meta.pkl -> reduzir colunas + compressão -----------------
dc_full = pd.read_pickle(ART / "cands_meta.pkl")

# quais colunas eu gostaria *se existirem*
wish = ["candidate_id", "nome", "text_cand", "cv_pt"]
keep_cols = [c for c in wish if c in dc_full.columns]
missing   = set(wish) - set(keep_cols)
if missing:
    print("Aviso: colunas ausentes em cands_meta.pkl ->", missing)

dc = dc_full[keep_cols].copy()
dc["candidate_id"] = dc["candidate_id"].astype(str)

# compactar
with lzma.open(ART / "cands_meta_light.pkl.xz", "wb") as f:
    pickle.dump(dc, f, protocol=pickle.HIGHEST_PROTOCOL)
# 3) V_cands.npz -> salvar com compressão máxima (se já veio sem)
Vc = sparse.load_npz(ART / "V_cands.npz")
sparse.save_npz(ART / "V_cands_compressed.npz", Vc, compressed=True)

print("Feito! Suba apenas *_light.*, *.parquet e *_compressed.npz")



# src/shrink_clusters.py
from pathlib import Path
import pandas as pd

df = pd.read_csv(ART / "clusters_assignments.csv", dtype={"candidate_id": str})
df.to_parquet(ART / "clusters_assignments.parquet", compression="gzip", index=False)
print("OK -> clusters_assignments.parquet criado")


Vc = sparse.load_npz(ART / "V_cands_compressed.npz")

# menos dimensões + float16
N_COMP = 200          # antes 300
svd = TruncatedSVD(n_components=N_COMP, random_state=42)
Vc_red = svd.fit_transform(Vc)
Vc_red32 = Vc_red.astype("float32")
comps32  = svd.components_.astype("float32")

np.savez_compressed(ART/"V_cands_svd.npz",
                    Vc=Vc_red32,
                    comps=comps32)

print("Novo V_cands_svd.npz salvo")
