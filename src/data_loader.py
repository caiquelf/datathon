import json, pathlib, zipfile, ijson
from typing import Any, Dict, List, Union

RAW_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "raw"

def unzip_all():
    for fname in ("vagas.zip","prospects.zip","applicants.zip"):
        z = RAW_DIR/fname
        if z.exists():
            with zipfile.ZipFile(z) as zp:
                zp.extractall(RAW_DIR)

def _load_json(path: pathlib.Path) -> Union[Dict[str, Any], List[Any]]:
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1); f.seek(0)
        if first == "{":
            return json.load(f)
        return [o for o in ijson.items(f, "item")]

def load_jobs():       return _load_json(RAW_DIR/"vagas.json")
def load_prospects():  return _load_json(RAW_DIR/"prospects.json")
def load_applicants(): return _load_json(RAW_DIR/"applicants.json")
