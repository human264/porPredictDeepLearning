import os, re
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

VEC_DIM = int(os.getenv("VEC_DIM", "512"))
_vectorizer = HashingVectorizer(
    analyzer="char_wb", ngram_range=(3,3),
    n_features=VEC_DIM, alternate_sign=False
)
_ws = re.compile(r"\s+")

def normalize(text: str) -> str:
    t = (text or "").strip().lower()
    t = re.sub(r"\bdn(\d+)", r"dn \1", t)
    t = re.sub(r"\bpn(\d+)", r"pn \1", t)
    t = re.sub(r"(?<=\d)(mm|cm|inch)\b", r" \1", t)
    t = t.replace("Ø"," diam ").replace("ø"," diam ")
    t = re.sub(r"(\d)x(?=\d)", r"\1 x ", t)
    t = re.sub(r"[^a-z0-9.\-/% ]+", " ", t)
    t = _ws.sub(" ", t).strip()
    return t

def text_to_vec(clean_text: str) -> np.ndarray:
    X = _vectorizer.transform([clean_text])
    dense = X.toarray()[0].astype("float32")
    n = np.linalg.norm(dense)
    if n > 0:
        dense /= n
    return dense
