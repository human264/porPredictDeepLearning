import os, re, unicodedata
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

VEC_DIM = int(os.getenv("VEC_DIM", "512"))

_vectorizer = HashingVectorizer(
    analyzer="char_wb", ngram_range=(3,3),
    n_features=VEC_DIM, alternate_sign=False, lowercase=True
)

_ws = re.compile(r"\s+")
_ALLOW = re.compile(r"[^0-9A-Za-z\u1100-\u11FF\u3130-\u318F\uAC00-\uD7A3.\-/% ]+")

def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKC", (text or "").strip()).lower()
    t = re.sub(r"\bdn(\d+)", r"dn \1", t)
    t = re.sub(r"\bpn(\d+)", r"pn \1", t)
    t = re.sub(r"(?<=\d)(mm|cm|inch)\b", r" \1", t)
    t = t.replace("Ø"," diam ").replace("ø"," diam ")
    t = re.sub(r"(\d)x(?=\d)", r"\1 x ", t)
    t = _ALLOW.sub(" ", t)
    t = _ws.sub(" ", t).strip()
    return t

def text_to_vec(clean_text: str) -> np.ndarray:
    X = _vectorizer.transform([clean_text])
    dense = X.toarray()[0].astype("float32")
    n = np.linalg.norm(dense)
    if n > 0:
        dense /= n
    return dense
