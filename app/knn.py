# app/knn.py — Category B + pgvector literal casting
import os
import numpy as np
from app.db import get_conn, dict_cur, get_schema

S = get_schema()

def _to_vector_literal(vec: np.ndarray) -> str:
    """
    pgvector 텍스트 리터럴로 변환: [0.12,0.34,...]
    숫자 포맷은 과도한 소수점 제거 (필요시 조정)
    """
    v = np.asarray(vec, dtype=float).ravel()
    # 과도한 소수점은 줄이고, NaN/Inf는 0.0으로 가드
    vals = []
    for x in v:
        if not np.isfinite(x):
            x = 0.0
        vals.append(f"{x:.8f}".rstrip("0").rstrip(".") or "0")
    return "[" + ",".join(vals) + "]"

def knn_topk(set_vec: np.ndarray, k=3, probes=None, active_only=True):
    """
    bundle_sets.label (카테고리 B 코드) 기준 KNN.
    반환: [{"label": "B1", "distance": 0.123}, ...]
    """
    probes = int(os.getenv("IVFFLAT_PROBES", "10")) if probes is None else int(probes)
    cond = f"WHERE label IN (SELECT code FROM {S}.categories_b WHERE active=TRUE)" if active_only else ""
    qvec_lit = _to_vector_literal(set_vec)  # ← 핵심: pgvector literal

    conn = get_conn()
    conn.autocommit = False
    try:
        cur = dict_cur(conn)
        cur.execute(f"SET LOCAL ivfflat.probes = {probes};")
        # 오른쪽 피연산자를 %s::vector 로 캐스팅
        cur.execute(
            f"""
            SELECT label, (set_embed <-> %s::vector) AS distance
            FROM {S}.bundle_sets
            {cond}
            ORDER BY set_embed <-> %s::vector
            LIMIT %s
            """,
            (qvec_lit, qvec_lit, k),
        )
        rows = cur.fetchall()
        cur.close()
        conn.commit()
        return rows
    finally:
        conn.close()
