# app/knn.py
import os
import numpy as np
from app.db import get_conn, dict_cur, get_schema

S = get_schema()

def knn_topk(set_vec: np.ndarray, k=3, probes=None, active_only=True):
    probes = int(os.getenv("IVFFLAT_PROBES", "10")) if probes is None else int(probes)
    cond = f"WHERE label IN (SELECT code FROM {S}.categories_b WHERE active=TRUE)" if active_only else ""
    qvec = np.asarray(set_vec, dtype=float).tolist()

    conn = get_conn(); conn.autocommit = False
    try:
        cur = dict_cur(conn)
        cur.execute(f"SET LOCAL ivfflat.probes = {probes};")
        cur.execute(f"""
          SELECT label, (set_embed <-> %s) AS distance
          FROM {S}.bundle_sets
          {cond}
          ORDER BY set_embed <-> %s
          LIMIT %s
        """, (qvec, qvec, k))
        rows = cur.fetchall()
        cur.close(); conn.commit()
        return rows
    finally:
        conn.close()
