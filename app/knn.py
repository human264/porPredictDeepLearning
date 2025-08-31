# # Category (actocode, actno) KNN over bundle_sets
# import os
# import numpy as np
# from app.db import get_conn, dict_cur, get_schema
#
# S = get_schema()
#
# def _to_vector_literal(vec: np.ndarray) -> str:
#     v = np.asarray(vec, dtype=float).ravel()
#     vals = []
#     for x in v:
#         if not np.isfinite(x):
#             x = 0.0
#         vals.append(f"{x:.8f}".rstrip("0").rstrip(".") or "0")
#     return "[" + ",".join(vals) + "]"
#
# def knn_topk(set_vec: np.ndarray, k=3, probes=None, active_only=True):
#     """
#     bundle_sets의 set_embed 기준으로 KNN.
#     반환: [{"actocode":"ACT","actno":"001","distance":0.123}, ...]
#     """
#     probes = int(os.getenv("IVFFLAT_PROBES", "10")) if probes is None else int(probes)
#     qvec_lit = _to_vector_literal(set_vec)
#     cond_active = ""
#     if active_only:
#         # 활동 코드 활성만
#         cond_active = f"""
#         WHERE (bs.label_actocode, bs.label_actno) IN (
#           SELECT actocode, actno FROM {S}.activity_codes WHERE active=TRUE
#         )
#         """
#
#     conn = get_conn()
#     conn.autocommit = False
#     try:
#         cur = dict_cur(conn)
#         cur.execute(f"SET LOCAL ivfflat.probes = {probes};")
#         cur.execute(
#             f"""
#             SELECT bs.label_actocode AS actocode,
#                    bs.label_actno    AS actno,
#                    (bs.set_embed <-> %s::vector) AS distance
#             FROM {S}.bundle_sets bs
#             {cond_active}
#             ORDER BY bs.set_embed <-> %s::vector
#             LIMIT %s
#             """,
#             (qvec_lit, qvec_lit, k),
#         )
#         rows = cur.fetchall()
#         cur.close()
#         conn.commit()
#         return rows
#     finally:
#         conn.close()
