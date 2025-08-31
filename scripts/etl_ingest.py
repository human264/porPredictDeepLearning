# scripts/etl_ingest.py
import os
import sys
import pathlib
import requests
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2 import sql
from dotenv import load_dotenv

# 프로젝트 루트 추가 (app.* 임포트용)
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from app.textvec import normalize, text_to_vec  # API 미사용 폴백 시 직접 DB 적재에 필요

load_dotenv()

DB = dict(
    host=os.getenv("DB_HOST", "localhost"),
    port=os.getenv("DB_PORT", "5432"),
    dbname=os.getenv("DB_NAME", "mydb"),
    user=os.getenv("DB_USER", "myuser"),
    password=os.getenv("DB_PASS", "mypass"),
)

# IPv6 이슈 회피: localhost 대신 127.0.0.1 기본
API = os.getenv("INGEST_API", "http://127.0.0.1:8000/bundle/ingest")
API_STATS = API.replace("/bundle/ingest", "/stats")

SCHEMA = os.getenv("DB_SCHEMA", "public")
VEC_DIM = int(os.getenv("VEC_DIM", "512"))
CHUNK = int(os.getenv("INGEST_CHUNK", "500"))


def api_ready(timeout_sec: int = 5) -> bool:
    """FastAPI /stats로 가용성 체크"""
    try:
        r = requests.get(API_STATS, timeout=timeout_sec)
        return r.status_code == 200
    except Exception:
        return False


def open_training_stream(rconn) -> RealDictCursor:
    """
    학습 데이터 스트리밍 커서 (읽기 전용 커넥션에서만 사용).
    - vw_mr_training 존재 시: 뷰 사용
    - 없으면: tb_por_detail ↔ tb_mr 조인으로 대체
    """
    with rconn.cursor() as chk:
        chk.execute("SELECT to_regclass(%s)", (f"{SCHEMA}.vw_mr_training",))
        has_view = chk.fetchone()[0] is not None

    # named server-side cursor WITH HOLD (commit 이후에도 유지)
    cur = rconn.cursor(name="train_stream", cursor_factory=RealDictCursor, withhold=True)
    cur.itersize = CHUNK

    if has_view:
        q = sql.SQL("SELECT label, items FROM {}.vw_mr_training;").format(
            sql.Identifier(SCHEMA)
        )
        cur.execute(q)
        print(f"[etl] using view {SCHEMA}.vw_mr_training")
    else:
        q = sql.SQL("""
            SELECT
              m.category_code AS label,
              ARRAY_AGG(CONCAT_WS(' ', d.item_name, d.spec_text) ORDER BY d.line_no) AS items
            FROM {}.tb_por_detail d
            JOIN {}.tb_mr m ON m.por_id = d.por_id
            WHERE m.category_code IS NOT NULL
            GROUP BY m.por_id, m.category_code
        """).format(sql.Identifier(SCHEMA), sql.Identifier(SCHEMA))
        cur.execute(q)
        print(f"[etl] fallback join {SCHEMA}.tb_por_detail ↔ {SCHEMA}.tb_mr")
    return cur


def ingest_via_api(rows) -> int:
    """API(/bundle/ingest)로 적재"""
    n = 0
    for r in rows:
        resp = requests.post(
            API,
            json={"label": r["label"], "items": r["items"]},
            timeout=30,
        )
        resp.raise_for_status()
        n += 1
    return n


def ingest_direct_db(wconn, rows) -> int:
    """
    API가 죽어있을 때 직접 DB에 삽입.
    - bundle_sets.set_embed, bundle_items.item_embed 모두 계산하여 저장
    (쓰기 전용 커넥션 사용)
    """
    import numpy as np

    def to_py_float_list(v: np.ndarray) -> list[float]:
        # np.float32 -> Python float 변환 (pgvector가 리스트[float]를 기대)
        return [float(x) for x in np.asarray(v, dtype="float64").ravel().tolist()]

    cur = wconn.cursor()
    inserted = 0
    for r in rows:
        label = r["label"]
        items = r["items"] or []
        clean = [normalize(x) for x in items]
        item_vecs = [text_to_vec(t) for t in clean]  # 각 요소 dtype=float32일 수 있음

        V = VEC_DIM
        if item_vecs:
            import numpy as np
            set_vec = np.mean(item_vecs, axis=0)
        else:
            import numpy as np
            set_vec = np.zeros(V, dtype="float32")

        # ✅ Python float 리스트로 변환
        set_vec_py = to_py_float_list(set_vec)

        cur.execute(
            "INSERT INTO bundle_sets(label, set_embed) VALUES (%s,%s) RETURNING id;",
            (label, set_vec_py),
        )
        set_id = cur.fetchone()[0]

        for raw, c, v in zip(items, clean, item_vecs):
            cur.execute(
                """
                INSERT INTO bundle_items(set_id, raw_text, clean_text, item_embed)
                VALUES (%s,%s,%s,%s)
                """,
                (set_id, raw, c, to_py_float_list(v)),
            )
        inserted += 1

    wconn.commit()
    cur.close()
    return inserted


def main():
    # 연결 분리: 읽기(rconn) / 쓰기(wconn)
    rconn = psycopg2.connect(**DB)
    wconn = psycopg2.connect(**DB)

    # 읽기 커넥션은 read only 세션 권장 (선택)
    try:
        with rconn.cursor() as c:
            c.execute("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY;")
    except Exception:
        pass  # 호환성 위해 실패해도 무시

    cur = open_training_stream(rconn)

    total = 0
    try:
        # API 준비 확인
        use_api = api_ready(timeout_sec=5)
        if use_api:
            print(f"[etl] API ready → {API}")
        else:
            print(f"[etl] API NOT ready → DIRECT DB ingest mode")

        while True:
            rows = cur.fetchmany(CHUNK)
            if not rows:
                break

            if use_api:
                try:
                    total += ingest_via_api(rows)
                except requests.exceptions.RequestException as e:
                    print(f"[etl] API error → switching to DIRECT mode: {e}")
                    total += ingest_direct_db(wconn, rows)
                    use_api = False
            else:
                total += ingest_direct_db(wconn, rows)
    finally:
        try:
            cur.close()
        except Exception:
            pass
        rconn.close()
        wconn.close()

    print("ingested bundles:", total)


if __name__ == "__main__":
    main()
