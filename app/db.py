# app/db.py
import os

# oracledb(신규) 또는 cx_Oracle(기존) 어느 쪽이든 동작하도록
try:
    import oracledb as cx_Oracle  # type: ignore
except ImportError:
    import cx_Oracle  # type: ignore


def _maybe_init_thick_client():
    """
    11g 환경에서 oracledb thin이 아닌 thick가 필요할 수 있으니
    ORACLE_CLIENT_LIB_DIR 환경변수가 있으면 thick 초기화 시도.
    """
    try:
        import oracledb  # type: ignore
        lib_dir = os.getenv("ORACLE_CLIENT_LIB_DIR")
        if lib_dir:
            oracledb.init_oracle_client(lib_dir=lib_dir)
    except Exception:
        pass


def _make_dsn() -> str:
    host = os.getenv("DB_HOST", "127.0.0.1")
    port = int(os.getenv("DB_PORT", "1521"))
    svc = os.getenv("DB_SERVICE_NAME") or os.getenv("DB_NAME")  # 서비스명
    sid = os.getenv("DB_SID")  # SID 사용 시
    if hasattr(cx_Oracle, "makedsn"):
        if svc:
            return cx_Oracle.makedsn(host, port, service_name=svc)
        if sid:
            return cx_Oracle.makedsn(host, port, sid=sid)
    # makedsn이 없다면 단순 문자열 연결
    return f"{host}:{port}/{svc or sid or ''}"


def get_conn():
    _maybe_init_thick_client()
    user = os.getenv("DB_USER", "myuser")
    password = os.getenv("DB_PASS", "mypass")
    dsn = os.getenv("DB_DSN") or _make_dsn()
    conn = cx_Oracle.connect(user=user, password=password, dsn=dsn,
                             encoding="UTF-8", nencoding="UTF-8")
    return conn


def dict_cur(conn):
    """
    cx_Oracle 커서를 dict로 반환하도록 rowfactory 설정
    (컬럼명은 소문자로 통일)
    """
    cur = conn.cursor()

    def _rf(*args):
        return {d[0].lower(): v for d, v in zip(cur.description, args)}

    cur.rowfactory = _rf
    return cur


def get_schema() -> str:
    # Oracle은 기본 대문자 스키마. 미지정 시 DB_USER 사용.
    return (os.getenv("DB_SCHEMA", "").strip() or os.getenv("DB_USER", "MYUSER")).upper()
