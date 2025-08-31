# app/ml/xgb_data.py
from typing import Tuple, Dict, Any
import pandas as pd

from app.db import get_conn, dict_cur, get_schema
from app.ml.xgb_config import (
    ID_COLS, NUMERIC_FEATURES, CATEGORICAL_FEATURES,
    TARGET_MR, TARGET_ACT
)

S = get_schema()

def fetch_training_df(target: str) -> pd.DataFrame:
    """
    target: 'mr' or 'act'
    헤더 단위 대표 라벨을 1개로 축약 후 tb_por_detail 라인과 조인하여 학습셋 구성.
    (Oracle 11g 호환)
    """
    conn = get_conn(); cur = dict_cur(conn)
    try:
        sql = f"""
        WITH mr_agg AS (
          SELECT pjtno, porser, porseq, revno, MIN(mrno) AS mrno
          FROM {S}.tb_mr
          GROUP BY pjtno, porser, porseq, revno
        ),
        act_agg AS (
          SELECT
              pjtno, porser, porseq, revno,
              MIN(
                CASE
                  WHEN actocode IS NOT NULL AND actno IS NOT NULL
                    THEN actocode || ':' || TO_CHAR(actno)
                  ELSE NULL
                END
              ) AS act_label
          FROM {S}.tb_mr
          GROUP BY pjtno, porser, porseq, revno
        )
        SELECT
          d.pjtno, d.porser, d.porseq, d.revno,
          d.mccsno, d.block, d.event, d.sign, d.duration, d.deptcode, d.shiptype,
          mr.mrno AS {TARGET_MR},
          act.act_label AS {TARGET_ACT}
        FROM {S}.tb_por_detail d
        LEFT JOIN mr_agg  mr
          ON (d.pjtno=mr.pjtno AND d.porser=mr.porser AND d.porseq=mr.porseq AND d.revno=mr.revno)
        LEFT JOIN act_agg act
          ON (d.pjtno=act.pjtno AND d.porser=act.porser AND d.porseq=act.porseq AND d.revno=act.revno)
        """
        cur.execute(sql)
        rows = cur.fetchall()
        df = pd.DataFrame(rows)
    finally:
        cur.close(); conn.close()

    if df.empty:
        raise ValueError("훈련 데이터가 비어 있습니다.")

    if "duration" in df.columns:
        df["duration"] = pd.to_numeric(df["duration"], errors="coerce")

    if target == "mr":
        df = df.dropna(subset=[TARGET_MR])
    elif target == "act":
        df = df.dropna(subset=[TARGET_ACT])
    else:
        raise ValueError("target은 'mr' 또는 'act' 이어야 합니다.")

    return df


def fetch_predict_df(header: Dict[str, Any]) -> pd.DataFrame:
    """지정한 헤더의 tb_por_detail 라인들을 로드하여 예측 입력으로 반환 (Oracle 바인딩 사용)"""
    conn = get_conn(); cur = dict_cur(conn)
    try:
        sql = f"""
        SELECT line_no, pjtno, porser, porseq, revno,
               mccsno, block, event, sign, duration, deptcode, shiptype
        FROM {S}.tb_por_detail
        WHERE pjtno=:pjtno AND porser=:porser AND porseq=:porseq AND revno=:revno
        ORDER BY line_no
        """
        cur.execute(sql, {
            "pjtno": header["pjtno"],
            "porser": header["porser"],
            "porseq": header["porseq"],
            "revno": header["revno"],
        })
        rows = cur.fetchall()
        df = pd.DataFrame(rows)
    finally:
        cur.close(); conn.close()

    if df.empty:
        raise ValueError("해당 헤더로 라인을 찾을 수 없습니다.")

    if "duration" in df.columns:
        df["duration"] = pd.to_numeric(df["duration"], errors="coerce")

    return df
