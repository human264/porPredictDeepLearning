# app/ml/xgb_model.py
import os
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from joblib import dump, load

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

import xgboost as xgb

from app.ml.xgb_config import (
    ID_COLS, NUMERIC_FEATURES, CATEGORICAL_FEATURES,
    TARGET_MR, TARGET_ACT, MODEL_DIR, MODEL_PATHS
)
from app.ml.xgb_data import fetch_training_df, fetch_predict_df

def _build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]), NUMERIC_FEATURES),
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore")),
            ]), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

def _build_clf() -> xgb.XGBClassifier:
    # 멀티클래스 자동 인식(Y가 0..K-1이면 softprob 사용)
    return xgb.XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        eval_metric="mlogloss",
        tree_method="hist",
    )

def _target_name(target: str) -> str:
    return TARGET_MR if target == "mr" else TARGET_ACT

def train_and_save_from_db(target: str) -> Dict[str, Any]:
    """
    target: 'mr' or 'act'
    DB에서 학습셋 구성 후 학습/저장
    """
    df = fetch_training_df(target)
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    y_name = _target_name(target)

    X = df[feature_cols].copy()
    y_raw = df[y_name].astype(str).fillna("")

    # 라벨 인코딩
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    pipe = Pipeline([
        ("prep", _build_preprocessor()),
        ("xgb", _build_clf()),
    ])
    pipe.fit(X, y)

    os.makedirs(MODEL_DIR, exist_ok=True)
    dump({
        "pipeline": pipe,
        "feature_cols": feature_cols,
        "label_encoder_classes": le.classes_,
        "target": target,
        "schema_version": 1,
    }, MODEL_PATHS[target])

    # 간단 리포트(훈련 셋 기준)
    pred = pipe.predict(X)
    report = classification_report(y, pred, output_dict=True, zero_division=0)

    return {
        "target": target,
        "saved_to": MODEL_PATHS[target],
        "n_samples": int(len(df)),
        "n_features": len(feature_cols),
        "n_classes": int(len(le.classes_)),
        "train_report": report,
    }

def load_model(target: str) -> Dict[str, Any]:
    path = MODEL_PATHS[target]
    if not os.path.exists(path):
        raise FileNotFoundError(f"모델이 없습니다. 먼저 학습하세요: {path}")
    return load(path)

def predict_header(target: str, header: Dict[str, Any], topk: int = 5) -> Dict[str, Any]:
    """
    특정 헤더의 tb_por_detail 라인들에 대해 타겟(mr/act) 예측
    """
    bundle = load_model(target)
    pipe: Pipeline = bundle["pipeline"]
    feature_cols: List[str] = bundle["feature_cols"]
    classes: np.ndarray = bundle["label_encoder_classes"]

    df = fetch_predict_df(header)
    X = df[feature_cols].copy()

    proba = pipe.predict_proba(X)  # shape: [N, K]
    pred_idx = np.argmax(proba, axis=1)
    pred_label = classes[pred_idx]

    # topk 후보
    topk = max(1, int(topk))
    topk_idx = np.argsort(-proba, axis=1)[:, :topk]
    topk_labels = [[classes[j] for j in row] for row in topk_idx]
    topk_probs = [[float(proba[i, j]) for j in row] for i, row in enumerate(topk_idx)]

    # 결과 합치기
    out_rows = []
    for i, r in df.iterrows():
        rec = {
            "line_no": r.get("line_no"),
            "pjtno": r.get("pjtno"), "porser": r.get("porser"),
            "porseq": r.get("porseq"), "revno": r.get("revno"),
            "pred": str(pred_label[i]),
            "proba": float(proba[i, pred_idx[i]]),
            "topk_labels": topk_labels[i],
            "topk_probs": topk_probs[i],
        }
        out_rows.append(rec)

    return {
        **header,
        "target": target,
        "count": len(out_rows),
        "results": out_rows,
    }
