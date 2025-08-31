#app/ml/xgb_config.py

# 예측에 사용하지 않는 식별자 (조회/매칭용)
ID_COLS = ["pjtno", "porser", "porseq", "revno"]

# 실제 예측 피처
NUMERIC_FEATURES = ["duration"]
CATEGORICAL_FEATURES = ["mccsno", "block", "event", "sign", "deptcode", "shiptype"]

# 라벨명(alias)
TARGET_MR = "mr_label"            # tb_mr.mrno
TARGET_ACT = "act_label"          # tb_mr.act_code||':'||tb_mr.act_no

MODEL_DIR = "models"
MODEL_PATHS = {
    "mr": f"{MODEL_DIR}/xgb_mr.joblib",
    "act": f"{MODEL_DIR}/xgb_act.joblib",
}
