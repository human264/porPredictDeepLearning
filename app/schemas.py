from typing import Any, List, Optional
from pydantic import BaseModel, ConfigDict, field_validator

class ApiModel(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

class PredictItemReq(BaseModel):
    pjtno: str
    porser: str
    porseq: str
    revno: str
    want_act: bool = True
    want_mr: bool = True
    topk: int = 3

class TrainResp(ApiModel):
    status: str
    model_version: int
    num_labels: int

class ItemPredictResp(ApiModel):
    pjtno: str
    porser: str
    porseq: str
    revno: str
    count: int
    results: Any
    item_act_model_version: int | None = None
    item_mr_model_version: int | None = None
