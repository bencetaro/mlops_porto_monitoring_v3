from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class Item(BaseModel):
    root: Optional[Dict[str, Any]] = None
    features: Optional[Dict[str, Any]] = None
    id: Optional[int] = None

class BatchRequest(BaseModel):
    root: Optional[List[Dict[str, Any]]] = None
    records: Optional[List[Dict[str, Any]]] = None
    items: Optional[List[Dict[str, Any]]] = None

class PredictionResponse(BaseModel):
    prediction: float
