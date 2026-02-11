from pydantic import BaseModel, ConfigDict
from typing import Any, Dict, Optional

class RenderEnvelope(BaseModel):
    """
    Mirrors the JSON payload you POST:
    {
      "endpoint": "...",
      "email_to": "...",
      "request": {...}
    }
    """
    model_config = ConfigDict(extra="allow")  # tolerate extra fields while testing

    endpoint: Optional[str] = None
    email_to: Optional[str] = None
    request: Dict[str, Any]
