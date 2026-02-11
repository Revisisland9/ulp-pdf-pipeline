from typing import Any, Dict

def s(v: Any) -> str:
    return "" if v is None else str(v)

def get_path(d: Dict[str, Any], *path, default=""):
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur

def fmt_phone(phone: str) -> str:
    digits = "".join(ch for ch in (phone or "") if ch.isdigit())
    if len(digits) == 10:
        return f"{digits[0:3]}-{digits[3:6]}-{digits[6:10]}"
    return phone or ""
