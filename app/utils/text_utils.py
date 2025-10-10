def safe_truncate(text: str, max_len: int = 10000) -> str:
    return text[:max_len]
