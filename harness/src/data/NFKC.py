from __future__ import annotations

import unicodedata
import re

def normalize_query(value: object) -> str:
    '''
    标准化搜索词，包括字符、大小写、货币单位、标点和空格。
    '''
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = text.replace("人民币", "元").replace("块钱", "元")
    text = re.sub(r"(?<=\d)\s*(?:rmb|cny|yuan)\b", "元", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[，,。；;：:！!？?、|]+", " ", text)
    return " ".join(text.split())