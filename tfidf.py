from typing import List
import re


def tokenize(text: str) -> List[str]:
    cleaned_text = re.sub(r"[^\w\s]", "", text)
    tokens = cleaned_text.lower().split()
    return tokens