"""Question routing uses a versioned public vocabulary, never a fixed goal order."""
import json
from pathlib import Path
from ...data.NFKC import normalize_query

TOPICS = json.loads(Path(__file__).with_name("topics.json").read_text())


def question_topics(question: str) -> list[str]:
    text = normalize_query(question)
    return [topic for topic, words in TOPICS.items() if any(normalize_query(w) in text for w in words)]
