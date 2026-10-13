"""Private immutable cache, addressed by all extraction inputs and configuration."""
import json
import sqlite3
from pathlib import Path
from ...agent.events import digest
from .models import ShoppingEpisode


class EpisodeStore:
    def __init__(self, path, *, readonly=False):
        self.path = Path(path).resolve()
        if readonly:
            self.connection = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(self.path)
            self.path.chmod(0o600)
            self.connection.execute("CREATE TABLE IF NOT EXISTS episodes(cache_key TEXT PRIMARY KEY, episode_id TEXT UNIQUE NOT NULL, payload TEXT NOT NULL, payload_hash TEXT NOT NULL)")
            self.connection.commit()

    def get(self, key):
        row = self.connection.execute("SELECT payload,payload_hash FROM episodes WHERE cache_key=?", (key,)).fetchone()
        return self._decode(row) if row else None

    @staticmethod
    def _decode(row):
        payload = json.loads(row[0])
        if digest(payload) != row[1]: raise ValueError("episode cache integrity mismatch")
        return ShoppingEpisode.model_validate(payload)

    def put(self, key, episode):
        payload = episode.model_dump(mode="json")
        existing = self.get(key)
        if existing is not None:
            if existing != episode: raise ValueError("refusing to replace a cached episode")
            return
        with self.connection:
            self.connection.execute("INSERT INTO episodes VALUES(?,?,?,?)", (key, episode.episode_id, episode.model_dump_json(), digest(payload)))

    def episodes(self):
        for row in self.connection.execute("SELECT payload,payload_hash FROM episodes ORDER BY episode_id"):
            yield self._decode(row)

    def close(self): self.connection.close()
    def __enter__(self): return self
    def __exit__(self, *_): self.close()
