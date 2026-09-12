"""Explicit caller-owned long-term preferences, isolated from task evidence."""

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import sqlite3

from .contracts import UserProfile, UserPreference


class PreferenceConflict(ValueError):
    pass


class PreferenceStore:
    """SQLite transactions prevent lost updates; no automatic model writes.

    Each profile carries a revision for compare-and-swap edits and a history of
    explicit caller changes. Task queries, candidate memory and labels are never
    accepted by this interface.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS profiles (id TEXT PRIMARY KEY, locale TEXT NOT NULL, revision INTEGER NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS preferences (profile_id TEXT NOT NULL, name TEXT NOT NULL, value TEXT NOT NULL, PRIMARY KEY(profile_id, name))"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS changes (profile_id TEXT NOT NULL, revision INTEGER NOT NULL, operation TEXT NOT NULL, name TEXT NOT NULL, value TEXT, source TEXT NOT NULL, updated_at TEXT NOT NULL, PRIMARY KEY(profile_id, revision))"
            )

    @contextmanager
    def _connection(self):
        connection = sqlite3.connect(self.path, timeout=15)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @staticmethod
    def _id(profile_id):
        if (
            not isinstance(profile_id, str)
            or not profile_id.strip()
            or len(profile_id) > 200
        ):
            raise ValueError("profile_id must contain 1-200 characters")
        return profile_id

    def load(self, profile_id: str) -> tuple[UserProfile, int]:
        profile_id = self._id(profile_id)
        with self._connection() as connection:
            connection.execute("BEGIN")
            row = connection.execute(
                "SELECT locale, revision FROM profiles WHERE id=?", (profile_id,)
            ).fetchone()
            preferences = connection.execute(
                "SELECT name, value FROM preferences WHERE profile_id=? ORDER BY name",
                (profile_id,),
            ).fetchall()
        return UserProfile(
            profile_id=profile_id,
            locale=row[0] if row else "zh-CN",
            preferences=[UserPreference(name=n, value=v) for n, v in preferences],
        ), (row[1] if row else 0)

    def set(
        self,
        profile_id: str,
        name: str,
        value: str,
        *,
        expected_revision: int | None = None,
    ) -> int:
        preference = UserPreference(name=name, value=value)
        if (
            not preference.name.strip()
            or not preference.value.strip()
            or len(name) > 200
            or len(value) > 2000
        ):
            raise ValueError("preference name/value is blank or too long")
        return self._change(profile_id, name, value, "set", expected_revision)

    def remove(
        self, profile_id: str, name: str, *, expected_revision: int | None = None
    ) -> int:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("preference name must not be blank")
        return self._change(profile_id, name, None, "remove", expected_revision)

    def _change(self, profile_id, name, value, operation, expected_revision):
        profile_id = self._id(profile_id)
        if expected_revision is not None and (
            type(expected_revision) is not int or expected_revision < 0
        ):
            raise ValueError("expected_revision must be a non-negative integer")
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT revision FROM profiles WHERE id=?", (profile_id,)
            ).fetchone()
            revision = row[0] if row else 0
            if expected_revision is not None and revision != expected_revision:
                raise PreferenceConflict("profile was updated; reload before editing")
            if operation == "remove":
                changed = connection.execute(
                    "DELETE FROM preferences WHERE profile_id=? AND name=?",
                    (profile_id, name),
                ).rowcount
                if not changed:
                    raise ValueError("preference does not exist")
            else:
                count = connection.execute(
                    "SELECT count(*) FROM preferences WHERE profile_id=?", (profile_id,)
                ).fetchone()[0]
                exists = connection.execute(
                    "SELECT 1 FROM preferences WHERE profile_id=? AND name=?",
                    (profile_id, name),
                ).fetchone()
                if count >= 50 and not exists:
                    raise ValueError("profile preference limit reached")
                connection.execute(
                    "INSERT INTO preferences VALUES (?, ?, ?) ON CONFLICT(profile_id, name) DO UPDATE SET value=excluded.value",
                    (profile_id, name, value),
                )
            revision += 1
            connection.execute(
                "INSERT INTO profiles VALUES (?, 'zh-CN', ?) ON CONFLICT(id) DO UPDATE SET revision=excluded.revision",
                (profile_id, revision),
            )
            connection.execute(
                "INSERT INTO changes VALUES (?, ?, ?, ?, ?, 'explicit_caller', ?)",
                (
                    profile_id,
                    revision,
                    operation,
                    name,
                    value,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            return revision

    def history(self, profile_id: str, *, limit: int = 20) -> list[dict]:
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("history limit must be 1-100")
        with self._connection() as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT revision, operation, name, value, source, updated_at FROM changes WHERE profile_id=? ORDER BY revision DESC LIMIT ?",
                (self._id(profile_id), limit),
            ).fetchall()
        return [dict(row) for row in rows]
