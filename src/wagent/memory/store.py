"""SQLite 记忆库：消息 / 事实 / 事件 / 画像 / 风格样本。

为什么先 SQLite 而不是向量库：
- 画像的核心是**结构化事实**（可更新、可冲突消解、可溯源），不是"相似度"；
- 向量检索是加法（L5 风格样本、L1 事件），后续接入 FAISS/Chroma 即可，
  但结构化这一层必须先有，否则会退化成"把聊天记录切块塞 prompt"。

约定：
- 时间一律 ISO8601 字符串（带时区），便于排序与时间衰减；
- 同一 ``(person, subject, predicate)`` 只保留一条 active 事实，
  写入新值会自动把旧值标记为 superseded 并记录有效期。
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path

from wagent.memory.types import (
    SELF_PERSON,
    Episode,
    FactCard,
    MessageRecord,
    PersonProfile,
    StyleSample,
    now_iso,
    text_fingerprint,
)

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    person TEXT NOT NULL,
    speaker TEXT NOT NULL,
    text TEXT NOT NULL,
    ts TEXT NOT NULL,
    session TEXT DEFAULT '',
    source TEXT DEFAULT 'import',
    fingerprint TEXT NOT NULL UNIQUE
);
CREATE INDEX IF NOT EXISTS idx_messages_person_ts ON messages(person, ts);

CREATE TABLE IF NOT EXISTS facts (
    id TEXT PRIMARY KEY,
    person TEXT NOT NULL,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    value TEXT NOT NULL,
    confidence REAL DEFAULT 0.8,
    valid_from TEXT,
    valid_to TEXT,
    status TEXT DEFAULT 'active',
    evidence_msg_ids TEXT DEFAULT '[]',
    evidence_text TEXT DEFAULT '',
    source TEXT DEFAULT 'own_message',
    created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_facts_person ON facts(person, status);
CREATE INDEX IF NOT EXISTS idx_facts_key ON facts(person, subject, predicate, status);

CREATE TABLE IF NOT EXISTS episodes (
    id TEXT PRIMARY KEY,
    person TEXT NOT NULL,
    summary TEXT NOT NULL,
    ts TEXT NOT NULL,
    topics TEXT DEFAULT '[]',
    msg_ids TEXT DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_episodes_person_ts ON episodes(person, ts);

CREATE TABLE IF NOT EXISTS profiles (
    person TEXT PRIMARY KEY,
    display_name TEXT DEFAULT '',
    relation TEXT DEFAULT '',
    traits TEXT DEFAULT '[]',
    preferences TEXT DEFAULT '[]',
    taboos TEXT DEFAULT '[]',
    notes TEXT DEFAULT '',
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS style_samples (
    id TEXT PRIMARY KEY,
    person TEXT NOT NULL,
    text TEXT NOT NULL,
    scene TEXT DEFAULT '',
    ts TEXT,
    fingerprint TEXT NOT NULL UNIQUE
);
CREATE INDEX IF NOT EXISTS idx_style_person ON style_samples(person);
"""


def _json_list(items: Iterable[str]) -> str:
    return json.dumps(list(items), ensure_ascii=False)


def _load_list(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return ()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return ()
    return tuple(str(item) for item in data) if isinstance(data, list) else ()


def parse_ts(value: str) -> datetime | None:
    """宽松解析 ISO 时间；失败返回 None。"""
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


class MemoryStore:
    """分层记忆库（单文件 SQLite）。"""

    def __init__(self, path: Path | str = ":memory:") -> None:
        self.path = str(path)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        self._conn.commit()

    # ------------------------------------------------------------ 基础

    @property
    def conn(self) -> sqlite3.Connection:
        return self._conn

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> MemoryStore:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def set_meta(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
        self._conn.commit()

    def get_meta(self, key: str, default: str = "") -> str:
        row = self._conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default

    def stats(self) -> dict[str, int]:
        tables = ("messages", "facts", "episodes", "profiles", "style_samples")
        out = {}
        for table in tables:
            row = self._conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()
            out[table] = int(row["n"])
        return out

    # ------------------------------------------------------------ 消息

    def add_messages(self, records: Sequence[MessageRecord]) -> int:
        """批量写入消息，按指纹去重；返回实际新增条数。"""
        inserted = 0
        for rec in records:
            fp = rec.fingerprint
            exists = self._conn.execute(
                "SELECT 1 FROM messages WHERE fingerprint=?", (fp,)
            ).fetchone()
            if exists:
                continue
            self._conn.execute(
                "INSERT INTO messages(id, person, speaker, text, ts, session, source, fingerprint) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (rec.id, rec.person, rec.speaker, rec.text, rec.ts, rec.session, rec.source, fp),
            )
            inserted += 1
        self._conn.commit()
        return inserted

    def recent_messages(self, person: str, *, limit: int = 20) -> list[MessageRecord]:
        rows = self._conn.execute(
            "SELECT * FROM messages WHERE person=? ORDER BY ts DESC LIMIT ?", (person, limit)
        ).fetchall()
        return [
            MessageRecord(
                id=row["id"],
                person=row["person"],
                speaker=row["speaker"],
                text=row["text"],
                ts=row["ts"],
                session=row["session"] or "",
                source=row["source"] or "import",
            )
            for row in reversed(rows)
        ]

    def messages_between(self, person: str, start: str, end: str) -> list[MessageRecord]:
        rows = self._conn.execute(
            "SELECT * FROM messages WHERE person=? AND ts>=? AND ts<? ORDER BY ts",
            (person, start, end),
        ).fetchall()
        return [
            MessageRecord(
                id=row["id"],
                person=row["person"],
                speaker=row["speaker"],
                text=row["text"],
                ts=row["ts"],
                session=row["session"] or "",
                source=row["source"] or "import",
            )
            for row in rows
        ]

    # ------------------------------------------------------------ 事实

    def add_fact(self, fact: FactCard) -> str:
        """写入事实；同一 key 的旧 active 事实会被 superseded。"""
        person, subject, predicate = fact.key()
        old_rows = self._conn.execute(
            "SELECT id, valid_from FROM facts "
            "WHERE person=? AND subject=? AND predicate=? AND status='active'",
            (person, subject, predicate),
        ).fetchall()
        ts = now_iso()
        for row in old_rows:
            if row["id"] == fact.id:
                continue
            self._conn.execute(
                "UPDATE facts SET status='superseded', valid_to=? WHERE id=?",
                (fact.valid_from or ts, row["id"]),
            )
        self._conn.execute(
            "INSERT OR REPLACE INTO facts("
            "id, person, subject, predicate, value, confidence, valid_from, valid_to, status,"
            "evidence_msg_ids, evidence_text, source, created_at"
            ") VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                fact.id,
                person,
                subject,
                predicate,
                fact.value,
                fact.confidence,
                fact.valid_from,
                fact.valid_to,
                fact.status,
                _json_list(fact.evidence_msg_ids),
                fact.evidence_text,
                fact.source,
                ts,
            ),
        )
        self._conn.commit()
        return fact.id

    def active_facts(self, person: str, *, limit: int = 200) -> list[FactCard]:
        rows = self._conn.execute(
            "SELECT * FROM facts WHERE person=? AND status='active' "
            "ORDER BY confidence DESC, valid_from DESC LIMIT ?",
            (person, limit),
        ).fetchall()
        return [self._row_to_fact(row) for row in rows]

    def search_facts(self, person: str, keywords: Sequence[str], *, limit: int = 20) -> list[FactCard]:
        """关键词检索（v1 用 LIKE；后续换 FTS5/向量）。"""
        keywords = [kw.strip() for kw in keywords if kw.strip()]
        if not keywords:
            return []
        clauses = " OR ".join("(value LIKE ? OR predicate LIKE ? OR evidence_text LIKE ?)" for _ in keywords)
        params: list[str] = [person]
        for kw in keywords:
            like = f"%{kw}%"
            params.extend([like, like, like])
        rows = self._conn.execute(
            f"SELECT * FROM facts WHERE person=? AND status='active' AND ({clauses}) LIMIT ?",
            (*params, limit),
        ).fetchall()
        return [self._row_to_fact(row) for row in rows]

    def supersede_fact(self, fact_id: str, *, valid_to: str | None = None) -> bool:
        cur = self._conn.execute(
            "UPDATE facts SET status='superseded', valid_to=? WHERE id=? AND status='active'",
            (valid_to or now_iso(), fact_id),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def delete_fact(self, fact_id: str) -> bool:
        cur = self._conn.execute("DELETE FROM facts WHERE id=?", (fact_id,))
        self._conn.commit()
        return cur.rowcount > 0

    @staticmethod
    def _row_to_fact(row: sqlite3.Row) -> FactCard:
        return FactCard(
            id=row["id"],
            person=row["person"],
            subject=row["subject"],
            predicate=row["predicate"],
            value=row["value"],
            confidence=float(row["confidence"] or 0.8),
            valid_from=row["valid_from"] or "",
            valid_to=row["valid_to"],
            status=row["status"],
            evidence_msg_ids=_load_list(row["evidence_msg_ids"]),
            evidence_text=row["evidence_text"] or "",
            source=row["source"] or "own_message",
        )

    # ------------------------------------------------------------ 事件

    def add_episode(self, episode: Episode) -> str:
        self._conn.execute(
            "INSERT OR REPLACE INTO episodes(id, person, summary, ts, topics, msg_ids) "
            "VALUES(?,?,?,?,?,?)",
            (
                episode.id,
                episode.person,
                episode.summary,
                episode.ts,
                _json_list(episode.topics),
                _json_list(episode.msg_ids),
            ),
        )
        self._conn.commit()
        return episode.id

    def recent_episodes(self, person: str, *, since: str | None = None, limit: int = 20) -> list[Episode]:
        if since:
            rows = self._conn.execute(
                "SELECT * FROM episodes WHERE person=? AND ts>=? ORDER BY ts DESC LIMIT ?",
                (person, since, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM episodes WHERE person=? ORDER BY ts DESC LIMIT ?", (person, limit)
            ).fetchall()
        return [
            Episode(
                id=row["id"],
                person=row["person"],
                summary=row["summary"],
                ts=row["ts"],
                topics=_load_list(row["topics"]),
                msg_ids=_load_list(row["msg_ids"]),
            )
            for row in rows
        ]

    # ------------------------------------------------------------ 画像

    def upsert_profile(self, profile: PersonProfile) -> None:
        self._conn.execute(
            "INSERT INTO profiles(person, display_name, relation, traits, preferences, taboos, notes, updated_at) "
            "VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(person) DO UPDATE SET "
            "display_name=excluded.display_name, relation=excluded.relation, traits=excluded.traits, "
            "preferences=excluded.preferences, taboos=excluded.taboos, notes=excluded.notes, "
            "updated_at=excluded.updated_at",
            (
                profile.person,
                profile.display_name,
                profile.relation,
                _json_list(profile.traits),
                _json_list(profile.preferences),
                _json_list(profile.taboos),
                profile.notes,
                profile.updated_at,
            ),
        )
        self._conn.commit()

    def get_profile(self, person: str) -> PersonProfile | None:
        row = self._conn.execute("SELECT * FROM profiles WHERE person=?", (person,)).fetchone()
        if row is None:
            return None
        return PersonProfile(
            person=row["person"],
            display_name=row["display_name"] or "",
            relation=row["relation"] or "",
            traits=_load_list(row["traits"]),
            preferences=_load_list(row["preferences"]),
            taboos=_load_list(row["taboos"]),
            notes=row["notes"] or "",
            updated_at=row["updated_at"] or "",
        )

    # ------------------------------------------------------------ 风格样本

    def add_style_samples(self, samples: Sequence[StyleSample]) -> int:
        inserted = 0
        for sample in samples:
            fp = text_fingerprint(sample.person, sample.text)
            exists = self._conn.execute(
                "SELECT 1 FROM style_samples WHERE fingerprint=?", (fp,)
            ).fetchone()
            if exists:
                continue
            self._conn.execute(
                "INSERT INTO style_samples(id, person, text, scene, ts, fingerprint) "
                "VALUES(?,?,?,?,?,?)",
                (sample.id, sample.person, sample.text, sample.scene, sample.ts, fp),
            )
            inserted += 1
        self._conn.commit()
        return inserted

    def style_samples(self, person: str = SELF_PERSON, *, limit: int = 50) -> list[StyleSample]:
        rows = self._conn.execute(
            "SELECT * FROM style_samples WHERE person=? ORDER BY ts DESC LIMIT ?", (person, limit)
        ).fetchall()
        return [
            StyleSample(
                id=row["id"],
                text=row["text"],
                person=row["person"],
                scene=row["scene"] or "",
                ts=row["ts"] or "",
            )
            for row in rows
        ]
