from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from ..schemas import Note, Speaker, SummarySnapshot, SummaryType, TranscriptSegment


SCHEMA = """
CREATE TABLE IF NOT EXISTS meetings (
  id TEXT PRIMARY KEY,
  started_at TEXT NOT NULL,
  ended_at TEXT,
  preferred_summary_language TEXT NOT NULL DEFAULT 'en'
);

CREATE TABLE IF NOT EXISTS transcript_segments (
  id TEXT PRIMARY KEY,
  meeting_id TEXT NOT NULL,
  start_time REAL NOT NULL,
  end_time REAL NOT NULL,
  text TEXT NOT NULL,
  original_language TEXT NOT NULL,
  speaker_id TEXT,
  confidence REAL NOT NULL,
    quality TEXT NOT NULL DEFAULT 'ok',
  translations TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS summary_snapshots (
  id TEXT PRIMARY KEY,
  meeting_id TEXT NOT NULL,
  summary_type TEXT NOT NULL,
  time_start REAL NOT NULL,
  time_end REAL NOT NULL,
  language TEXT NOT NULL,
  content TEXT NOT NULL,
  source_segment_ids TEXT NOT NULL DEFAULT '[]',
  is_latest INTEGER NOT NULL DEFAULT 0,
  translations TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS notes (
  id TEXT PRIMARY KEY,
  meeting_id TEXT NOT NULL,
  content TEXT NOT NULL,
  source_type TEXT,
  source_id TEXT,
    time_seconds REAL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS speakers (
  id TEXT PRIMARY KEY,
  meeting_id TEXT NOT NULL,
  display_name TEXT,
  inferred_label TEXT NOT NULL,
  confidence REAL NOT NULL DEFAULT 0.5,
  is_local_user INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_segments_meeting ON transcript_segments(meeting_id, start_time);
CREATE INDEX IF NOT EXISTS idx_snapshots_meeting_type ON summary_snapshots(meeting_id, summary_type, created_at);
CREATE INDEX IF NOT EXISTS idx_notes_meeting ON notes(meeting_id, created_at);
CREATE INDEX IF NOT EXISTS idx_speakers_meeting ON speakers(meeting_id);
"""


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


class Storage:
    """Thin sync SQLite wrapper. Guarded by an internal lock so it is safe to call
    from both the WebSocket loop and background worker threads.
    """

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        with self._lock:
            self._conn.executescript(SCHEMA)
            try:
                self._conn.execute("ALTER TABLE transcript_segments ADD COLUMN quality TEXT NOT NULL DEFAULT 'ok'")
            except sqlite3.OperationalError:
                pass
            try:
                self._conn.execute("ALTER TABLE notes ADD COLUMN time_seconds REAL")
                self._conn.commit()
            except sqlite3.OperationalError:
                pass
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def create_meeting(self, meeting_id: str, preferred_summary_language: str = "en") -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO meetings (id, started_at, preferred_summary_language) VALUES (?, ?, ?)",
                (meeting_id, _iso(_utc_now()), preferred_summary_language),
            )
            self._conn.commit()

    def end_meeting(self, meeting_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE meetings SET ended_at = ? WHERE id = ?",
                (_iso(_utc_now()), meeting_id),
            )
            self._conn.commit()

    def update_meeting_summary_language(self, meeting_id: str, preferred_summary_language: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE meetings SET preferred_summary_language = ? WHERE id = ?",
                (preferred_summary_language, meeting_id),
            )
            self._conn.commit()

    def get_meeting(self, meeting_id: str) -> Optional[dict[str, str | None]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT id, started_at, ended_at, preferred_summary_language FROM meetings WHERE id = ?",
                (meeting_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "id": row[0],
            "started_at": row[1],
            "ended_at": row[2],
            "preferred_summary_language": row[3],
        }

    def list_meetings(self, limit: int = 100) -> list[dict[str, object]]:
        """Recent meetings (newest first) with a segment count and first-line preview.

        Used by the meeting-history UI. Counts and previews are derived with
        correlated subqueries so a single round-trip returns everything the list
        needs without loading full transcripts.
        """
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT
                  m.id,
                  m.started_at,
                  m.ended_at,
                  m.preferred_summary_language,
                  (SELECT COUNT(*) FROM transcript_segments s WHERE s.meeting_id = m.id) AS segment_count,
                  (SELECT s2.text FROM transcript_segments s2 WHERE s2.meeting_id = m.id
                     ORDER BY s2.start_time ASC LIMIT 1) AS first_text
                FROM meetings m
                ORDER BY m.started_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            {
                "id": r[0],
                "started_at": r[1],
                "ended_at": r[2],
                "preferred_summary_language": r[3],
                "segment_count": r[4],
                "first_text": r[5],
            }
            for r in rows
        ]

    def delete_meeting(self, meeting_id: str) -> None:
        """Delete a meeting and all of its child rows (transcript, summaries, notes, speakers)."""
        with self._lock:
            for table in ("transcript_segments", "summary_snapshots", "notes", "speakers"):
                self._conn.execute(f"DELETE FROM {table} WHERE meeting_id = ?", (meeting_id,))
            self._conn.execute("DELETE FROM meetings WHERE id = ?", (meeting_id,))
            self._conn.commit()

    def insert_segment(self, seg: TranscriptSegment) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO transcript_segments
                                    (id, meeting_id, start_time, end_time, text, original_language, speaker_id, confidence, quality, translations, created_at)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    seg.id,
                    seg.meeting_id,
                    seg.start_time,
                    seg.end_time,
                    seg.text,
                    seg.original_language,
                    seg.speaker_id,
                    seg.confidence,
                    seg.quality,
                    json.dumps(dict(seg.translations)),
                    _iso(seg.created_at),
                ),
            )
            self._conn.commit()

    def delete_segment(self, segment_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM transcript_segments WHERE id = ?", (segment_id,))
            self._conn.commit()

    def update_segment_translations(self, segment_id: str, translations: dict[str, str]) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE transcript_segments SET translations = ? WHERE id = ?",
                (json.dumps(dict(translations)), segment_id),
            )
            self._conn.commit()

    def list_segments(self, meeting_id: str) -> list[TranscriptSegment]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id, meeting_id, start_time, end_time, text, original_language, speaker_id, confidence, quality, translations, created_at
                FROM transcript_segments WHERE meeting_id = ? ORDER BY start_time ASC
                """,
                (meeting_id,),
            ).fetchall()
        return [
            TranscriptSegment(
                id=r[0],
                meeting_id=r[1],
                start_time=r[2],
                end_time=r[3],
                text=r[4],
                original_language=r[5],
                speaker_id=r[6],
                confidence=r[7],
                quality=r[8],
                translations=json.loads(r[9] or "{}"),
                created_at=datetime.fromisoformat(r[10]),
            )
            for r in rows
        ]

    def insert_snapshot(self, snap: SummarySnapshot) -> None:
        with self._lock:
            # Clear is_latest on any prior snapshot of the same type for this meeting.
            self._conn.execute(
                "UPDATE summary_snapshots SET is_latest = 0 WHERE meeting_id = ? AND summary_type = ?",
                (snap.meeting_id, snap.summary_type),
            )
            self._conn.execute(
                """
                INSERT OR REPLACE INTO summary_snapshots
                  (id, meeting_id, summary_type, time_start, time_end, language, content,
                   source_segment_ids, is_latest, translations, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snap.id,
                    snap.meeting_id,
                    snap.summary_type,
                    snap.time_start,
                    snap.time_end,
                    snap.language,
                    snap.content,
                    json.dumps(list(snap.source_segment_ids)),
                    1 if snap.is_latest else 0,
                    json.dumps(dict(snap.translations)),
                    _iso(snap.created_at),
                ),
            )
            self._conn.commit()

    def list_snapshots(
        self,
        meeting_id: str,
        summary_type: Optional[SummaryType] = None,
    ) -> list[SummarySnapshot]:
        query = (
            "SELECT id, meeting_id, summary_type, time_start, time_end, language, content, "
            "source_segment_ids, is_latest, translations, created_at FROM summary_snapshots "
            "WHERE meeting_id = ?"
        )
        params: list[object] = [meeting_id]
        if summary_type is not None:
            query += " AND summary_type = ?"
            params.append(summary_type)
        query += " ORDER BY created_at ASC"
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [
            SummarySnapshot(
                id=r[0],
                meeting_id=r[1],
                summary_type=r[2],
                time_start=r[3],
                time_end=r[4],
                language=r[5],
                content=r[6],
                source_segment_ids=json.loads(r[7] or "[]"),
                is_latest=bool(r[8]),
                translations=json.loads(r[9] or "{}"),
                created_at=datetime.fromisoformat(r[10]),
            )
            for r in rows
        ]

    def insert_note(self, note: Note) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO notes (id, meeting_id, content, source_type, source_id, time_seconds, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    note.id,
                    note.meeting_id,
                    note.content,
                    note.source_type,
                    note.source_id,
                    note.time_seconds,
                    _iso(note.created_at),
                ),
            )
            self._conn.commit()

    def list_notes(self, meeting_id: str) -> list[Note]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, meeting_id, content, source_type, source_id, time_seconds, created_at "
                "FROM notes WHERE meeting_id = ? ORDER BY created_at ASC",
                (meeting_id,),
            ).fetchall()
        return [
            Note(
                id=r[0],
                meeting_id=r[1],
                content=r[2],
                source_type=r[3],
                source_id=r[4],
                time_seconds=r[5],
                created_at=datetime.fromisoformat(r[6]),
            )
            for r in rows
        ]

    def bulk_insert_segments(self, segments: Iterable[TranscriptSegment]) -> None:
        for s in segments:
            self.insert_segment(s)

    def upsert_speaker(self, speaker: Speaker) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO speakers
                  (id, meeting_id, display_name, inferred_label, confidence, is_local_user)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    speaker.id,
                    speaker.meeting_id,
                    speaker.display_name,
                    speaker.inferred_label,
                    speaker.confidence,
                    1 if speaker.is_local_user else 0,
                ),
            )
            self._conn.commit()

    def list_speakers(self, meeting_id: str) -> list[Speaker]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, meeting_id, display_name, inferred_label, confidence, is_local_user "
                "FROM speakers WHERE meeting_id = ?",
                (meeting_id,),
            ).fetchall()
        return [
            Speaker(
                id=r[0],
                meeting_id=r[1],
                display_name=r[2],
                inferred_label=r[3],
                confidence=r[4],
                is_local_user=bool(r[5]),
            )
            for r in rows
        ]
