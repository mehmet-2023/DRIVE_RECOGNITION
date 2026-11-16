from __future__ import annotations

import base64
import json
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

EMBEDDING_SIZE = 128


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class IndexStore:
    """Yerel, hızlı SQLite indeksi. Sunucu uykuya dalınca kaybolur; Postgres ile senkronlanır."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._version = 0
        self._cache: tuple[int, np.ndarray, list[dict]] | None = None

        self.conn = sqlite3.connect(
            str(self.path),
            check_same_thread=False,
            timeout=60,
        )
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA temp_store=MEMORY")
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self.conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT
                );

                CREATE TABLE IF NOT EXISTS images (
                    file_id TEXT PRIMARY KEY,
                    name TEXT,
                    path TEXT,
                    status TEXT NOT NULL DEFAULT 'done',
                    face_count INTEGER NOT NULL DEFAULT 0,
                    width INTEGER NOT NULL DEFAULT 0,
                    height INTEGER NOT NULL DEFAULT 0,
                    error TEXT,
                    updated_at TEXT NOT NULL,
                    claimed_at TEXT
                );

                CREATE TABLE IF NOT EXISTS faces (
                    file_id TEXT NOT NULL,
                    face_idx INTEGER NOT NULL,
                    x REAL NOT NULL,
                    y REAL NOT NULL,
                    w REAL NOT NULL,
                    h REAL NOT NULL,
                    det_score REAL NOT NULL,
                    embedding BLOB NOT NULL,
                    PRIMARY KEY (file_id, face_idx)
                );

                CREATE INDEX IF NOT EXISTS idx_faces_file_id ON faces(file_id);
                CREATE INDEX IF NOT EXISTS idx_images_status ON images(status);
                """
            )
            columns = {
                str(row[1])
                for row in self.conn.execute("PRAGMA table_info(images)").fetchall()
            }
            if "claimed_at" not in columns:
                self.conn.execute("ALTER TABLE images ADD COLUMN claimed_at TEXT")
            self.conn.commit()

    def _bump(self) -> None:
        self._version += 1
        self._cache = None

    # ------------------------------------------------------------------- writes

    def save_result(
        self,
        file_id: str,
        name: str,
        path: str,
        width: int,
        height: int,
        faces: list[dict],
        status: str,
        error: str | None = None,
        *,
        commit: bool = True,
    ) -> None:
        clean_faces: list[dict] = []
        for face in faces or []:
            embedding = face.get("embedding")
            if embedding is None:
                continue
            array = np.asarray(embedding, dtype=np.float32).reshape(-1)
            if array.size != EMBEDDING_SIZE:
                continue
            bbox = face.get("bbox") or (0, 0, 0, 0)
            clean_faces.append(
                {
                    "bbox": tuple(float(value) for value in bbox[:4]),
                    "det_score": float(face.get("det_score") or 0.0),
                    "embedding": array,
                }
            )

        with self._lock:
            self.conn.execute(
                """
                INSERT INTO images
                    (file_id, name, path, status, face_count, width, height, error, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(file_id) DO UPDATE SET
                    name = excluded.name,
                    path = excluded.path,
                    status = excluded.status,
                    face_count = excluded.face_count,
                    width = excluded.width,
                    height = excluded.height,
                    error = excluded.error,
                    updated_at = excluded.updated_at,
                    claimed_at = NULL
                """,
                (
                    file_id,
                    name,
                    path,
                    status,
                    len(clean_faces),
                    int(width),
                    int(height),
                    error,
                    _now_iso(),
                ),
            )
            self.conn.execute("DELETE FROM faces WHERE file_id = ?", (file_id,))
            for index, face in enumerate(clean_faces):
                x, y, w, h = face["bbox"]
                self.conn.execute(
                    """
                    INSERT INTO faces
                        (file_id, face_idx, x, y, w, h, det_score, embedding)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        file_id,
                        index,
                        x,
                        y,
                        w,
                        h,
                        face["det_score"],
                        face["embedding"].astype(np.float32).tobytes(),
                    ),
                )
            if commit:
                self.conn.commit()
                self._bump()

    def save_error(
        self,
        file_id: str,
        name: str,
        path: str,
        error: str,
        *,
        commit: bool = True,
    ) -> None:
        self.save_result(
            file_id=file_id,
            name=name,
            path=path,
            width=0,
            height=0,
            faces=[],
            status="error",
            error=error,
            commit=commit,
        )

    def claim_images(
        self,
        file_ids: list[str],
        *,
        force: bool = False,
        stale_minutes: int = 30,
    ) -> set[str]:
        """SQL'de atomik olarak sahiplenir; başka bir indeksleme işliyorsa atlar."""
        if not file_ids:
            return set()

        now = datetime.now(timezone.utc)
        now_iso = now.isoformat(timespec="seconds")
        stale_iso = (now - timedelta(minutes=max(1, int(stale_minutes)))).isoformat(
            timespec="seconds"
        )
        claimed: set[str] = set()

        with self._lock:
            try:
                for file_id in file_ids:
                    cursor = self.conn.execute(
                        """
                        INSERT OR IGNORE INTO images
                            (file_id, name, path, status, face_count,
                             width, height, error, updated_at, claimed_at)
                        VALUES (?, '', '', 'processing', 0, 0, 0, NULL, ?, ?)
                        """,
                        (file_id, now_iso, now_iso),
                    )
                    if cursor.rowcount:
                        claimed.add(file_id)
                        continue

                    if force:
                        cursor = self.conn.execute(
                            """
                            UPDATE images
                            SET claimed_at = ?,
                                status = CASE
                                    WHEN status IN ('done', 'no_face')
                                    THEN status
                                    ELSE 'processing'
                                END
                            WHERE file_id = ?
                              AND (claimed_at IS NULL OR claimed_at < ?)
                            """,
                            (now_iso, file_id, stale_iso),
                        )
                    else:
                        cursor = self.conn.execute(
                            """
                            UPDATE images
                            SET status = 'processing',
                                updated_at = ?,
                                claimed_at = ?
                            WHERE file_id = ?
                              AND status NOT IN ('done', 'no_face')
                              AND (claimed_at IS NULL OR claimed_at < ?)
                            """,
                            (now_iso, now_iso, file_id, stale_iso),
                        )

                    if cursor.rowcount:
                        claimed.add(file_id)

                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise

            if claimed:
                self._bump()

        return claimed

    def set_meta(self, key: str, value: str) -> None:
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO meta (key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (key, value),
            )
            self.conn.commit()

    def import_rows(self, rows: list[dict]) -> int:
        if not rows:
            return 0
        imported = 0
        for row in rows:
            payload = row.get("faces") or []
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except ValueError:
                    payload = []

            faces: list[dict] = []
            for face in payload:
                try:
                    raw = base64.b64decode(str(face.get("emb") or ""))
                    embedding = np.frombuffer(raw, dtype=np.float32)
                except Exception:
                    continue
                if embedding.size != EMBEDDING_SIZE:
                    continue
                faces.append(
                    {
                        "bbox": (
                            float(face.get("x") or 0.0),
                            float(face.get("y") or 0.0),
                            float(face.get("w") or 0.0),
                            float(face.get("h") or 0.0),
                        ),
                        "det_score": float(face.get("score") or 0.0),
                        "embedding": embedding.copy(),
                    }
                )

            self.save_result(
                file_id=str(row.get("file_id") or ""),
                name=str(row.get("name") or ""),
                path=str(row.get("path") or ""),
                width=int(row.get("width") or 0),
                height=int(row.get("height") or 0),
                faces=faces,
                status=str(row.get("status") or "done"),
                error=row.get("error"),
                commit=False,
            )
            imported += 1

        with self._lock:
            self.conn.commit()
            self._bump()
        return imported

    # -------------------------------------------------------------------- reads

    def has_image(self, file_id: str) -> bool:
        with self._lock:
            row = self.conn.execute(
                "SELECT 1 FROM images WHERE file_id = ? LIMIT 1", (file_id,)
            ).fetchone()
        return row is not None

    def sample_image(self) -> tuple[str, str] | None:
        """En sığ yollu bir fotoğrafın (file_id, path) değerini döndürür."""
        with self._lock:
            row = self.conn.execute(
                "SELECT file_id, path FROM images "
                "WHERE path IS NOT NULL AND path != '' "
                "ORDER BY LENGTH(path) ASC LIMIT 1"
            ).fetchone()
        if not row:
            return None
        return str(row[0]), str(row[1] or "")

    def done_ids(self) -> set[str]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT file_id FROM images WHERE status IN ('done', 'no_face')"
            ).fetchall()
        return {str(row[0]) for row in rows}

    def meta_get(self, key: str, default: str | None = None) -> str | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT value FROM meta WHERE key = ?", (key,)
            ).fetchone()
        return str(row[0]) if row else default

    def stats(self) -> dict:
        with self._lock:
            total = int(
                self.conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
            )
            done = int(
                self.conn.execute(
                    "SELECT COUNT(*) FROM images WHERE status = 'done'"
                ).fetchone()[0]
            )
            no_face = int(
                self.conn.execute(
                    "SELECT COUNT(*) FROM images WHERE status = 'no_face'"
                ).fetchone()[0]
            )
            errors = int(
                self.conn.execute(
                    "SELECT COUNT(*) FROM images WHERE status = 'error'"
                ).fetchone()[0]
            )
            faces = int(self.conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0])
            last_indexed = self.meta_get("last_indexed")
            folder_url = self.meta_get("folder_url")
        return {
            "images": total,
            "done": done,
            "no_face": no_face,
            "errors": errors,
            "faces": faces,
            "last_indexed": last_indexed,
            "folder_url": folder_url,
        }

    def load_matrix(self) -> tuple[np.ndarray, list[dict]]:
        with self._lock:
            if self._cache is not None and self._cache[0] == self._version:
                return self._cache[1], self._cache[2]

            rows = self.conn.execute(
                """
                SELECT f.file_id, i.name, i.path, i.width, i.height,
                       f.face_idx, f.x, f.y, f.w, f.h, f.det_score, f.embedding
                FROM faces f
                JOIN images i ON i.file_id = f.file_id
                WHERE i.status = 'done'
                """
            ).fetchall()

        embeddings: list[np.ndarray] = []
        metas: list[dict] = []
        for row in rows:
            embedding = np.frombuffer(row[11], dtype=np.float32)
            if embedding.size != EMBEDDING_SIZE:
                continue
            embeddings.append(embedding)
            metas.append(
                {
                    "file_id": str(row[0]),
                    "name": str(row[1] or ""),
                    "path": str(row[2] or ""),
                    "width": int(row[3] or 0),
                    "height": int(row[4] or 0),
                    "face_idx": int(row[5]),
                    "bbox": (
                        float(row[6]),
                        float(row[7]),
                        float(row[8]),
                        float(row[9]),
                    ),
                    "det_score": float(row[10]),
                }
            )

        if embeddings:
            matrix = np.stack(embeddings).astype(np.float32)
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            np.divide(matrix, norms, out=matrix, where=norms > 1e-8)
        else:
            matrix = np.zeros((0, EMBEDDING_SIZE), dtype=np.float32)

        with self._lock:
            self._cache = (self._version, matrix, metas)
        return matrix, metas

    def export_rows(self) -> list[dict]:
        """Postgres/JSONB için base64 embeddingli satırlar üretir."""
        with self._lock:
            face_rows = self.conn.execute(
                """
                SELECT file_id, face_idx, x, y, w, h, det_score, embedding
                FROM faces ORDER BY file_id, face_idx
                """
            ).fetchall()
            image_rows = self.conn.execute(
                """
                SELECT file_id, name, path, status, width, height, error
                FROM images
                """
            ).fetchall()

        faces_by_file: dict[str, list[dict]] = {}
        for row in face_rows:
            embedding = np.frombuffer(row[7], dtype=np.float32)
            faces_by_file.setdefault(str(row[0]), []).append(
                {
                    "x": float(row[2]),
                    "y": float(row[3]),
                    "w": float(row[4]),
                    "h": float(row[5]),
                    "score": float(row[6]),
                    "emb": base64.b64encode(
                        embedding.astype(np.float32).tobytes()
                    ).decode("ascii"),
                }
            )

        rows: list[dict] = []
        for row in image_rows:
            rows.append(
                {
                    "file_id": str(row[0]),
                    "name": str(row[1] or ""),
                    "path": str(row[2] or ""),
                    "status": str(row[3] or "done"),
                    "width": int(row[4] or 0),
                    "height": int(row[5] or 0),
                    "error": row[6],
                    "faces": faces_by_file.get(str(row[0]), []),
                }
            )
        return rows

    # ------------------------------------------------------------ import/export

    def export_bytes(self) -> bytes:
        temporary = self.path.with_name(self.path.stem + ".export.sqlite")
        temporary.unlink(missing_ok=True)
        with self._lock:
            self.conn.commit()
            destination = sqlite3.connect(str(temporary))
            try:
                self.conn.backup(destination)
            finally:
                destination.close()
        try:
            return temporary.read_bytes()
        finally:
            temporary.unlink(missing_ok=True)

    def merge_sqlite_bytes(self, data: bytes) -> int:
        temporary = self.path.with_name(self.path.stem + ".upload.sqlite")
        temporary.write_bytes(data)
        try:
            with self._lock:
                self.conn.commit()
                self.conn.execute("ATTACH DATABASE ? AS upload_db", (str(temporary),))
                try:
                    self.conn.execute(
                        """
                        INSERT OR REPLACE INTO images
                            (file_id, name, path, status, face_count,
                             width, height, error, updated_at)
                        SELECT file_id, name, path, status, face_count,
                               width, height, error, updated_at
                        FROM upload_db.images
                        """
                    )
                    self.conn.execute(
                        """
                        INSERT OR REPLACE INTO faces
                            (file_id, face_idx, x, y, w, h, det_score, embedding)
                        SELECT file_id, face_idx, x, y, w, h, det_score, embedding
                        FROM upload_db.faces
                        """
                    )
                    self.conn.commit()
                except Exception:
                    self.conn.rollback()
                    raise
                finally:
                    self.conn.execute("DETACH DATABASE upload_db")
                self._bump()
                return int(
                    self.conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
                )
        except sqlite3.DatabaseError as exc:
            raise ValueError(
                "Yüklenen dosya geçerli bir indeks veritabanı değil."
            ) from exc
        finally:
            temporary.unlink(missing_ok=True)

    def clear(self) -> None:
        with self._lock:
            self.conn.execute("DELETE FROM faces")
            self.conn.execute("DELETE FROM images")
            self.conn.execute("DELETE FROM meta")
            self.conn.commit()
            self._bump()

    def close(self) -> None:
        with self._lock:
            try:
                self.conn.commit()
            finally:
                self.conn.close()
