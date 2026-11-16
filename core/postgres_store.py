from __future__ import annotations

import json
import re
import threading
from typing import Any

import psycopg2
from psycopg2.extensions import make_dsn
from psycopg2.extras import Json, execute_values


class PostgresError(Exception):
    """Supabase/Postgres bağlantı ve sorgu hataları."""


_DISCONNECT_HINTS = (
    "server closed the connection unexpectedly",
    "connection already closed",
    "ssl connection has been closed",
    "terminating connection",
    "connection reset by peer",
    "could not receive data from server",
    "connection not open",
    "broken pipe",
)


def _is_disconnect(exc: Exception) -> bool:
    """Bağlantı kopması kaynaklı hataları ayırt eder (yeniden bağlanıp denenir)."""
    if isinstance(exc, (psycopg2.OperationalError, psycopg2.InterfaceError)):
        return True
    text = str(exc).lower()
    return any(hint in text for hint in _DISCONNECT_HINTS)


def _friendly_error(exc: Exception) -> str:
    text = str(exc).strip() or exc.__class__.__name__
    lowered = text.lower()
    hint = ""

    if "password authentication failed" in lowered:
        hint = " Supabase şifresini kontrol edin."
    elif "could not translate host name" in lowered or "name or service not known" in lowered:
        hint = (
            " Host adresini kontrol edin. Supabase'te Dashboard → Connect → "
            "Session pooler bağlantısını kullanmak daha güvenilirdir."
        )
    elif "connection refused" in lowered or "network is unreachable" in lowered:
        hint = (
            " Supabase doğrudan bağlantısı yalnızca IPv6 olabilir. "
            "Dashboard → Connect → Session pooler (port 5432) bağlantısını deneyin."
        )
    elif "timeout" in lowered or "timed out" in lowered:
        hint = " Ağ/port erişimini ve Supabase pooler adresini kontrol edin."
    elif "ssl" in lowered:
        hint = " sslmode=require ayarını kontrol edin."
    elif "does not exist" in lowered and "relation" in lowered:
        hint = " supabase_setup.sql dosyasındaki tabloyu oluşturun."

    return text + hint


class PostgresStore:
    """Supabase Postgres üzerinde kalıcı indeks saklar. Embedding'ler JSONB içinde base64'tür."""

    def __init__(
        self,
        config: dict[str, Any],
        table: str = "drive_face_index",
        connect_timeout: int = 10,
    ) -> None:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table):
            raise ValueError("Geçersiz tablo adı.")
        self.table = table
        self.connect_timeout = connect_timeout
        self._lock = threading.RLock()
        self._conn = None
        self._schema_ready = False

        dsn = str(config.get("dsn") or "").strip()
        if not dsn:
            dsn = make_dsn(
                host=config.get("host"),
                port=config.get("port"),
                dbname=config.get("dbname") or config.get("database") or "postgres",
                user=config.get("user"),
                password=config.get("password"),
                sslmode=config.get("sslmode") or "require",
            )
        self.dsn = dsn

    @property
    def enabled(self) -> bool:
        return bool(self.dsn)

    def _connect_locked(self):
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(
                self.dsn,
                connect_timeout=self.connect_timeout,
            )
            self._conn.autocommit = False
        return self._conn

    def _ensure_schema_locked(self, conn) -> None:
        if self._schema_ready:
            return
        with conn.cursor() as cursor:
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                    folder_key TEXT NOT NULL,
                    file_id TEXT NOT NULL,
                    name TEXT,
                    path TEXT,
                    width INTEGER NOT NULL DEFAULT 0,
                    height INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'done',
                    error TEXT,
                    faces JSONB NOT NULL DEFAULT '[]'::jsonb,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    claimed_at TIMESTAMPTZ,
                    PRIMARY KEY (folder_key, file_id)
                )
                """
            )
            cursor.execute(
                f"""
                ALTER TABLE {self.table}
                ADD COLUMN IF NOT EXISTS claimed_at TIMESTAMPTZ
                """
            )
            cursor.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {self.table}_folder_key_idx
                ON {self.table} (folder_key)
                """
            )
            cursor.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {self.table}_file_id_idx
                ON {self.table} (file_id)
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS drive_folders (
                    folder_key TEXT PRIMARY KEY,
                    meta JSONB NOT NULL DEFAULT '{}'::jsonb,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS drive_indexed_folders (
                    folder_id TEXT PRIMARY KEY,
                    root_key TEXT NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS drive_indexed_folders_root_idx
                ON drive_indexed_folders (root_key)
                """
            )
        conn.commit()
        self._schema_ready = True

    def _discard_connection(self) -> None:
        conn = self._conn
        self._conn = None
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    def _run(self, operation):
        with self._lock:
            for attempt in range(2):
                try:
                    conn = self._connect_locked()
                    self._ensure_schema_locked(conn)
                    result = operation(conn)
                    conn.commit()
                    return result
                except psycopg2.Error as exc:
                    try:
                        if self._conn is not None and not self._conn.closed:
                            self._conn.rollback()
                    except Exception:
                        pass
                    self._discard_connection()
                    # Boşta kalan bağlantıyı pooler kapatmış olabilir:
                    # bir kez taze bağlantıyla tekrar dene.
                    if attempt == 0 and _is_disconnect(exc):
                        continue
                    raise PostgresError(_friendly_error(exc)) from exc

    def ping(self) -> bool:
        def operation(conn):
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1")
                return cursor.fetchone()[0] == 1

        return bool(self._run(operation))

    def count(self, folder_key: str) -> int:
        def operation(conn):
            with conn.cursor() as cursor:
                cursor.execute(
                    f"SELECT COUNT(*) FROM {self.table} WHERE folder_key = %s",
                    (folder_key,),
                )
                return int(cursor.fetchone()[0])

        return int(self._run(operation))

    def upsert(self, folder_key: str, rows: list[dict]) -> int:
        if not rows:
            return 0

        def operation(conn):
            values = [
                (
                    folder_key,
                    str(row.get("file_id") or ""),
                    str(row.get("name") or ""),
                    str(row.get("path") or ""),
                    int(row.get("width") or 0),
                    int(row.get("height") or 0),
                    str(row.get("status") or "done"),
                    row.get("error"),
                    Json(row.get("faces") or []),
                )
                for row in rows
                if row.get("file_id")
            ]
            if not values:
                return 0

            with conn.cursor() as cursor:
                execute_values(
                    cursor,
                    f"""
                    INSERT INTO {self.table}
                        (folder_key, file_id, name, path, width, height,
                         status, error, faces)
                    VALUES %s
                    ON CONFLICT (folder_key, file_id) DO UPDATE SET
                        name = EXCLUDED.name,
                        path = EXCLUDED.path,
                        width = EXCLUDED.width,
                        height = EXCLUDED.height,
                        status = EXCLUDED.status,
                        error = EXCLUDED.error,
                        faces = EXCLUDED.faces,
                        updated_at = now()
                    """,
                    values,
                    page_size=200,
                )
            return len(values)

        return int(self._run(operation))

    def fetch(self, folder_key: str) -> list[dict]:
        def operation(conn):
            with conn.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT file_id, name, path, width, height, status, error, faces
                    FROM {self.table}
                    WHERE folder_key = %s
                    """,
                    (folder_key,),
                )
                output: list[dict] = []
                for row in cursor.fetchall():
                    faces = row[7]
                    if isinstance(faces, str):
                        try:
                            faces = json.loads(faces)
                        except ValueError:
                            faces = []
                    output.append(
                        {
                            "file_id": str(row[0]),
                            "name": str(row[1] or ""),
                            "path": str(row[2] or ""),
                            "width": int(row[3] or 0),
                            "height": int(row[4] or 0),
                            "status": str(row[5] or "done"),
                            "error": row[6],
                            "faces": faces or [],
                        }
                    )
                return output

        return self._run(operation)

    def fetch_existing(self, file_ids: list[str]) -> list[dict]:
        if not file_ids:
            return []

        def operation(conn):
            with conn.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT DISTINCT ON (file_id)
                        file_id, name, path, width, height, status, error, faces
                    FROM {self.table}
                    WHERE file_id = ANY(%s::text[])
                      AND status IN ('done', 'no_face')
                    ORDER BY file_id, updated_at DESC
                    """,
                    (file_ids,),
                )
                output: list[dict] = []
                for row in cursor.fetchall():
                    faces = row[7]
                    if isinstance(faces, str):
                        try:
                            faces = json.loads(faces)
                        except ValueError:
                            faces = []
                    output.append(
                        {
                            "file_id": str(row[0]),
                            "name": str(row[1] or ""),
                            "path": str(row[2] or ""),
                            "width": int(row[3] or 0),
                            "height": int(row[4] or 0),
                            "status": str(row[5] or "done"),
                            "error": row[6],
                            "faces": faces or [],
                        }
                    )
                return output

        return self._run(operation)

    def delete_folder(self, folder_key: str) -> int:
        def operation(conn):
            with conn.cursor() as cursor:
                cursor.execute(
                    f"DELETE FROM {self.table} WHERE folder_key = %s",
                    (folder_key,),
                )
                return cursor.rowcount or 0

        return int(self._run(operation))

    # ------------------------------------------------------------- folder meta

    def set_folder_meta(self, folder_key: str, key: str, value: str) -> None:
        def operation(conn):
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO drive_folders (folder_key, meta, updated_at)
                    VALUES (%s, jsonb_build_object(%s, %s::text), now())
                    ON CONFLICT (folder_key) DO UPDATE SET
                        meta = drive_folders.meta
                               || jsonb_build_object(%s, %s::text),
                        updated_at = now()
                    """,
                    (folder_key, key, value, key, value),
                )

        self._run(operation)

    def get_folder_meta(
        self, folder_key: str, key: str, default: str | None = None
    ) -> str | None:
        def operation(conn):
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT meta ->> %s FROM drive_folders WHERE folder_key = %s",
                    (key, folder_key),
                )
                row = cursor.fetchone()
                if row is None or row[0] is None:
                    return default
                return str(row[0])

        return self._run(operation)

    def list_folders(self) -> list[dict]:
        def operation(conn):
            with conn.cursor() as cursor:
                cursor.execute("SELECT folder_key, meta FROM drive_folders")
                output: list[dict] = []
                for folder_key, meta in cursor.fetchall():
                    if isinstance(meta, str):
                        try:
                            meta = json.loads(meta)
                        except ValueError:
                            meta = {}
                    output.append(
                        {"folder_key": str(folder_key), "meta": meta or {}}
                    )
                return output

        return self._run(operation)

    def delete_folder_meta(self, folder_key: str) -> None:
        def operation(conn):
            with conn.cursor() as cursor:
                cursor.execute(
                    "DELETE FROM drive_folders WHERE folder_key = %s",
                    (folder_key,),
                )

        self._run(operation)

    def folder_stats(self, folder_key: str) -> dict:
        def operation(conn):
            with conn.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT
                        COUNT(*),
                        COUNT(*) FILTER (WHERE status = 'done'),
                        COUNT(*) FILTER (WHERE status = 'no_face'),
                        COUNT(*) FILTER (WHERE status = 'error'),
                        COALESCE(SUM(jsonb_array_length(faces)), 0)
                    FROM {self.table}
                    WHERE folder_key = %s
                    """,
                    (folder_key,),
                )
                total, done, no_face, errors, faces = cursor.fetchone()
                return {
                    "images": int(total or 0),
                    "done": int(done or 0),
                    "no_face": int(no_face or 0),
                    "errors": int(errors or 0),
                    "faces": int(faces or 0),
                }

        return self._run(operation)

    def sample_image(self, folder_key: str) -> tuple[str, str] | None:
        def operation(conn):
            with conn.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT file_id, path FROM {self.table}
                    WHERE folder_key = %s AND path IS NOT NULL AND path <> ''
                    ORDER BY LENGTH(path) ASC LIMIT 1
                    """,
                    (folder_key,),
                )
                row = cursor.fetchone()
                if row is None:
                    return None
                return str(row[0]), str(row[1] or "")

        return self._run(operation)

    def has_image(self, folder_key: str, file_id: str) -> bool:
        def operation(conn):
            with conn.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT 1 FROM {self.table}
                    WHERE folder_key = %s AND file_id = %s LIMIT 1
                    """,
                    (folder_key, file_id),
                )
                return cursor.fetchone() is not None

        return bool(self._run(operation))

    def done_ids(self, folder_key: str) -> set[str]:
        def operation(conn):
            with conn.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT file_id FROM {self.table}
                    WHERE folder_key = %s AND status IN ('done', 'no_face')
                    """,
                    (folder_key,),
                )
                return {str(row[0]) for row in cursor.fetchall()}

        return self._run(operation)

    def claim_images(
        self,
        folder_key: str,
        file_ids: list[str],
        *,
        force: bool = False,
        stale_minutes: int = 30,
    ) -> set[str]:
        if not file_ids:
            return set()
        stale = max(1, int(stale_minutes))

        def operation(conn):
            claimed: set[str] = set()
            with conn.cursor() as cursor:
                for file_id in file_ids:
                    cursor.execute(
                        f"""
                        INSERT INTO {self.table}
                            (folder_key, file_id, name, path, status,
                             updated_at, claimed_at)
                        VALUES (%s, %s, '', '', 'processing', now(), now())
                        ON CONFLICT (folder_key, file_id) DO NOTHING
                        """,
                        (folder_key, file_id),
                    )
                    if cursor.rowcount:
                        claimed.add(file_id)
                        continue

                    if force:
                        cursor.execute(
                            f"""
                            UPDATE {self.table}
                            SET claimed_at = now(),
                                status = CASE
                                    WHEN status IN ('done', 'no_face')
                                    THEN status
                                    ELSE 'processing'
                                END
                            WHERE folder_key = %s AND file_id = %s
                              AND (claimed_at IS NULL
                                   OR claimed_at < now()
                                      - interval '{stale} minutes')
                            """,
                            (folder_key, file_id),
                        )
                    else:
                        cursor.execute(
                            f"""
                            UPDATE {self.table}
                            SET status = 'processing',
                                updated_at = now(),
                                claimed_at = now()
                            WHERE folder_key = %s AND file_id = %s
                              AND status NOT IN ('done', 'no_face')
                              AND (claimed_at IS NULL
                                   OR claimed_at < now()
                                      - interval '{stale} minutes')
                            """,
                            (folder_key, file_id),
                        )
                    if cursor.rowcount:
                        claimed.add(file_id)
            return claimed

        return self._run(operation)

    def register_indexed_folders(
        self, folder_ids: list[str], root_key: str
    ) -> int:
        ids = [str(folder_id) for folder_id in folder_ids if folder_id]
        if not ids:
            return 0

        def operation(conn):
            with conn.cursor() as cursor:
                execute_values(
                    cursor,
                    """
                    INSERT INTO drive_indexed_folders (folder_id, root_key)
                    VALUES %s
                    ON CONFLICT (folder_id) DO UPDATE SET
                        root_key = EXCLUDED.root_key,
                        updated_at = now()
                    """,
                    [(folder_id, root_key) for folder_id in ids],
                    page_size=500,
                )
            return len(ids)

        return int(self._run(operation))

    def is_indexed_folder(self, folder_id: str) -> bool:
        if not folder_id:
            return False

        def operation(conn):
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT 1 FROM drive_indexed_folders WHERE folder_id = %s LIMIT 1",
                    (folder_id,),
                )
                return cursor.fetchone() is not None

        return bool(self._run(operation))

    def clear_folder(self, folder_key: str) -> None:
        def operation(conn):
            with conn.cursor() as cursor:
                cursor.execute(
                    f"DELETE FROM {self.table} WHERE folder_key = %s",
                    (folder_key,),
                )
                cursor.execute(
                    "DELETE FROM drive_folders WHERE folder_key = %s",
                    (folder_key,),
                )
                cursor.execute(
                    "DELETE FROM drive_indexed_folders WHERE root_key = %s",
                    (folder_key,),
                )

        self._run(operation)

    def close(self) -> None:
        with self._lock:
            if self._conn is not None and not self._conn.closed:
                self._conn.close()
