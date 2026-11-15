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
                    PRIMARY KEY (folder_key, file_id)
                )
                """
            )
            cursor.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {self.table}_folder_key_idx
                ON {self.table} (folder_key)
                """
            )
        conn.commit()
        self._schema_ready = True

    def _run(self, operation):
        with self._lock:
            try:
                conn = self._connect_locked()
                self._ensure_schema_locked(conn)
                result = operation(conn)
                conn.commit()
                return result
            except psycopg2.Error as exc:
                if self._conn is not None and not self._conn.closed:
                    try:
                        self._conn.rollback()
                    except Exception:
                        pass
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

    def delete_folder(self, folder_key: str) -> int:
        def operation(conn):
            with conn.cursor() as cursor:
                cursor.execute(
                    f"DELETE FROM {self.table} WHERE folder_key = %s",
                    (folder_key,),
                )
                return cursor.rowcount or 0

        return int(self._run(operation))

    def close(self) -> None:
        with self._lock:
            if self._conn is not None and not self._conn.closed:
                self._conn.close()
