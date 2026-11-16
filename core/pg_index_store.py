from __future__ import annotations

import base64
from datetime import datetime, timezone

import numpy as np

from .config import EMBEDDING_SIZE
from .postgres_store import PostgresStore


def _encode_faces(faces: list[dict]) -> list[dict]:
    payload: list[dict] = []
    for face in faces or []:
        embedding = face.get("embedding")
        if embedding is None:
            continue
        array = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if array.size != EMBEDDING_SIZE:
            continue
        bbox = face.get("bbox") or (0, 0, 0, 0)
        payload.append(
            {
                "x": float(bbox[0]),
                "y": float(bbox[1]),
                "w": float(bbox[2]),
                "h": float(bbox[3]),
                "score": float(face.get("det_score") or 0.0),
                "emb": base64.b64encode(
                    array.astype(np.float32).tobytes()
                ).decode("ascii"),
            }
        )
    return payload


class PgIndexStore:
    """Tek bir klasörün indeksini Supabase üzerinde tutan store. Yerel SQL yoktur."""

    def __init__(self, postgres: PostgresStore, folder_key: str) -> None:
        self.pg = postgres
        self.folder_key = folder_key
        self._cache: tuple[np.ndarray, list[dict]] | None = None
        self._pending: list[dict] = []
        self._pending_limit = 100

    def flush(self) -> None:
        if not self._pending:
            return
        rows = self._pending
        self._pending = []
        self.pg.upsert(self.folder_key, rows)

    # ------------------------------------------------------------------- meta

    def set_meta(self, key: str, value: str) -> None:
        self.pg.set_folder_meta(self.folder_key, key, str(value))

    def meta_get(self, key: str, default: str | None = None) -> str | None:
        return self.pg.get_folder_meta(self.folder_key, key, default)

    # ------------------------------------------------------------------ writes

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
        self._cache = None
        self._pending.append(
            {
                "file_id": file_id,
                "name": name,
                "path": path,
                "width": int(width),
                "height": int(height),
                "status": status,
                "error": error,
                "faces": _encode_faces(faces),
            }
        )
        if len(self._pending) >= self._pending_limit:
            self.flush()

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
        self.flush()
        return self.pg.claim_images(
            self.folder_key,
            file_ids,
            force=force,
            stale_minutes=stale_minutes,
        )

    def import_rows(self, rows: list[dict]) -> int:
        if not rows:
            return 0
        self._cache = None
        self._pending.extend(rows)
        if len(self._pending) >= self._pending_limit:
            self.flush()
        return len(rows)

    def set_indexed_at(self) -> None:
        self.set_meta(
            "last_indexed",
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )

    # ------------------------------------------------------------------- reads

    def has_image(self, file_id: str) -> bool:
        self.flush()
        return self.pg.has_image(self.folder_key, file_id)

    def sample_image(self) -> tuple[str, str] | None:
        self.flush()
        return self.pg.sample_image(self.folder_key)

    def done_ids(self) -> set[str]:
        self.flush()
        return self.pg.done_ids(self.folder_key)

    def stats(self) -> dict:
        self.flush()
        return self.pg.folder_stats(self.folder_key)

    def load_matrix(self) -> tuple[np.ndarray, list[dict]]:
        if self._cache is not None:
            return self._cache

        self.flush()
        rows = self.pg.fetch(self.folder_key)
        embeddings: list[np.ndarray] = []
        metas: list[dict] = []
        for row in rows:
            if str(row.get("status") or "") != "done":
                continue
            for face in row.get("faces") or []:
                try:
                    raw = base64.b64decode(str(face.get("emb") or ""))
                    embedding = np.frombuffer(raw, dtype=np.float32)
                except Exception:
                    continue
                if embedding.size != EMBEDDING_SIZE:
                    continue
                embeddings.append(embedding.copy())
                metas.append(
                    {
                        "file_id": str(row.get("file_id") or ""),
                        "name": str(row.get("name") or ""),
                        "path": str(row.get("path") or ""),
                        "width": int(row.get("width") or 0),
                        "height": int(row.get("height") or 0),
                        "bbox": (
                            float(face.get("x") or 0.0),
                            float(face.get("y") or 0.0),
                            float(face.get("w") or 0.0),
                            float(face.get("h") or 0.0),
                        ),
                        "det_score": float(face.get("score") or 0.0),
                    }
                )

        if embeddings:
            matrix = np.stack(embeddings).astype(np.float32)
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            np.divide(matrix, norms, out=matrix, where=norms > 1e-8)
        else:
            matrix = np.zeros((0, EMBEDDING_SIZE), dtype=np.float32)

        self._cache = (matrix, metas)
        return matrix, metas

    def clear(self) -> None:
        self._cache = None
        self._pending = []
        self.pg.clear_folder(self.folder_key)

    def close(self) -> None:
        return None
