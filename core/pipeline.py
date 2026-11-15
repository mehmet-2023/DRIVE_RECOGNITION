from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

import numpy as np

from .drive import DriveClient, DriveItem
from .face_engine import FaceEngine
from .index_store import EMBEDDING_SIZE, IndexStore


@dataclass(slots=True)
class IndexOptions:
    include_subfolders: bool = True
    thumb_size: int = 768
    det_threshold: float = 0.60
    min_face: int = 40
    max_faces: int = 8
    max_images: int = 0
    force: bool = False
    workers: int = 8
    sync_batch: int = 100


def encode_row(
    file_id: str,
    name: str,
    path: str,
    width: int,
    height: int,
    status: str,
    error: str | None,
    faces: list[dict],
) -> dict:
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

    return {
        "file_id": file_id,
        "name": name,
        "path": path,
        "width": int(width),
        "height": int(height),
        "status": status,
        "error": error,
        "faces": payload,
    }


def build_index(
    client: DriveClient,
    engine: FaceEngine,
    store: IndexStore,
    folder_id: str,
    resource_key: str | None,
    options: IndexOptions,
    *,
    on_status: Callable[[str], None] | None = None,
    on_progress: Callable[[int, int, int, dict], None] | None = None,
    on_batch: Callable[[list[dict]], None] | None = None,
) -> dict:
    if on_status:
        on_status("Klasör okunuyor...")

    items: list[DriveItem] = []
    seen: set[str] = set()
    for item in client.walk_images(
        folder_id,
        include_subfolders=options.include_subfolders,
        resource_key=resource_key,
    ):
        if item.id in seen:
            continue
        seen.add(item.id)
        items.append(item)
        if options.max_images and len(items) >= int(options.max_images):
            break

    total_found = len(items)
    if on_status:
        on_status(f"{total_found} fotoğraf bulundu.")

    done_ids = set() if options.force else store.done_ids()
    todo = [item for item in items if item.id not in done_ids]
    skipped = total_found - len(todo)

    if on_status:
        on_status(
            f"{len(todo)} fotoğraf işlenecek, {skipped} fotoğraf zaten kayıtlı."
        )

    counters = {"done": 0, "no_face": 0, "error": 0, "faces": 0}
    processed = 0
    realtime_skipped = 0
    batch: list[dict] = []
    sync_errors: list[str] = []
    workers = max(1, min(int(options.workers), 12))
    executor = ThreadPoolExecutor(max_workers=workers)

    def flush_batch(force: bool = False) -> None:
        nonlocal batch
        if not batch:
            return
        if not force and len(batch) < max(1, int(options.sync_batch)):
            return
        if on_batch is not None:
            try:
                on_batch(batch)
            except Exception as exc:
                sync_errors.append(str(exc)[:300])
        batch = []

    try:
        for start in range(0, len(todo), 64):
            chunk = todo[start : start + 64]
            claimed_ids = store.claim_images(
                [item.id for item in chunk],
                force=options.force,
            )
            realtime_skipped += len(chunk) - len(claimed_ids)
            chunk = [item for item in chunk if item.id in claimed_ids]
            if not chunk:
                continue

            future_map = {
                executor.submit(
                    client.fetch_image,
                    item.id,
                    options.thumb_size,
                    resource_key,
                ): item
                for item in chunk
            }

            for future in as_completed(future_map):
                item = future_map[future]
                image = None
                error_text: str | None = None

                try:
                    image = future.result()
                except Exception as exc:
                    error_text = str(exc)[:300]

                if image is None:
                    error_text = error_text or "Görsel indirilemedi veya çözülemedi."
                    store.save_error(item.id, item.name, item.path, error_text)
                    counters["error"] += 1
                    row = encode_row(
                        item.id,
                        item.name,
                        item.path,
                        0,
                        0,
                        "error",
                        error_text,
                        [],
                    )
                else:
                    height, width = image.shape[:2]
                    try:
                        faces, _ = engine.analyze(
                            image,
                            det_threshold=options.det_threshold,
                            min_face=options.min_face,
                            max_faces=options.max_faces,
                            input_max_side=options.thumb_size,
                        )
                    except Exception as exc:
                        error_text = f"Yüz analizi hatası: {exc}"[:300]
                        store.save_error(item.id, item.name, item.path, error_text)
                        counters["error"] += 1
                        row = encode_row(
                            item.id,
                            item.name,
                            item.path,
                            width,
                            height,
                            "error",
                            error_text,
                            [],
                        )
                    else:
                        status = "done" if faces else "no_face"
                        store.save_result(
                            item.id,
                            item.name,
                            item.path,
                            width,
                            height,
                            faces,
                            status,
                        )
                        if faces:
                            counters["done"] += 1
                            counters["faces"] += len(faces)
                        else:
                            counters["no_face"] += 1
                        row = encode_row(
                            item.id,
                            item.name,
                            item.path,
                            width,
                            height,
                            status,
                            None,
                            faces,
                        )

                batch.append(row)
                processed += 1
                if on_progress is not None:
                    on_progress(
                        processed,
                        len(todo),
                        skipped + realtime_skipped,
                        counters,
                    )
                flush_batch()
    finally:
        executor.shutdown(wait=True)
        flush_batch(force=True)

    skipped += realtime_skipped
    store.set_meta("folder_id", folder_id)
    store.set_meta(
        "last_indexed",
        datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )

    return {
        "total_found": total_found,
        "processed": processed,
        "skipped": skipped,
        "counters": counters,
        "sync_errors": sync_errors,
        "warnings": list(client.warnings),
    }


def search_embeddings(
    matrix: np.ndarray,
    metas: list[dict],
    probes: list[np.ndarray],
    threshold: float = 0.50,
    max_results: int | None = None,
) -> list[dict]:
    if matrix.size == 0 or not probes or not metas:
        return []

    probe_matrix = np.stack(
        [np.asarray(probe, dtype=np.float32).reshape(-1) for probe in probes]
    )
    norms = np.linalg.norm(probe_matrix, axis=1, keepdims=True)
    np.divide(probe_matrix, norms, out=probe_matrix, where=norms > 1e-8)

    similarities = matrix @ probe_matrix.T
    best_scores = similarities.max(axis=1)
    best_probes = similarities.argmax(axis=1)

    candidate_indices = np.where(best_scores >= float(threshold))[0]
    best_by_file: dict[str, dict] = {}

    for raw_index in candidate_indices:
        index = int(raw_index)
        meta = metas[index]
        score = float(best_scores[index])
        current = best_by_file.get(meta["file_id"])
        if current is None or score > current["score"]:
            best_by_file[meta["file_id"]] = {
                **meta,
                "score": score,
                "probe_index": int(best_probes[index]),
            }

    results = sorted(
        best_by_file.values(),
        key=lambda item: item["score"],
        reverse=True,
    )
    if max_results is None:
        return results
    return results[: max(1, int(max_results))]
