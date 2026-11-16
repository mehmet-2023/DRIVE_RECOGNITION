from __future__ import annotations

import hashlib
import io
import json
import zipfile
from datetime import datetime, timezone

import numpy as np
import streamlit as st

try:
    from starlette.requests import ClientDisconnect as _ClientDisconnect  # type: ignore
except ImportError:
    _ClientDisconnect = None  # type: ignore

# Yakalanacak istisna sınıfları—ClientDisconnect yoksa sadece boş tuple
_DISCONNECT_EXC = (_ClientDisconnect,) if _ClientDisconnect is not None else ()

st.set_page_config(
    page_title="Drive Yüz Tanıma",
    page_icon="⬛",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from core.config import (  # noqa: E402
    APP_BRAND,
    APP_TITLE,
    DEFAULT_DET_THRESHOLD,
    DEFAULT_MATCH_THRESHOLD,
    DEFAULT_MAX_FACES,
    DEFAULT_MIN_FACE,
    DEFAULT_THUMB_SIZE,
    drive_view_url,
    folder_key,
    get_secret,
    get_service_account_info,
    get_supabase_config,
)
from core.drive import DriveClient, DriveError, parse_drive_folder_url  # noqa: E402
from core.face_engine import FaceEngine, ensure_models  # noqa: E402
from core.image_utils import (  # noqa: E402
    bgr_to_rgb,
    decode_image_bytes,
    encode_jpeg_bytes,
)
from core.pg_index_store import PgIndexStore  # noqa: E402
from core.pipeline import IndexOptions, build_index, search_embeddings  # noqa: E402
from core.postgres_store import PostgresError, PostgresStore  # noqa: E402

PREVIEW_THUMB_SIZE = 384

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

    :root {
        --ink: #0a0a0a;
        --paper: #ffffff;
        --soft: #f3f3f0;
        --muted: #686868;
        --line: #0a0a0a;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .stApp, div[data-testid="stAppViewContainer"] {
        background: var(--paper);
        color: var(--ink);
    }
    div[data-testid="stHeader"] { background: transparent; }
    div[data-testid="stToolbar"],
    div[data-testid="stDecoration"],
    div[data-testid="stSidebar"],
    div[data-testid="stSidebarCollapsedControl"],
    #MainMenu, footer { display: none !important; }

    .block-container {
        max-width: 1080px;
        padding: 0 24px 72px;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Space Mono', monospace;
        color: var(--ink);
        letter-spacing: -.045em;
    }

    .hero {
        background: var(--ink);
        color: var(--paper);
        padding: 56px 40px 48px;
        margin: 0 0 56px;
    }
    .hero-top {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        gap: 24px;
        margin-bottom: 40px;
    }
    .brand, .kicker {
        font-family: 'Space Mono', monospace;
        font-size: .68rem;
        letter-spacing: .2em;
        text-transform: uppercase;
    }
    .brand { color: #fff; font-weight: 700; }
    .kicker { color: #9a9a9a; }
    .hero h1 {
        color: #fff;
        font-size: clamp(2.4rem, 6vw, 4.8rem);
        line-height: .95;
        letter-spacing: -.07em;
        margin: 0 0 20px;
        max-width: 820px;
    }
    .hero p {
        color: #c9c9c9;
        font-size: 1rem;
        line-height: 1.55;
        max-width: 560px;
        margin: 0;
    }

    .step-head {
        display: flex;
        gap: 16px;
        align-items: flex-start;
        margin: 0 0 20px;
    }
    .step-no {
        font-family: 'Space Mono', monospace;
        font-size: .68rem;
        letter-spacing: .12em;
        border: 1px solid var(--line);
        padding: 6px 7px;
        line-height: 1;
        white-space: nowrap;
    }
    .step-title {
        font-family: 'Space Mono', monospace;
        font-weight: 700;
        font-size: 1.05rem;
        letter-spacing: -.03em;
        line-height: 1.2;
    }
    .step-desc {
        color: var(--muted);
        font-size: .9rem;
        line-height: 1.45;
        margin-top: 3px;
        max-width: 640px;
    }
    .step-note {
        color: var(--muted);
        font-size: .78rem;
        line-height: 1.4;
        margin-top: 6px;
        padding-left: 10px;
        border-left: 2px solid var(--line);
        max-width: 640px;
    }
    .rule {
        border-top: 1px solid var(--line);
        margin: 48px 0 28px;
    }
    .hint {
        color: var(--muted);
        font-size: .8rem;
        line-height: 1.45;
        margin-top: 8px;
    }
    .footer {
        border-top: 1px solid var(--line);
        margin-top: 48px;
        padding-top: 16px;
        color: var(--muted);
        font-size: .72rem;
        font-family: 'Space Mono', monospace;
        letter-spacing: .04em;
        display: flex;
        justify-content: space-between;
        gap: 16px;
        flex-wrap: wrap;
    }
    .no-image {
        border: 1px dashed var(--line);
        color: var(--muted);
        font-family: 'Space Mono', monospace;
        font-size: .68rem;
        text-transform: uppercase;
        letter-spacing: .08em;
        padding: 36px 12px;
        text-align: center;
    }

    div[data-baseweb="input"],
    div[data-baseweb="base-input"],
    div[data-baseweb="select"] > div {
        border-radius: 0 !important;
        border: 1px solid var(--line) !important;
        background: var(--paper) !important;
        box-shadow: none !important;
    }
    div[data-baseweb="input"] input,
    div[data-baseweb="select"] input {
        font-family: 'Inter', sans-serif;
        color: var(--ink) !important;
    }
    div[data-baseweb="input"]:focus-within,
    div[data-baseweb="select"] > div:focus-within {
        box-shadow: 0 0 0 1px var(--line) !important;
    }

    div[data-testid="stButton"] > button,
    div[data-testid="stDownloadButton"] > button,
    div[data-testid="stLinkButton"] > a {
        border-radius: 0 !important;
        border: 1px solid var(--line) !important;
        background: var(--paper) !important;
        color: var(--ink) !important;
        font-family: 'Space Mono', monospace !important;
        font-size: .68rem !important;
        letter-spacing: .1em !important;
        text-transform: uppercase !important;
        padding: .65rem .9rem !important;
        box-shadow: none !important;
        transition: background .15s ease, color .15s ease;
    }
    div[data-testid="stButton"] > button:hover,
    div[data-testid="stDownloadButton"] > button:hover,
    div[data-testid="stLinkButton"] > a:hover {
        background: var(--ink) !important;
        color: var(--paper) !important;
    }
    div[data-testid="stButton"] > button[kind="primary"] {
        background: var(--ink) !important;
        color: var(--paper) !important;
        padding: 1.05rem 1.25rem !important;
        font-size: .78rem !important;
    }
    div[data-testid="stButton"] > button[kind="primary"]:hover {
        background: var(--paper) !important;
        color: var(--ink) !important;
    }

    div[data-testid="stExpander"] details {
        border: 1px solid var(--line) !important;
        border-radius: 0 !important;
        background: var(--paper) !important;
    }
    div[data-testid="stExpander"] summary {
        font-family: 'Space Mono', monospace;
        font-size: .68rem;
        letter-spacing: .1em;
        text-transform: uppercase;
        padding: 14px 16px;
    }
    div[data-testid="stExpander"] summary:hover { background: var(--soft); }

    section[data-testid="stFileUploaderDropzone"] {
        border: 1px dashed var(--line);
        border-radius: 0;
        background: var(--soft);
    }

    div[data-testid="stAlert"] {
        border-radius: 0;
        border: 1px solid var(--line);
        background: var(--soft);
        color: var(--ink);
    }
    div[data-testid="stProgress"] div[role="progressbar"] > div {
        background: var(--ink);
    }
    div[data-testid="stStatus"] {
        border: 1px solid var(--line);
        border-radius: 0;
        background: var(--paper);
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border: 1px solid var(--line) !important;
        border-radius: 0 !important;
        background: var(--paper);
        padding: 10px;
    }
    div[data-testid="stImage"] img { border-radius: 0; }
    div[data-testid="stRadio"] label {
        font-family: 'Space Mono', monospace;
        font-size: .75rem;
    }

    @media (max-width: 700px) {
        .block-container { padding: 1.75rem 16px 56px; }
        .hero { padding: 36px 22px; margin-bottom: 36px; }
        .hero-top { flex-direction: column; gap: 10px; margin-bottom: 28px; }
        .step-head { gap: 12px; }
        .footer { flex-direction: column; gap: 6px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def flash(kind: str, message: str) -> None:
    st.session_state["_flash"] = (kind, message)


def show_flash() -> None:
    item = st.session_state.pop("_flash", None)
    if not item:
        return
    kind, message = item
    renderer = {
        "success": st.success,
        "warning": st.warning,
        "error": st.error,
        "info": st.info,
    }.get(kind, st.info)
    renderer(message)


def step_header(
    number: str, title: str, description: str, note: str = ""
) -> None:
    note_html = f'<div class="step-note">{note}</div>' if note else ""
    st.markdown(
        f"""
        <div class="step-head">
          <div class="step-no">{number}</div>
          <div>
            <div class="step-title">{title}</div>
            <div class="step-desc">{description}</div>
            {note_html}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def rule() -> None:
    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_engine_cached() -> FaceEngine | None:
    """None döner; hata mesajı çağıran tarafta gösterilmeli."""
    try:
        yunet_path, sface_path = ensure_models()
        return FaceEngine(yunet_path, sface_path)
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def get_drive_client_cached(api_key: str, credentials_identity: str) -> DriveClient:
    info = get_service_account_info()
    return DriveClient(api_key=api_key or None, service_account_info=info)


@st.cache_resource(show_spinner=False)
def get_store_cached(key: str) -> PgIndexStore | None:
    """None döner; pg bağlantısı yoksa veya hata oluşursa crash olmaz."""
    if pg is None:
        return None
    try:
        return PgIndexStore(pg, key)
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def get_postgres_cached(config_identity: str) -> PostgresStore:
    config = get_supabase_config()
    if not config:
        raise PostgresError("Kalıcı kayıt ayarı bulunamadı.")
    return PostgresStore(config)


@st.cache_data(show_spinner=False, max_entries=24)
def analyze_upload_cached(data: bytes, det_threshold: float, min_face: int) -> dict:
    image_bgr = decode_image_bytes(data)
    if image_bgr is None:
        return {
            "error": "Fotoğraf açılamadı. JPG, PNG veya WebP dene.",
            "faces": [],
        }

    engine = get_engine_cached()
    if engine is None:
        return {
            "error": "Yüz modeli yüklenemedi. İnternet bağlantısını kontrol edip sayfayı yenile.",
            "faces": [],
        }
    faces, _ = engine.analyze(
        image_bgr,
        det_threshold=det_threshold,
        min_face=max(20, int(min_face)),
        max_faces=10,
        input_max_side=1280,
    )
    return {"error": None, "faces": faces}


@st.cache_data(ttl=1800, max_entries=48, show_spinner=False)
def get_display_image(
    file_id: str,
    size: int,
    api_key: str,
    credentials_identity: str,
    resource_key: str | None,
) -> np.ndarray | None:
    client = get_drive_client_cached(api_key, credentials_identity)
    return client.fetch_image(file_id, size, resource_key)


@st.cache_data(ttl=1800, max_entries=48, show_spinner=False)
def get_download_file(
    file_id: str,
    api_key: str,
    credentials_identity: str,
    resource_key: str | None,
) -> tuple[bytes, str] | None:
    client = get_drive_client_cached(api_key, credentials_identity)
    return client.fetch_file_bytes(file_id, resource_key)


@st.cache_data(ttl=1800, max_entries=8, show_spinner=False)
def build_preview_zip(
    entries: tuple[tuple[str, str], ...],
    api_key: str,
    credentials_identity: str,
    resource_key: str | None,
) -> bytes:
    """Eşleşen fotoğrafların önizleme (thumbnail) hallerini ZIP'ler."""
    buffer = io.BytesIO()
    used: dict[str, int] = {}
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
        for file_id, name in entries:
            try:
                image = get_display_image(
                    file_id,
                    PREVIEW_THUMB_SIZE,
                    api_key,
                    credentials_identity,
                    resource_key,
                )
            except Exception:
                image = None
            if image is None:
                continue
            data = encode_jpeg_bytes(image, quality=80)
            if not data:
                continue
            base = (name or file_id).strip() or file_id
            if "." in base:
                base = base.rsplit(".", 1)[0] + ".jpg"
            else:
                base = base + ".jpg"
            count = used.get(base, 0)
            used[base] = count + 1
            if count:
                stem, dot, ext = base.rpartition(".")
                base = f"{stem}_{count}.{ext}" if dot else f"{base}_{count}"
            archive.writestr(base, data)
    return buffer.getvalue()


@st.cache_data(ttl=3600, max_entries=128, show_spinner=False)
def get_folder_meta_cached(
    folder_id: str,
    api_key: str,
    credentials_identity: str,
) -> dict:
    client = get_drive_client_cached(api_key, credentials_identity)
    return client.get_folder_meta(folder_id)


@st.cache_data(ttl=3600, max_entries=128, show_spinner=False)
def get_folder_count_cached(
    folder_id: str,
    resource_key: str | None,
    api_key: str,
    credentials_identity: str,
) -> int:
    client = get_drive_client_cached(api_key, credentials_identity)
    try:
        return client.count_images(folder_id, resource_key, True)
    except Exception:
        return -1


@st.cache_data(ttl=3600, max_entries=256, show_spinner=False)
def resolve_folder_meta_cached(
    file_id: str,
    depth: int,
    api_key: str,
    credentials_identity: str,
) -> dict:
    client = get_drive_client_cached(api_key, credentials_identity)
    try:
        return client.resolve_folder_from_file(file_id, depth)
    except Exception:
        return {}


def _format_drive_date(value: str) -> str:
    if not value:
        return ""
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return value
    return parsed.astimezone(timezone.utc).strftime("%d.%m.%Y")


def list_saved_folders() -> list[dict]:
    """Daha önce eklenmiş klasörleri Supabase'ten Drive tarihine göre sıralı döndürür."""
    if pg is None:
        return []

    try:
        rows = pg.list_folders()
    except Exception:
        return []

    folders: list[dict] = []
    for row in rows:
        key = str(row.get("folder_key") or "")
        meta = row.get("meta") or {}
        if not key:
            continue
        try:
            hidden_flag = meta.get("folder_hidden")
            if hidden_flag is None:
                pg.set_folder_meta(key, "folder_hidden", "0")
                hidden_flag = "0"
            if str(hidden_flag) == "1":
                continue

            store = PgIndexStore(pg, key)
            image_count = int(store.stats().get("images") or 0)

            folder_url = str(meta.get("folder_url") or "")
            folder_id = str(meta.get("folder_id") or "") or None
            title = str(meta.get("folder_title") or "")
            drive_date = str(meta.get("folder_date") or "")
            added_at = str(meta.get("added_at") or "")
            folder_count_meta = str(meta.get("folder_count") or "")

            resource_key = None
            if not folder_id and folder_url:
                try:
                    folder_id, resource_key = parse_drive_folder_url(folder_url)
                except DriveError:
                    folder_id = None

            if not folder_id:
                sample = store.sample_image()
                if sample is not None:
                    sample_id, sample_path = sample
                    resolved = resolve_folder_meta_cached(
                        sample_id,
                        sample_path.count(" / "),
                        api_key,
                        credentials_identity,
                    )
                    resolved_id = resolved.get("id")
                    if resolved_id:
                        folder_id = resolved_id
                        folder_url = folder_url or (
                            "https://drive.google.com/drive/folders/"
                            f"{resolved_id}"
                        )
                        store.set_meta("folder_id", resolved_id)
                        store.set_meta("folder_url", folder_url)
                        if resolved.get("name"):
                            title = title or resolved["name"]
                            store.set_meta("folder_title", resolved["name"])
                        resolved_date = resolved.get(
                            "created_time"
                        ) or resolved.get("modified_time")
                        if resolved_date:
                            drive_date = drive_date or resolved_date
                            store.set_meta("folder_date", resolved_date)

            if folder_id and (not title or not drive_date):
                try:
                    drive_meta = get_folder_meta_cached(
                        folder_id, api_key, credentials_identity
                    )
                except Exception:
                    drive_meta = {}
                title = title or drive_meta.get("name") or ""
                drive_date = (
                    drive_date
                    or drive_meta.get("created_time")
                    or drive_meta.get("modified_time")
                    or ""
                )

            count = int(folder_count_meta) if folder_count_meta.isdigit() else 0
            if not count and folder_id:
                drive_count = get_folder_count_cached(
                    folder_id, resource_key, api_key, credentials_identity
                )
                if drive_count >= 0:
                    count = drive_count
                    store.set_meta("folder_count", str(drive_count))
            if not count:
                count = image_count

            if not folder_id and image_count <= 0:
                continue

            folders.append(
                {
                    "key": key,
                    "url": folder_url,
                    "folder_id": folder_id,
                    "linked": bool(folder_id),
                    "title": title or f"Klasör {key[:8]}",
                    "count": count,
                    "indexed": image_count,
                    "date": drive_date or added_at,
                    "date_label": _format_drive_date(drive_date or added_at),
                }
            )
        except Exception:
            continue

    folders.sort(key=lambda item: item["date"], reverse=True)
    return folders


service_account_info = get_service_account_info()
supabase_config = get_supabase_config()
api_key = str(get_secret("GOOGLE_DRIVE_API_KEY") or "")
credentials_identity = str((service_account_info or {}).get("client_email") or "none")

try:
    client = get_drive_client_cached(api_key, credentials_identity)
except Exception:
    client = DriveClient(api_key=api_key or None)

pg: PostgresStore | None = None
if supabase_config:
    try:
        config_identity = hashlib.sha1(
            json.dumps(supabase_config, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        pg = get_postgres_cached(config_identity)
    except Exception:
        pg = None

if pg is None:
    st.error(
        "Supabase bağlantısı kurulamadı. Uygulama tüm verileri Supabase'te "
        "tutuyor; SUPABASE_DB_URL ayarını kontrol edip tekrar dene."
    )
    st.stop()


def run_indexing() -> None:
    if store is None or not folder_id or not key:
        st.warning("Önce bir Drive klasör linki gir.")
        return

    status_box = st.empty()
    status_box.caption("Başlıyor...")
    progress = st.progress(0.0, text="Klasör okunuyor...")

    try:
        with st.spinner("Hazırlanıyor..."):
            engine = get_engine_cached()
        if engine is None:
            status_box.caption("Model yüklenemedi.")
            st.error(
                "Yüz tanıma modeli indirilemedi veya başlatılamadı. "
                "İnternet bağlantısını kontrol edip sayfayı yenile."
            )
            return

        store.set_meta("folder_url", folder_url.strip())
        store.set_meta("folder_id", folder_id)
        store.set_meta("folder_hidden", "1" if hide_folder else "0")
        store.set_meta(
            "added_at",
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        try:
            folder_meta = get_folder_meta_cached(
                folder_id, api_key, credentials_identity
            )
        except Exception:
            folder_meta = {}
        if folder_meta.get("name"):
            store.set_meta("folder_title", folder_meta["name"])
        drive_date = folder_meta.get("created_time") or folder_meta.get(
            "modified_time"
        )
        if drive_date:
            store.set_meta("folder_date", drive_date)

        def on_status(message: str) -> None:
            status_box.caption(message)

        def on_progress(done: int, total: int, skipped: int, counters: dict) -> None:
            fraction = (done / total) if total else 1.0
            progress.progress(
                min(1.0, fraction),
                text=f"{done}/{total} fotoğraf · {counters['faces']} yüz",
            )

        scanned_folders: list[str] = []

        summary = build_index(
            client=client,
            engine=engine,
            store=store,
            folder_id=folder_id,
            resource_key=resource_key,
            options=IndexOptions(
                include_subfolders=include_subfolders,
                thumb_size=thumb_size,
                det_threshold=det_threshold,
                min_face=min_face,
                max_faces=max_faces,
                max_images=int(max_images),
                force=force_reindex,
                workers=8,
                sync_batch=100,
            ),
            on_status=on_status,
            on_progress=on_progress,
            on_batch=lambda _rows: store.flush(),
            load_cached=pg.fetch_existing if pg is not None else None,
            on_folder=scanned_folders.append,
        )

        store.flush()
        store.set_meta("folder_count", str(summary["total_found"]))
        if pg is not None and scanned_folders:
            try:
                pg.register_indexed_folders(scanned_folders, key)
            except Exception:
                pass
        progress.progress(1.0, text="Bitti")

        counters = summary["counters"]
        realtime_skipped = int(summary.get("skipped_realtime") or 0)
        cached_skipped = int(summary.get("skipped_cached") or 0)
        already_indexed = max(
            0,
            summary["skipped"] - realtime_skipped - cached_skipped,
        )
        message = (
            f"{summary['processed']} yeni fotoğraf tarandı, "
            f"{already_indexed} zaten kayıtlıydı, "
            f"{counters['faces']} yüz bulundu."
        )
        if cached_skipped:
            message += (
                f" {cached_skipped} fotoğraf başka klasördeki indeksten "
                "yeniden kullanıldı."
            )
        if realtime_skipped:
            message += (
                f" {realtime_skipped} fotoğraf eşzamanlı tarama "
                "nedeniyle atlandı."
            )
        warn_list = list(summary.get("warnings") or [])
        if warn_list:
            message += " Uyarı: " + " ".join(warn_list[:3])
        status_box.caption("Tarama bitti.")
        flash("success", message)
        st.rerun()
    except DriveError as exc:
        status_box.caption("Tarama tamamlanamadı.")
        st.error(str(exc))
    except Exception:
        status_box.caption("Tarama tamamlanamadı.")
        st.error(
            "Fotoğraflar taranamadı. Klasör linkini ve paylaşım ayarını "
            "kontrol edip tekrar dene."
        )


st.markdown(
    f"""
    <div class="hero">
      <div class="hero-top">
        <div class="brand">{APP_BRAND}</div>
        <div class="kicker">{APP_TITLE}</div>
      </div>
      <h1>Fotoğrafta<br>kendini bul.</h1>
      <p>Klasörünü ekle ya da kayıtlı klasörlerden seç, yüzünü yükle,
      eşleşen fotoğrafları gör.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

show_flash()

step_header(
    "01",
    "Fotoğrafların olduğu klasör",
    "Daha önce eklediğin bir klasörü seç ya da yeni bir Drive klasör "
    "linki yapıştır. Yeni bir yükleme ise gizli tutmayı işaretleyebilirsin.",
    "Klasör Drive'da 'Bağlantıya sahip herkes' olarak paylaşılmış olmalı.",
)


def select_saved_folder(item_key: str, url: str) -> None:
    st.session_state["active_folder_key"] = item_key
    st.session_state["folder_url"] = url or ""


def update_saved_folder(item_key: str, url: str) -> None:
    st.session_state["active_folder_key"] = item_key
    st.session_state["folder_url"] = url or ""
    st.session_state["_pending_force"] = False


try:
    saved_folders = list_saved_folders()
except Exception:
    saved_folders = []

indexing_locked = st.session_state.get("_pending_force") is not None

folder_mode = "Yeni klasör linki"
if saved_folders:
    folder_mode = st.radio(
        "Klasör seçimi",
        ["Yeni klasör linki", "Mevcut ekli klasörler"],
        horizontal=True,
        label_visibility="collapsed",
        key="folder_mode",
        disabled=indexing_locked,
    )

selected_saved_key: str | None = None
folder_url = ""

if folder_mode == "Mevcut ekli klasörler":
    st.caption(f"{len(saved_folders)} klasör kayıtlı · en yeni yükleme üstte")
    selected_saved_key = st.session_state.get("active_folder_key")

    for item in saved_folders:
        is_active = item["key"] == selected_saved_key
        with st.container(border=True):
            info_col, action_col = st.columns([4, 1.2])
            with info_col:
                st.markdown(f"**{item['title']}**")
                meta_bits = [f"{item['count']} fotoğraf"]
                if item["linked"] and item["indexed"] != item["count"]:
                    meta_bits.append(f"{item['indexed']} taranmış")
                if item["date_label"]:
                    meta_bits.append(f"Drive yükleme: {item['date_label']}")
                st.caption(" · ".join(meta_bits))
            with action_col:
                st.button(
                    "Seçili" if is_active else "Bu klasörü seç",
                    key=f"pick_{item['key']}",
                    width="stretch",
                    disabled=is_active or indexing_locked,
                    on_click=select_saved_folder,
                    args=(item["key"], item["url"]),
                )
                st.button(
                    "Güncelle",
                    key=f"update_{item['key']}",
                    width="stretch",
                    disabled=indexing_locked or not item["linked"],
                    help="Bu klasördeki yeni/eksik fotoğrafları tarar.",
                    on_click=update_saved_folder,
                    args=(item["key"], item["url"]),
                )
    folder_url = st.session_state.get("folder_url", "")
else:
    folder_url = st.text_input(
        "Drive klasör linki",
        key="folder_url",
        placeholder="https://drive.google.com/drive/folders/...",
        label_visibility="collapsed",
    )

folder_id: str | None = None
resource_key: str | None = None
folder_error: str | None = None
if folder_url.strip():
    try:
        folder_id, resource_key = parse_drive_folder_url(folder_url)
    except DriveError as exc:
        folder_error = str(exc)

include_subfolders = True
thumb_size = DEFAULT_THUMB_SIZE
det_threshold = DEFAULT_DET_THRESHOLD
min_face = DEFAULT_MIN_FACE
max_faces = DEFAULT_MAX_FACES
max_images = 0
force_reindex = False
hide_folder = False

key: str | None = None
store: PgIndexStore | None = None
if selected_saved_key:
    key = selected_saved_key
    store = get_store_cached(key)
    if store is None:
        st.warning("Seçili klasör yüklenemedi. Supabase bağlantısını kontrol et.")
elif folder_id:
    key = folder_key(folder_id, resource_key, include_subfolders, thumb_size)
    store = get_store_cached(key)

if folder_id and store is not None:
    try:
        needs_meta = (
            not store.meta_get("folder_title") or not store.meta_get("folder_url")
        ) and store.stats().get("images", 0) > 0
    except Exception:
        needs_meta = False

    if needs_meta:
        try:
            entered_meta = get_folder_meta_cached(
                folder_id, api_key, credentials_identity
            )
        except Exception:
            entered_meta = {}
        try:
            if entered_meta.get("name"):
                store.set_meta("folder_title", entered_meta["name"])
            entered_date = entered_meta.get("created_time") or entered_meta.get(
                "modified_time"
            )
            if entered_date:
                store.set_meta("folder_date", entered_date)
            store.set_meta("folder_url", folder_url.strip())
            store.set_meta("folder_id", folder_id)
        except Exception:
            pass

if folder_error:
    st.caption(
        "Klasör linki okunamadı. Drive'da klasörü açıp adres çubuğundaki "
        "linki kopyala."
    )
elif selected_saved_key:
    st.caption("Seçili klasör hazır. Sonuçlar 02'de yüz yükleyince görünür.")
elif folder_id:
    st.caption(
        "İlk tarama fotoğraf sayısına göre birkaç dakika sürebilir. "
        "Eklediğin klasörler 'Mevcut ekli klasörler' listesinde birikir."
    )
else:
    st.caption("Bir klasör seç ya da yeni bir Drive klasör linki yapıştır.")

if folder_id and not folder_error:
    hide_folder = st.checkbox(
        "Bu klasörü gizli tut",
        key=f"hide_folder_{folder_id}",
        help=(
            "Yeni bir yükleme ise işaretle: bu klasör 'Mevcut ekli "
            "klasörler' listesinde görünmez. Gizli klasörleri yalnızca "
            "linki tekrar girerek açabilirsin."
        ),
        disabled=indexing_locked,
    )

scan_clicked = st.button(
    "Taramayı başlat",
    type="primary",
    width="stretch",
    disabled=not folder_id or indexing_locked,
)

has_index = False
if store is not None:
    try:
        has_index = store.stats()["images"] > 0
    except Exception:
        has_index = False

rescan_clicked = False
if has_index:
    rescan_clicked = st.button(
        "Yeniden tara",
        width="stretch",
        help="Kayıtlı fotoğrafları atlamadan tüm klasörü baştan tarar.",
        disabled=indexing_locked,
    )

already_indexed = False
if scan_clicked and selected_saved_key is None and folder_id and pg is not None:
    try:
        already_indexed = pg.is_indexed_folder(folder_id)
    except Exception:
        already_indexed = False

if already_indexed:
    st.warning(
        "Bu klasör, daha önce indekslenmiş bir klasörün alt klasörü. "
        "Fotoğrafları zaten kayıtlı olduğu için tekrar indekslenmez. "
        "Üst klasörü 'Mevcut ekli klasörler' listesinden açıp 'Güncelle' "
        "ile yenileyebilirsin."
    )
elif scan_clicked or rescan_clicked:
    st.session_state["_pending_force"] = rescan_clicked
    st.rerun()

if indexing_locked:
    force_reindex = bool(st.session_state.pop("_pending_force", False))
    try:
        run_indexing()
    except _DISCONNECT_EXC:  # type: ignore[misc]  # noqa: E722
        st.warning("Bağlantı kesildi; tarama durdu. Tekrar başlatabilirsin.")
    except Exception as _run_exc:
        st.error(f"Beklenmeyen hata: {_run_exc}")

rule()

step_header(
    "02",
    "Yüzünü yükle",
    "Net, önden çekilmiş bir fotoğraf seç veya kamerayla anlık çek.",
    "Seçilen resimler hiçbir yere gönderilmez; yalnızca taranan fotoğraflarla karşılaştırılır.",
)

face_input = st.radio(
    "Yüz fotoğrafı ekle",
    ["Fotoğraf yükle", "Kamera ile çek"],
    horizontal=True,
    label_visibility="collapsed",
    key="face_input_mode",
)

if face_input == "Kamera ile çek":
    face_file = st.camera_input(
        "Yüz fotoğrafı çek",
        key="face_camera",
        help="Kameranı kullanarak anlık bir fotoğraf çek.",
    )
else:
    face_file = st.file_uploader(
        "Yüz fotoğrafı",
        type=["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"],
        key="face_upload",
        help="Gözlük, maske veya gölge yoksa sonuç daha iyi olur.",
    )

probe_embedding: np.ndarray | None = None

if face_file is not None:
    upload_result = None
    try:
        file_bytes = face_file.getvalue()
    except _DISCONNECT_EXC:  # type: ignore[misc]  # noqa: E722
        # Kullanıcı yükleme sırasında bağlantıyı kesti — sessizce çık
        file_bytes = None
    except Exception:
        file_bytes = None

    if file_bytes:
        try:
            with st.spinner("Yüz aranıyor..."):
                upload_result = analyze_upload_cached(
                    file_bytes,
                    det_threshold,
                    min_face,
                )
        except _DISCONNECT_EXC:  # type: ignore[misc]  # noqa: E722
            # Yükleme iptal edildi, uygulama devam eder
            upload_result = None
        except Exception:
            st.error(
                "Yüz modeli yüklenemedi. İnternet bağlantısını kontrol edip "
                "tekrar dene."
            )

    if upload_result is not None:
        if upload_result.get("error"):
            st.caption(upload_result["error"])
        else:
            faces = upload_result["faces"]
            if not faces:
                st.caption("Yüz bulunamadı. Daha net bir fotoğraf dene.")
            else:
                probe_embedding = faces[0]["embedding"]
                if len(faces) == 1:
                    st.caption("Yüz bulundu.")
                else:
                    st.caption(
                        "Birden fazla yüz bulundu. En belirgin yüz kullanılıyor."
                    )

rule()

step_header(
    "03",
    "Eşleşen fotoğraflar",
    "Sonuçlar benzerlik sırasına göre gelir.",
)

if store is None:
    st.caption("Önce 01'de bir klasör seç veya Drive klasör linki gir.")
elif face_file is None:
    st.caption("Yüz fotoğrafını yükleyince veya çekince sonuçlar burada görünür.")
elif probe_embedding is None:
    st.caption("Yüz bulunamadı. Daha net bir fotoğraf yükle veya çek.")
else:
    with st.spinner("Sonuçlar hazırlanıyor..."):
        try:
            matrix, metas = store.load_matrix()
        except _DISCONNECT_EXC:  # type: ignore[misc]  # noqa: E722
            st.warning("Bağlantı kesildi. Sayfayı yenilemeyi dene.")
            st.stop()
        except Exception:
            st.error(
                "Kayıtlı indeks okunamadı. Supabase bağlantısını kontrol edip "
                "sayfayı yenile."
            )
            st.stop()

    if matrix.shape[0] == 0:
        st.caption("Bu klasörde henüz yüz yok. Önce 01'deki taramayı başlat.")
    else:
        match_threshold = float(
            st.session_state.get(
                "advanced_match_threshold",
                DEFAULT_MATCH_THRESHOLD,
            )
        )

        results = search_embeddings(
            matrix,
            metas,
            [probe_embedding],
            threshold=match_threshold,
        )

        if results:
            st.markdown(
                f"**{len(results)} fotoğraf bulundu.** "
                f"En yüksek benzerlik: %{results[0]['score'] * 100:.0f}."
            )
        else:
            st.markdown("**Eşleşme bulunamadı.**")

        if not results:
            st.caption("Gelişmiş ayarlardan hassasiyeti düşürmeyi dene.")
        else:
            if st.button(
                f"Tümünü indir · önizleme ({len(results)})",
                key=f"zip_request_{key}",
                width="stretch",
                help=(
                    "Eşleşen fotoğrafların küçük (önizleme) hallerini tek ZIP "
                    "dosyası olarak indirir. Orijinal kalite için kartlardaki "
                    "'Fotoğrafı indir' butonunu kullan."
                ),
            ):
                st.session_state["_zip_request_key"] = key

            if st.session_state.get("_zip_request_key") == key:
                zip_bytes = b""
                try:
                    with st.spinner("Önizlemeler paketleniyor..."):
                        zip_bytes = build_preview_zip(
                            tuple((r["file_id"], r["name"]) for r in results),
                            api_key,
                            credentials_identity,
                            resource_key,
                        )
                except _DISCONNECT_EXC:  # type: ignore[misc]  # noqa: E722
                    zip_bytes = b""
                    st.warning("Bağlantı kesildi; ZIP oluşturulamadı. Tekrar dene.")
                except Exception:
                    zip_bytes = b""
                if zip_bytes:
                    st.download_button(
                        "ZIP dosyasını indir",
                        zip_bytes,
                        file_name="eslesen_onizlemeler.zip",
                        mime="application/zip",
                        width="stretch",
                        key=f"zip_download_{key}",
                    )
                else:
                    st.caption("Önizleme paketi hazırlanamadı, tekrar dene.")

            columns = st.columns(3, gap="medium")

            for result_index, result in enumerate(results):
                with columns[result_index % 3]:
                    with st.container(border=True):
                        try:
                            image_bgr = get_display_image(
                                result["file_id"],
                                PREVIEW_THUMB_SIZE,
                                api_key,
                                credentials_identity,
                                resource_key,
                            )
                        except _DISCONNECT_EXC:  # type: ignore[misc]  # noqa: E722
                            image_bgr = None
                        except Exception:
                            image_bgr = None
                        if image_bgr is not None:
                            st.image(
                                bgr_to_rgb(image_bgr),
                                width="stretch",
                            )
                        else:
                            st.markdown(
                                '<div class="no-image">Önizleme yok</div>',
                                unsafe_allow_html=True,
                            )

                        st.markdown(f"**%{result['score'] * 100:.0f} eşleşme**")
                        st.caption(result["name"])

                        try:
                            download_file = get_download_file(
                                result["file_id"],
                                api_key,
                                credentials_identity,
                                resource_key,
                            )
                        except _DISCONNECT_EXC:  # type: ignore[misc]  # noqa: E722
                            download_file = None
                        except Exception:
                            download_file = None
                        if download_file is not None:
                            download_bytes, download_mime = download_file
                            st.download_button(
                                "Fotoğrafı indir",
                                download_bytes,
                                file_name=result["name"],
                                mime=download_mime,
                                width="stretch",
                                key=f"download_{result['file_id']}",
                            )

                        st.link_button(
                            "Drive'da aç",
                            drive_view_url(result["file_id"]),
                            width="stretch",
                        )

rule()

with st.expander("Nasıl çalışır?"):
    st.markdown(
        """
        - Eklediğin klasörler kaydedilir; sonraki girişlerde listeden seçebilirsin.
        - Klasörler Drive'daki yükleme tarihine göre, en yeni üstte sıralanır.
        - 'Gizli tut' işaretlenen yeni yüklemeler listede görünmez.
        - Daha önce eklenmiş tüm klasörler varsayılan olarak herkese açıktır.
        - Klasördeki fotoğrafların küçük önizlemeleri alınır.
        - Yüzler burada işlenir ve birbirleriyle karşılaştırılır.
        - Üçüncü taraf yüz tanıma servisi kullanılmaz.
        - Sonuçları her zaman gözle doğrula; sistem kesin kimlik iddiası taşımaz.
        - Yalnızca yetkili olduğun fotoğraflarda kullan.
        """
    )

with st.expander("Gelişmiş ayarlar"):
    st.slider(
        "Eşleşme hassasiyeti",
        min_value=0.30,
        max_value=0.70,
        value=DEFAULT_MATCH_THRESHOLD,
        step=0.01,
        key="advanced_match_threshold",
        help="Yükseldikçe daha az ama daha güvenli sonuç verir.",
    )
    st.caption("Değişiklikler sonuçlara hemen uygulanır.")

st.markdown(
    f"""
    <div class="footer">
      <span>{APP_BRAND}</span>
      <span>Yalnızca yetkili olduğun fotoğraflarda kullan.</span>
    </div>
    """,
    unsafe_allow_html=True,
)
