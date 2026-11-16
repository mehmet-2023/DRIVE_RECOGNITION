from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import streamlit as st

APP_TITLE = "Drive Yüz Tanıma Sistemi"
APP_BRAND = "BY OCAL"

EMBEDDING_SIZE = 128

MODEL_DIR = Path(os.environ.get("DRIVE_FACE_MODEL_DIR", "/tmp/drive_face_models"))

YUNET_FILE = "face_detection_yunet_2023mar.onnx"
SFACE_FILE = "face_recognition_sface_2021dec.onnx"
YUNET_URLS = (
    "https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/"
    f"face_detection_yunet/{YUNET_FILE}",
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    f"face_detection_yunet/{YUNET_FILE}",
)
SFACE_URLS = (
    "https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/"
    f"face_recognition_sface/{SFACE_FILE}",
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    f"face_recognition_sface/{SFACE_FILE}",
)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".jpe",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
    ".heic",
    ".heif",
    ".avif",
    ".cr2",
    ".cr3",
    ".nef",
    ".arw",
    ".dng",
    ".orf",
    ".raf",
    ".rw2",
}

DRIVE_FOLDER_MIME = "application/vnd.google-apps.folder"
DRIVE_SCOPE = "https://www.googleapis.com/auth/drive.readonly"

DEFAULT_THUMB_SIZE = 768
DEFAULT_DET_THRESHOLD = 0.60
DEFAULT_MIN_FACE = 40
DEFAULT_MAX_FACES = 8
DEFAULT_MATCH_THRESHOLD = 0.50

THUMB_OPTIONS = {
    "Hızlı (512 px)": 512,
    "Dengeli (768 px)": 768,
    "Yüksek kalite (1024 px)": 1024,
}


def _load_dotenv() -> None:
    """Küçük, bağımlılıksız .env yükleyici. Var olan env değerlerini ezmez."""
    for candidate in (Path(".env"), Path(__file__).resolve().parent.parent / ".env"):
        if not candidate.exists():
            continue
        try:
            for raw_line in candidate.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                if line.startswith("export "):
                    line = line[len("export ") :]
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key:
                    os.environ.setdefault(key, value)
        except OSError:
            continue
        break


_load_dotenv()


def get_secret(name: str, default: Any = None) -> Any:
    """Streamlit secrets -> environment -> default sırasıyla okur."""
    try:
        secret = st.secrets.get(name)  # type: ignore[attr-defined]
        if secret not in (None, ""):
            return secret
    except Exception:
        pass
    value = os.environ.get(name)
    return default if value in (None, "") else value


def get_service_account_info() -> dict[str, Any] | None:
    """Service account bilgisini secrets, JSON env, dosya env veya yerel dosyadan bulur."""
    try:
        if "gcp_service_account" in st.secrets:
            info = dict(st.secrets["gcp_service_account"])
            if info.get("type") == "service_account" and info.get("private_key"):
                info["private_key"] = str(info["private_key"]).replace("\\n", "\n")
                return info
    except Exception:
        pass

    raw = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    if raw:
        try:
            info = json.loads(raw)
            if info.get("type") == "service_account":
                return info
        except (TypeError, ValueError):
            pass

    for env_name in ("GOOGLE_APPLICATION_CREDENTIALS", "GCP_CREDENTIALS_FILE"):
        path_value = os.environ.get(env_name)
        if not path_value:
            continue
        path = Path(path_value)
        if path.exists():
            try:
                info = json.loads(path.read_text(encoding="utf-8"))
                if info.get("type") == "service_account":
                    return info
            except (OSError, ValueError):
                pass

    local = Path(__file__).resolve().parent.parent / "credentials.json"
    if local.exists():
        try:
            info = json.loads(local.read_text(encoding="utf-8"))
            if info.get("type") == "service_account":
                return info
        except (OSError, ValueError):
            pass

    return None


def get_supabase_config() -> dict[str, Any] | None:
    """Supabase Postgres bağlantı bilgisini secrets/env üzerinden okur."""
    dsn = get_secret("SUPABASE_DB_URL") or get_secret("DATABASE_URL")
    if dsn:
        return {"dsn": str(dsn)}

    try:
        if "supabase" in st.secrets:
            section = dict(st.secrets["supabase"])
            if section.get("host") or section.get("dsn"):
                return section
    except Exception:
        pass

    host = get_secret("SUPABASE_HOST") or get_secret("PGHOST")
    if host:
        password = get_secret("SUPABASE_PASSWORD") or get_secret("PGPASSWORD")
        return {
            "host": str(host),
            "port": int(get_secret("SUPABASE_PORT", get_secret("PGPORT", 5432))),
            "dbname": str(get_secret("SUPABASE_DB", get_secret("PGDATABASE", "postgres"))),
            "user": str(get_secret("SUPABASE_USER", get_secret("PGUSER", "postgres"))),
            "password": str(password or ""),
            "sslmode": str(get_secret("SUPABASE_SSLMODE", "require")),
        }

    return None


def folder_key(
    folder_id: str,
    resource_key: str | None,
    include_subfolders: bool,
    thumb_size: int,
) -> str:
    raw = f"{folder_id}|{resource_key or ''}|{int(include_subfolders)}|{int(thumb_size)}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def drive_view_url(file_id: str) -> str:
    return f"https://drive.google.com/file/d/{file_id}/view"


def drive_download_url(file_id: str) -> str:
    return f"https://drive.usercontent.google.com/download?id={file_id}&export=download"
