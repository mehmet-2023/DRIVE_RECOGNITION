from __future__ import annotations

import csv
import hashlib
import io
import json

import cv2
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="Etkinlikler İçin Drive Yüz Tanıma Sistemi | BY OCAL",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

from core.config import (  # noqa: E402
    APP_BRAND,
    APP_TITLE,
    DEFAULT_DET_THRESHOLD,
    DEFAULT_MATCH_THRESHOLD,
    DEFAULT_MAX_FACES,
    DEFAULT_MIN_FACE,
    DEFAULT_THUMB_SIZE,
    INDEX_DIR,
    THUMB_OPTIONS,
    drive_download_url,
    drive_view_url,
    folder_key,
    get_secret,
    get_service_account_info,
    get_supabase_config,
)
from core.drive import DriveClient, DriveError, parse_drive_folder_url  # noqa: E402
from core.face_engine import FaceEngine, ensure_models  # noqa: E402
from core.image_utils import bgr_to_rgb, decode_image_bytes, draw_label_box, resize_max_side  # noqa: E402
from core.index_store import IndexStore  # noqa: E402
from core.pipeline import IndexOptions, build_index, search_embeddings  # noqa: E402
from core.postgres_store import PostgresError, PostgresStore  # noqa: E402

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.2rem; max-width: 1500px; }
    .ocal-hero {
        display: flex; justify-content: space-between; align-items: center;
        gap: 24px; padding: 26px 30px; border-radius: 22px;
        background:
            radial-gradient(circle at top left, rgba(245,158,11,.22), transparent 45%),
            linear-gradient(135deg, #111A2C 0%, #0B1220 100%);
        border: 1px solid rgba(245,158,11,.25); margin-bottom: 18px;
    }
    .ocal-kicker {
        color: #F59E0B; font-size: .82rem; letter-spacing: .14em;
        text-transform: uppercase; font-weight: 700;
    }
    .ocal-title { font-size: 2.1rem; font-weight: 800; color: #F8FAFC; margin-top: 4px; }
    .ocal-sub { color: #94A3B8; margin-top: 6px; max-width: 820px; }
    .ocal-badge {
        font-size: 1.05rem; font-weight: 800; letter-spacing: .12em;
        color: #0B1220; background: #F59E0B; padding: 12px 18px;
        border-radius: 14px; white-space: nowrap;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    @media (max-width: 700px) {
        .ocal-hero { flex-direction: column; align-items: flex-start; }
        .ocal-title { font-size: 1.5rem; }
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


@st.cache_resource(show_spinner=False)
def get_engine_cached() -> FaceEngine:
    yunet_path, sface_path = ensure_models()
    return FaceEngine(yunet_path, sface_path)


@st.cache_resource(show_spinner=False)
def get_drive_client_cached(api_key: str, credentials_identity: str) -> DriveClient:
    info = get_service_account_info()
    return DriveClient(api_key=api_key or None, service_account_info=info)


@st.cache_resource(show_spinner=False)
def get_store_cached(key: str) -> IndexStore:
    return IndexStore(INDEX_DIR / f"{key}.sqlite")


@st.cache_resource(show_spinner=False)
def get_postgres_cached(config_identity: str) -> PostgresStore:
    config = get_supabase_config()
    if not config:
        raise PostgresError("Supabase yapılandırması bulunamadı.")
    return PostgresStore(config)


@st.cache_data(show_spinner=False, max_entries=24)
def analyze_upload_cached(
    data: bytes, det_threshold: float, min_face: int
) -> dict:
    image_bgr = decode_image_bytes(data)
    if image_bgr is None:
        return {
            "error": "Görsel açılamadı. JPG, PNG veya WebP deneyin.",
            "image": None,
            "faces": [],
            "scale": 1.0,
        }

    engine = get_engine_cached()
    faces, _ = engine.analyze(
        image_bgr,
        det_threshold=det_threshold,
        min_face=max(20, int(min_face)),
        max_faces=10,
        input_max_side=1280,
    )
    preview_bgr, scale = resize_max_side(image_bgr, 900)
    return {
        "error": None,
        "image": bgr_to_rgb(preview_bgr),
        "faces": faces,
        "scale": scale,
    }


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


def annotate_result(image_bgr: np.ndarray, result: dict) -> np.ndarray:
    image = image_bgr.copy()
    stored_width = int(result.get("width") or image.shape[1])
    stored_height = int(result.get("height") or image.shape[0])
    scale_x = image.shape[1] / stored_width if stored_width else 1.0
    scale_y = image.shape[0] / stored_height if stored_height else 1.0
    x, y, w, h = result["bbox"]
    draw_label_box(
        image,
        int(x * scale_x),
        int(y * scale_y),
        int(w * scale_x),
        int(h * scale_y),
        f"%{result['score'] * 100:.1f}",
        color=(245, 158, 11),
    )
    return bgr_to_rgb(image)


def results_to_csv(results: list[dict]) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer, delimiter=";")
    writer.writerow(["Dosya", "Klasör", "Benzerlik (%)", "Drive Linki"])
    for result in results:
        writer.writerow(
            [
                result["name"],
                result["path"],
                f"{result['score'] * 100:.2f}",
                drive_view_url(result["file_id"]),
            ]
        )
    return buffer.getvalue().encode("utf-8-sig")


service_account_info = get_service_account_info()
supabase_config = get_supabase_config()

# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### 🎛️ Kontrol Paneli")
    st.caption("Drive klasörü → indeks → yüz arama")

    folder_url = st.text_input(
        "Google Drive klasör linki",
        placeholder="https://drive.google.com/drive/folders/...",
        help="Klasör 'Bağlantıya sahip herkes → Görüntüleyen' olarak paylaşılmalı.",
    )

    folder_id: str | None = None
    resource_key: str | None = None
    if folder_url.strip():
        try:
            folder_id, resource_key = parse_drive_folder_url(folder_url)
        except DriveError as exc:
            st.error(str(exc))

    include_subfolders = st.toggle("Alt klasörleri dahil et", value=True)
    thumb_label = st.selectbox(
        "Görsel kalitesi",
        options=list(THUMB_OPTIONS.keys()),
        index=1,
        help="Yüksek kalite daha iyi küçük yüz tespiti sağlar; daha yavaştır.",
    )
    thumb_size = THUMB_OPTIONS[thumb_label]

    max_images = st.number_input(
        "Maksimum fotoğraf (0 = tümü)",
        min_value=0,
        max_value=100_000,
        value=0,
        step=500,
        help="İlk denemede 500-1000 girerek hızı test edebilirsiniz.",
    )
    det_threshold = st.slider(
        "Yüz tespit eşiği",
        min_value=0.40,
        max_value=0.90,
        value=DEFAULT_DET_THRESHOLD,
        step=0.05,
        help="Yüksek değer daha az ama daha net yüzleri indeksler.",
    )
    min_face = st.slider(
        "Minimum yüz boyutu (px)",
        min_value=24,
        max_value=120,
        value=DEFAULT_MIN_FACE,
        step=4,
        help="Grup fotoğraflarındaki küçük yüzler için düşürebilirsiniz.",
    )
    max_faces = st.slider(
        "Fotoğraf başına maks. yüz",
        min_value=1,
        max_value=20,
        value=DEFAULT_MAX_FACES,
        step=1,
    )
    force_reindex = st.checkbox(
        "Mevcut kayıtları yeniden işle",
        value=False,
        help="Kapalıyken daha önce indekslenen fotoğraflar atlanır.",
    )

    with st.expander("Gelişmiş: farklı Drive API anahtarı"):
        api_key_override = st.text_input(
            "Drive API anahtarı",
            type="password",
            value="",
            help=(
                "Sunucu secrets ayarında anahtar varsa boş bırakın. "
                "Buraya girilen anahtar tarayıcıya gönderilir; paylaşmayın."
            ),
        )

    st.divider()
    st.markdown("#### Bağlantı durumu")

    if service_account_info:
        st.caption("🔐 Drive: Service account ✅")
        email = service_account_info.get("client_email")
        if email:
            st.caption(f"`{email}`")
    elif api_key_override.strip() or get_secret("GOOGLE_DRIVE_API_KEY"):
        st.caption("🔐 Drive: API anahtarı ✅")
    else:
        st.caption("🔐 Drive: Anahtarsız public HTML ⚠️")

    if supabase_config:
        st.caption("🗄️ Kalıcı DB: Supabase Postgres ✅")
    else:
        st.caption("🗄️ Kalıcı DB: kapalı ⚠️")
        st.caption("Uyku/reboot sonrası indeks kaybolur. Supabase bilgilerini secrets'a ekleyin.")

    st.divider()
    index_clicked = st.button(
        "🚀 İndeksi Oluştur / Güncelle",
        type="primary",
        use_container_width=True,
        disabled=not folder_id,
    )
    st.caption("İlk tarama fotoğraf sayısına göre birkaç dakika sürebilir. Sayfayı kapatmayın.")

# ---------------------------------------------------------------------- clients
api_key = (api_key_override or "").strip() or str(
    get_secret("GOOGLE_DRIVE_API_KEY") or ""
)
credentials_identity = (
    str((service_account_info or {}).get("client_email") or "none")
)

try:
    client = get_drive_client_cached(api_key, credentials_identity)
except Exception as exc:
    st.error(f"Drive istemcisi kurulamadı: {exc}")
    client = DriveClient(api_key=api_key or None)

key: str | None = None
store: IndexStore | None = None
if folder_id:
    key = folder_key(folder_id, resource_key, include_subfolders, thumb_size)
    store = get_store_cached(key)

pg: PostgresStore | None = None
if supabase_config:
    try:
        config_identity = hashlib.sha1(
            json.dumps(supabase_config, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        pg = get_postgres_cached(config_identity)
    except Exception as exc:
        st.sidebar.error(f"Postgres: {exc}")


def ensure_remote_loaded(
    target_store: IndexStore | None,
    postgres: PostgresStore | None,
    target_key: str | None,
    *,
    force: bool = False,
) -> int:
    if target_store is None or postgres is None or not target_key:
        return 0

    loaded_flag = f"_remote_loaded_{target_key}"
    if not force and st.session_state.get(loaded_flag):
        return 0
    if not force and target_store.stats()["images"] > 0:
        st.session_state[loaded_flag] = True
        return 0

    try:
        rows = postgres.fetch(target_key)
    except Exception as exc:
        st.session_state[f"_remote_error_{target_key}"] = str(exc)
        st.session_state[loaded_flag] = True
        return 0

    if rows:
        target_store.import_rows(rows)
    st.session_state[loaded_flag] = True
    return len(rows)


def run_indexing() -> None:
    if store is None or not folder_id or not key:
        st.error("Önce geçerli bir Drive klasör linki girin.")
        return

    status = st.status("İndeksleme hazırlanıyor...", expanded=True)
    progress = st.progress(0.0, text="Drive klasörü listeleniyor...")

    try:
        with st.spinner("Yüz modelleri hazırlanıyor..."):
            engine = get_engine_cached()

        postgres_ready = False
        if pg is not None:
            status.write("Postgres bağlantısı kontrol ediliyor...")
            try:
                pg.ping()
                postgres_ready = True
                remote_count = ensure_remote_loaded(store, pg, key)
                if remote_count:
                    status.write(f"Postgres'ten {remote_count} kayıt yüklendi.")
            except Exception as exc:
                status.write(f"⚠️ Postgres kullanılamıyor, yalnızca yerel indeks: {exc}")
        else:
            status.write(
                "ℹ️ Kalıcı DB kapalı. İndeks yalnızca geçici diskte tutulacak."
            )

        def on_status(message: str) -> None:
            status.write(message)

        def on_progress(done: int, total: int, skipped: int, counters: dict) -> None:
            fraction = (done / total) if total else 1.0
            progress.progress(
                min(1.0, fraction),
                text=(
                    f"{done}/{total} işlendi · {skipped} zaten indekste · "
                    f"{counters['faces']} yüz"
                ),
            )

        def on_batch(rows: list[dict]) -> None:
            if postgres_ready and pg is not None:
                pg.upsert(key, rows)

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
            on_batch=on_batch,
        )

        store.set_meta("folder_url", folder_url.strip())
        progress.progress(1.0, text="Tamamlandı")

        counters = summary["counters"]
        message = (
            f"{summary['processed']} yeni görsel işlendi · "
            f"{summary['skipped']} zaten indeksteydi · "
            f"{counters['faces']} yüz indekslendi · "
            f"{counters['no_face']} yüzsüz · {counters['error']} hata."
        )
        if summary["sync_errors"]:
            message += f" Postgres uyarısı: {summary['sync_errors'][0]}"
        if summary["warnings"]:
            message += f" Drive uyarısı: {summary['warnings'][0]}"

        status.update(label="İndeks hazır ✅", state="complete", expanded=False)
        flash("success", message)
        st.rerun()
    except Exception as exc:
        status.update(label="İndeksleme hatası", state="error", expanded=True)
        st.error(str(exc))


show_flash()

st.markdown(
    f"""
    <div class="ocal-hero">
      <div>
        <div class="ocal-kicker">{APP_TITLE}</div>
        <div class="ocal-title">Fotoğraflarda kendini bul.</div>
        <div class="ocal-sub">
          Drive klasörünü tara; yüzünü yükle, eşleşen fotoğrafları saniyeler içinde gör.
          Tüm işlemler kendi altyapında, üçüncü taraf yüz API'si olmadan çalışır.
        </div>
      </div>
      <div class="ocal-badge">{APP_BRAND}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("Gizlilik ve kullanım notu", expanded=False):
    st.markdown(
        """
        Bu uygulama yalnızca **yetkili olduğunuz** etkinlik fotoğraflarında kullanılmalıdır.
        Yüz verisi biyometrik veridir; KVKK/GDPR kapsamında gerekli bilgilendirme ve
        açık rızayı sağlamak sizin sorumluluğunuzdadır. Görseller üçüncü taraf yüz
        tanıma servislerine gönderilmez; yalnızca Google Drive'dan thumbnail olarak
        alınır ve sunucu belleğinde işlenir.
        """
    )

if index_clicked:
    run_indexing()

tab_search, tab_index, tab_help = st.tabs(
    ["🔎 Yüz Ara", "🗂️ İndeks & Kalıcılık", "ℹ️ Nasıl Çalışır?"]
)

# ---------------------------------------------------------------------- search tab
with tab_search:
    st.subheader("1. Yüz fotoğrafını yükle")
    uploaded_files = st.file_uploader(
        "Aranacak kişinin net fotoğraf(lar)ı",
        type=["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"],
        accept_multiple_files=True,
        help="Aynı kişinin 2-5 farklı fotoğrafını yüklerseniz doğruluk artar.",
    )

    analyzed: list[tuple[int, str, dict]] = []
    probe_embeddings: list[np.ndarray] = []

    if uploaded_files:
        try:
            with st.spinner("Yüz modelleri hazırlanıyor..."):
                get_engine_cached()
        except Exception as exc:
            st.error(f"Yüz motoru hazırlanamadı: {exc}")
            st.stop()

        with st.spinner("Yüklenen fotoğraflar analiz ediliyor..."):
            for upload_index, uploaded_file in enumerate(uploaded_files):
                result = analyze_upload_cached(
                    uploaded_file.getvalue(),
                    det_threshold,
                    min_face,
                )
                analyzed.append((upload_index, uploaded_file.name, result))

        preview_columns = st.columns(min(3, len(analyzed)))
        for column_index, (upload_index, name, result) in enumerate(analyzed):
            with preview_columns[column_index % len(preview_columns)]:
                if result.get("image") is None:
                    st.warning(f"{name}: {result.get('error')}")
                    continue

                preview = result["image"].copy()
                for face_index, face in enumerate(result["faces"]):
                    x, y, w, h = [
                        int(value * result["scale"]) for value in face["bbox"]
                    ]
                    draw_label_box(
                        preview,
                        x,
                        y,
                        w,
                        h,
                        f"Yüz {face_index + 1}",
                        color=(245, 158, 11),
                    )

                st.image(
                    preview,
                    caption=f"{upload_index + 1}. {name}",
                    use_container_width=True,
                )
                if not result["faces"]:
                    st.caption("Bu fotoğrafta yüz bulunamadı.")

        face_options: list[tuple[str, np.ndarray]] = []
        for upload_index, _name, result in analyzed:
            for face_index, face in enumerate(result["faces"]):
                label = (
                    f"{upload_index + 1}. fotoğraf · Yüz {face_index + 1} · "
                    f"tespit %{face['det_score'] * 100:.0f}"
                )
                face_options.append((label, face["embedding"]))

        labels = [label for label, _embedding in face_options]
        if labels:
            selected_labels = st.multiselect(
                "Aramada kullanılacak yüzler",
                options=labels,
                default=labels,
                help=(
                    "Tek kişi arıyorsanız ilgili yüzü seçin. Birden fazla yüz "
                    "seçilirse her biri için en iyi eşleşme alınır."
                ),
            )
            probe_embeddings = [
                embedding
                for label, embedding in face_options
                if label in selected_labels
            ]
        else:
            st.warning(
                "Kullanılabilir yüz bulunamadı. Daha net, önden çekilmiş ve "
                "yüzün büyük olduğu bir fotoğraf deneyin."
            )

    st.divider()
    st.subheader("2. Benzerlik eşiği ve sonuçlar")

    col_threshold, col_count = st.columns([2, 1])
    with col_threshold:
        match_threshold = st.slider(
            "Benzerlik eşiği",
            min_value=0.20,
            max_value=0.70,
            value=DEFAULT_MATCH_THRESHOLD,
            step=0.01,
            help=(
                "SFace için resmi eşik 0.363'tür. Yanlış eşleşmeyi azaltmak için "
                "0.40-0.50 aralığı önerilir. Yüksek eşik daha az sonuç getirir."
            ),
        )
    with col_count:
        max_results = st.select_slider(
            "Maksimum sonuç",
            options=[20, 50, 100, 200, 500],
            value=100,
        )

    if store is None:
        st.info("Soldan geçerli bir Google Drive klasör linki girin.")
    elif not uploaded_files:
        st.info("Aramak için en az bir yüz fotoğrafı yükleyin.")
    elif not probe_embeddings:
        st.warning("Aramak için en az bir yüz seçin.")
    else:
        with st.spinner("Kalıcı indeks kontrol ediliyor..."):
            loaded = ensure_remote_loaded(store, pg, key)

        load_error = st.session_state.pop(f"_remote_error_{key}", None)
        if load_error:
            st.warning(f"Postgres'ten veri yüklenemedi: {load_error}")
        if loaded:
            st.toast(f"Postgres'ten {loaded} kayıt yüklendi.")

        matrix, metas = store.load_matrix()
        stats = store.stats()

        if matrix.shape[0] == 0:
            st.warning(
                "İndekste henüz yüz yok. Sağdaki **İndeks & Kalıcılık** "
                "sekmesinden indeksi oluşturun."
            )
        else:
            results = search_embeddings(
                matrix,
                metas,
                probe_embeddings,
                threshold=match_threshold,
                max_results=max_results,
            )

            metric_a, metric_b, metric_c = st.columns(3)
            metric_a.metric("Eşleşen fotoğraf", len(results))
            metric_b.metric("İndeksteki yüz", f"{matrix.shape[0]:,}".replace(",", "."))
            metric_c.metric(
                "En yüksek benzerlik",
                f"%{results[0]['score'] * 100:.1f}" if results else "—",
            )

            if stats["errors"]:
                st.caption(
                    f"ℹ️ İndekste {stats['errors']} görsel işlenemedi. "
                    "Gerekirse 'Mevcut kayıtları yeniden işle' ile tekrar deneyin."
                )

            if not results:
                st.info(
                    "Eşik üzerinde eşleşme bulunamadı. Eşiği 0.30-0.35 civarına "
                    "düşürmeyi deneyin; ancak yanlış eşleşme riski artar."
                )
            else:
                top_left, top_right = st.columns([3, 1])
                with top_left:
                    st.success(
                        f"{len(results)} fotoğrafta eşleşme bulundu. "
                        "Sonuçlar benzerlik skoruna göre sıralıdır."
                    )
                with top_right:
                    st.download_button(
                        "📥 Sonuçları CSV indir",
                        data=results_to_csv(results),
                        file_name="yuz_eslesmeleri.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

                display_limit = int(st.session_state.get("_display_limit", 60))
                visible_results = results[:display_limit]
                columns = st.columns(3, gap="medium")

                for result_index, result in enumerate(visible_results):
                    with columns[result_index % 3]:
                        with st.container(border=True):
                            image_bgr = get_display_image(
                                result["file_id"],
                                thumb_size,
                                api_key,
                                credentials_identity,
                                resource_key,
                            )
                            if image_bgr is not None:
                                st.image(
                                    annotate_result(image_bgr, result),
                                    use_container_width=True,
                                )
                            else:
                                st.info(
                                    "Önizleme alınamadı. Drive'da aç butonunu kullanın."
                                )

                            st.markdown(
                                f"**%{result['score'] * 100:.1f} benzerlik**"
                            )
                            st.caption(
                                f"{result['name']}\n\n"
                                f"{result['path'] or 'Kök klasör'}"
                            )
                            button_a, button_b = st.columns(2)
                            with button_a:
                                st.link_button(
                                    "Drive'da aç",
                                    drive_view_url(result["file_id"]),
                                    use_container_width=True,
                                )
                            with button_b:
                                st.link_button(
                                    "İndir",
                                    drive_download_url(result["file_id"]),
                                    use_container_width=True,
                                )

                if len(results) > display_limit:
                    if st.button(
                        f"Daha fazla göster ({len(results) - display_limit} sonuç)",
                        use_container_width=True,
                    ):
                        st.session_state["_display_limit"] = display_limit + 60
                        st.rerun()

# ----------------------------------------------------------------------- index tab
with tab_index:
    st.subheader("İndeks durumu")

    if store is None:
        st.info("Soldan geçerli bir Google Drive klasör linki girin.")
    else:
        stats = store.stats()
        metric_1, metric_2, metric_3, metric_4 = st.columns(4)
        metric_1.metric("Fotoğraf", f"{stats['images']:,}".replace(",", "."))
        metric_2.metric("İndekslenen yüz", f"{stats['faces']:,}".replace(",", "."))
        metric_3.metric("Yüzsüz fotoğraf", f"{stats['no_face']:,}".replace(",", "."))
        metric_4.metric("Hata", f"{stats['errors']:,}".replace(",", "."))

        if stats["last_indexed"]:
            st.caption(f"Son indeks: `{stats['last_indexed']}`")
        folder_url_meta = store.meta_get("folder_url")
        if folder_url_meta:
            st.caption(f"Klasör: {folder_url_meta}")
        st.caption(f"Kalıcı anahtar: `{key}`")

        if stats["images"] == 0 and pg is not None:
            st.info(
                "Yerel indeks boş. **Postgres'ten çek** ile kalıcı indeksi "
                "bu oturuma yükleyebilirsiniz."
            )

        st.divider()
        action_1, action_2, action_3 = st.columns(3)
        with action_1:
            if st.button(
                "🚀 İndeksi oluştur / güncelle",
                use_container_width=True,
                type="primary",
            ):
                run_indexing()
        with action_2:
            if st.button(
                "☁️ Postgres'e gönder",
                use_container_width=True,
                disabled=pg is None or stats["images"] == 0,
            ):
                try:
                    sent = pg.upsert(key, store.export_rows())
                    flash("success", f"{sent} kayıt Postgres'e gönderildi.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Postgres'e gönderilemedi: {exc}")
        with action_3:
            if st.button(
                "☁️ Postgres'ten çek",
                use_container_width=True,
                disabled=pg is None,
            ):
                try:
                    rows = pg.fetch(key)
                    imported = store.import_rows(rows)
                    st.session_state[f"_remote_loaded_{key}"] = True
                    flash("success", f"{imported} kayıt Postgres'ten yüklendi.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Postgres'ten çekilemedi: {exc}")

        if pg is not None:
            pg_col_1, pg_col_2 = st.columns(2)
            with pg_col_1:
                if st.button(
                    "🔌 Postgres bağlantısını test et",
                    use_container_width=True,
                ):
                    try:
                        pg.ping()
                        count = pg.count(key)
                        st.success(
                            f"Bağlantı başarılı. Bu klasör için {count} kayıt var."
                        )
                    except Exception as exc:
                        st.error(str(exc))
            with pg_col_2:
                try:
                    remote_count = pg.count(key)
                    st.caption(f"Supabase'de bu klasör için {remote_count} kayıt.")
                except Exception:
                    st.caption("Supabase durumu sorgulanamadı.")
        else:
            st.warning(
                "Kalıcı DB kapalı. Streamlit uykuya dalınca yerel indeks silinir. "
                "Supabase bilgilerini `.streamlit/secrets.toml` veya Streamlit "
                "Cloud secrets bölümüne ekleyin."
            )

        st.divider()
        st.markdown("#### Yedekleme")

        try:
            export_data = store.export_bytes()
            st.download_button(
                "💾 İndeksi indir (.sqlite)",
                data=export_data,
                file_name=f"drive_face_index_{key}.sqlite",
                mime="application/x-sqlite3",
            )
        except Exception as exc:
            st.warning(f"İndeks dışa aktarılamadı: {exc}")

        uploaded_index = st.file_uploader(
            "İndeks dosyası yükle (.sqlite)",
            type=["sqlite", "sqlite3", "db"],
            key="index_upload",
        )
        if uploaded_index is not None:
            if st.button("Yüklenen indeksi birleştir"):
                try:
                    total = store.merge_sqlite_bytes(uploaded_index.getvalue())
                    flash("success", f"İndeks birleştirildi. Toplam {total} kayıt.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"İndeks yüklenemedi: {exc}")

        st.divider()
        st.markdown("#### Sıfırlama")
        confirm_reset = st.checkbox(
            "Bu klasörün yerel indeksini silmeyi onaylıyorum.",
            key="confirm_reset",
        )
        delete_remote = st.checkbox(
            "Postgres'teki kayıtları da sil",
            key="delete_remote",
            disabled=pg is None,
        )
        if st.button(
            "🗑️ İndeksi sıfırla",
            disabled=not confirm_reset,
        ):
            store.clear()
            if delete_remote and pg is not None:
                try:
                    pg.delete_folder(key)
                except Exception as exc:
                    st.warning(f"Postgres silme hatası: {exc}")
            st.session_state[f"_remote_loaded_{key}"] = False
            st.session_state.pop("_display_limit", None)
            flash("success", "İndeks sıfırlandı.")
            st.rerun()

# ------------------------------------------------------------------------ help tab
with tab_help:
    st.subheader("Nasıl çalışır?")
    st.markdown(
        """
        **1. Drive klasörü.** Klasör Google Drive'da
        *Bağlantıya sahip herkes → Görüntüleyen* olarak paylaşılmalı.
        Alt klasörler dahil edilebilir.

        **2. İndeksleme.** Uygulama klasördeki görselleri Drive API veya public
        HTML üzerinden listeler. Her görselin **tam dosyası indirilmez**; Drive
        thumbnail'ı alınır. Görseldeki yüzler YuNet ile bulunur, SFace ile
        128 boyutlu embedding'e çevrilir. Embedding'ler yerel SQLite'ta ve
        (yapılandırıldıysa) Supabase Postgres'te saklanır.

        **3. Arama.** Yüklediğiniz yüzün embedding'i, indeksteki tüm yüzlerle
        kosinüs benzerliği üzerinden karşılaştırılır. Sonuçlar fotoğraf başına
        en iyi skora göre gruplanır. Bu işlem 10.000 yüz için bile milisaniyeler
        sürer.

        ### Doğruluk ve eşik
        - SFace'in resmi kosinüs eşiği **0.363**'tür (LFW ~%99,6).
        - Etkinlik fotoğraflarında önerilen aralık **0.40 - 0.50**.
        - Eşik yükseldikçe yanlış eşleşme azalır, kaçırma artar.
        - Sonuçları her zaman gözle doğrulayın; sistem kesin kimlik iddiası taşımaz.
        - Aynı kişinin birden fazla fotoğrafını yüklemek doğruluğu artırır.
        - Çok küçük, bulanık veya yandan çekilmiş yüzler kaçabilir.

        ### Kalıcılık
        Streamlit Community Cloud ücretsiz katmanında yerel disk kalıcı değildir
        ve uygulama 12 saat trafik almazsa uykuya dalar. Bu yüzden indeks:
        - Supabase Postgres'e otomatik yazılır (yapılandırıldıysa),
        - ayrıca `.sqlite` olarak indirilebilir / yüklenebilir.

        Supabase doğrudan bağlantısı IPv6 gerektirebilir. Streamlit Cloud'da
        bağlantı kurulamazsa **Dashboard → Connect → Session pooler** bağlantı
        dizesini `SUPABASE_DB_URL` olarak kullanın.

        ### Gizlilik
        - Üçüncü taraf yüz tanıma API'si yoktur.
        - Görseller ve embedding'ler yalnızca sizin altyapınızda işlenir.
        - Drive görselleri yalnızca thumbnail olarak çekilir.
        - Yüz verisi biyometriktir; yetkiniz olan fotoğraflarda ve KVKK/GDPR'a
          uygun şekilde kullanın.

        ### Teknik
        - Tespit: **YuNet** (MIT lisansı)
        - Tanıma: **SFace** (Apache 2.0)
        - Arayüz: **Streamlit**
        - Kalıcı depo: **Supabase Postgres / SQLite**
        """
    )
    st.caption(f"{APP_TITLE} · {APP_BRAND}")
