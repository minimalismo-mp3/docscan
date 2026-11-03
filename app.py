import io

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

st.set_page_config(page_title="DocScan Pro", layout="centered")
st.title("📄 DocScan Pro")

MAX_SIDE = 2000  # lado mayor máximo (px): mantiene el procesado rápido

# ---------- Estado ----------
if "pages" not in st.session_state:
    st.session_state.pages = []      # lista de imágenes PIL ya procesadas
if "pdf_bytes" not in st.session_state:
    st.session_state.pdf_bytes = b""
if "uid" not in st.session_state:
    st.session_state.uid = 0         # cambia las keys de los widgets para limpiarlos


# ---------- Procesamiento ----------
def load_image(fuente) -> np.ndarray:
    """Abre la imagen, corrige la rotación EXIF (fotos de celular) y la reduce."""
    img = Image.open(fuente)
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    return np.array(img)


def flatten_illumination(channel: np.ndarray) -> np.ndarray:
    """Quita sombras: estima el fondo (papel) y lo resta al canal."""
    bg = cv2.dilate(channel, np.ones((7, 7), np.uint8))
    bg = cv2.medianBlur(bg, 21)
    diff = 255 - cv2.absdiff(channel, bg)
    return cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)


def process_image(img_rgb: np.ndarray, mode: str) -> Image.Image:
    if mode == "Blanco y Negro":
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        flat = flatten_illumination(gray)
        out = cv2.adaptiveThreshold(
            flat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 12
        )
        return Image.fromarray(out, mode="L")

    # Magic Color: aplana la iluminación en cada canal y sube un poco el contraste
    channels = [flatten_illumination(c) for c in cv2.split(img_rgb)]
    out = cv2.merge(channels)
    out = cv2.convertScaleAbs(out, alpha=1.15, beta=-10)
    # Nitidez suave (unsharp mask)
    soft = cv2.GaussianBlur(out, (0, 0), 2)
    out = cv2.addWeighted(out, 1.5, soft, -0.5, 0)
    return Image.fromarray(out)


def build_pdf(pages) -> bytes:
    buf = io.BytesIO()
    pages[0].save(
        buf,
        format="PDF",
        save_all=True,
        append_images=pages[1:],
        resolution=200,
    )
    return buf.getvalue()


# ---------- Entrada ----------
uid = st.session_state.uid
usar_camara = st.toggle("Usar cámara", value=False)

fuente = None
if usar_camara:
    # La cámara solo se carga si la activas (evita que la página se quede cargando)
    fuente = st.camera_input("Capturar documento", key=f"cam_{uid}")
if fuente is None:
    fuente = st.file_uploader(
        "Subir imagen o tomar foto",
        type=["jpg", "jpeg", "png"],
        key=f"file_{uid}",
    )

# ---------- Vista previa y agregar ----------
if fuente is not None:
    filtro = st.radio("Modo de escaneo:", ["Blanco y Negro", "Magic Color"], horizontal=True)
    procesada = process_image(load_image(fuente), filtro)
    st.image(procesada, caption="Vista previa")

    if st.button("➕ Agregar al PDF", type="primary"):
        st.session_state.pages.append(procesada)
        st.session_state.pdf_bytes = build_pdf(st.session_state.pages)
        st.session_state.uid += 1  # limpia cámara/uploader para la siguiente página
        st.rerun()

# ---------- Exportación ----------
if st.session_state.pages:
    st.write("---")
    st.write(f"### Documento actual: {len(st.session_state.pages)} página(s)")

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "📥 Descargar PDF",
            data=st.session_state.pdf_bytes,
            file_name="Documento_Escaneado.pdf",
            mime="application/pdf",
        )
    with col2:
        if st.button("🗑️ Borrar todo"):
            st.session_state.pages = []
            st.session_state.pdf_bytes = b""
            st.session_state.uid += 1
            st.rerun()
