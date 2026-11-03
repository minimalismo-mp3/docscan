import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="DocScan Pro", layout="centered")
st.title("📄 DocScan Pro")

# Inicializar estado para guardar múltiples páginas
if 'pages' not in st.session_state:
    st.session_state.pages = []

def process_image(img_array, mode):
    # 1. Quitar sombras (Estimación de iluminación local)
    if mode == "Blanco y Negro":
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        # Umbral adaptativo para aislar letras del fondo oscurecido
        processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 15)
        processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
    else:
        # Magic Color: Dividir imagen original por una versión muy difuminada para neutralizar sombras
        blur = cv2.GaussianBlur(img_array, (21, 21), 0)
        processed = cv2.divide(img_array, blur, scale=255)
        processed = cv2.convertScaleAbs(processed, alpha=1.2, beta=0) # Contraste

    # 2. Aumentar Nitidez (Filtro Laplaciano)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(processed, -1, kernel)
    return sharpened

# El componente nativo de cámara de Streamlit
foto = st.camera_input("Capturar documento")
archivo = st.file_uploader("O subir archivo de la galería", type=["jpg", "jpeg", "png"])

fuente = foto if foto else archivo

if fuente:
    # Leer la imagen subida/capturada
    img = Image.open(fuente).convert('RGB')
    img_array = np.array(img)
    
    filtro = st.radio("Modo de escaneo:", ["Blanco y Negro", "Magic Color"], horizontal=True)
    
    if st.button("Procesar y Agregar al PDF"):
        procesada = process_image(img_array, mode=filtro)
        st.session_state.pages.append(procesada)
        st.success(f"✅ Página {len(st.session_state.pages)} agregada al documento.")

# Panel de Exportación
if st.session_state.pages:
    st.write("---")
    st.write(f"### Documento actual: {len(st.session_state.pages)} páginas")
    
    # Previsualizar la última página procesada
    st.image(st.session_state.pages[-1], caption="Vista previa de la última página", use_container_width=True)
    
    # Generar PDF en un buffer de memoria
    pdf_buffer = io.BytesIO()
    pil_pages = [Image.fromarray(p) for p in st.session_state.pages]
    
    pil_pages[0].save(
        pdf_buffer, 
        format='PDF', 
        save_all=True, 
        append_images=pil_pages[1:]
    )
    
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="📥 Descargar PDF Completo",
            data=pdf_buffer.getvalue(),
            file_name="Documento_Escaneado.pdf",
            mime="application/pdf"
        )
    with col2:
        if st.button("🗑️ Borrar Todo"):
            st.session_state.pages = []
            st.rerun()