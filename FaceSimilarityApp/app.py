# python -m streamlit run app.py

import streamlit as st
import cv2
import numpy as np
import os
import tempfile
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import euclidean

st.set_page_config(
    page_title="Deteksi Kemiripan Wajah",
    layout="wide",
    initial_sidebar_state="expanded"
)

def switch_theme():
    new_theme = "light" if st.session_state.theme == "dark" else "dark"
    st.session_state.theme = new_theme
    st.session_state.theme_override = True
    st.query_params["theme"] = new_theme

THEME_DETECT_JS = """
<script>
(function() {
    const params = new URLSearchParams(window.location.search);
    if (params.has('theme')) return;
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    params.set('theme', prefersDark ? 'dark' : 'light');
    window.location.replace(window.location.pathname + '?' + params.toString());
})();
</script>
"""

query_params = st.query_params
detected_theme = query_params.get("theme", "dark")

if "theme" not in st.session_state:
    st.session_state.theme = detected_theme
if "theme_override" not in st.session_state:
    st.session_state.theme_override = False

is_dark = (st.session_state.theme == "dark")

if is_dark:
    bg             = "#0A0E1A"
    app_bg         = "linear-gradient(135deg, #0A0E1A 0%, #0D1321 50%, #0A0E1A 100%)"
    sidebar_bg     = "linear-gradient(180deg, #111827 0%, #0D1521 100%)"
    metric_bg      = "rgba(15,23,42,0.9)"
    result_bg      = "rgba(15,23,42,0.95)"
    track_bg       = "rgba(30,41,59,0.8)"
    border         = "rgba(99,102,241,0.2)"
    border_soft    = "rgba(99,102,241,0.1)"
    accent         = "#6366F1"
    accent_glow    = "rgba(99,102,241,0.7)"
    accent_dim     = "rgba(99,102,241,0.15)"
    cyan           = "#22D3EE"
    green          = "#34D399"
    text_primary   = "#F8FAFC"
    text_body      = "#CBD5E1"
    text_muted     = "#64748B"
    text_faint     = "#475569"
    text_badge     = "#818CF8"
    text_mono_v    = "#A5B4FC"
    footer_color   = "#1E293B"
    grad_title     = "linear-gradient(135deg, #F8FAFC 0%, #A5B4FC 50%, #22D3EE 100%)"
else:
    bg             = "#F8FAFC"
    app_bg         = "linear-gradient(135deg, #F8FAFC 0%, #EEF2FF 50%, #F8FAFC 100%)"
    sidebar_bg     = "linear-gradient(180deg, #FFFFFF 0%, #F1F5F9 100%)"
    metric_bg      = "rgba(255,255,255,0.95)"
    result_bg      = "rgba(255,255,255,0.98)"
    track_bg       = "rgba(226,232,240,0.8)"
    border         = "rgba(99,102,241,0.25)"
    border_soft    = "rgba(99,102,241,0.12)"
    accent         = "#4F46E5"
    accent_glow    = "rgba(79,70,229,0.4)"
    accent_dim     = "rgba(99,102,241,0.08)"
    cyan           = "#0891B2"
    green          = "#059669"
    text_primary   = "#0F172A"
    text_body      = "#1E293B"
    text_muted     = "#475569"
    text_faint     = "#64748B"
    text_badge     = "#4F46E5"
    text_mono_v    = "#4338CA"
    footer_color   = "#94A3B8"
    grad_title     = "linear-gradient(135deg, #0F172A 0%, #4338CA 50%, #0891B2 100%)"

css = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

.stApp {{
    background: {app_bg};
    min-height: 100vh;
}}
[data-testid="stSidebar"] {{
    background: {sidebar_bg};
    border-right: 1px solid {border};
}}
[data-testid="stSidebar"] * {{ color: {text_body} !important; }}

.hero-header {{ text-align: center; padding: 2.5rem 1rem 1.5rem; margin-bottom: 0.5rem; }}
.hero-badge {{
    display: inline-block;
    background: {accent_dim};
    border: 1px solid {border};
    color: {text_badge};
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem; font-weight: 600;
    letter-spacing: 0.15em;
    padding: 0.3rem 1rem; border-radius: 20px;
    margin-bottom: 1rem; text-transform: uppercase;
}}
.hero-title {{
    font-size: 2.8rem; font-weight: 700;
    background: {grad_title};
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    margin: 0.5rem 0; line-height: 1.15; letter-spacing: -0.02em;
}}
.hero-subtitle {{
    color: {text_muted}; font-size: 1rem;
    max-width: 560px; margin: 0 auto; line-height: 1.6;
}}

.section-label {{ display: flex; align-items: center; gap: 0.6rem; margin: 2rem 0 1rem; }}
.section-dot {{
    width: 8px; height: 8px; border-radius: 50%;
    background: {accent}; box-shadow: 0 0 10px {accent_glow}; flex-shrink: 0;
}}
.section-title {{ color: {text_primary}; font-size: 1.1rem; font-weight: 600; letter-spacing: -0.01em; margin: 0; }}
.section-sub {{ color: {text_faint}; font-size: 0.8rem; font-family: 'JetBrains Mono', monospace; margin-left: auto; }}

.metric-row {{ display: flex; gap: 1rem; margin: 1.2rem 0; }}
.metric-tile {{
    flex: 1; background: {metric_bg}; border: 1px solid {border};
    border-radius: 12px; padding: 1.1rem 1.2rem;
    position: relative; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}}
.metric-tile::before {{
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, {accent}, {cyan});
}}
.metric-label {{
    color: {text_faint}; font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem; font-weight: 500; letter-spacing: 0.08em;
    text-transform: uppercase; margin-bottom: 0.4rem;
}}
.metric-value {{ color: {text_primary}; font-size: 1.5rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; letter-spacing: -0.02em; }}
.metric-value.accent {{ color: {cyan}; }}
.metric-value.green {{ color: {green}; }}

.img-label {{
    color: {text_muted}; font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem; font-weight: 500; letter-spacing: 0.1em;
    text-transform: uppercase; margin-bottom: 0.6rem; display: block;
}}

.result-card {{
    background: {result_bg}; border: 1px solid {border_soft};
    border-radius: 16px; padding: 1rem; text-align: center;
    position: relative; overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}}
.result-card.best {{ border-color: {border}; box-shadow: 0 0 24px {accent_dim}; }}
.rank-badge {{
    display: inline-block; font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem; font-weight: 600; letter-spacing: 0.12em;
    padding: 0.25rem 0.7rem; border-radius: 20px; margin-bottom: 0.8rem; text-transform: uppercase;
}}
.rank-badge.gold {{ background: {accent_dim}; border: 1px solid {border}; color: {text_badge}; }}
.rank-badge.silver {{ background: rgba(8,145,178,0.08); border: 1px solid rgba(8,145,178,0.25); color: {cyan}; }}
.rank-badge.bronze {{ background: rgba(148,163,184,0.08); border: 1px solid rgba(148,163,184,0.2); color: {text_muted}; }}
.identity-name {{ color: {text_primary}; font-size: 1rem; font-weight: 600; margin: 0.6rem 0 0.3rem; }}
.identity-file {{ color: {text_faint}; font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; }}
.score-container {{ margin: 0.8rem 0 0.4rem; }}
.score-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem; }}
.score-label-text {{ color: {text_faint}; font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.08em; }}
.score-number {{ font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; font-weight: 600; color: {text_mono_v}; }}
.score-track {{ height: 6px; background: {track_bg}; border-radius: 99px; overflow: hidden; }}
.score-fill {{ height: 100%; border-radius: 99px; background: linear-gradient(90deg, {accent}, {cyan}); }}

[data-testid="stFileUploader"] {{
    background: {accent_dim} !important;
    border: 1.5px dashed {border} !important;
    border-radius: 12px !important;
}}
.stDownloadButton {{
    display: flex !important;
    justify-content: center !important;
    width: 100% !important;
    margin-top: 1.5rem !important;
}}
.stDownloadButton > button {{
    background: {accent_dim} !important; 
    border: 1px solid {border} !important;
    color: {text_mono_v} !important; 
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important; 
    font-size: 0.85rem !important;
    padding: 0.6rem 2.5rem !important;
    transition: all 0.3s ease !important;
}}
.stDownloadButton > button:hover {{
    background: {border} !important;
    color: {text_primary} !important;
}}

hr {{ border: none !important; border-top: 1px solid {border_soft} !important; margin: 2rem 0 !important; }}
p, label, .stMarkdown {{ color: {text_body} !important; }}
h1, h2, h3 {{ color: {text_primary} !important; }}
::-webkit-scrollbar {{ width: 5px; }}
::-webkit-scrollbar-track {{ background: {bg}; }}
::-webkit-scrollbar-thumb {{ background: {border}; border-radius: 99px; }}
</style>
"""
st.markdown(css, unsafe_allow_html=True)

#=========================
# FUNGSI
#=========================
if not st.session_state.theme_override:
    st.markdown(THEME_DETECT_JS, unsafe_allow_html=True)

# ── KOMPRESI PCA BERWARNA (RGB) ──────────────────────────────────────────────
def compress_image_pca_color(image, n_components=150):
    """Kompresi PCA per channel RGB, hasil tetap berwarna."""
    img_rgb = image.convert("RGB")
    arr = np.array(img_rgb).astype(float) / 255.0  # (H, W, 3)

    channels = []
    for c in range(3):
        matrix = arr[:, :, c]
        mean = np.mean(matrix, axis=0)
        centered = matrix - mean
        cov = np.cov(centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, idx]
        components = eigvecs[:, :n_components]
        projected = np.dot(centered, components)
        reconstructed = np.dot(projected, components.T) + mean
        reconstructed = np.clip(reconstructed * 255, 0, 255).astype(np.uint8)
        channels.append(reconstructed)

    color_img = np.stack(channels, axis=2)
    return Image.fromarray(color_img)
    return compressed

#Fungsi kompres PCA Gray
def compress_image_pca_gray(image, n_components=150):
    """
    Kompresi PCA grayscale.
    Input : RGB
    Output: Grayscale
    """

    img_gray = image.convert("L")

    matrix = np.array(img_gray).astype(float) / 255.0

    mean = np.mean(matrix, axis=0)

    centered = matrix - mean

    cov = np.cov(
        centered,
        rowvar=False
    )

    eigvals, eigvecs = np.linalg.eigh(cov)

    idx = np.argsort(eigvals)[::-1]

    eigvecs = eigvecs[:, idx]

    n_components = min(
        n_components,
        eigvecs.shape[1]
    )

    components = eigvecs[:, :n_components]

    projected = np.dot(
        centered,
        components
    )

    reconstructed = np.dot(
        projected,
        components.T
    ) + mean

    reconstructed = np.clip(
        reconstructed * 255,
        0,
        255
    ).astype(np.uint8)

    return Image.fromarray(
        reconstructed,
        mode="L"
    )

#===========================================
#  PREPROCESSING UNTUK PENCOCOKAN
#===========================================
def preprocess_image_color(path, img_size=(100, 100)):
    """Baca gambar berwarna, resize, dan flatten menjadi vektor."""
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # BGR → RGB
    img = cv2.resize(img, img_size)
    return img.flatten().astype(float)            # panjang = 100*100*3 = 30000

def preprocess_image(path, img_size=(100, 100)):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Gagal membaca gambar: {path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades +
        "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    # Konversi ke RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        img_rgb = img_rgb[y:y+h, x:x+w]
    # Histogram equalization per channel RGB
    r, g, b = cv2.split(img_rgb)
    r = cv2.equalizeHist(r)
    g = cv2.equalizeHist(g)
    b = cv2.equalizeHist(b)
    img_rgb = cv2.merge([r, g, b])

    img_rgb = cv2.resize(img_rgb, img_size)
    return img_rgb.flatten().astype(float)

@st.cache_resource(show_spinner=False)
def load_dataset(dataset_path, img_size=(100, 100)):
    data, labels, filenames = [], [], []
    for person_name in os.listdir(dataset_path):
        person_folder = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_folder):
            continue
        for file in os.listdir(person_folder):
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                path = os.path.join(person_folder, file)
                vector = preprocess_image(path, img_size)
                data.append(vector)
                labels.append(person_name)
                filenames.append(file)
    return np.array(data), labels, filenames

def center_data(X):
    mean_face = np.mean(X, axis=0)
    return X - mean_face, mean_face

def compute_pca_svd(X_centered, num_components=50):
    U, S, VT = np.linalg.svd(X_centered, full_matrices=False)
    max_components = min(num_components, VT.shape[0])
    return VT[:num_components]


def project_faces(X_centered, eigenfaces):
    return np.dot(X_centered, eigenfaces.T)

def extract_feature(image_path, mean_face, eigenfaces, img_size=(100, 100)):
    """Ekstrak fitur dari gambar berwarna."""
    vector = preprocess_image_color(image_path, img_size)
    centered = vector - mean_face
    return np.dot(centered, eigenfaces.T)

def recognize_cosine_topk(test_feature, database_features, labels, filenames, k=1):
    results = []
    test_feature = test_feature.reshape(1, -1)
    for feature, label, file in zip(database_features, labels, filenames):
        feature = feature.reshape(1, -1)
        score = cosine_similarity(test_feature, feature)[0][0]
        results.append((label, file, score))
    results.sort(key=lambda x: x[2], reverse=True)
    return results[:k]

def compare_two_faces(
    img1_path,
    img2_path,
    mean_face,
    eigenfaces
):

    feature1 = extract_feature(
        img1_path,
        mean_face,
        eigenfaces,
        IMG_SIZE
    )

    feature2 = extract_feature(
        img2_path,
        mean_face,
        eigenfaces,
        IMG_SIZE
    )

    similarity = cosine_similarity(
        feature1.reshape(1, -1),
        feature2.reshape(1, -1)
    )[0][0]

    return similarity

def compute_pca_svd_variance(X_centered):
    U, S, VT = np.linalg.svd(
        X_centered,
        full_matrices=False
    )
    explained_variance = (
        S**2
    ) / np.sum(S**2)
    cumulative_variance = np.cumsum(
        explained_variance
    )
    return VT, cumulative_variance

def predict_cosine(
    test_feature,
    database_features,
    labels
):
    scores = cosine_similarity(
        test_feature.reshape(1, -1),
        database_features
    )[0]
    idx = np.argmax(scores)
    return labels[idx]


def predict_euclidean(
    test_feature,
    database_features,
    labels
):
    distances = []
    for feature in database_features:
        distances.append(
            euclidean(
                test_feature,
                feature
            )
        )
    idx = np.argmin(distances)
    return labels[idx]


def show_mean_face(
    mean_face
):
    mean_img = mean_face.reshape(
        IMG_SIZE[0],
        IMG_SIZE[1],
        3
    )
    mean_img = cv2.cvtColor(
        mean_img.astype(np.uint8),
        cv2.COLOR_RGB2GRAY
    )
    st.image(
        mean_img,
        caption="Mean Face Dataset",
        width=300, clamp=True
    )

def extract_face_image(path, img_size=(100, 100)):
    img = cv2.imread(path)

    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades +
        "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        img_rgb = img_rgb[y:y+h, x:x+w]

    img_rgb = cv2.resize(img_rgb, img_size)

    return img_rgb

#========================================
# ── SIDEBAR 
#========================================
with st.sidebar:
    col_toggle, col_space = st.columns([1.5, 4])
    
    with col_toggle:
        toggle_icon = "☀️" if is_dark else "🌙"
        toggle_help = "Beralih ke Light Mode" if is_dark else "Beralih ke Dark Mode"
        st.button(
            toggle_icon,
            key="theme_toggle",
            help=toggle_help,
            use_container_width=True,
            on_click=switch_theme
        )
    btn_bg = "rgba(250,204,21,0.18)" if is_dark else "rgba(99,102,241,0.22)"
    btn_border = "rgba(250,204,21,0.6)" if is_dark else "rgba(99,102,241,0.6)"
    btn_shadow = "0 0 10px rgba(250,204,21,0.35)" if is_dark else "0 0 10px rgba(99,102,241,0.4)"
    
    st.markdown(f"""
    <style>
    [data-testid="stSidebar"] button[kind="secondary"] {{
        background: {btn_bg} !important;
        border: 1px solid {btn_border} !important;
        box-shadow: {btn_shadow} !important;
        border-radius: 10px !important;
        height: 42px !important;
        font-size: 1.2rem !important;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: all 0.3s ease;
        margin-bottom: 1.5rem !important;
    }}
    [data-testid="stSidebar"] button[kind="secondary"]:hover {{
        filter: brightness(1.2);
        border-color: {btn_border} !important;
        color: inherit !important;
    }}
    </style>
    """, unsafe_allow_html=True)

    st.markdown(
        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.7rem;color:{text_faint};'
        f'text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.3rem;">Komponen PCA</div>',
        unsafe_allow_html=True
    )

    menu = st.radio(
        "Menu",
        [
            "EDA & Evaluasi",
            "Kompres Foto",
            "Deteksi Wajah dengan dataset",
            "Deteksi Wajah (Lama dan Sekarang)"
        ]
    )

# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown(
    f'<div class="hero-header">'
    f'<div class="hero-badge">⬡ PCA · SVD · Cosine Similarity · Kompres · Euclidean Distance</div>'
    f'<div class="hero-title">Deteksi Kemiripan Wajah</div>'
    f'<div class="hero-subtitle">Kompresi gambar dengan Principal Component Analysis dan temukan '
    f'identitas wajah terdekat dari dataset dan Foto Dulu Dan sekarang menggunakan eigenface projection.</div>'
    f'</div>',
    unsafe_allow_html=True
)

DATASET_PATH = "FaceSimilarityApp/dataset"
IMG_SIZE = (100, 100)

if not os.path.exists(DATASET_PATH):
    st.error("⚠️ Direktori `dataset` tidak ditemukan. Pastikan folder tersedia di lokasi yang sama dengan skrip ini.")
    st.stop()

X, labels, filenames = load_dataset(
    DATASET_PATH,
    IMG_SIZE
)

X_train, X_test, y_train, y_test, train_files, test_files = train_test_split(
    X,
    labels,
    filenames,
    test_size=0.2,
    stratify=labels,
    random_state=42
)

#=======================
# MENU
#=======================

#menu ke 1
if menu == "EDA & Evaluasi":
    
    st.subheader("Laporan Model")
    st.write(
            f"Komponen PCA"
        )
    jumlah_komponen_evaluasi = st.slider(
            label="", min_value=1, max_value=150, value=10, step=1,
            label_visibility="collapsed"
        )

    st.write(
        f"Total Data : {len(X)} gambar"
    )

    st.write(
        f"Data Latih (80%) : {len(X_train)} gambar"
    )

    st.write(
        f"Data Uji (20%) : {len(X_test)} gambar"
    )

    X_centered_train, mean_face = center_data(
        X_train
    )

    eigenfaces_all, cumulative_variance = (
        compute_pca_svd_variance(
            X_centered_train
        )
    )

    komponen = len(X_train)
    total_var = cumulative_variance[jumlah_komponen_evaluasi - 1] * 100

    st.metric(
        "Total Explained Variance",
        f"{total_var:.2f}%"
    )

    st.subheader(
        "Visualisasi Mean Face"
    )

    show_mean_face(
        mean_face
    )
    
    eigenfaces = eigenfaces_all[
        :jumlah_komponen_evaluasi
    ]

    database_features = project_faces(
        X_centered_train,
        eigenfaces
    )
    correct_cosine = 0

    for x_test, true_label in zip(
        X_test,
        y_test
    ):

        centered = (
            x_test -
            mean_face
        )

        feature = np.dot(
            centered,
            eigenfaces.T
        )

        pred = predict_cosine(
            feature,
            database_features,
            y_train
        )

        if pred == true_label:
            correct_cosine += 1

    accuracy_cosine = (
        correct_cosine /
        len(X_test)
    ) * 100
    correct_euclidean = 0

    for x_test, true_label in zip(
        X_test,
        y_test
    ):

        centered = (
            x_test -
            mean_face
        )

        feature = np.dot(
            centered,
            eigenfaces.T
        )

        pred = predict_euclidean(
            feature,
            database_features,
            y_train
        )
        if pred == true_label:
            correct_euclidean += 1

    accuracy_euclidean = (
        correct_euclidean /
        len(X_test)
    ) * 100
    
    col1, col2 = st.columns(2)

    col1.metric(
        "Akurasi Cosine Similarity",
        f"{accuracy_cosine:.2f}%"
    )

    col2.metric(
        "Akurasi Euclidean Distance",
        f"{accuracy_euclidean:.2f}%"
    )

#menu ke 2
elif menu == "Kompres Foto":
    st.subheader("Kompresi Gambar")

    st.markdown(
            f'<div class="section-label">'
            f'<div class="section-dot"></div>'
            f'<span class="section-title">Unggah Gambar</span>'
            f'<span class="section-sub">JPG / JPEG / PNG</span>'
            f'</div>',
            unsafe_allow_html=True
        )
    
    uploaded_file_kompres = st.file_uploader(
        label="", type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        help="Format yang didukung: JPG, JPEG, PNG"
    )
    
    if uploaded_file_kompres:
        st.markdown(
            f'</div></div>',
            unsafe_allow_html=True
        )
        format_output = st.selectbox("Pilih Format Output:", ["JPEG", "PNG", "WEBP"])
        
        if format_output in ["JPEG", "WEBP"]:
            # slide PCA
            st.markdown(
                        f'<div class="section-label">'
                        f'<div class="section-dot"></div>'
                        f'<span class="section-title">Kualitas</span>'
                        f'<span class="section-sub">%</span>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
            
            kualitas = st.slider(
                    label="Quality", min_value=1, max_value=95, value=10, step=1,
                    label_visibility="collapsed"
                )
            
            st.markdown(
                f'<div style="background:{accent_dim};border:1px solid {border};border-radius:10px;padding:1rem;margin-top:0.8rem;">'
                f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.65rem;color:{accent};'
                f'text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.5rem;">Parameter</div>'
                f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                f'<span style="color:{text_muted};font-size:1.5rem;">Quality</span>'
                f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:2rem;font-weight:600;color:{text_mono_v};">{kualitas}</span>'
                f'</div></div>',
                unsafe_allow_html=True
            )
    
        image = Image.open(uploaded_file_kompres)
        st.markdown("<hr>", unsafe_allow_html=True)

        st.markdown(
            f'<div class="section-label">'
            f'<div class="section-dot"></div>'
            f'<span class="section-title">Kompresi Gambar</span>'
            f'<span class="section-sub">MEMBANDINGKAN</span>'
            f'</div>',
            unsafe_allow_html=True
        )
        
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown('<span class="img-label">↳ Gambar Asli</span>', unsafe_allow_html=True)
            st.image(image, use_container_width=True)
    
        with st.spinner("Mengompresi gambar..."):
            # Kompres langsung ke JPEG dengan kualitas tertentu
            temp_buffer = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            if format_output == "JPEG":
                image.save(temp_buffer.name, format_output, quality=kualitas, optimize=True)
            elif format_output == "WebP":
                image.save(temp_buffer.name, format_output, quality=kualitas)
            else:
                image.save(temp_buffer.name, format_output, optimize=True)
            compressed = Image.open(temp_buffer.name)

        with col2:
            st.markdown('<span class="img-label">↳ Hasil Kompres</span>', unsafe_allow_html=True)
            st.image(compressed, use_container_width=True)
        
        size_before = len(uploaded_file_kompres.getvalue())
        temp_buffer = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        compressed.save(temp_buffer.name)
        size_after = os.path.getsize(temp_buffer.name)
        reduction = ((size_before - size_after) / size_before) * 100
        reduction_color = "green" if reduction > 0 else "accent"

        st.markdown(
            f'<div class="metric-row">'
            f'<div class="metric-tile" style="text-align:center;">'
            f'<div class="metric-label">Ukuran Asli</div>'
            f'<div class="metric-value">{size_before/1024:.1f} KB</div>'
            f'</div>'
            f'<div class="metric-tile" style="text-align:center;">'
            f'<div class="metric-label">Setelah Kompresi</div>'
            f'<div class="metric-value accent">{size_after/1024:.1f} KB</div>'
            f'</div>'
            f'<div class="metric-tile" style="text-align:center;">'
            f'<div class="metric-label">Rasio Pengurangan</div>'
            f'<div class="metric-value {reduction_color}">{abs(reduction):.1f}%</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True
        )


        st.markdown("<br>", unsafe_allow_html=True)
        col_left, col_btn, col_right = st.columns([1, 1, 1])
        with col_btn:
            with open(temp_buffer.name, "rb") as f:
                st.download_button(
                    label=f"⬇ Unduh Gambar ({format_output})",
                    data=f,
                    file_name=f"compressed_{uploaded_file_kompres.name.split('.')[0]}.{format_output.lower()}",
                    mime=f"image/{format_output.lower()}",
                    use_container_width=True
                )

#menu ke 3
elif menu == "Deteksi Wajah dengan dataset":
    
    st.subheader("Deteksi Wajah")
    jumlah_komponen_wajah_dataset = st.slider(
                label="", min_value=1, max_value=150, value=50, step=1,
                label_visibility="collapsed"
            )
    
    st.markdown(
            f'<div class="section-label">'
            f'<div class="section-dot"></div>'
            f'<span class="section-title">Unggah Gambar</span>'
            f'<span class="section-sub">JPG / JPEG / PNG</span>'
            f'</div>',
            unsafe_allow_html=True
        )
    
    uploaded_file_wajah_dataset = st.file_uploader(
        label="", type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
        help="Format yang didukung: JPG, JPEG, PNG"
    )

    if uploaded_file_wajah_dataset:
        with st.spinner("Memuat dataset dan membangun ruang eigenface..."):
            X, labels, filenames = load_dataset(DATASET_PATH, IMG_SIZE)
            X_centered, mean_face_gray = center_data(X)
            eigenfaces = compute_pca_svd(X_centered, jumlah_komponen_wajah_dataset)
            database_features = project_faces(X_centered, eigenfaces)

        image = Image.open(uploaded_file_wajah_dataset)
        st.markdown("<hr>", unsafe_allow_html=True)

        st.markdown(
            f'<div class="section-label">'
            f'<div class="section-dot"></div>'
            f'<span class="section-title">Kompresi Gambar via PCA</span>'
            f'<span class="section-sub">RGB · Berwarna</span>'
            f'</div>',
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown('<span class="img-label">↳ Gambar Asli</span>', unsafe_allow_html=True)
            st.image(image, use_container_width=True)

        with st.spinner("Mengompresi gambar dengan PCA (RGB)..."):
            compressed = compress_image_pca_gray(image, jumlah_komponen_wajah_dataset)
            compressed_temp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            compressed.save(compressed_temp.name)

        with col2:
            st.markdown('<span class="img-label">↳ Hasil Rekonstruksi PCA (Berwarna)</span>', unsafe_allow_html=True)
            st.image(compressed, use_container_width=True)

        size_before = len(uploaded_file_wajah_dataset.getvalue())
        temp_buffer = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        compressed.save(temp_buffer.name)
        size_after = os.path.getsize(temp_buffer.name)
        reduction = ((size_before - size_after) / size_before) * 100
        reduction_color = "green" if reduction > 0 else "accent"

        st.markdown(
            f'<div class="metric-row">'
            f'<div class="metric-tile" style="text-align:center;">'
            f'<div class="metric-label">Ukuran Asli</div>'
            f'<div class="metric-value">{size_before/1024:.1f} KB</div>'
            f'</div>'
            f'<div class="metric-tile" style="text-align:center;">'
            f'<div class="metric-label">Setelah Kompresi</div>'
            f'<div class="metric-value accent">{size_after/1024:.1f} KB</div>'
            f'</div>'
            f'<div class="metric-tile" style="text-align:center;">'
            f'<div class="metric-label">Rasio Pengurangan</div>'
            f'<div class="metric-value {reduction_color}">{abs(reduction):.1f}%</div>'
            f'</div>'
            f'<div class="metric-tile" style="text-align:center;">'
            f'<div class="metric-tile"><div class="metric-label">Komponen PCA</div>'
            f'<div class="metric-value" style="color:{text_mono_v};">{jumlah_komponen_wajah_dataset}<span style="font-size:0.9rem;color:{text_faint};"> dims</span></div></div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True
        )


        st.markdown("<br>", unsafe_allow_html=True)
        col_left, col_btn, col_right = st.columns([1, 1, 1])
        with col_btn:
            with open(temp_buffer.name, "rb") as f:
                st.download_button(
                    label="⬇ Unduh Gambar Rekonstruksi PCA",
                    data=f,
                    file_name=f"pca_{uploaded_file_wajah_dataset.name}",
                    mime="image/jpeg",
                    use_container_width=True
                )

        st.markdown("<hr>", unsafe_allow_html=True)

        st.markdown(
            f'<div class="section-label">'
            f'<div class="section-dot"></div>'
            f'<span class="section-title">Hasil Pencocokan Wajah</span>'
            f'<span class="section-sub">Top-1 · Cosine Similarity · RGB</span>'
            f'</div>',
            unsafe_allow_html=True
        )

        # Simpan gambar ASLI (bukan yang dikompres) ke file sementara untuk pencocokan
        temp_original = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        image.convert("RGB").save(temp_original.name)

        with st.spinner("Menghitung kemiripan di ruang eigenface..."):
            test_feature = extract_feature(compressed_temp.name, mean_face_gray, eigenfaces, IMG_SIZE)
            # ==========================
            # VISUALISASI MEAN FACE
            # ==========================

            st.markdown("### Visualisasi Mean Face vs Wajah Upload")

            upload_face = extract_face_image(
                compressed_temp.name,
                IMG_SIZE
            )
            
            upload_face_gray = cv2.cvtColor(
                upload_face,
                cv2.COLOR_RGB2GRAY
            )

            mean_face_img = mean_face_gray.reshape(
                IMG_SIZE[0],
                IMG_SIZE[1],
                3
            )

            mean_face_img = cv2.cvtColor(
                mean_face_img.astype(np.uint8),
                cv2.COLOR_RGB2GRAY
            )

            # normalisasi mean face agar bisa dilihat
            mean_face_img = mean_face_img - mean_face_img.min()
            mean_face_img = mean_face_img / mean_face_img.max()
            mean_face_img = (mean_face_img * 255).astype(np.uint8)

            col_a, col_b = st.columns(2)

            with col_a:
                st.image(
                    upload_face,
                    caption="Wajah Upload", #grayscale
                    use_container_width=True,
                    clamp = True
                )

            with col_b:
                st.image(
                    mean_face_img,
                    caption="Mean Face Dataset",
                    use_container_width=True
                )

            # ==========================
            # KEMIRIPAN DENGAN MEAN FACE
            # ==========================

            upload_vector = upload_face_gray.flatten().astype(float)

            mean_vector = mean_face_img.flatten().astype(float)

            similarity_mean = cosine_similarity(
                upload_vector.reshape(1, -1),
                mean_vector.reshape(1, -1)
            )[0][0]

            st.metric(
                "Kemiripan Upload vs Mean Face",
                f"{similarity_mean*100:.2f}%"
            )
            results = recognize_cosine_topk(test_feature, database_features, labels, filenames, k=2)

        badge_info = [
            ("gold", "Kecocokan Pertama"),
            ("silver", "Kecocokan Kedua")
        ]

        res_cols = st.columns(3, gap="medium")
        for i, (lbl, file, score) in enumerate(results):
            with res_cols[i]:
                img_path = os.path.join(DATASET_PATH, lbl, file)
                badge_class, badge_text = badge_info[i]
                is_best = "best" if i == 0 else ""

                st.markdown(
                    f'<div class="result-card {is_best}">'
                    f'<div class="rank-badge {badge_class}">{badge_text}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                
                st.image(img_path, use_container_width=False)

                st.markdown(
                    f'<div style="text-align:center;margin-top:0.5rem;">'
                    f'<div class="identity-name">{lbl}</div>'
                    f'<div class="identity-file" style="margin-bottom:0.8rem;">{file}</div>'
                    f'<div class="score-container">'
                    f'<div class="score-header">'
                    f'<span class="score-label-text">Kemiripan</span>'
                    f'<span class="score-number">{score:.4f}</span>'
                    f'</div>'
                    f'</div></div>',
                    unsafe_allow_html=True
                )

        st.markdown(
            f'<div style="text-align:center;padding:3rem 0 1.5rem;border-top:1px solid {border_soft};margin-top:2rem;">'
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.65rem;color:{footer_color};'
            f'letter-spacing:0.12em;text-transform:uppercase;">PCA · SVD · Eigenfaces · Cosine Similarity · RGB</div>'
            f'</div>',
            unsafe_allow_html=True
        )

# menu ke 4
elif menu == "Deteksi Wajah (Lama dan Sekarang)":
    st.header("DETEKSI WAJAH LAMA DENGAN YANG SEKARANG")
    
    st.markdown(
                f'</div></div>',
                unsafe_allow_html=True
            )
    
    st.markdown(
                f'<div class="section-label">'
                f'<div class="section-dot"></div>'
                f'<span class="section-title">Kualitas</span>'
                f'</div>',
                unsafe_allow_html=True
            )
    
    komponen_dulu_sekarang = st.slider(
            label="Quality", min_value=1, max_value=150, value=50, step=1,
            label_visibility="collapsed"
        )
    
    st.markdown(
        f'<div style="background:{accent_dim};border:1px solid {border};border-radius:10px;padding:1rem;margin-top:0.8rem;">'
        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.65rem;color:{accent};'
        f'text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.5rem;">Parameter Aktif</div>'
        f'<div style="display:flex;justify-content:space-between;align-items:center;">'
        f'<span style="color:{text_muted};font-size:1.5rem;">n_components</span>'
        f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:2rem;font-weight:600;color:{text_mono_v};">{komponen_dulu_sekarang}</span>'
        f'</div></div>',
        unsafe_allow_html=True
    )
    
    st.markdown(
        """
        <div style="display:flex;gap:2rem;justify-content:center;">
            <div style="flex:1;text-align:center;">
                <div class="section-label">
                    <div class="section-dot"></div>
                    <span class="section-title">Unggah Gambar Dulu</span>
                    <span class="section-sub">JPG / JPEG / PNG</span>
                </div>
            </div>
            <div style="flex:1;text-align:center;">
                <div class="section-label">
                    <div class="section-dot"></div>
                    <span class="section-title">Unggah Gambar Sekarang</span>
                    <span class="section-sub">JPG / JPEG / PNG</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)
    with col1:
        uploaded_file_wajah_dulu = st.file_uploader(
            label="dulu", type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
            help="Format yang didukung: JPG, JPEG, PNG"
        )
    with col2:
        uploaded_file_wajah_sekarang = st.file_uploader(
            label="sekarang", type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
            help="Format yang didukung: JPG, JPEG, PNG"
        )

    col_dulu, col_sekarang = st.columns(2)
    with col_dulu:
        if uploaded_file_wajah_dulu:
            image = Image.open(uploaded_file_wajah_dulu)
            st.markdown("<hr>", unsafe_allow_html=True)
        
            st.markdown(
                f'<div class="section-label">'
                f'<div class="section-dot"></div>'
                f'<span class="section-title">Kompresi Gambar via PCA (Sekarang)</span>'
                f'<span class="section-sub">GRAY</span>'
                f'</div>',
                unsafe_allow_html=True
            )
        
            col1, col2 = st.columns(2, gap="large")
            with col1:
                st.markdown('<span class="img-label">↳ Gambar Asli</span>', unsafe_allow_html=True)
                st.image(image, use_container_width=True)
        
            with st.spinner("Mengompresi gambar dengan PCA..."):
                compressed = compress_image_pca_gray(image, komponen_dulu_sekarang)
                compressed_temp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                compressed.save(compressed_temp.name)

            with col2:
                st.markdown('<span class="img-label">↳ Hasil Rekonstruksi PCA</span>', unsafe_allow_html=True)
                st.image(compressed, use_container_width=True)
        
            size_before = len(uploaded_file_wajah_dulu.getvalue())
            temp_buffer = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            compressed.save(temp_buffer.name)
            size_after = os.path.getsize(temp_buffer.name)
            reduction = ((size_before - size_after) / size_before) * 100
            reduction_color = "green" if reduction > 0 else "accent"
        
            st.markdown(
                f'<div class="metric-row">'
                f'<div class="metric-tile" style="text-align:center;">'
                f'<div class="metric-label">Ukuran Asli</div>'
                f'<div class="metric-value">{size_before/1024:.1f} KB</div>'
                f'</div>'
                f'<div class="metric-tile" style="text-align:center;">'
                f'<div class="metric-label">Setelah Kompresi</div>'
                f'<div class="metric-value accent">{size_after/1024:.1f} KB</div>'
                f'</div>'
                f'<div class="metric-tile" style="text-align:center;">'
                f'<div class="metric-label">Rasio Pengurangan</div>'
                f'<div class="metric-value {reduction_color}">{abs(reduction):.1f}%</div>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True
            )

    with col_sekarang:
        if uploaded_file_wajah_sekarang:
            image = Image.open(uploaded_file_wajah_sekarang)
            st.markdown("<hr>", unsafe_allow_html=True)

            st.markdown(
                f'<div class="section-label">'
                f'<div class="section-dot"></div>'
                f'<span class="section-title">Kompresi Gambar via PCA (Sekarang)</span>'
                f'<span class="section-sub">GRAY</span>'
                f'</div>',
                unsafe_allow_html=True
            )

            col1, col2 = st.columns(2, gap="large")
            with col1:
                st.markdown('<span class="img-label">↳ Gambar Asli</span>', unsafe_allow_html=True)
                st.image(image, use_container_width=True)
        
            with st.spinner("Mengompresi gambar dengan PCA..."):
                compressed = compress_image_pca_gray(image, komponen_dulu_sekarang)
                compressed_temp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                compressed.save(compressed_temp.name)
        
            with col2:
                st.markdown('<span class="img-label">↳ Hasil Rekonstruksi PCA</span>', unsafe_allow_html=True)
                st.image(compressed, use_container_width=True)
        
            size_before = len(uploaded_file_wajah_sekarang.getvalue())
            temp_buffer = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            compressed.save(temp_buffer.name)
            size_after = os.path.getsize(temp_buffer.name)
            reduction = ((size_before - size_after) / size_before) * 100
            reduction_color = "green" if reduction > 0 else "accent"
        
            st.markdown(
                f'<div class="metric-row">'
                f'<div class="metric-tile" style="text-align:center;">'
                f'<div class="metric-label">Ukuran Asli</div>'
                f'<div class="metric-value">{size_before/1024:.1f} KB</div>'
                f'</div>'
                f'<div class="metric-tile" style="text-align:center;">'
                f'<div class="metric-label">Setelah Kompresi</div>'
                f'<div class="metric-value accent">{size_after/1024:.1f} KB</div>'
                f'</div>'
                f'<div class="metric-tile" style="text-align:center;">'
                f'<div class="metric-label">Rasio Pengurangan</div>'
                f'<div class="metric-value {reduction_color}">{abs(reduction):.1f}%</div>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True
            )
            
    # ====================================================
    # PERBANDINGAN FOTO DULU VS SEKARANG
    # ====================================================

    if uploaded_file_wajah_dulu and uploaded_file_wajah_sekarang:
        with st.spinner("Membangun model PCA..."):

            X, labels, filenames = load_dataset(
                DATASET_PATH,
                IMG_SIZE
            )

            X_centered, mean_face = center_data(X)

            eigenfaces = compute_pca_svd(
                X_centered,
                komponen_dulu_sekarang
            )

            database_features = project_faces(
                X_centered,
                eigenfaces
            )

        st.markdown("<hr>", unsafe_allow_html=True)

        st.subheader("Hasil Perbandingan Wajah")

        temp_dulu = tempfile.NamedTemporaryFile(
            suffix=".jpg",
            delete=False
        )

        temp_dulu.write(
            uploaded_file_wajah_dulu.getvalue()
        )

        temp_dulu.close()

        temp_sekarang = tempfile.NamedTemporaryFile(
            suffix=".jpg",
            delete=False
        )

        temp_sekarang.write(
            uploaded_file_wajah_sekarang.getvalue()
        )

        temp_sekarang.close()

        # Dataset PCA dibangun sekali
        X, labels, filenames = load_dataset(
            DATASET_PATH,
            IMG_SIZE
        )

        X_centered, mean_face = center_data(X)

        eigenfaces = compute_pca_svd(
            X_centered,
            komponen_dulu_sekarang
        )

        feature_dulu = extract_feature(
            temp_dulu.name,
            mean_face,
            eigenfaces,
            IMG_SIZE
        )

        feature_sekarang = extract_feature(
            temp_sekarang.name,
            mean_face,
            eigenfaces,
            IMG_SIZE
        )

        similarity = cosine_similarity(
            feature_dulu.reshape(1, -1),
            feature_sekarang.reshape(1, -1)
        )[0][0]

        st.metric(
            "Kemiripan Wajah",
            f"{similarity * 100:.2f}%"
        )

        if similarity >= 0.90:
            st.success(
                "Sangat mirip (kemungkinan orang yang sama)"
            )

        elif similarity >= 0.75:
            st.info(
                "Mirip"
            )

        elif similarity >= 0.60:
            st.warning(
                "Cukup mirip"
            )

        else:
            st.error(
                "Kurang mirip"
            )
