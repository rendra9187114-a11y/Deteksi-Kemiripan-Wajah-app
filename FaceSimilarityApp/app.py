import streamlit as st
import cv2
import numpy as np
import os
import tempfile

from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# KONFIGURASI
# =========================

IMG_SIZE = (100, 100)
JUMLAH_KOMPONEN = 75

# =========================
# PCA KOMPRESI
# =========================

def compress_image_pca(image, n_components=50):

    img_gray = image.convert("L")

    matrix = np.array(img_gray).astype(float)

    matrix_norm = matrix / 255.0

    mean = np.mean(matrix_norm, axis=0)

    centered = matrix_norm - mean

    cov = np.cov(centered, rowvar=False)

    eigvals, eigvecs = np.linalg.eigh(cov)

    idx = np.argsort(eigvals)[::-1]

    eigvecs = eigvecs[:, idx]

    components = eigvecs[:, :n_components]

    projected = np.dot(centered, components)

    reconstructed = np.dot(projected, components.T) + mean

    reconstructed = np.clip(
        reconstructed * 255,
        0,
        255
    ).astype(np.uint8)

    return Image.fromarray(reconstructed)

# =========================
# PREPROCESS
# =========================

def preprocess_image(path):

    img = cv2.imread(path)

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.resize(gray, IMG_SIZE)

    return gray.flatten()

# =========================
# LOAD DATASET
# =========================

@st.cache_resource
def load_dataset(dataset_path):

    data = []
    labels = []
    filenames = []

    for person_name in os.listdir(dataset_path):

        person_folder = os.path.join(
            dataset_path,
            person_name
        )

        if not os.path.isdir(person_folder):
            continue

        for file in os.listdir(person_folder):

            if file.lower().endswith(
                (".jpg", ".jpeg", ".png")
            ):

                path = os.path.join(
                    person_folder,
                    file
                )

                vector = preprocess_image(path)

                data.append(vector)

                labels.append(person_name)

                filenames.append(file)

    return np.array(data), labels, filenames

# =========================
# PCA SVD
# =========================

def center_data(X):

    mean_face = np.mean(X, axis=0)

    return X - mean_face, mean_face


def compute_pca_svd(X_centered,
                    num_components=50): 

    U, S, VT = np.linalg.svd(
        X_centered,
        full_matrices=False
    )

    return VT[:num_components]


def project_faces(X_centered,
                  eigenfaces):

    return np.dot(
        X_centered,
        eigenfaces.T
    )

# =========================
# FEATURE TEST
# =========================

def extract_feature(
        image_path,
        mean_face,
        eigenfaces
):

    vector = preprocess_image(image_path)

    centered = vector - mean_face

    return np.dot(
        centered,
        eigenfaces.T
    )

# =========================
# TOP K COSINE
# =========================

def recognize_cosine_topk(
        test_feature,
        database_features,
        labels,
        filenames,
        k=3
):

    results = []

    test_feature = test_feature.reshape(1, -1)

    for feature, label, file in zip(
            database_features,
            labels,
            filenames
    ):

        feature = feature.reshape(1, -1)

        score = cosine_similarity(
            test_feature,
            feature
        )[0][0]

        results.append(
            (
                label,
                file,
                score
            )
        )

    results.sort(
        key=lambda x: x[2],
        reverse=True
    )

    return results[:k]

# =========================
# UI
# =========================

st.set_page_config(
    page_title="Deteksi Kemiripan Wajah",
    layout="wide"
)

st.title(
    "Deteksi Kemiripan Wajah Menggunakan PCA-SVD"
)

DATASET_PATH = "FaceSimilarityApp/dataset"

if not os.path.exists(DATASET_PATH):

    st.error(
        "Folder dataset tidak ditemukan."
    )

    st.stop()

uploaded_file = st.file_uploader(
    "Upload gambar wajah",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Gambar Asli")
        st.image(
            image,
            use_container_width=True
        )

    compressed = compress_image_pca(
        image,
        JUMLAH_KOMPONEN
    )

    with col2:
        st.subheader("Hasil Kompresi")
        st.image(
            compressed,
            use_container_width=True
        )

    # ukuran file
    size_before = len(
        uploaded_file.getvalue()
    )

    temp_buffer = tempfile.NamedTemporaryFile(
        suffix=".jpg",
        delete=False
    )

    compressed.save(
        temp_buffer.name
    )

    size_after = os.path.getsize(
        temp_buffer.name
    )

    reduction = (
        (size_before - size_after)
        / size_before
    ) * 100

    st.subheader("Informasi Kompresi")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Ukuran Awal",
        f"{size_before/1024:.2f} KB"
    )

    c2.metric(
        "Ukuran Kompresi",
        f"{size_after/1024:.2f} KB"
    )

    c3.metric(
        "Pengurangan",
        f"{reduction:.2f}%"
    )

    st.divider()

    st.subheader(
        "Deteksi Kemiripan Wajah"
    )

    X, labels, filenames = load_dataset(
        DATASET_PATH
    )

    X_centered, mean_face = center_data(X)

    eigenfaces = compute_pca_svd(
        X_centered,
        JUMLAH_KOMPONEN
    )

    database_features = project_faces(
        X_centered,
        eigenfaces
    )

    test_feature = extract_feature(
        temp_buffer.name,
        mean_face,
        eigenfaces
    )

    results = recognize_cosine_topk(
        test_feature,
        database_features,
        labels,
        filenames,
        k=3
    )

    st.subheader(
        "Top 3 Kemiripan"
    )

    for rank, (
        label,
        file,
        score
    ) in enumerate(
        results,
        start=1
    ):

        st.write(
            f"{rank}. {file} "
            f"({label}) "
            f"- Similarity: "
            f"{score:.4f}"
        )

    best_label, best_file, best_score = results[0]

    best_path = os.path.join(
        DATASET_PATH,
        best_label,
        best_file
    )

    st.success(
        f"Hasil Terbaik: "
        f"{best_label} "
        f"({best_score:.4f})"
    )

    st.image(
        best_path,
        caption=f"{best_file}",
        width=300
    )

    with open(
        temp_buffer.name,
        "rb"
    ) as f:

        st.download_button(
            "Download Hasil Kompresi",
            f,
            file_name=f"kompresi_{uploaded_file.name}"
        )
