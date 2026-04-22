#app.py — Skin Lesion Clinical Decision Support Tool
#XGBoost + Original Feature Extraction Pipeline

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import cv2
import math
import shap
import matplotlib.pyplot as plt
from PIL import Image
from skimage.color import rgb2gray
import mahotas as mh

st.set_page_config(
    page_title="Skin Lesion Clinical Decision Support",
    page_icon="🔬",
    layout="wide"
)

st.markdown("""
    <style>
    .stAlert { display: none; }
    .stApp { background-color: #f0f6ff; color: #1a1a2e; }
    .stSidebar { background-color: #1a3a6b; }
    .stSidebar * { color: #ffffff !important; }
    h1, h2, h3, h4, h5, h6 { color: #1a3a6b !important; }
    p { color: #1a1a2e; }

    [data-testid="stFileUploader"] {
        background-color: #dbeafe;
        border: 2px dashed #1a3a6b;
        border-radius: 12px;
        padding: 10px;
    }
    [data-testid="stFileUploader"] * { color: #1a3a6b !important; }

    .stat-card {
        background-color: #1a3a6b;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        color: #ffffff;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .stat-card h2 { color: #ffffff !important; font-size: 2.2rem; margin: 0; }
    .stat-card p { color: #cce0ff !important; margin: 5px 0 0 0; font-size: 0.95rem; }

    .result-card {
        background-color: #dbeafe;
        padding: 20px;
        border-radius: 12px;
        border-left: 6px solid #1a3a6b;
        margin-bottom: 10px;
    }
    .result-card h3 { color: #1a3a6b !important; margin: 0 0 8px 0; font-size: 1.6rem; }
    .result-card p { color: #1a1a2e; margin: 4px 0; }

    .clinical-note {
        background-color: #dbeafe;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #1a3a6b;
        color: #1a1a2e;
        margin: 8px 0;
        font-size: 0.95rem;
        line-height: 1.6;
    }

    .risk-factor-card {
        background-color: #dbeafe;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #1a3a6b;
        color: #1a1a2e;
        margin: 8px 0;
    }

    .combined-risk-high {
        background-color: #fff0f0;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #c0392b;
        color: #1a1a2e;
        margin: 8px 0;
        font-size: 0.95rem;
    }

    .combined-risk-moderate {
        background-color: #fff8e1;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #f39c12;
        color: #1a1a2e;
        margin: 8px 0;
        font-size: 0.95rem;
    }

    .combined-risk-low {
        background-color: #f0fff4;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #27ae60;
        color: #1a1a2e;
        margin: 8px 0;
        font-size: 0.95rem;
    }

    .abcd-card {
        background-color: #dbeafe;
        padding: 15px;
        border-radius: 10px;
        border-top: 5px solid #1a3a6b;
        text-align: center;
        color: #1a1a2e;
    }
    .abcd-card h4 { color: #1a3a6b !important; margin: 0 0 8px 0; }
    .abcd-card h2 { color: #1a3a6b !important; margin: 5px 0; font-size: 1.6rem; }
    .abcd-card p { color: #1a1a2e; margin: 0; font-size: 0.9rem; }

    .abcd-summary {
        background-color: #dbeafe;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #1a3a6b;
        color: #1a1a2e;
        margin-top: 12px;
        font-size: 0.95rem;
    }

    .diff-card {
        background-color: #dbeafe;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #1a3a6b;
        color: #1a1a2e;
    }
    .diff-card h4 { color: #1a3a6b !important; margin: 0 0 5px 0; }
    .diff-card p { margin: 3px 0; color: #1a1a2e; }

    .confidence-note {
        background-color: #dbeafe;
        padding: 10px 15px;
        border-radius: 8px;
        border-left: 4px solid #1a3a6b;
        color: #1a1a2e;
        font-size: 0.85rem;
        margin-top: 8px;
    }

    .low-confidence-warning {
        background-color: #fff0f0;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #c0392b;
        color: #1a1a2e;
        font-size: 0.95rem;
        margin-top: 8px;
    }

    .disclaimer {
        background-color: #fff8e1;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #f39c12;
        color: #1a1a2e;
        font-size: 0.9rem;
    }

    [data-testid="stDataFrame"] table { background-color: #dbeafe !important; color: #1a1a2e !important; }
    [data-testid="stDataFrame"] th { background-color: #1a3a6b !important; color: #ffffff !important; font-weight: bold; }
    [data-testid="stDataFrame"] td { background-color: #dbeafe !important; color: #1a1a2e !important; }
    [data-testid="stDataFrame"] tr:nth-child(even) td { background-color: #bfdbfe !important; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load("xgb_best_model.pkl")

model = load_model()

CLASS_NAMES = [
    "pigmented benign keratosis", "melanoma", "basal cell carcinoma",
    "nevus", "squamous cell carcinoma", "vascular lesion",
    "actinic keratosis", "dermatofibroma", "seborrheic keratosis"
]

MALIGNANT = ["melanoma", "basal cell carcinoma", "squamous cell carcinoma", "actinic keratosis"]

RECOMMENDED_ACTION = {
    "melanoma": "🔴 Urgent biopsy recommended. Refer to oncology immediately.",
    "basal cell carcinoma": "🟠 Surgical excision recommended. Schedule dermatology follow-up.",
    "squamous cell carcinoma": "🔴 Biopsy and excision recommended. Urgent dermatology referral.",
    "actinic keratosis": "🟠 Cryotherapy or topical treatment recommended. Monitor closely.",
    "nevus": "🟢 Routine monitoring recommended. Re-evaluate if changes occur.",
    "pigmented benign keratosis": "🟢 No immediate action required. Routine skin check advised.",
    "vascular lesion": "🟡 Clinical evaluation recommended to determine treatment need.",
    "dermatofibroma": "🟢 No treatment required unless symptomatic. Monitor for changes.",
    "seborrheic keratosis": "🟢 No treatment required. Reassure patient of benign nature."
}

CLINICAL_NOTES = {
    "melanoma": "Melanoma is the most serious form of skin cancer originating from melanocytes. Key warning signs include asymmetry, irregular borders, color variation, diameter greater than 6mm, and evolution over time. Metastatic potential is high if not detected early.",
    "basal cell carcinoma": "The most common skin cancer, originating from basal cells of the epidermis. Typically presents as a pearly or waxy bump. Rarely metastasizes but can cause significant local tissue destruction if untreated. Most commonly found on sun-exposed areas.",
    "squamous cell carcinoma": "Originates from squamous cells in the outer layer of skin. Can develop from actinic keratosis. Has metastatic potential, particularly in immunocompromised patients. Presents as a firm red nodule or flat lesion with a scaly crust.",
    "actinic keratosis": "A precancerous condition caused by cumulative UV radiation exposure. Appears as rough, scaly patches. Approximately 5-10% may progress to squamous cell carcinoma if untreated. Early intervention is key to preventing malignant transformation.",
    "nevus": "A common benign melanocytic lesion. Most nevi are harmless but require monitoring for signs of dysplasia. Atypical nevi with irregular features may warrant excision and histological examination.",
    "pigmented benign keratosis": "A benign epidermal lesion with a characteristic stuck-on appearance. Common in older adults. No malignant potential but may be mistaken for melanoma. Dermoscopy can help differentiate from malignant lesions.",
    "vascular lesion": "Includes a spectrum of benign vascular abnormalities characterized by red to purple coloration due to blood vessel prominence. Most require no treatment but clinical evaluation is recommended to rule out rare malignant variants.",
    "dermatofibroma": "A common benign fibrous nodule typically found on the lower extremities. Characterized by a positive dimple sign on lateral compression. No malignant potential. Treatment is only required if symptomatic or cosmetically concerning.",
    "seborrheic keratosis": "A very common benign epidermal growth with a waxy stuck-on appearance that increases in frequency with age. No malignant potential. May be mistaken for melanoma — dermoscopy aids in differentiation."
}

FEATURE_MAP = {
    "avg_r": "Average redness of the lesion — higher values indicate a redder lesion surface",
    "avg_g": "Average green tone of the lesion surface",
    "avg_b": "Average blue tone of the lesion surface",
    "hist_feature_0": "Proportion of very dark red pixels in the lesion",
    "hist_feature_1": "Proportion of dark red pixels in the lesion",
    "hist_feature_2": "Proportion of dark red pixels in the lesion",
    "hist_feature_3": "Proportion of low-intensity red pixels",
    "hist_feature_4": "Proportion of low-intensity red pixels",
    "hist_feature_5": "Proportion of low-to-mid intensity red pixels",
    "hist_feature_6": "Proportion of very dark green pixels in the lesion",
    "hist_feature_7": "Proportion of dark green pixels in the lesion",
    "hist_feature_8": "Proportion of mid-intensity green pixels",
    "hist_feature_9": "Proportion of mid-intensity green pixels",
    "hist_feature_10": "Proportion of mid-intensity green pixels",
    "hist_feature_11": "Proportion of mid-intensity green pixels",
    "hist_feature_12": "Proportion of mid-to-bright green pixels",
    "hist_feature_13": "Proportion of mid-to-bright green pixels",
    "hist_feature_14": "Proportion of mid-to-bright green pixels",
    "hist_feature_15": "Proportion of bright green pixels",
    "hist_feature_16": "Proportion of bright green pixels",
    "hist_feature_17": "Proportion of bright green pixels",
    "hist_feature_18": "Proportion of very bright green pixels",
    "hist_feature_19": "Proportion of very bright green pixels",
    "hist_feature_20": "Proportion of very bright green pixels in the lesion",
    "hist_feature_21": "Proportion of very bright green pixels",
    "hist_feature_22": "Proportion of extremely bright green pixels",
    "hist_feature_23": "Proportion of extremely bright green pixels",
    "hist_feature_24": "Proportion of extremely bright green pixels",
    "hist_feature_25": "Proportion of near-white green pixels",
    "hist_feature_26": "Proportion of near-white green pixels",
    "hist_feature_27": "Proportion of near-white green pixels",
    "hist_feature_28": "Proportion of near-white green pixels",
    "hist_feature_29": "Proportion of near-white green pixels",
    "hist_feature_30": "Proportion of near-white green pixels",
    "hist_feature_31": "Proportion of near-white green pixels",
    "hist_feature_32": "Proportion of very dark blue pixels in the lesion",
    "hist_feature_33": "Proportion of dark blue pixels",
    "hist_feature_34": "Proportion of mid-intensity red pixels in the lesion",
    "hist_feature_35": "Proportion of mid-to-high intensity red pixels",
    "hist_feature_36": "Proportion of high intensity red pixels",
    "hist_feature_37": "Proportion of bright red pixels — may indicate inflammation or vascularity",
    "hist_feature_38": "Proportion of very bright red pixels — strongly associated with vascular lesions",
    "hist_feature_39": "Proportion of near-white red pixels",
    "hist_feature_40": "Proportion of near-white red pixels",
    "hist_feature_41": "Proportion of near-white red pixels",
    "hist_feature_42": "Proportion of mid-intensity blue pixels in the lesion",
    "hist_feature_43": "Proportion of mid-intensity blue pixels",
    "hist_feature_44": "Proportion of mid-intensity blue pixels",
    "hist_feature_45": "Proportion of mid-to-high blue pixels",
    "hist_feature_46": "Proportion of high intensity blue pixels",
    "hist_feature_47": "Proportion of bright blue pixels",
    "hist_feature_48": "Proportion of very bright blue pixels",
    "hist_feature_49": "Proportion of very bright blue pixels",
    "hist_feature_50": "Proportion of very bright blue pixels",
    "hist_feature_51": "Proportion of near-white blue pixels",
    "hist_feature_52": "Proportion of near-white blue pixels",
    "hist_feature_53": "Proportion of near-white blue pixels",
    "hist_feature_54": "Proportion of near-white blue pixels",
    "hist_feature_55": "Proportion of near-white blue pixels",
    "hist_feature_56": "Proportion of near-white blue pixels",
    "hist_feature_57": "Proportion of near-white blue pixels",
    "hist_feature_58": "Proportion of near-white blue pixels",
    "hist_feature_59": "Proportion of near-white blue pixels",
    "hist_feature_60": "Proportion of near-white blue pixels",
    "hist_feature_61": "Proportion of near-white blue pixels",
    "hist_feature_62": "Proportion of near-white blue pixels",
    "hist_feature_63": "Proportion of near-white blue pixels",
    "hist_feature_64": "Proportion of very dark blue pixels",
    "hist_feature_65": "Proportion of dark blue pixels in the lesion",
    "hist_feature_66": "Proportion of low-intensity blue pixels",
    "hist_feature_67": "Proportion of low-intensity blue pixels",
    "hist_feature_68": "Proportion of low-to-mid blue pixels",
    "hist_feature_69": "Proportion of mid blue pixels",
    "hist_feature_70": "Proportion of mid blue pixels",
    "hist_feature_71": "Proportion of mid blue pixels",
    "hist_feature_72": "Proportion of mid blue pixels",
    "hist_feature_73": "Proportion of mid-to-high blue pixels",
    "hist_feature_74": "Proportion of high blue pixels",
    "hist_feature_75": "Proportion of bright blue pixels",
    "hist_feature_76": "Proportion of bright blue pixels",
    "hist_feature_77": "Proportion of very bright blue pixels",
    "hist_feature_78": "Proportion of very bright blue pixels",
    "hist_feature_79": "Proportion of very bright blue pixels",
    "hist_feature_80": "Proportion of near-white blue pixels",
    "hist_feature_81": "Proportion of near-white blue pixels",
    "hist_feature_82": "Proportion of near-white blue pixels",
    "hist_feature_83": "Proportion of near-white blue pixels",
    "hist_feature_84": "Proportion of near-white blue pixels",
    "hist_feature_85": "Proportion of near-white blue pixels",
    "hist_feature_86": "Proportion of near-white blue pixels",
    "hist_feature_87": "Proportion of near-white blue pixels",
    "hist_feature_88": "Proportion of near-white blue pixels",
    "hist_feature_89": "Proportion of near-white blue pixels",
    "hist_feature_90": "Proportion of near-white blue pixels",
    "hist_feature_91": "Proportion of very bright blue pixels in the lesion",
    "hist_feature_92": "Proportion of near-white blue pixels",
    "hist_feature_93": "Proportion of near-white blue pixels",
    "hist_feature_94": "Proportion of near-white blue pixels",
    "hist_feature_95": "Proportion of near-white blue pixels",
    "haralick_feature_0": "Surface texture contrast — how sharply color and brightness change across the lesion",
    "haralick_feature_1": "Surface texture correlation — how regularly patterns repeat across the lesion surface",
    "haralick_feature_2": "Surface texture energy — how smooth and uniform the lesion surface appears",
    "haralick_feature_3": "Surface texture homogeneity — how consistent the lesion surface texture is overall",
    "haralick_feature_4": "Surface texture complexity — how irregular and varied the lesion surface pattern is",
    "haralick_feature_5": "Surface texture variation — how much the surface brightness varies across the lesion",
    "haralick_feature_6": "Average intensity of neighboring areas on the lesion surface",
    "haralick_feature_7": "Variation in intensity between neighboring areas on the lesion surface",
    "haralick_feature_8": "Complexity of intensity patterns between neighboring lesion areas",
    "haralick_feature_9": "Variation in brightness differences between adjacent lesion areas",
    "haralick_feature_10": "Complexity of brightness differences between adjacent lesion areas",
    "haralick_feature_11": "Statistical correlation between surface patterns across the lesion",
    "haralick_feature_12": "Alternative measure of surface pattern correlation across the lesion",
    "canny_r": "Sharpness of lesion edges in the red channel — higher values indicate more defined borders",
    "canny_g": "Sharpness of lesion edges in the green channel — helps assess border irregularity",
    "canny_b": "Sharpness of lesion edges in the blue channel — helps assess border irregularity",
    "gaussian_r": "Smoothed red color intensity — reduces noise to better capture underlying color patterns",
    "gaussian_g": "Smoothed green color intensity — reduces noise to better capture underlying color patterns",
    "gaussian_b": "Smoothed blue color intensity — reduces noise to better capture underlying color patterns",
    "laplacian_r": "Rate of color change at lesion borders in the red channel — captures border sharpness",
    "laplacian_g": "Rate of color change at lesion borders in the green channel — captures border sharpness",
    "laplacian_b": "Rate of color change at lesion borders in the blue channel — captures border sharpness",
    "sobel_r": "Strength of the lesion boundary in the red channel — measures how pronounced the border is",
    "sobel_g": "Strength of the lesion boundary in the green channel — measures border prominence",
    "sobel_b": "Strength of the lesion boundary in the blue channel — measures border prominence",
    "min_enc_circle_radius": "Lesion size — radius of the smallest circle that fully encloses the lesion",
    "min_enc_circle_area": "Lesion area — total area of the smallest circle enclosing the lesion",
    "min_area_rect_width": "Lesion width measured from its bounding rectangle",
    "min_area_rect_height": "Lesion height measured from its bounding rectangle",
    "min_area_rect_aspect_ratio": "Lesion shape ratio — compares width to height to assess elongation or asymmetry",
    "min_area_rect_angle": "Orientation angle of the lesion within the image frame",
    "ratio_rg": "Ratio of red to green color — high values suggest redness associated with inflammation or vascularity",
    "ratio_rb": "Ratio of red to blue color — elevated values may indicate vascular or inflammatory lesions",
    "ratio_gb": "Ratio of green to blue color across the lesion surface",
    "contrast_rg": "Difference between red and green color intensity — reflects color variation in the lesion",
    "contrast_rb": "Difference between red and blue intensity — strong indicator of lesion redness or vascularity",
    "contrast_gb": "Difference between green and blue intensity across the lesion surface",
    "brightness": "Overall lightness of the lesion — darker lesions may indicate higher pigmentation",
    "color_range": "Spread of colors across the lesion — wide range may indicate multiple color zones as seen in melanoma",
    "color_std": "Consistency of color across the lesion — low consistency may suggest irregular pigmentation",
    "asymmetry_ratio": "Lesion asymmetry — values far from 1.0 indicate an uneven shape, a key melanoma warning sign",
    "border_irregularity": "Lesion border irregularity — higher values indicate a more poorly defined or uneven border",
    "diameter_estimate": "Estimated lesion size — larger lesions may carry higher clinical significance"
}

num_bins = 32

def extract_features(image):
    img_pil = image.convert("RGB")
    img_array = np.array(img_pil)
    avg_r = np.mean(img_array[:, :, 0])
    avg_g = np.mean(img_array[:, :, 1])
    avg_b = np.mean(img_array[:, :, 2])
    img_cv2 = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    hist_b = cv2.calcHist([img_cv2], [0], None, [num_bins], [0, 256])
    hist_g = cv2.calcHist([img_cv2], [1], None, [num_bins], [0, 256])
    hist_r = cv2.calcHist([img_cv2], [2], None, [num_bins], [0, 256])
    histogram_features = np.concatenate((hist_b, hist_g, hist_r)).flatten()
    gray_img = rgb2gray(img_array)
    gray_img_uint8 = (gray_img * 255).astype(np.uint8)
    haralick_features = mh.features.haralick(gray_img_uint8).mean(0)
    canny_features_rgb, gaussian_features_rgb, laplacian_features_rgb, sobel_features_rgb = [], [], [], []
    for i in range(3):
        channel = img_array[:, :, i].astype(np.uint8)
        canny_features_rgb.append(np.mean(cv2.Canny(channel, 100, 200)))
        gaussian_features_rgb.append(np.mean(cv2.GaussianBlur(channel, (5, 5), 0)))
        laplacian_features_rgb.append(np.mean(np.abs(cv2.Laplacian(channel, cv2.CV_64F))))
        sobelx = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=3)
        sobel_features_rgb.append(np.mean(np.sqrt(sobelx**2 + sobely**2)))
    contours, _ = cv2.findContours(gray_img_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_enc_circle_radius = min_enc_circle_area = min_area_rect_width = 0.0
    min_area_rect_height = min_area_rect_aspect_ratio = min_area_rect_angle = 0.0
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        (_, _), radius = cv2.minEnclosingCircle(largest_contour)
        min_enc_circle_radius = radius
        min_enc_circle_area = math.pi * (radius ** 2)
        rect = cv2.minAreaRect(largest_contour)
        min_area_rect_width = rect[1][0]
        min_area_rect_height = rect[1][1]
        min_area_rect_angle = rect[2]
        if min_area_rect_height > 0:
            min_area_rect_aspect_ratio = min_area_rect_width / min_area_rect_height
    return (
        [avg_r, avg_g, avg_b] +
        histogram_features.tolist() +
        haralick_features.tolist() +
        canny_features_rgb + gaussian_features_rgb +
        laplacian_features_rgb + sobel_features_rgb +
        [min_enc_circle_radius, min_enc_circle_area,
         min_area_rect_width, min_area_rect_height,
         min_area_rect_aspect_ratio, min_area_rect_angle]
    )

def engineer_features(df):
    df["ratio_rg"] = df["avg_r"] / (df["avg_g"] + 1e-6)
    df["ratio_rb"] = df["avg_r"] / (df["avg_b"] + 1e-6)
    df["ratio_gb"] = df["avg_g"] / (df["avg_b"] + 1e-6)
    df["contrast_rg"] = df["avg_r"] - df["avg_g"]
    df["contrast_rb"] = df["avg_r"] - df["avg_b"]
    df["contrast_gb"] = df["avg_g"] - df["avg_b"]
    df["brightness"] = (df["avg_r"] + df["avg_g"] + df["avg_b"]) / 3
    df["color_range"] = df[["avg_r","avg_g","avg_b"]].max(axis=1) - df[["avg_r","avg_g","avg_b"]].min(axis=1)
    df["color_std"] = df[["avg_r","avg_g","avg_b"]].std(axis=1)
    df["asymmetry_ratio"] = df["min_area_rect_width"] / (df["min_area_rect_height"] + 1e-6)
    df["border_irregularity"] = df["min_enc_circle_area"] / (df["min_area_rect_width"] * df["min_area_rect_height"] + 1e-6)
    df["diameter_estimate"] = (df["min_area_rect_width"] + df["min_area_rect_height"]) / 2
    return df

ORIGINAL_COLS = (
    ["avg_r", "avg_g", "avg_b"] +
    [f"hist_feature_{i}" for i in range(96)] +
    [f"haralick_feature_{i}" for i in range(13)] +
    ["canny_r", "canny_g", "canny_b"] +
    ["gaussian_r", "gaussian_g", "gaussian_b"] +
    ["laplacian_r", "laplacian_g", "laplacian_b"] +
    ["sobel_r", "sobel_g", "sobel_b"] +
    ["min_enc_circle_radius", "min_enc_circle_area",
     "min_area_rect_width", "min_area_rect_height",
     "min_area_rect_aspect_ratio", "min_area_rect_angle"]
)

#sidebar
with st.sidebar:
    st.image("uta_logo.png", use_container_width=True)
    st.divider()
    st.markdown("### About This Tool")
    st.markdown("This clinical decision support tool uses a machine learning model trained on dermoscopic images to assist in the preliminary classification of skin lesions.")
    st.divider()
    st.markdown("### How To Use")
    st.markdown("""
    1. Upload a dermoscopic image
    2. Enter patient risk factors
    3. Review the classification result
    4. Review the ABCD analysis
    5. Review key visual indicators
    6. Use findings to support clinical judgment
    """)
    st.divider()
    st.markdown("**Developed by:**")
    st.markdown("Sidhantaa Sarna")
    st.markdown("Tiffany De La Cruz")
    st.markdown("Diego Maldonado")
    st.markdown("**Faculty Mentor:** Dr. Masoud Rostami")
    st.markdown("DATA 4382 — University of Texas at Arlington")

#main
st.title("🔬 Skin Lesion Clinical Decision Support")
st.markdown("##### An AI-assisted dermoscopic image analysis tool for preliminary skin lesion classification")
st.markdown("##### Supporting clinical decision-making through explainable AI")

s1, s2, s3 = st.columns(3)
with s1:
    st.markdown('<div class="stat-card"><h2>60.4%</h2><p>Diagnostic Accuracy</p></div>', unsafe_allow_html=True)
with s2:
    st.markdown('<div class="stat-card"><h2>0.884</h2><p>Predictive Confidence (ROC-AUC)</p></div>', unsafe_allow_html=True)
with s3:
    st.markdown('<div class="stat-card"><h2>9</h2><p>Lesion Classes</p></div>', unsafe_allow_html=True)

st.divider()

#patient risk factors
st.subheader("👤 Patient Risk Factors")
st.markdown("*Enter patient clinical history to supplement the image-based classification. These factors are not used in the model prediction but are used to generate a combined clinical risk assessment.*")

rf_col1, rf_col2, rf_col3 = st.columns(3)

with rf_col1:
    age_over_50 = st.checkbox("Age over 50")
    fair_skin = st.checkbox("Fair skin (Fitzpatrick type I–II)")
    family_history = st.checkbox("Family history of melanoma")

with rf_col2:
    sun_exposure = st.checkbox("History of chronic sun exposure")
    prev_skin_cancer = st.checkbox("Previous skin cancer diagnosis")
    immunocompromised = st.checkbox("Immunocompromised status")

with rf_col3:
    changing_lesion = st.checkbox("Lesion has changed recently")
    bleeding_lesion = st.checkbox("Lesion is bleeding or ulcerated")
    multiple_lesions = st.checkbox("Multiple atypical lesions present")

risk_factors = [
    age_over_50, fair_skin, family_history,
    sun_exposure, prev_skin_cancer, immunocompromised,
    changing_lesion, bleeding_lesion, multiple_lesions
]
risk_factor_names = [
    "Age over 50", "Fair skin (Fitzpatrick type I–II)", "Family history of melanoma",
    "History of chronic sun exposure", "Previous skin cancer diagnosis", "Immunocompromised status",
    "Lesion has changed recently", "Lesion is bleeding or ulcerated", "Multiple atypical lesions present"
]
active_risk_factors = [name for name, val in zip(risk_factor_names, risk_factors) if val]
risk_factor_count = len(active_risk_factors)

st.divider()

uploaded_file = st.file_uploader("📤 Upload a dermoscopic image to begin analysis", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Uploaded Dermoscopic Image", use_container_width=True)

    with col2:
        with st.spinner("Analyzing image..."):
            features = extract_features(image)
            df_features = pd.DataFrame([features], columns=ORIGINAL_COLS)
            df_features = engineer_features(df_features)

            prediction = model.predict(df_features)[0]
            probabilities = model.predict_proba(df_features)[0]
            predicted_class = CLASS_NAMES[prediction]
            confidence = round(probabilities[prediction] * 100, 1)

            if confidence >= 60:
                conf_label = "High"
                conf_color = "green"
            elif confidence >= 40:
                conf_label = "Moderate"
                conf_color = "orange"
            else:
                conf_label = "Low"
                conf_color = "red"

            is_malignant = predicted_class in MALIGNANT

            st.markdown(f"""
            <div class="result-card">
                <h3>{predicted_class.title()}</h3>
                <p><strong>Risk Category:</strong> {"🔴 Malignant" if is_malignant else "🟢 Benign"}</p>
                <p><strong>Model Certainty:</strong> {conf_label} — {confidence}%</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="confidence-note">
            <strong>Model Certainty Guide:</strong> High (≥60%) — strong signal detected &nbsp;|&nbsp;
            Moderate (40–60%) — some uncertainty, review carefully &nbsp;|&nbsp;
            Low (&lt;40%) — low confidence, clinical judgment essential
            </div>
            """, unsafe_allow_html=True)

            if confidence < 40:
                st.markdown("""
                <div class="low-confidence-warning">
                ⚠️ <strong>Low Model Certainty:</strong> The model is not confident in this classification.
                Do not rely on this result without thorough clinical examination. Consider biopsy or specialist referral.
                </div>
                """, unsafe_allow_html=True)

            if is_malignant:
                st.error("🔴 HIGH RISK — Malignant lesion detected. Immediate clinical review recommended.")
            else:
                st.success("🟢 LOW RISK — Benign lesion detected. Routine monitoring recommended.")

            st.markdown("**Recommended Clinical Action:**")
            st.markdown(f'<div class="clinical-note">{RECOMMENDED_ACTION.get(predicted_class, "Consult a dermatologist.")}</div>', unsafe_allow_html=True)

            st.markdown("**Clinical Context:**")
            st.markdown(f'<div class="clinical-note">{CLINICAL_NOTES.get(predicted_class, "No clinical note available.")}</div>', unsafe_allow_html=True)

    st.divider()

    #combined risk assessment
    st.subheader("⚕️ Combined Clinical Risk Assessment")
    st.markdown("*Integrates the model classification with patient risk factors to provide a holistic clinical picture.*")

    total_concern = risk_factor_count + (2 if is_malignant else 0) + (1 if confidence < 40 else 0)

    if active_risk_factors:
        risk_list = ", ".join(active_risk_factors)
    else:
        risk_list = "No additional risk factors selected"

    if total_concern >= 4:
        combined_class = "combined-risk-high"
        combined_icon = "🔴"
        combined_label = "High Overall Concern"
        combined_msg = f"Model classifies this as {predicted_class.title()} with {conf_label.lower()} confidence. Patient presents with {risk_factor_count} additional clinical risk factor(s): {risk_list}. Elevated overall concern — immediate clinical evaluation strongly recommended."
    elif total_concern >= 2:
        combined_class = "combined-risk-moderate"
        combined_icon = "🟠"
        combined_label = "Moderate Overall Concern"
        combined_msg = f"Model classifies this as {predicted_class.title()} with {conf_label.lower()} confidence. Patient presents with {risk_factor_count} additional clinical risk factor(s): {risk_list}. Moderate concern — close monitoring and clinical evaluation recommended."
    else:
        combined_class = "combined-risk-low"
        combined_icon = "🟢"
        combined_label = "Low Overall Concern"
        combined_msg = f"Model classifies this as {predicted_class.title()} with {conf_label.lower()} confidence. {risk_list}. Low overall concern — routine monitoring advised."

    st.markdown(f"""
    <div class="{combined_class}">
    <strong>{combined_icon} {combined_label}:</strong> {combined_msg}
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    #ABCD
    st.subheader("🔍 ABCD Analysis")
    st.markdown("*Computed from image features using the classical ABCD dermatological framework. Values are image-derived estimates and should be interpreted alongside clinical examination.*")

    asymmetry = df_features["asymmetry_ratio"].values[0]
    border = df_features["border_irregularity"].values[0]
    color = df_features["color_range"].values[0]
    diameter = df_features["diameter_estimate"].values[0]

    a1, a2, a3, a4 = st.columns(4)

    with a1:
        st.markdown(f"""
        <div class="abcd-card">
        <h4>A — Asymmetry</h4>
        <h2>{round(asymmetry, 2)}</h2>
        <p style="font-size:0.75rem; color:#64748b;">Scale: 0–2 | Closer to 1.0 = more symmetric</p>
        <p>{"⚠️ Asymmetric" if asymmetry > 1.2 or asymmetry < 0.8 else "✅ Symmetric"}</p>
        </div>""", unsafe_allow_html=True)

    with a2:
        st.markdown(f"""
        <div class="abcd-card">
        <h4>B — Border</h4>
        <h2>{round(border, 2)}</h2>
        <p style="font-size:0.75rem; color:#64748b;">Scale: 0–5+ | Higher = more irregular</p>
        <p>{"⚠️ Irregular" if border > 1.5 else "✅ Regular"}</p>
        </div>""", unsafe_allow_html=True)

    with a3:
        st.markdown(f"""
        <div class="abcd-card">
        <h4>C — Color</h4>
        <h2>{round(color, 1)}</h2>
        <p style="font-size:0.75rem; color:#64748b;">Scale: 0–255 | Higher = more color variation</p>
        <p>{"⚠️ High Variation" if color > 50 else "✅ Uniform"}</p>
        </div>""", unsafe_allow_html=True)

    with a4:
        st.markdown(f"""
        <div class="abcd-card">
        <h4>D — Diameter</h4>
        <h2>{round(diameter, 1)} px</h2>
        <p style="font-size:0.75rem; color:#64748b;">px = pixels | Higher = larger lesion</p>
        <p>{"⚠️ Large" if diameter > 200 else "✅ Small-Medium"}</p>
        </div>""", unsafe_allow_html=True)

    abcd_flags = sum([
        asymmetry > 1.2 or asymmetry < 0.8,
        border > 1.5,
        color > 50,
        diameter > 200
    ])

    if abcd_flags >= 3:
        abcd_risk = "🔴 High concern — Multiple warning signs detected. Clinical evaluation strongly recommended."
    elif abcd_flags == 2:
        abcd_risk = "🟠 Moderate concern — Some warning signs present. Consider biopsy and close monitoring."
    elif abcd_flags == 1:
        abcd_risk = "🟡 Low-moderate concern — One warning sign detected. Routine monitoring advised."
    else:
        abcd_risk = "🟢 Low concern — No major warning signs detected. Routine monitoring advised."

    st.markdown(f"""
    <div class="abcd-summary">
    <strong>ABCD Summary ({abcd_flags}/4 warning signs raised):</strong> {abcd_risk}
    </div>""", unsafe_allow_html=True)

    st.divider()

    #differential diagnosis
    st.subheader("📊 Differential Diagnosis")
    st.markdown("*Top 3 most likely classifications based on image analysis:*")

    top3 = np.argsort(probabilities)[::-1][:3]
    d1, d2, d3 = st.columns(3)

    for idx, col in zip(top3, [d1, d2, d3]):
        cname = CLASS_NAMES[idx]
        prob = round(probabilities[idx] * 100, 1)
        risk = "🔴 Malignant" if cname in MALIGNANT else "🟢 Benign"
        with col:
            st.markdown(f"""
            <div class="diff-card">
            <h4>{cname.title()}</h4>
            <p><strong>{prob}% confidence</strong></p>
            <p>{risk}</p>
            </div>""", unsafe_allow_html=True)

    st.divider()

    #probability table
    st.subheader("📋 Full Probability Distribution")
    prob_df = pd.DataFrame({
        "Lesion Type": [c.title() for c in CLASS_NAMES],
        "Probability (%)": [round(p * 100, 1) for p in probabilities],
        "Risk Category": ["🔴 Malignant" if c in MALIGNANT else "🟢 Benign" for c in CLASS_NAMES]
    }).sort_values("Probability (%)", ascending=False)
    st.dataframe(prob_df, use_container_width=True, hide_index=True)

    st.divider()

    #key visual indicators
    st.subheader("🧬 Key Visual Indicators")
    st.markdown("*The following image characteristics were most influential in determining the classification. Each feature reflects a measurable property of the uploaded lesion image.*")

    with st.spinner("Identifying key visual indicators..."):
        explainer = shap.TreeExplainer(model.named_steps["xgb"])
        shap_values = explainer.shap_values(df_features)

        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_vals_to_plot = shap_values[:, :, prediction]
        elif isinstance(shap_values, list):
            shap_vals_to_plot = shap_values[prediction]
        else:
            shap_vals_to_plot = shap_values

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('#f0f6ff')
        ax.set_facecolor('#f0f6ff')
        shap.summary_plot(
            shap_vals_to_plot,
            df_features,
            plot_type="bar",
            show=False,
            max_display=10,
            color="#1a3a6b"
        )
        ax.set_xlabel("Influence on Classification Decision", color="#1a1a2e", fontsize=12)
        ax.tick_params(colors="#1a1a2e")
        st.pyplot(fig)
        plt.close()

        st.markdown("**What each indicator means clinically:**")
        feature_importance = pd.DataFrame({
            "Feature": df_features.columns,
            "Impact": np.abs(shap_vals_to_plot[0])
        }).sort_values("Impact", ascending=False).head(10)

        feature_importance["Clinical Description"] = feature_importance["Feature"].map(
            lambda x: FEATURE_MAP.get(x, "Image-derived measurement contributing to lesion classification")
        )
        feature_importance["Relative Influence"] = (
            feature_importance["Impact"] / feature_importance["Impact"].sum() * 100
        ).round(1).astype(str) + "%"

        st.dataframe(
            feature_importance[["Feature", "Clinical Description", "Relative Influence"]],
            use_container_width=True,
            hide_index=True
        )

    st.divider()

    st.markdown("""
    <div class="disclaimer">
    ⚠️ <strong>Clinical Disclaimer:</strong> This tool is intended for research and decision support purposes only.
    It is not a substitute for professional clinical diagnosis. All findings must be reviewed and validated
    by a qualified dermatologist before any clinical decisions are made. Performance may vary on images
    outside the training distribution.
    </div>""", unsafe_allow_html=True)

    st.divider()

    st.markdown("### 📚 References & Clinical Sources")
    st.markdown("""
    - American Academy of Dermatology. *Melanoma: Signs and symptoms.* [aad.org](https://www.aad.org)
    - American Cancer Society. *Basal and Squamous Cell Skin Cancer.* [cancer.org](https://www.cancer.org)
    - Skin Cancer Foundation. *Actinic Keratosis.* [skincancer.org](https://www.skincancer.org)
    - Nachbar F, et al. *The ABCD rule of dermatoscopy.* Journal of the American Academy of Dermatology, 1994.
    - Tschandl P, et al. *The HAM10000 dataset.* Scientific Data, 2018.
    """)

    st.divider()
    st.caption("Developed by Sidhantaa Sarna, Tiffany De La Cruz, Diego Maldonado | Faculty Mentor: Dr. Masoud Rostami | DATA 4382 Capstone | University of Texas at Arlington")