"""
╔══════════════════════════════════════════════════════════════╗
║   Arrhythmia Detector — Streamlit App                       ║
║   Model  : CNN + BiLSTM Hybrid                              ║
║   Input  : .npy file  (shape: (200,) or (1,200) or (1,200,1))║
║   Output : Normal / Arrhythmia + probability                ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import butter, filtfilt

# ── Must be the very first Streamlit call ─────────────────────────────────────
st.set_page_config(
    page_title="Arrhythmia Detector",
    page_icon="🫀",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Suppress TF logs before importing ────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow.keras.models import load_model

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH   = "model/arrhythmia_cnn_bilstm_model.h5"
SEGMENT_LEN  = 200
THRESHOLD    = 0.40      # same as training

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL  (cached so it only loads once per session)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_keras_model():
    if not os.path.exists(MODEL_PATH):
        return None
    m = load_model(MODEL_PATH)
    m.make_predict_function()
    return m

# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING  (identical to training pipeline)
# ─────────────────────────────────────────────────────────────────────────────
def bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=360, order=4):
    nyq  = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, signal)

def zscore_normalize(signal):
    return (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

def preprocess(raw: np.ndarray) -> np.ndarray:
    """Filter → normalise → reshape to (1, 200, 1)."""
    filtered   = bandpass_filter(raw.astype(np.float32))
    normalised = zscore_normalize(filtered)
    return normalised.reshape(1, SEGMENT_LEN, 1)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 780px; }

/* ── Header ── */
.app-header {
    text-align: center;
    padding: 2rem 1rem 1.2rem;
    border-bottom: 1px solid #21262d;
    margin-bottom: 2rem;
}
.app-icon {
    font-size: 2.8rem;
    display: block;
    margin-bottom: 0.4rem;
}
.app-title {
    font-size: 1.75rem;
    font-weight: 700;
    color: #e6edf3;
    letter-spacing: -0.5px;
    margin: 0;
}
.app-sub {
    font-size: 0.85rem;
    color: #8b949e;
    margin-top: 0.3rem;
}

/* ── Verdict boxes ── */
.verdict-normal {
    background: rgba(74,222,128,0.08);
    border: 1.5px solid rgba(74,222,128,0.35);
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    text-align: center;
    margin: 1.2rem 0;
}
.verdict-arrhythmia {
    background: rgba(248,113,113,0.08);
    border: 1.5px solid rgba(248,113,113,0.35);
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    text-align: center;
    margin: 1.2rem 0;
}
.verdict-emoji  { font-size: 2.2rem; display: block; margin-bottom: 0.3rem; }
.verdict-label-normal {
    font-size: 1.5rem; font-weight: 700; color: #4ade80; display: block;
}
.verdict-label-arr {
    font-size: 1.5rem; font-weight: 700; color: #f87171; display: block;
}
.verdict-desc { font-size: 0.82rem; color: #8b949e; margin-top: 0.25rem; }

/* ── Metric pills ── */
.metrics-row {
    display: flex;
    gap: 10px;
    margin: 1rem 0;
    justify-content: center;
    flex-wrap: wrap;
}
.metric-pill {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 0.55rem 1.1rem;
    text-align: center;
    min-width: 110px;
}
.metric-num {
    display: block;
    font-size: 1.1rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    color: #2dd4bf;
}
.metric-lbl {
    display: block;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #484f58;
    font-weight: 600;
    margin-top: 2px;
}

/* ── Upload area ── */
.upload-hint {
    font-size: 0.8rem;
    color: #484f58;
    text-align: center;
    margin-top: 0.4rem;
}

/* ── Section label ── */
.section-label {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    color: #484f58;
    margin-bottom: 0.5rem;
}

/* ── Warning box ── */
.warn-box {
    background: rgba(251,191,36,0.07);
    border: 1px solid rgba(251,191,36,0.25);
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    font-size: 0.83rem;
    color: #fbbf24;
    margin-bottom: 1.2rem;
}

/* ── Model missing box ── */
.model-missing {
    background: rgba(248,113,113,0.07);
    border: 1px solid rgba(248,113,113,0.25);
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    font-size: 0.85rem;
    color: #f87171;
    margin: 1rem 0;
}
.model-missing code {
    background: rgba(248,113,113,0.12);
    border-radius: 4px;
    padding: 1px 5px;
    font-family: monospace;
}

/* ── Footer ── */
.app-footer {
    text-align: center;
    font-size: 0.73rem;
    color: #30363d;
    border-top: 1px solid #21262d;
    padding-top: 1.2rem;
    margin-top: 2.5rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <span class="app-icon">🫀</span>
  <p class="app-title">Arrhythmia Detector</p>
  <p class="app-sub">CNN + BiLSTM Hybrid &nbsp;·&nbsp; MIT-BIH Arrhythmia Dataset &nbsp;·&nbsp; 360 Hz</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────────────────────
model = load_keras_model()

if model is None:
    st.markdown(f"""
    <div class="model-missing">
      <strong>⚠️ Model file not found</strong><br><br>
      Place your trained model at:<br>
      <code>{MODEL_PATH}</code><br><br>
      From your Colab notebook, run:<br>
      <code>model.save("arrhythmia_cnn_bilstm_model.h5")</code><br>
      then move the file into the <code>model/</code> folder.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# FILE UPLOAD
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<p class="section-label">Upload ECG Segment (.npy)</p>', unsafe_allow_html=True)

uploaded = st.file_uploader(
    label      = "Upload a .npy file",
    type       = ["npy"],
    label_visibility = "collapsed",
    help       = "NumPy array containing exactly 200 float values (one R-peak-centred ECG beat at 360 Hz)."
)

st.markdown(
    '<p class="upload-hint">Expected shape: (200,) &nbsp;·&nbsp; '
    'Float32/64 values &nbsp;·&nbsp; One R-peak-centred heartbeat window at 360 Hz</p>',
    unsafe_allow_html=True
)

# ── Sample download ───────────────────────────────────────────────────────────
with st.expander("📥 Need a sample .npy file to test?"):
    st.markdown("""
    Run this in your Colab notebook after training to export a test sample:

    ```python
    import numpy as np

    # Pick one Normal and one Arrhythmia beat from your test set
    normal_idx     = np.where(y_test == 0)[0][0]
    arrhythmia_idx = np.where(y_test == 1)[0][0]

    # X_test_dl has shape (N, 200, 1) — flatten to (200,)
    np.save("normal_sample.npy",     X_test_dl[normal_idx].flatten())
    np.save("arrhythmia_sample.npy", X_test_dl[arrhythmia_idx].flatten())

    print("Saved normal_sample.npy and arrhythmia_sample.npy")
    ```

    Download both files from Colab's file browser and upload them here.
    """)

# ─────────────────────────────────────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────────────────────────────────────
if uploaded is not None:

    # ── 1. Load and validate ─────────────────────────────────────────────────
    try:
        raw = np.load(uploaded, allow_pickle=False).flatten().astype(np.float32)
    except Exception as e:
        st.error(f"❌ Could not read .npy file: {e}")
        st.stop()

    if raw.shape[0] != SEGMENT_LEN:
        st.error(
            f"❌ Expected **{SEGMENT_LEN}** values, got **{raw.shape[0]}**. "
            f"Please upload a single 200-sample ECG window."
        )
        st.stop()

    # ── 2. Preprocess ────────────────────────────────────────────────────────
    processed      = preprocess(raw)
    display_signal = zscore_normalize(bandpass_filter(raw))   # for chart

    # ── 3. Inference ─────────────────────────────────────────────────────────
    with st.spinner("Running inference…"):
        prob  = float(model.predict(processed, verbose=0)[0][0])

    label = int(prob >= THRESHOLD)
    pred  = "Arrhythmia" if label == 1 else "Normal"
    conf  = prob if label == 1 else (1.0 - prob)

    # ─────────────────────────────────────────────────────────────────────────
    # RESULT
    # ─────────────────────────────────────────────────────────────────────────

    # ── Verdict banner ───────────────────────────────────────────────────────
    if label == 0:
        st.markdown(f"""
        <div class="verdict-normal">
          <span class="verdict-emoji">✅</span>
          <span class="verdict-label-normal">Normal</span>
          <span class="verdict-desc">Normal sinus rhythm &nbsp;·&nbsp; {conf*100:.1f}% confidence</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="verdict-arrhythmia">
          <span class="verdict-emoji">⚠️</span>
          <span class="verdict-label-arr">Arrhythmia</span>
          <span class="verdict-desc">Arrhythmic beat detected &nbsp;·&nbsp; {conf*100:.1f}% confidence</span>
        </div>
        """, unsafe_allow_html=True)

    # ── Metric pills ──────────────────────────────────────────────────────────
    normal_pct = (1 - prob) * 100
    arr_pct    = prob * 100

    st.markdown(f"""
    <div class="metrics-row">
      <div class="metric-pill">
        <span class="metric-num">{prob:.4f}</span>
        <span class="metric-lbl">P(Arrhythmia)</span>
      </div>
      <div class="metric-pill">
        <span class="metric-num">{conf*100:.1f}%</span>
        <span class="metric-lbl">Confidence</span>
      </div>
      <div class="metric-pill">
        <span class="metric-num">{THRESHOLD:.2f}</span>
        <span class="metric-lbl">Threshold</span>
      </div>
      <div class="metric-pill">
        <span class="metric-num">{normal_pct:.1f}%</span>
        <span class="metric-lbl">P(Normal)</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Probability bar ───────────────────────────────────────────────────────
    st.markdown('<p class="section-label" style="margin-top:1.2rem;">Probability Distribution</p>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Normal",      f"{normal_pct:.1f}%")
        st.progress(float(1 - prob))
    with col2:
        st.metric("Arrhythmia",  f"{arr_pct:.1f}%")
        st.progress(float(prob))

    # ── ECG Signal Chart ──────────────────────────────────────────────────────
    st.markdown('<p class="section-label" style="margin-top:1.4rem;">ECG Signal — Preprocessed (Bandpass + Z-score)</p>',
                unsafe_allow_html=True)

    line_color = "#f87171" if label == 1 else "#4ade80"
    fill_color = "#f8717120" if label == 1 else "#4ade8020"

    fig, ax = plt.subplots(figsize=(9, 2.8))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    x = np.arange(SEGMENT_LEN)

    # Shaded fill
    ax.fill_between(x, display_signal, alpha=0.15,
                    color=line_color)

    # Signal line
    ax.plot(x, display_signal, color=line_color, linewidth=1.4, zorder=3)

    # R-peak marker (centre = sample 100)
    ax.axvline(x=100, color="#2dd4bf", linewidth=1, linestyle="--", alpha=0.6, label="R-peak")
    ax.scatter([100], [display_signal[100]], color="#2dd4bf", s=40, zorder=5)

    # Threshold line at y=0 (baseline)
    ax.axhline(y=0, color="#30363d", linewidth=0.8, linestyle="-")

    ax.set_xlim(0, SEGMENT_LEN - 1)
    ax.set_xlabel("Sample index  (100 = R-peak)", color="#484f58", fontsize=8)
    ax.set_ylabel("Amplitude (normalised)", color="#484f58", fontsize=8)
    ax.tick_params(colors="#484f58", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#21262d")

    legend = ax.legend(fontsize=7, facecolor="#161b22", edgecolor="#30363d",
                       labelcolor="#8b949e", loc="upper right")

    # Label annotation
    ax.text(0.01, 0.93, pred,
            transform=ax.transAxes,
            fontsize=9, fontweight="bold",
            color=line_color, va="top")

    plt.tight_layout(pad=0.5)
    st.pyplot(fig)
    plt.close(fig)

    # ── Disclaimer ────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="warn-box">
      ⚠️ <strong>Research use only.</strong>
      This tool is not a certified clinical diagnostic instrument.
      Always consult a qualified cardiologist for medical decisions.
    </div>
    """, unsafe_allow_html=True)

else:
    # ── Placeholder when nothing uploaded ────────────────────────────────────
    st.markdown("""
    <div style="text-align:center; padding: 2.5rem 1rem; color: #484f58;">
      <div style="font-size:2.5rem; margin-bottom:0.7rem;">📂</div>
      <p style="font-size:0.88rem;">Upload a <strong style="color:#8b949e;">.npy</strong> file above to get a prediction.</p>
      <p style="font-size:0.78rem; margin-top:0.4rem;">
        Array must contain exactly <strong style="color:#8b949e;">200 float values</strong><br>
        (one ECG heartbeat window centred on an R-peak)
      </p>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
  CNN + BiLSTM · MIT-BIH Arrhythmia Dataset · TensorFlow / Keras · Streamlit
</div>
""", unsafe_allow_html=True)
