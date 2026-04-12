# 🫀 Arrhythmia Detector — Streamlit App

CNN + BiLSTM Hybrid model for classifying ECG heartbeats as **Normal** or **Arrhythmia**.

---

## Project Structure

```
arrhythmia-streamlit/
├── app.py               ← Streamlit application (single file)
├── requirements.txt     ← Python dependencies
├── README.md
└── model/
    └── arrhythmia_cnn_bilstm_model.h5   ← PUT YOUR MODEL HERE
```

---

## Step 1 — Export your model + test samples from Colab

Run this at the end of your training notebook:

```python
import numpy as np, os

# ── Save model ──────────────────────────────────────────────
os.makedirs("model", exist_ok=True)
model.save("model/arrhythmia_cnn_bilstm_model.h5")
print("Model saved!")

# ── Export test samples as .npy ─────────────────────────────
os.makedirs("sample_inputs", exist_ok=True)

normal_idx     = np.where(y_test == 0)[0][0]
arrhythmia_idx = np.where(y_test == 1)[0][0]

# X_test_dl has shape (N, 200, 1) — flatten to (200,)
np.save("sample_inputs/normal_sample.npy",
        X_test_dl[normal_idx].flatten())
np.save("sample_inputs/arrhythmia_sample.npy",
        X_test_dl[arrhythmia_idx].flatten())

print("Saved normal_sample.npy and arrhythmia_sample.npy")
```

Download:
- `model/arrhythmia_cnn_bilstm_model.h5`
- `sample_inputs/normal_sample.npy`
- `sample_inputs/arrhythmia_sample.npy`

---

## Step 2 — Run locally

```bash
# Clone / download this repo
cd arrhythmia-streamlit

# Place your model at:
#   model/arrhythmia_cnn_bilstm_model.h5

# Install dependencies
pip install -r requirements.txt

# Launch
streamlit run app.py
```

App opens at `http://localhost:8501`

---

## Step 3 — Deploy on Streamlit Community Cloud (free, public URL)

1. Push this folder to a **GitHub repo** (public or private)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your repo → set main file to `app.py`
4. Upload your `.h5` model:
   - Option A: Use **Git LFS** if model < 100 MB:
     ```bash
     git lfs install
     git lfs track "*.h5"
     git add .gitattributes model/arrhythmia_cnn_bilstm_model.h5
     git commit -m "add model"
     git push
     ```
   - Option B: Use **Streamlit Secrets** + load from Google Drive URL  
     (see below)
5. Click **Deploy** — get a free public URL instantly

### Option B: Load model from Google Drive

```python
# In app.py, replace load_keras_model() with:
import gdown, os

@st.cache_resource(show_spinner="Downloading model…")
def load_keras_model():
    url = "https://drive.google.com/uc?id=YOUR_FILE_ID"
    gdown.download(url, "model/model.h5", quiet=False)
    return load_model("model/model.h5")
```

Add `gdown` to `requirements.txt`.

---

## Input Format

| Property | Value |
|---|---|
| File type | `.npy` |
| Array shape | `(200,)` or `(1, 200)` or `(1, 200, 1)` — all accepted |
| Data type | float32 / float64 |
| Sampling rate | 360 Hz |
| Content | One ECG heartbeat window: 100 samples before + 100 after R-peak |

---

## Model Details

| | |
|---|---|
| **Architecture** | 3× Conv1D → 2× BiLSTM → Dense → Sigmoid |
| **Dataset** | MIT-BIH Arrhythmia Database |
| **Accuracy** | 94% |
| **Recall** | 96% |
| **Threshold** | 0.40 |

> ⚠️ For research / academic use only. Not a clinical diagnostic tool.
