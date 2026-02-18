# ECG-Based Arrhythmia Detection

## Overview
This project focuses on detecting cardiac arrhythmia from ECG signals using data-driven classification techniques. The system processes raw ECG recordings, extracts heartbeat-level signal windows, and performs automated classification to identify abnormal heart rhythms.

The project is designed to support both traditional machine learning and deep learning approaches for medical signal analysis.

---

## Dataset
**MIT-BIH Arrhythmia Database (PhysioNet)**

- 48 ECG recordings  
- Approx. 30 minutes per recording  
- Sampling Frequency: 360 Hz  
- Dual-channel ECG signals  
- ~110,000 annotated heartbeats  

### Data File Types
- `.dat` — ECG waveform signal  
- `.hea` — Signal metadata  
- `.atr` — Beat annotations  

---

## System Workflow
1. Raw ECG signal acquisition  
2. Signal filtering and normalization  
3. Heartbeat segmentation using annotations  
4. Signal window extraction  
5. Model-based classification  
6. Prediction of Normal vs Arrhythmia  

---

## Current Implementation
- ECG preprocessing using WFDB tools  
- Beat-level classification pipeline  
- Binary Classification:
  - Normal Beat
  - Arrhythmia Beat
- Web interface built using Streamlit  

---

## Future Scope
- Integration of Deep Learning models (CNN, LSTM)  
- Explainable AI integration for medical interpretability  
- Real-time ECG monitoring support  
- Multi-class arrhythmia classification  

---

## Deployment
The application is deployed using Streamlit Cloud. Users can upload ECG beat data (.npy format) and receive real-time prediction results.

---

## Tech Stack
- Python  
- Streamlit  
- NumPy  
- Pandas  
- Scikit-learn  
- SciPy  
- WFDB  

---

## Note
The dataset is not included in this repository due to size limitations. It can be downloaded from PhysioNet.

---

## Author
**Riya Khandelwal**  
Manipal University Jaipur
