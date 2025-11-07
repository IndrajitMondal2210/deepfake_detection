## 🎯 Project Overview

This project detects DeepFakes by analyzing:

* 🎥 **Visual content:** Face manipulation detection
* 🔊 **Audio content:** Voice synthesis detection
* 👄 **Audio–Visual Synchronization:** Lip-sync mismatch detection
* 🔗 **Multi-modal Fusion:** Combined decision-making

---

## 🧩 Classification Categories

| Label        | Category | Description                               |
| :----------- | :------- | :---------------------------------------- |
| **0 — Real** | (A, B)   | RealVideo–RealAudio / RealVideo–FakeAudio |
| **1 — Fake** | (C, D)   | FakeVideo–RealAudio / FakeVideo–FakeAudio |

> The model performs **binary classification** — determining whether a video is **authentic (Real)** or **manipulated (Fake)**.

---

## 📁 Project Structure

```
deepfake_detection_2/
├── data/
│   ├── raw/                         # Original FakeAVCeleb dataset
│   │   ├── RealVideo-RealAudio/     # Category A: Authentic
│   │   ├── RealVideo-FakeAudio/     # Category B: Voice cloned
│   │   ├── FakeVideo-RealAudio/     # Category C: Face swapped
│   │   ├── FakeVideo-FakeAudio/     # Category D: Both fake
│   │   └── meta_data.csv            # Dataset metadata
│   └── processed/                   # Preprocessed features
│       ├── frames/                  # Extracted video frames
│       ├── faces/                   # Cropped face regions
│       ├── audio/                   # Extracted audio files
│       ├── audio_features/          # Mel-spectrograms
│       ├── sync/                    # Lip-sync alignment data
│       └── fusion_features/         # Combined multi-modal features
│
├── src/
│   ├── preprocessing/               # Data preprocessing scripts
│   ├── models/                      # Model architectures
│   ├── training/                    # Training scripts
│   ├── evaluation/                  # Evaluation and metrics
│   └── api/                         # (Future) FastAPI deployment
│
├── models/                          # Saved model weights
├── results/                         # Evaluation results
├── notebooks/                       # Jupyter notebooks for experiments
└── requirements.txt                 # Python dependencies
```

---

## 🔄 Complete Workflow

### 1️⃣ **Data Preprocessing**

| Step                   | Script                                    | Description                                            | Output                           |
| ---------------------- | ----------------------------------------- | ------------------------------------------------------ | -------------------------------- |
| **Frame Extraction**   | `src/preprocessing/extract_frames.py`     | Extracts frames from videos                            | `data/processed/frames/`         |
| **Face Detection**     | `src/preprocessing/face_detection.py`     | Detects and crops faces                                | `data/processed/faces/`          |
| **Audio Extraction**   | `src/preprocessing/extract_audio.py`      | Extracts and converts audio to WAV                     | `data/processed/audio/`          |
| **Feature Extraction** | `src/preprocessing/feature_extraction.py` | Converts audio to mel-spectrograms; creates embeddings | `data/processed/audio_features/` |
| **Sync Data Creation** | `src/preprocessing/sync_preprocess.py`    | Aligns lips and audio for synchronization dataset      | `data/processed/sync/`           |

---

### 2️⃣ **Model Training**

#### 🧍‍♂️ Visual Model (`src/training/train_visual.py`)

* **Architecture:** ResNet-18 based CNN
* **Purpose:** Detects face manipulation artifacts
* **Input:** Face crops (224×224)
* **Output:** Real / Fake classification
* **Saved at:** `models/visual_model/visual_model.pth`

---

#### 🔊 Audio Model (`src/training/train_audio.py`)

* **Architecture:** CNN for spectrogram analysis
* **Purpose:** Detects voice cloning or synthetic speech
* **Input:** Mel-spectrograms (64×160)
* **Output:** Real / Fake audio classification
* **Saved at:** `models/audio_model/audio_model.pth`

---

#### 👄 Sync Model (`src/training/train_sync.py`)

* **Architecture:** CNN-based (similar to SyncNet)
* **Purpose:** Detects lip–speech mismatches
* **Input:** Lip movement + audio alignment
* **Output:** Synced / Unsynced classification
* **Saved at:** `models/sync_model/sync_model.pth`

---

#### 🔗 Fusion Model (`src/training/train_fusion_transformer.py`)

* **Architecture:** Transformer-based encoder (tokenized fusion vector + positional encoding + TransformerEncoder)
* **Purpose:** Combines **visual**, **audio**, and **sync** features to make the final Real/Fake decision.
* **Input:** Concatenated embeddings → split into token sequences for the Transformer.
* **Output:** **Binary classification (Real vs Fake)**

**Training Details:**

* **Loss:** CrossEntropyLoss
* **Optimizer:** Adam
* **LR Scheduler:** StepLR
* **Batch Size:** 16
* **Learning Rate:** 1e-4
* **Epochs:** 6
* **Saved Model:** `models/fusion_model/fusion_transformer.pth`

**Notes:**

* The fusion vector is padded and reshaped into `(seq_len, d_model)` tokens.
* Transformer learns cross-modal relationships through attention.

---

### 3️⃣ **Model Evaluation**

| Component               | Script                             | Description                                                       | Output                    |
| ----------------------- | ---------------------------------- | ----------------------------------------------------------------- | ------------------------- |
| **Performance Metrics** | `src/evaluation/evaluate.py`       | Calculates accuracy, precision, recall, F1, ROC, confusion matrix | `results/evaluation/`     |
| **Explainability**      | `src/evaluation/explainability.py` | Grad-CAM & feature heatmaps                                       | `results/explainability/` |

---

### 4️⃣ **API Deployment (Future Work)**

| File                                                                    | Description                                  |
| ----------------------------------------------------------------------- | -------------------------------------------- |
| `src/api/main.py`                                                       | FastAPI endpoint `/predict` for video upload |
| `src/api/inference.py`                                                  | Loads models, runs full inference pipeline   |
| **Planned Output:** JSON with Real/Fake prediction and confidence score |                                              |

**Example:**

```json
{
  "visual_pred": "Real",
  "audio_pred": "Fake",
  "fusion_pred": "Fake",
  "confidence": 0.97
}
```

---

## 📊 Dataset Information

**FakeAVCeleb Dataset**

* Generated using **Faceswap**, **FSGAN**, **Wav2Lip**, and **RTVC**.
* Each video is categorized by the authenticity of **video** and **audio** components.

| Technique | Description         |
| --------- | ------------------- |
| Faceswap  | Face replacement    |
| FSGAN     | Face reenactment    |
| Wav2Lip   | Lip synchronization |
| RTVC      | Voice cloning       |

---

## 📝 Usage Example (After API Integration)

```python
import requests

with open("test_video.mp4", "rb") as f:
    response = requests.post("http://localhost:8000/predict", files={"file": f})

result = response.json()
print(f"Final Prediction: {result['fusion_pred']} ({result['confidence']:.2f})")
```

---

## 🎓 Research & Real-World Applications

* Social Media Content Verification
* News & Broadcast Authenticity Checking
* Digital Forensics & Law Enforcement
* Media Literacy & Awareness
* Academic Research on DeepFake Detection

---

## 🚀 Future Enhancements

* Real-time video stream analysis
* FastAPI deployment (inference endpoint)
* Mobile and cloud deployment support
* Integration of new deepfake generation types
* Cross-modal contrastive learning (future research)
