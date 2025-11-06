# Deepfake Detection System

A comprehensive multi-modal deepfake detection system that analyzes both visual and audio components of videos to identify manipulated content. The system uses four specialized models to classify videos into four categories based on authenticity.

## 🎯 Project Overview

This project detects deepfakes by analyzing:
- **Visual content**: Face manipulation detection
- **Audio content**: Voice synthesis detection  
- **Audio-Visual synchronization**: Lip-sync mismatch detection
- **Multi-modal fusion**: Combined decision making

### Classification Categories
- **A**: RealVideo-RealAudio (Authentic)
- **B**: RealVideo-FakeAudio (Voice cloned)
- **C**: FakeVideo-RealAudio (Face swapped)
- **D**: FakeVideo-FakeAudio (Both manipulated)

## 📁 Project Structure

```
deepfake_detction_2/
├── data/                           # Dataset storage
│   ├── raw/                        # Original FakeAVCeleb dataset
│   │   ├── RealVideo-RealAudio/    # Category A: Authentic videos
│   │   ├── RealVideo-FakeAudio/    # Category B: Voice cloned
│   │   ├── FakeVideo-RealAudio/    # Category C: Face swapped
│   │   ├── FakeVideo-FakeAudio/    # Category D: Both fake
│   │   └── meta_data.csv           # Dataset metadata
│   └── processed/                  # Preprocessed features
│       ├── frames/                 # Extracted video frames
│       ├── faces/                  # Cropped face regions
│       ├── audio/                  # Extracted audio files
│       ├── audio_features/         # Audio spectrograms
│       ├── sync/                   # Lip-sync alignment data
│       └── fusion_features/        # Combined features
├── src/                            # Source code
│   ├── preprocessing/              # Data preprocessing
│   ├── models/                     # Model architectures
│   ├── training/                   # Training scripts
│   ├── evaluation/                 # Evaluation & metrics
│   └── api/                        # FastAPI deployment
├── models/                         # Trained model weights
├── results/                        # Evaluation results
├── notebooks/                      # Jupyter notebooks
└── requirements.txt                # Dependencies
```

## 🔄 Complete Workflow

### 1. Data Preprocessing

#### Frame Extraction (`src/preprocessing/extract_frames.py`)
- Extracts frames from videos at regular intervals
- Converts video files to image sequences
- **Input**: MP4 videos from `data/raw/`
- **Output**: Frame images in `data/processed/frames/`

#### Face Detection (`src/preprocessing/face_detection.py`)
- Detects and crops face regions from frames
- Uses computer vision techniques to isolate faces
- **Input**: Frame images
- **Output**: Cropped face images in `data/processed/faces/`

#### Audio Extraction (`src/preprocessing/extract_audio.py`)
- Separates audio tracks from video files
- Converts to standard format (WAV, 16kHz)
- **Input**: MP4 videos
- **Output**: Audio files in `data/processed/audio/`

#### Feature Extraction (`src/preprocessing/feature_extraction.py`)
- Converts audio to mel-spectrograms
- Creates visual embeddings from face images
- **Input**: Audio files and face images
- **Output**: Feature vectors in `data/processed/audio_features/`

#### Sync Preprocessing (`src/preprocessing/sync_preprocess.py`)
- Aligns lip movements with audio
- Creates synchronization datasets
- **Input**: Face images and audio
- **Output**: Sync data in `data/processed/sync/`

### 2. Model Training

#### Visual Model (`src/training/train_visual.py`)
- **Architecture**: ResNet-18 based CNN
- **Purpose**: Detects face manipulation artifacts
- **Input**: Face images (224x224)
- **Output**: Real/Fake classification
- **Saved**: `models/visual_model/visual_model.pth`

#### Audio Model (`src/training/train_audio.py`)
- **Architecture**: CNN for spectrogram analysis
- **Purpose**: Identifies synthetic voice patterns
- **Input**: Mel-spectrograms (64x160)
- **Output**: Real/Fake audio classification
- **Saved**: `models/audio_model/audio_model.pth`

#### Sync Model (`src/training/train_sync.py`)
- **Architecture**: LSTM-based sequence model
- **Purpose**: Detects lip-sync mismatches
- **Input**: Lip movement + audio alignment
- **Output**: Sync/Unsync classification
- **Saved**: `models/sync_model/sync_model.pth`

**Fusion Model (src/training/train_fusion.py)**
**Architecture:** Transformer-based encoder (tokenized 1D fusion vector + positional encoding + TransformerEncoder)
**Purpose:** Combine visual, audio and sync feature embeddings to produce a single robust decision using cross-token attention.
**Input:** Concatenated feature vector per video (visual_embedding || audio_embedding || sync_embedding). The 1D vector is split/padded into seq_len tokens of size d_model before feeding to the Transformer.
**Output:** Final 4-category classification (A, B, C, D)
**Training details:**
  - Criterion: CrossEntropyLoss
  - Optimizer: Adam
  - LR Scheduler: StepLR (optional)
  - Batch size, learning rate, epochs: configured in training script (defaults: batch_size=16, lr=1e-4, epochs=6)
**Saved:** models/fusion_model/fusion_transformer.pth
**Notes:**
  - The script tokenizes the fusion vector into seq_len tokens (seq_len = ceil(input_dim / d_model)) and pads the vector if needed.
  - Class labels mapping must follow the metadata.csv `category` field:
      A -> 0
      B -> 1
      C -> 2
      D -> 3
  - When loading the checkpoint, reconstruct the FusionTransformer with the same input_dim and d_model used during training and set num_classes=4.


### 3. Model Evaluation

#### Performance Metrics (`src/evaluation/evaluate.py`)
- Calculates accuracy, precision, recall, F1-score
- Generates ROC curves and confusion matrices
- **Output**: Metrics saved in `results/evaluation/`

#### Explainability (`src/evaluation/explainability.py`)
- Generates Grad-CAM visualizations
- Shows which regions influence decisions
- **Output**: Heatmaps in `results/explainability/`

### 4. API Deployment

#### FastAPI Server (`src/api/main.py`)
- **Endpoint**: `POST /predict`
- **Input**: Video file upload
- **Process**: 
  1. Extract frames and audio
  2. Run all four models
  3. Combine predictions
- **Output**: JSON with predictions and confidence scores

#### Inference Pipeline (`src/api/inference.py`)
- Handles video preprocessing
- Loads trained models
- Performs real-time prediction

## 🚀 Quick Start

### Installation
```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation
```bash
# Extract frames from videos
python src/preprocessing/extract_frames.py

# Detect and crop faces
python src/preprocessing/face_detection.py

# Extract audio features
python src/preprocessing/extract_audio.py
python src/preprocessing/feature_extraction.py
```

### Training Models
```bash
# Train individual models
python src/training/train_visual.py
python src/training/train_audio.py
python src/training/train_sync.py

# Train fusion model
python src/training/train_fusion.py
```

### Evaluation
```bash
# Evaluate all models
python src/evaluation/evaluate.py

# Generate explainability results
python src/evaluation/explainability.py
```

### Run API
```bash
# Start FastAPI server
uvicorn src.api.main:app --reload

# API will be available at http://localhost:8000
# Upload video to /predict endpoint
```

## 📊 Dataset Information

**FakeAVCeleb Dataset** contains videos generated using:
- **Faceswap**: Face replacement
- **FSGAN**: Face reenactment  
- **Wav2Lip**: Lip synchronization
- **RTVC**: Voice cloning

Each video is categorized based on authenticity of visual and audio components.

## 🎯 Model Performance

Results are saved in `results/evaluation/`:
- Confusion matrices for each model
- ROC curves showing detection performance
- Score distributions for real vs fake content

## 🔧 Key Features

- **Multi-modal Analysis**: Combines visual, audio, and sync detection
- **Real-time Inference**: FastAPI deployment for live detection
- **Explainable AI**: Grad-CAM visualizations show decision reasoning
- **Comprehensive Evaluation**: Multiple metrics and visualizations
- **Modular Design**: Each component can be trained/evaluated independently

## 📝 Usage Example

```python
# Upload video to API
import requests

with open("test_video.mp4", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict",
        files={"file": f}
    )

result = response.json()
print(f"Visual: {result['visual_pred']} ({result['visual_prob']:.3f})")
print(f"Audio: {result['audio_pred']} ({result['audio_prob']:.3f})")
print(f"Final: {result['fusion_pred']} ({result['fusion_prob']:.3f})")
```

## 🎓 Research Applications

This system can be used for:
- Social media content verification
- News authenticity checking
- Digital forensics
- Media literacy education
- Academic research on deepfake detection

## 📈 Future Enhancements

- Real-time video stream processing
- Mobile app deployment
- Additional deepfake generation methods
- Improved model architectures
- Cloud deployment options
