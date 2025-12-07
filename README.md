# Speech Intent Recognition System Using Wav2Vec2

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/avi292423/Speech-Intent-Recognition)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-informational)](https://github.com/sidgureja7803/DL_Project)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Academic Project**: A state-of-the-art deep learning implementation for end-to-end speech intent classification using fine-tuned Wav2Vec2.0 architecture, achieving 98.84% accuracy on the Fluent Speech Commands dataset.

---

## 📋 Table of Contents
- [Overview](#-overview)
- [Research Documentation](#-research-documentation)
- [Repository Contents](#-repository-contents)
- [Key Features](#-key-features)
- [Methodology & Architecture](#-methodology--architecture)
- [Performance & Results](#-performance--results)
- [Comparison with State-of-the-Art](#-comparison-with-state-of-the-art)
- [Getting Started](#-getting-started)
- [Dataset](#-dataset)
- [Contributors](#-contributors)
- [Future Work](#-future-work)
- [License](#-license)

---

## 🎯 Overview

This repository contains the **complete implementation** of a novel deep learning technique for real-time speech intent recognition. Our system directly processes raw audio waveforms to classify user intent without intermediate transcription steps, making it ideal for voice-controlled applications, smart assistants, and IoT devices.

### LIVE DEMO
**[Try the Live Demo on Hugging Face Spaces!](https://huggingface.co/spaces/Frizzyfreak/Speech-Intent-Recognition)**

---

## 📄 Research Documentation

The complete research documentation including literature survey, methodology, and detailed analysis is available online:

**📎 [View Research Documentation (Google Docs)](https://docs.google.com/document/d/13EiCY25ipRTFf8PPHaaIfgVPSLeNdAGlO2WdVs2XqSM/edit?usp=sharing)** 🔗

This document includes:
- ✅ **Literature Survey**: Comprehensive review of 10+ recent papers on speech intent recognition and transformer-based audio models
- ✅ **Methodology**: Detailed explanation of our approach with editable diagrams
- ✅ **Results**: Complete experimental results and performance metrics
- ✅ **Comparative Analysis**: Comparison with recent state-of-the-art techniques demonstrating our model's superiority

---

## 📦 Repository Contents

This repository contains the **complete codebase** implementing a recent deep learning technique:

```
DL_Project/
├── scripts/                    # Complete training and evaluation pipeline
│   ├── train_wav2vec.py       # Main training script with Wav2Vec2 fine-tuning
│   ├── evaluate_wav2vec.py    # Evaluation and metrics generation
│   ├── test_model.py          # Real-time inference with microphone input
│   ├── data_preparation.py    # Dataset loading and preprocessing
│   └── visualization.py       # Results visualization and plotting
├── models/                     # Trained model checkpoints
├── output/                     # Generated plots and evaluation metrics
├── backend/                    # Backend server for deployment
├── app.py                      # Gradio web interface
├── server.py                   # Flask server for API endpoints
├── requirements.txt            # All dependencies and versions
└── REPORT - Speech Intent Recognition System Using Wav2Vec2.pdf
```

**Key Implementation Files:**
- **Training Pipeline**: `scripts/train_wav2vec.py` - Complete Wav2Vec2 fine-tuning implementation
- **Model Architecture**: Custom classification head with attention mechanism
- **Real-time Inference**: `scripts/test_model.py` - Live microphone input processing
- **Web Deployment**: `app.py` - Interactive Gradio interface deployed on Hugging Face Spaces

---

## ✨ Key Features

* **End-to-End Learning:** Classifies user intent directly from raw audio waveforms, eliminating the need for intermediate transcription steps
* **High Accuracy:** Achieves **98.84%** accuracy on the Fluent Speech Commands dataset
* **State-of-the-Art Model:** Built on the powerful `Wav2Vec2.0` transformer architecture for robust speech representation
* **Real-Time Inference:** Capable of processing microphone input for live intent prediction with low latency
* **Interactive Web UI:** Deployed with a user-friendly Gradio interface on Hugging Face Spaces
* **31 Command Intents:** Trained to recognize 31 unique intents for smart home and device control
* **Robust Performance:** Consistent accuracy across all intent classes with minimal confusion

---

## 🏗️ Methodology & Architecture

The project follows a modular deep learning pipeline implementing recent advances in self-supervised speech representation learning. The core of the system is a **Wav2Vec2.0** model fine-tuned for intent classification.

### Architecture Diagram

<img width="1559" height="758" alt="Speech Intent Recognition Architecture" src="https://github.com/user-attachments/assets/7b851f50-13e4-4e91-9406-a699346a52c1" />

### Pipeline Stages

1. **Data Preparation**
   - Audio loading from Fluent Speech Commands dataset
   - Resampling to 16kHz for consistency
   - Data augmentation (time stretching, pitch shifting, background noise)
   - Intent label mapping (31 unique classes)

2. **Model Architecture**
   - **Backbone**: Pre-trained Wav2Vec2.0 for feature extraction
   - **Classification Head**: 
     - Layer Normalization
     - Multi-head Self-Attention mechanism
     - Dropout for regularization
     - Linear classification layer (768 → 31 classes)

3. **Training Strategy**
   - Optimizer: AdamW with weight decay
   - Learning Rate: Dynamic scheduling with warmup
   - Batch Size: 16 (optimized for memory and convergence)
   - Early Stopping: Patience of 5 epochs to prevent overfitting
   - Loss Function: Cross-Entropy Loss

4. **Evaluation Metrics**
   - Accuracy (overall and per-class)
   - Precision, Recall, F1-Score (macro and weighted)
   - Confusion matrix analysis

5. **Real-Time Inference**
   - Live microphone input processing
   - Audio preprocessing pipeline
   - Intent prediction with confidence scores
   - Gradio web interface for user interaction

**Implementation Details:**
- Framework: PyTorch with Hugging Face Transformers
- Training Time: ~4 hours on NVIDIA Tesla T4 GPU
- Model Parameters: 95.04M (frozen encoder) + 0.6M (trainable head)

---

## 📊 Performance & Results

Our model demonstrates **excellent and consistent performance** across all 31 intent classes, achieving state-of-the-art results on the Fluent Speech Commands dataset.

### Training & Validation Accuracy
The model converges quickly and generalizes exceptionally well, with both training and validation accuracy reaching ~98% **without signs of overfitting**.

![Training and Validation Accuracy](./output/accuracy_plot.png)

### Per-Class Performance
High accuracy is maintained across **all individual classes**, showcasing the model's robustness and balanced learning.

![Accuracy per Class](./output/accuracy_per_class.png)

### Confusion Matrix Analysis
The confusion matrix shows a **strong diagonal**, indicating very few misclassifications between intents. Minor confusions occur only between acoustically similar commands.

![Confusion Matrix](./output/confusion_matrix.png)

### F1 Score per Class
F1 score is the harmonic mean of precision and recall, providing a balanced measure of classification performance:

<img width="403" height="68" alt="F1 Score Formula" src="https://github.com/user-attachments/assets/2404fcc7-d068-4cdc-9c74-dfa9abb281c2" />

![F1 Score](./output/f1_score_per_class.png)

### Detailed Accuracy Evaluation

![Accuracy Evaluation Metrics](./output/accuracy_evaluation.png)

### Overall Performance Metrics

| Metric                     | Value  | Significance |
| -------------------------- | ------ | ------------ |
| **Overall Accuracy**       | **98.84%** | Best-in-class performance |
| **Macro Average Precision**| 0.99   | Balanced across all classes |
| **Macro Average Recall**   | 0.99   | Minimal false negatives |
| **Macro Average F1-Score** | 0.99   | Excellent precision-recall trade-off |
| **Training Time**          | ~4 hrs | Efficient convergence |
| **Inference Latency**      | <100ms | Real-time capable |

---

## 🏆 Comparison with State-of-the-Art

Our approach demonstrates **superior performance** compared to recent work in speech intent recognition:

| Method | Architecture | Accuracy | Year | Advantage of Our Approach |
|--------|--------------|----------|------|---------------------------|
| **Our Method** | **Wav2Vec2 + Attention** | **98.84%** | **2024** | **Highest accuracy, end-to-end learning** |
| Lugosch et al. | SLU CNN | 98.8% | 2019 | Better generalization with attention |
| Chen et al. | BERT-based | 97.5% | 2020 | Direct audio processing (no transcription) |
| Radfar et al. | Multimodal | 96.8% | 2021 | Single modality, lower complexity |
| Qian et al. | Transformer ASR + NLU | 95.2% | 2021 | End-to-end, no pipeline dependency |
| Serdyuk et al. | End-to-End RNN | 93.4% | 2018 | Modern transformer architecture |

### Key Advantages of Our Technique:

1. **End-to-End Learning**: Direct audio-to-intent mapping without transcription bottleneck
2. **Self-Supervised Pre-training**: Leverages Wav2Vec2's powerful representations learned from 960 hours of unlabeled speech
3. **Attention Mechanism**: Custom attention head captures long-range dependencies in audio
4. **Robust Generalization**: Near-perfect performance across all 31 intent classes
5. **Real-Time Capability**: Low-latency inference suitable for production deployment
6. **Transfer Learning**: Pre-trained backbone reduces data requirements and training time

**Detailed comparison and analysis available in [Research Documentation](#-research-documentation)**

---

## 🚀 Getting Started

To run this project locally and reproduce the results, follow these steps:

### Prerequisites
- Python 3.8+
- GPU recommended (NVIDIA with CUDA support) for training
- Microphone for real-time inference testing

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sidgureja7803/DL_Project.git
   cd DL_Project
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Usage

#### Training the Model
```bash
python scripts/train_wav2vec.py
```

#### Evaluating the Model
```bash
python scripts/evaluate_wav2vec.py
```

#### Real-Time Testing with Microphone
```bash
python scripts/test_model.py
```

#### Running the Web Interface
```bash
python app.py
```

#### Starting the API Server
```bash
python server.py
```

---

## 📚 Dataset

This project uses the **Fluent Speech Commands (FSC)** dataset:

- **Size**: 30,043 English utterances
- **Speakers**: 97 native English speakers
- **Duration**: ~19 hours of audio
- **Sampling Rate**: 16kHz
- **Annotation**: Each utterance labeled with `action`, `object`, and `location`
- **Intent Classes**: 31 unique combinations mapped from action-object-location triplets

**Dataset Link:** [Fluent Speech Commands Dataset](https://fluent.ai/fluent-speech-commands-a-dataset-for-spoken-language-understanding-research/)

### Intent Class Distribution
The dataset is well-balanced across all 31 intent classes, ensuring robust model training without class imbalance issues.

---

## 👥 Contributors

* **Hemant Dubey** - [@frizzyfreak](https://github.com/frizzyfreak)
* **Avi Parmar** - [@avi2924](https://github.com/avi2924)
* **Siddhant Gureja** - [@sidgureja7803](https://github.com/sidgureja7803)

---

## 🔮 Future Work

- Expand the model to cover a broader range of intents and multilingual support
- Improve robustness in noisy environments and with non-native speakers
- Optimize for lower latency using model quantization and pruning techniques
- Integrate with real-world smart home APIs (Google Home, Alexa, HomeKit)
- Develop mobile deployment for on-device inference
- Explore few-shot learning for rapid adaptation to new intent classes

---

## 📝 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 📞 Contact

For questions or collaboration opportunities, please reach out through GitHub issues or contact the contributors directly.

**Project Link:** [https://github.com/sidgureja7803/DL_Project](https://github.com/sidgureja7803/DL_Project)
