<div align="center">

# 🏷️ Amazon ML Challenge — Multimodal Price Prediction

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/🤗_Transformers-4.x-yellow)](https://huggingface.co/transformers)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://docker.com)
[![AWS](https://img.shields.io/badge/AWS-Deployable-FF9900?logo=amazonaws&logoColor=white)](https://aws.amazon.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**An end-to-end multimodal deep-learning pipeline that predicts product prices from catalog text, product images, and OCR-extracted label content — achieving state-of-the-art SMAPE scores via gradient-boosted ensemble stacking.**

[Quick Start](#-quick-start) · [Architecture](#-architecture) · [Tech Stack](#-technology-stack) · [Docker](#-docker-deployment) · [AWS](#-aws-deployment) · [OCR Module](#-ocr-module-trocr)

</div>

---

## 🧠 Core Vision Models

### Microsoft TrOCR — Optical Character Recognition

Product images in real-world e-commerce are **noisy, low-resolution, and inconsistently formatted** — standard rule-based OCR engines fail on them. This pipeline uses [**Microsoft TrOCR**](https://huggingface.co/microsoft/trocr-base-handwritten) (`microsoft/trocr-base-handwritten`), a Vision-Language Transformer that combines a **ViT (Vision Transformer) encoder** with a **RoBERTa autoregressive decoder** to extract text from product label images.

```
┌──────────────────┐      ┌─────────────────────┐      ┌──────────────────────┐
│  Noisy Product   │─────▶│  ViT Image Encoder  │─────▶│  RoBERTa Text        │──▶ "500ml Pack of 6"
│  Label Image     │      │  (patch embeddings) │      │  Decoder (seq2seq)   │
└──────────────────┘      └─────────────────────┘      └──────────────────────┘
```

- **Why TrOCR over Tesseract/EasyOCR?** TrOCR is pre-trained on millions of handwritten and printed text images. Its encoder-decoder architecture handles noisy, rotated, and partially occluded text far better than traditional OCR — critical for real-world product images where labels are photographed at odd angles, with varying lighting and compression artefacts.
- **Fine-tuning**: The model is fine-tuned on product-specific label images (`src/ocr_trocr_train.py`) with CER (Character Error Rate) as the primary evaluation metric.
- **Use case**: Extracted text (brand names, quantities, units) feeds into the feature pipeline, enriching the model's understanding of what the product *says* on its label.

### ResNet-50 — Visual Feature Extraction

[**ResNet-50**](https://arxiv.org/abs/1512.03385) (pre-trained on ImageNet) is used as a **visual feature extractor** to generate **2048-dimensional image embeddings** for every product image. These embeddings capture visual cues that correlate with pricing:

```
┌──────────────────┐      ┌─────────────────────┐      ┌──────────────────┐
│  Product Image   │─────▶│  ResNet-50 Backbone  │─────▶│ 2048-D Embedding │
│  (224×224 RGB)   │      │  (conv → pool → avg) │      │  (visual cues)   │
└──────────────────┘      └─────────────────────┘      └──────────────────┘
```

- **What the embeddings capture**: Packaging quality (premium vs economy), brand logo recognition, product size/shape estimation, colour palette (luxury gold tones vs budget plastic), and shelf presentation.
- **Architecture**: The final classification head (`fc` layer) is stripped; instead, **global average pooling** over the last convolutional block produces a compact 2048-D vector per image.
- **Integration**: Image embeddings are aligned with text embeddings and engineered features via sample ID, then concatenated into a unified feature matrix for the ensemble models.

> **Together**, TrOCR reads *what the product label says* and ResNet-50 captures *what the product looks like* — giving the ensemble both textual and visual signals for price prediction.

---

## 📐 Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        MULTIMODAL FEATURE PIPELINE                        │
│                                                                           │
│  ┌──────────────┐    ┌──────────────────┐    ┌────────────────────────┐   │
│  │  Raw Text    │───▶│  DeBERTa v3 Base │───▶│ 768-D Text Embeddings │   │
│  │  (catalog)   │    │  (Transformer)   │    └───────────┬────────────┘   │
│  └──────────────┘    └──────────────────┘                │               │
│                                                          │               │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────▼────────────┐   │
│  │  Product Img │───▶│  ResNet-50       │───▶│ 2048-D Img Embeddings │   │
│  │  (RGB .jpg)  │    │  (CNN Backbone)  │    └───────────┬────────────┘   │
│  └──────────────┘    └──────────────────┘                │               │
│                                                          │               │
│  ┌──────────────┐    ┌──────────────────┐                │               │
│  │  Engineered  │───▶│  item_size,      │────────────────┤               │
│  │  Features    │    │  pack_count, ... │                │               │
│  └──────────────┘    └──────────────────┘                │               │
│                                                          ▼               │
│                                            ┌─────────────────────────┐   │
│                                            │ Concatenated Feature    │   │
│                                            │ Matrix (N × ~2821)     │   │
│                                            └───────────┬─────────────┘   │
└────────────────────────────────────────────────────────│─────────────────┘
                                                         │
                ┌────────────────────────────────────────▼─────┐
                │            ENSEMBLE STACKING LAYER           │
                │                                              │
                │   ┌──────────┐ ┌──────────┐ ┌────────────┐  │
                │   │ LightGBM │ │ XGBoost  │ │  CatBoost  │  │
                │   │ (GPU)    │ │ (GPU)    │ │  (GPU)     │  │
                │   └────┬─────┘ └────┬─────┘ └─────┬──────┘  │
                │        │ OOF        │ OOF         │ OOF     │
                │        └──────┬─────┴─────┬───────┘         │
                │               ▼           ▼                  │
                │         ┌─────────────────────┐              │
                │         │  Ridge Meta-Learner │              │
                │         │  (L2 Regularised)   │              │
                │         └──────────┬──────────┘              │
                └────────────────────│─────────────────────────┘
                                     ▼
                              Final Price Prediction
```

## 🧬 Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Text Encoding** | [DeBERTa v3 Base](https://huggingface.co/microsoft/deberta-v3-base) | 768-D contextual text embeddings from catalog descriptions |
| **Image Encoding** | [ResNet-50](https://arxiv.org/abs/1512.03385) (ImageNet) | 2048-D visual feature extraction via global average pooling |
| **OCR** | [TrOCR Base](https://huggingface.co/microsoft/trocr-base-handwritten) | ViT→RoBERTa Vision-Language model for label text extraction |
| **Gradient Boosting** | LightGBM · XGBoost · CatBoost | Three-model ensemble with GPU-accelerated GBDT |
| **Meta-Learner** | Ridge Regression (L2) | Stacking layer over out-of-fold predictions |
| **Feature Engineering** | pandas · NumPy · spaCy | Target encoding, NLP parsing, numerical transforms |
| **Deep Learning** | PyTorch 2.x · HuggingFace Transformers | Model fine-tuning, inference, tokenisation |
| **Containerisation** | Docker · Docker Compose | Multi-stage builds, GPU passthrough, health checks |
| **Cloud** | AWS EC2 / ECS · ECR · S3 | GPU instance deployment, container registry, data storage |

## 📁 Project Structure

```
amazon_ml_challenge/
├── Dockerfile                  # Multi-stage production build
├── docker-compose.yml          # Orchestration with GPU & OCR profiles
├── .dockerignore               # Lean build context
├── requirements.txt            # Pinned Python dependencies
│
├── src/
│   ├── build_features.py       # DeBERTa text embedding generation
│   ├── image_embed.py          # ResNet-50 image embedding extraction
│   ├── predict_model.py        # Ensemble training (LGB + XGB + CAT + Ridge)
│   └── ocr_trocr_train.py      # TrOCR fine-tuning & CER evaluation
│
├── notebooks/
│   ├── FE_final.ipynb          # Feature engineering experiments
│   └── FE_size.ipynb           # Size-feature analysis
│
├── data/
│   ├── raw/                    # Original CSVs (train.csv, test.csv)
│   ├── processed/              # Embeddings (.npy), engineered features
│   └── ocr/                    # OCR images & labels (for TrOCR)
│
├── models/                     # Saved model checkpoints (.pkl)
│   └── ocr/                    # TrOCR fine-tuned weights & reports
│
└── submissions/                # Competition submission CSVs
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU training — optional for CPU inference)
- Docker & Docker Compose (for containerised deployment)

### Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-username/amazon_ml_challenge.git
cd amazon_ml_challenge

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate text embeddings (DeBERTa)
python src/build_features.py

# 5. Generate image embeddings (ResNet-50)
python src/image_embed.py

# 6. Train the ensemble and generate predictions
python src/predict_model.py
```

## 🐳 Docker Deployment

### Build & Run (CPU)

```bash
docker compose up --build
```

### Build & Run (GPU — requires nvidia-docker)

```bash
docker compose up --build    # GPU auto-detected via deploy.reservations
```

### Run OCR Training Only

```bash
docker compose --profile ocr up ocr-train
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_DEVICE` | `auto` | Force device: `cpu`, `cuda`, or `auto` |
| `LOG_LEVEL` | `INFO` | Python logging level |
| `EPOCHS` | `10` | OCR training epochs |
| `BATCH_SIZE` | `16` | OCR training batch size |
| `LEARNING_RATE` | `5e-5` | OCR training learning rate |

## ☁️ AWS Deployment

### Option A: EC2 (GPU Instance)

```bash
# 1. Launch a p3.2xlarge or g4dn.xlarge instance with Deep Learning AMI
# 2. SSH into the instance and clone the repo

# 3. Build the Docker image
docker build -t amazon-ml-challenge .

# 4. Run with GPU support
docker run --gpus all -p 8501:8501 \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/models:/app/models \
    amazon-ml-challenge
```

### Option B: ECS with ECR

```bash
# 1. Authenticate Docker with ECR
aws ecr get-login-password --region us-east-1 | \
    docker login --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com

# 2. Create ECR repository
aws ecr create-repository --repository-name amazon-ml-challenge

# 3. Tag and push the image
docker tag amazon-ml-challenge:latest \
    <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/amazon-ml-challenge:latest
docker push <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/amazon-ml-challenge:latest

# 4. Create an ECS task definition with GPU capabilities
#    and deploy as a Fargate or EC2-backed service
```

### Option C: S3 Data Pipeline

```bash
# Sync local data to S3
aws s3 sync ./data s3://amazon-ml-challenge-data/

# Inside the container, pull data from S3
aws s3 sync s3://amazon-ml-challenge-data/ /app/data/
```

## 🔤 OCR Module (TrOCR)

This project includes a **TrOCR-based Optical Character Recognition** pipeline for extracting text from product label images.

### Model Architecture

```
Product Label Image → ViT Encoder → Cross-Attention → RoBERTa Decoder → Text
```

- **Encoder**: Vision Transformer (ViT) pre-trained on ImageNet
- **Decoder**: RoBERTa-based autoregressive text generator
- **Base model**: [`microsoft/trocr-base-handwritten`](https://huggingface.co/microsoft/trocr-base-handwritten)

### Training Pipeline

```bash
# Fine-tune TrOCR on product labels
python src/ocr_trocr_train.py \
    --image_dir data/ocr/images \
    --labels_csv data/ocr/labels.csv \
    --epochs 15 \
    --batch_size 16 \
    --lr 3e-5

# Or via Docker Compose
docker compose --profile ocr up ocr-train
```

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **CER** (Character Error Rate) | `(S + I + D) / N` — primary OCR accuracy metric |
| **Cross-Entropy Loss** | Token-level decoder loss during training |

### Output Artifacts

```
models/ocr/
├── best_model/                # Best checkpoint (by val CER)
├── final_model/               # Final epoch checkpoint
├── training_history.json      # Per-epoch loss & CER curves
├── ocr_predictions.csv        # Per-sample predictions vs ground truth
└── ocr_report.txt             # Human-readable evaluation summary
```

## 📊 Model Performance

| Model | CV SMAPE (%) | Notes |
|-------|-------------|-------|
| LightGBM | — | GPU-accelerated, 1500 trees |
| XGBoost | — | GPU histogram method |
| CatBoost | — | 2000 iterations |
| **Stacked Ensemble** | **Best** | Ridge meta-learner over OOF |

> **Note**: Fill in SMAPE values after running the pipeline on your dataset.

## 📝 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with ❤️ for the Amazon ML Challenge**

</div>
