# 🍔👁 Food Vision — Image Classification with EfficientNetB2 (PyTorch)

> Classifying **101 food categories** with **EfficientNetB2** + Transfer Learning, beating the [DeepFood](https://arxiv.org/abs/1606.05675) paper baseline by nearly 10 points.

---

## 📌 Overview

Food Vision is an end-to-end image classification project trained on the full [Food101 dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) — 101,000 images across 101 food categories. The model uses **EfficientNetB2** pretrained on ImageNet and is fine-tuned using a two-phase training strategy with data augmentation and label smoothing.

| Metric | Value |
|---|---|
| **Dataset** | Food101 (101 classes, 101k images) |
| **Base Model** | EfficientNetB2 (ImageNet weights) |
| **DeepFood Paper Baseline** | ~77.4% |
| **This Model — Feature Extraction** | 64.56% |
| **This Model — Fine-Tuned** | **87.20%** |
| **Framework** | PyTorch / Torchvision |
| **Training Hardware** | NVIDIA RTX 4050 |

---

## 🗂️ Project Structure

```
food-vision-101/
│
├── Food_Vision_PyTorch.ipynb   # Main notebook (full pipeline)
└── README.md
```

---

## 🔄 Pipeline

```
Food101 Dataset (via torchvision.datasets)
        │
        ▼
Data Transforms
(TrivialAugmentWide + EfficientNetB2 default transforms)
        │
        ▼
DataLoader (batch_size=32, shuffle=True)
        │
        ▼
Phase 1: Feature Extraction
(EfficientNetB2 frozen, train classifier head only, 5 epochs)
        │
        ▼
Phase 2: Fine-Tuning
(all layers unfrozen, lr=1e-4, 5 epochs)
        │
        ▼
Evaluation + Sample Predictions
```

---

## 🧠 Model Architecture

- **Backbone:** EfficientNetB2 (pretrained, `torchvision.models`)
- **Classifier Head:** `Dropout(0.3)` → `Linear(1408, 101)`
- **Loss:** `CrossEntropyLoss(label_smoothing=0.1)`

```
Input → EfficientNetB2 (frozen/unfrozen) → GlobalAvgPool → Dropout(0.3) → Linear(101) → Softmax
```

---

## ⚙️ Training Configuration

| Setting | Phase 1 (Feature Extract) | Phase 2 (Fine-Tune) |
|---|---|---|
| Base model frozen | ✅ Yes | ❌ No |
| Epochs | 5 | 5 |
| Optimizer | Adam (lr=1e-3) | Adam (lr=1e-4) |
| Loss | CrossEntropyLoss (ls=0.1) | CrossEntropyLoss (ls=0.1) |
| Augmentation | TrivialAugmentWide | TrivialAugmentWide |

---

## 🚀 How to Run

### Option 1 — Local (Recommended)

```bash
# Clone the repo
git clone https://github.com/RehanRaza/food-vision-101.git
cd food-vision-101

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torchinfo tqdm matplotlib jupyter

# Launch notebook
jupyter notebook Food_Vision_PyTorch.ipynb
```

### Option 2 — Google Colab

1. Open `Food_Vision_PyTorch.ipynb` in [Google Colab](https://colab.research.google.com/)
2. Set runtime to **GPU** → Runtime → Change runtime type → T4 GPU
3. Run all cells top to bottom

> ⚠️ **Note:** Training on full Food101 takes ~1.5–2 hrs on Colab free T4. Consider reducing epochs or using a local GPU.

---

## 📦 Dependencies

```
torch >= 1.12.0
torchvision >= 0.13.0
torchinfo
tqdm
matplotlib
```

---

## 📊 Results

| Model | Top-1 Accuracy |
|---|---|
| Original Food101 Paper (2014) | ~56.4% |
| DeepFood Paper (2016) | ~77.4% |
| Food Vision — Feature Extraction | 64.56% |
| Food Vision — Fine-Tuned ✅ | **87.20%** |

---

## 📚 References

- [Food101 Dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) — Bossard et al., 2014
- [DeepFood Paper](https://arxiv.org/abs/1606.05675) — Liu et al., 2016
- [EfficientNet](https://arxiv.org/abs/1905.11946) — Tan & Le, 2019
- [PyTorch EfficientNetB2](https://pytorch.org/vision/stable/models/efficientnet.html)

---

## 👤 Author

**Rehan Raza**  
[GitHub](https://github.com/RehanRaza) · [LinkedIn](https://linkedin.com/in/RehanRaza)
