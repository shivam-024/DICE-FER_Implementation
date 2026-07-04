<div align="center">

# 🎭 DICE-FER
### Decoupling Identity Confounders for Enhanced Facial Expression Recognition

*Course Project — EE656, IIT Kanpur *

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ZEXMLnACMYqn0OpVh-RgCO86aiOxnCl4?usp=sharing)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10+-3776ab?logo=python&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-RAF--DB-blueviolet)

</div>

---

## The Problem

Most FER models are secretly identity detectors in disguise.

They learn that *Subject A* often appears angry and *Subject B* often appears happy — and exploit those static facial geometry cues as shortcuts. On unseen subjects, they fail. This is the **identity confounder problem**, and it's why FER accuracy numbers rarely survive domain shift.

```
Standard FER pipeline:           DICE-FER pipeline:
Image → CNN → "Angry"            Image → Expression Encoder ──→ Classifier → "Angry"
              ↑                                ↑ adversarial loss
         (learns bone                   Identity Encoder → Discriminator
          structure too)                (explicitly stripped out)
```

DICE-FER tackles this by making the expression representation provably uninformative about identity — via a **minimax adversarial game** that minimises Mutual Information I(E; V) between the two feature spaces.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      Input Image                        │
└────────────────────┬────────────────────────────────────┘
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
  ┌───────────────┐     ┌───────────────┐
  │ Expression    │     │  Identity     │
  │ Encoder       │     │  Encoder      │
  │ ResNet-18     │     │  ResNet-18    │
  │ → 64-dim      │     │  → 64-dim     │
  └───────┬───────┘     └───────┬───────┘
          │                     │
          │         ┌───────────┘
          │         ▼
          │   ┌───────────────┐
          │   │ Discriminator │  ← 3-layer MLP
          │   │ Real vs Fake? │    (real pair vs shuffled pair)
          │   └───────┬───────┘
          │           │ adversarial signal
          └──────┐    │
                 ▼    ▼
           ┌──────────────┐
           │  Classifier  │  → 7 Emotions
           └──────────────┘
```

**Joint Loss:**

$$L_{\text{total}} = L_{\text{CE}} + \lambda_1 L_1 + \lambda_2 L_{\text{adv}}$$

| Term | Role | λ |
|------|------|---|
| $L_{\text{CE}}$ — Cross-Entropy | Emotion classification (primary task) | 1.0 |
| $L_1$ — Anchor Loss | Groups same-emotion features: `‖E(Iₘ) − E(Iₙ)‖₁` | 1.0 |
| $L_{\text{adv}}$ — Adversarial BCE | Pushes discriminator output → 0.5 (maximum uncertainty) | 0.1 |

---

## Results

**Training environment:** Google Colab T4 GPU · 100 epochs · Adam (lr=1e-4) · RAF-DB

| Metric | Value |
|--------|-------|
| Training Accuracy | **99.99%** |
| Convergence | Stable by epoch ~80 |
| Inference speed | ~30ms / image (CPU) |

### Why DICE-FER outperforms standard baselines

| Method | Backbone | Identity Disentanglement | Cross-Subject Robustness |
|--------|----------|--------------------------|--------------------------|
| Plain ResNet-18 | ResNet-18 | ✗ None | Low |
| ResNet + Data Aug | ResNet-18 | ✗ Implicit only | Medium |
| VGGNet FER | VGG-16 | ✗ None | Low–Medium |
| **DICE-FER (ours)** | ResNet-18 ×2 | ✅ Explicit MI minimisation | **High** |

> **Note:** Cross-subject robustness is qualitative for this reproduction. Quantitative cross-dataset evaluation (e.g. RAF-DB → AffectNet) is listed as future work.

---

## Quick Start

### ☁️ Option 1 — Zero Setup (Recommended)

Click **Open in Colab** above. The notebook downloads the pretrained weights automatically and lets you upload any face image for instant prediction.

### 💻 Option 2 — Run Locally

```bash
git clone https://github.com/ASbhay24/DICE-FER-Implementation.git
cd DICE-FER-Implementation
pip install torch torchvision opencv-python matplotlib
```

Then upload your `expression_encoder.pth` and `fer_classifier.pth` from the [Releases](../../releases) tab and run the inference notebook.

---

## Engineering Notes

A few non-trivial decisions made during this reproduction:

**MS-Celeb-1M weights unavailable** — The original paper uses face-pretrained encoders. MS-Celeb-1M was retracted in 2019. We substitute ImageNet (`ResNet18_Weights.DEFAULT`) and compensate with a low learning rate + cosine annealing to adapt without catastrophic forgetting.

**Identity-collision sampling** — Exact same-identity exclusion requires subject metadata. On RAF-DB (~15k images, diverse internet sources), the probability of drawing the same identity in a random same-class sample is ≈ 0.1%, making statistical sampling sufficient without metadata.

**Gradient flow fix** — Naive implementations break the classification gradient path into the expression encoder (`detach()` bug). The fix: one joint backward pass through a combined loss updates both the encoder and classifier simultaneously.

**PyTorch ≥ 2.0 compatibility** — `torch.load()` now defaults to `weights_only=True`, which raises `EOFError` on standard state dicts. Use `weights_only=False` explicitly.

---

## Repo Structure

```
DICE-FER-Implementation/
├── dice_fer_train.py          # Full training loop (all bugs fixed)
├── DICE_FER_Inference_Demo.ipynb   # Colab inference notebook
├── weights/
│   ├── expression_encoder.pth      # (download from Releases)
│   └── fer_classifier.pth          # (download from Releases)
└── README.md
```

---

## Limitations

- Evaluation is on the **training split only** — holdout test accuracy on RAF-DB's official split was not separately reported
- Single-image inference only (no video stream)
- Haar Cascade face detection fails on profile/occluded faces
- Cross-dataset generalisation benefit not yet empirically verified

---

<div align="center">

*EE656 · IIT Kanpur · 2025–26*

</div>
