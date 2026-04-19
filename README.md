# Image-Inpainting-Using-Diffusion-Models
# 🧠 Symmetry-Aware Image Inpainting using Stable Diffusion + LoRA

---

## 📌 Problem Statement

Image inpainting aims to reconstruct missing or corrupted regions in an image. While modern generative models can produce visually appealing outputs, they often suffer from:

* ❌ Lack of structural consistency
* ❌ Hallucinated or unrealistic content
* ❌ Poor reconstruction for structured objects (faces, monuments)
* ❌ Sensitivity to mask size and missing context

This project focuses on building a **robust inpainting framework** that can:

* Restore **human faces** with realistic features
* Reconstruct **monuments** while preserving symmetry and structure
* Maintain **context-awareness** across varying mask sizes

---

## 🎯 Objective

To develop a **transfer learning-based inpainting system** using diffusion models that:

* Learns from structured datasets (faces & monuments)
* Handles varying levels of missing regions (5% → 50%)
* Produces high-quality reconstructions with minimal artifacts

---

## 🏗️ Model Architecture

We build upon:

👉 **Stable Diffusion Inpainting Model**

* Pretrained model: `runwayml/stable-diffusion-inpainting`
* Components:

  * **VAE** → encodes images to latent space
  * **UNet** → performs denoising (core learning module)
  * **Text Encoder** → guides generation via prompts

---

## 🔧 Key Modification: LoRA Fine-Tuning

Instead of training the full model (very expensive), we use:

👉 **LoRA (Low-Rank Adaptation)**

### Why LoRA?

* Efficient fine-tuning
* Reduces GPU memory usage
* Faster convergence
* Keeps pretrained knowledge intact

### Applied on:

* Attention layers:

  * `to_q`, `to_k`, `to_v`, `to_out`

This allows the model to **adapt to domain-specific structures** like faces and monuments.

---

## 🧪 Training Strategy

We trained **two separate models**:

### 1. 👤 Face Inpainting

* Dataset: **CelebA (~200K images)**
* Focus:

  * Facial symmetry
  * Skin texture consistency
  * Natural feature reconstruction

---

### 2. 🏛️ Monument Inpainting

* Dataset: Custom curated (~few thousand images)
* Focus:

  * Structural alignment
  * Symmetry preservation
  * Texture continuity (stone, carvings)

---

## 🎭 Masking Strategy

To simulate missing regions:

* Small masks (5%) → multiple patches
* Large masks (10%–50%) → square regions

This helps evaluate performance under **progressive difficulty**.

---

## ⚙️ Training Details

* Image size: 256 × 256
* Batch size: 4 (with gradient accumulation)
* Learning rate: ~6e-6 to 8e-6
* Inference steps: 50–75
* Guidance scale: 3.0 – 7.5

---

## 📉 Loss Functions

We used a combination of:

### 1. Reconstruction Loss (L1 / MSE)

* Ensures pixel-level accuracy

### 2. Perceptual Loss (optional in tuning phase)

* Helps preserve high-level features
* Improves realism

---

## 🧠 Key Idea

👉 Early training learns **structure**
👉 Later training refines **details**

This staged learning improves both:

* Geometry (important for monuments)
* Texture (important for faces)

---

## 📊 Evaluation Metrics

We evaluate using:

* **PSNR (Peak Signal-to-Noise Ratio)** → pixel accuracy
* **SSIM (Structural Similarity Index)** → perceptual quality

---

# 📈 RESULTS

---

## 👤 Face Dataset (CelebA)

### Without Prompt

| Mask % | PSNR  | SSIM  |
| ------ | ----- | ----- |
| 5%     | 32.24 | 0.947 |
| 10%    | 27.95 | 0.922 |
| 20%    | 23.08 | 0.862 |
| 30%    | 20.12 | 0.789 |
| 40%    | 17.87 | 0.707 |
| 50%    | 15.47 | 0.623 |

### With Prompt

| Mask % | PSNR  | SSIM  |
| ------ | ----- | ----- |
| 5%     | 32.43 | 0.948 |
| 10%    | 27.72 | 0.920 |
| 20%    | 23.43 | 0.867 |
| 30%    | 19.79 | 0.794 |
| 40%    | 18.12 | 0.717 |
| 50%    | 15.93 | 0.624 |

---

## 👤 Real Face Images (Generalization)

| Mask % | PSNR (On) | SSIM (On) |
| ------ | --------- | --------- |
| 5%     | 33.26     | 0.955     |
| 10%    | 28.86     | 0.936     |
| 20%    | 23.72     | 0.874     |
| 30%    | 22.42     | 0.810     |
| 40%    | 19.41     | 0.754     |
| 50%    | 17.09     | 0.693     |

👉 Shows strong **generalization beyond training dataset**

---

## 🏛️ Monument Results

| Mask % | Prompt ON | Prompt OFF |
| ------ | --------- | ---------- |
| 3%     | 30.19     | 27.81      |
| 10%    | 29.21     | 26.96      |
| 15%    | 28.26     | 26.62      |
| 20%    | 27.19     | 25.96      |

👉 Prompts significantly improve **structural reconstruction**

---

## 📌 Observations

* Performance decreases as mask size increases (expected)
* Prompt guidance improves:

  * Semantic consistency
  * Structural recovery
* Faces show smoother degradation than monuments
* Monuments require stronger structural priors

---

## 🖼️ Sample Outputs

(Add your saved images here)

* Original | Mask | Masked | Output comparisons
* Both prompt ON and OFF

---

## ⚠️ Limitations

* Struggles with very low-resolution inputs
* Large missing regions (>50%) lead to hallucination
* Monuments require stronger symmetry constraints

---

## 🚀 Future Work

* Add **symmetry-aware loss** (important for monuments)
* Integrate **super-resolution preprocessing**
* Improve mask-aware attention
* Reduce hallucination using structural priors

---

## 🧠 Key Takeaways

* Diffusion + LoRA is powerful for domain-specific tasks
* Transfer learning significantly reduces training cost
* Structured objects need **context + symmetry awareness**
* Prompt engineering plays a critical role

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
python train_faces.py
python train_monuments.py
python inference.py
```

---

## 📂 Project Structure

```
├── train_faces.py
├── train_monuments.py
├── inference.py
├── evaluate.py
├── notebooks/
├── results/
├── requirements.txt
└── README.md
```

---

## 🙌 Final Note

This project demonstrates a **practical and scalable approach** to image inpainting using modern generative models, with a strong focus on **real-world robustness and structured reconstruction**.

---
