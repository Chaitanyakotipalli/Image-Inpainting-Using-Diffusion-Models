# Image-Inpainting-Using-Diffusion-Models
#  Symmetry-Aware Image Inpainting using Stable Diffusion + LoRA

---

##  Problem Statement

Image inpainting aims to reconstruct missing or corrupted regions in an image. While modern generative models can produce visually appealing outputs, they often suffer from:

*  Lack of structural consistency
*  Hallucinated or unrealistic content
*  Poor reconstruction for structured objects (faces, monuments)
*  Sensitivity to mask size and missing context

This project focuses on building a **robust inpainting framework** that can:

* Restore **human faces** with realistic features
* Reconstruct **monuments** while preserving symmetry and structure
* Maintain **context-awareness** across varying mask sizes

---

##  Objective

To develop a **transfer learning-based inpainting system** using diffusion models that:

* Learns from structured datasets (faces & monuments)
* Handles varying levels of missing regions (5% → 50%)
* Produces high-quality reconstructions with minimal artifacts

---

## 🏗️ Model Architecture

We build upon:

 **Stable Diffusion Inpainting Model**

* Pretrained model: `runwayml/stable-diffusion-inpainting`
* Components:

  * **VAE** → encodes images to latent space
  * **UNet** → performs denoising (core learning module)
  * **Text Encoder** → guides generation via prompts

---

## 🔧 Key Modification: LoRA Fine-Tuning

Instead of training the full model we use:

 **LoRA (Low-Rank Adaptation)**

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

 Early training learns **structure**
 Later training refines **details**

This staged learning improves both:

* Geometry (important for monuments)
* Texture (important for faces)

---

## 📊 Evaluation Metrics

We evaluate using:

* **PSNR (Peak Signal-to-Noise Ratio)** → pixel accuracy
* **SSIM (Structural Similarity Index)** → perceptual quality

---
## 🏛️ Monument Inpainting Architecture (SD2 + Canny-Guided Structure Control)

Monument reconstruction is fundamentally different from face inpainting. While faces rely more on texture and learned priors, monuments require strict preservation of **geometry, edges, and symmetry**. Standard diffusion models tend to generate visually plausible but **structurally incorrect outputs** (hallucinated shapes, broken alignment).

To solve this, we designed a **structure-aware inpainting pipeline** using:

* **Stable Diffusion 2 (SD2)**
* **Canny Edge Conditioning**

---

### 🔹 Base Model

* **Stable Diffusion 2 (SD2)**
* Chosen for:

  * Better latent representations
  * Improved handling of complex textures and scenes
  * More stable generation compared to SD1 for structured objects

---

### 🔹 Key Idea

Instead of relying only on generative capability, we inject **explicit structural information** into the model using **Canny edge maps**.

 This ensures that the generated content follows the **true geometric layout** of the monument.

---

## 🔧 Pipeline (as implemented)

### 1. Input Image

* Monument image is loaded and resized (e.g., 256×256)
* Acts as the base reference

---

### 2. Mask Generation

* A region (3%–20%) is masked
* Simulates missing/damaged parts of the structure

---

### 3. Canny Edge Extraction (Core Step)

We apply **Canny Edge Detection** on the original image to extract structural features:

* Detects:

  * boundaries
  * edges
  * contours
  * repeating patterns

 This produces a **binary edge map** representing the structure

---

###  Why Canny?

Canny was specifically used because:

* It captures **sharp structural boundaries**
* Works well for architectural images (clear edges)
* Removes unnecessary texture noise
* Provides a **clean geometric guide**

Without Canny:

* Model relies only on learned priors
* Leads to:

  * distorted buildings
  * broken symmetry
  * unrealistic geometry

With Canny:

* Model follows **actual structure of the monument**
* Reduces hallucination significantly

---

### 4. Conditioning the Diffusion Model

The model is conditioned using:

* Masked image
* Mask
* **Canny edge map (structure guidance)**
* Optional text prompt

 The edge map acts as a **hard structural constraint**

---

### 5. Diffusion Process

Inside SD2:

* Image is encoded into latent space via VAE
* UNet performs iterative denoising
* During reconstruction:

  * masked regions are filled
  * generation is guided by:

    * surrounding context
    * **edge structure (Canny)**
    * prompt (if enabled)

---

### 6. Output Generation

* Latent output is decoded back to image space
* Final output preserves:

  * alignment of structures
  * edge continuity
  * architectural symmetry

---

##  Design Insight

| Without Canny         | With Canny              |
| --------------------- | ----------------------- |
| blurry shapes         | sharp structures        |
| hallucinated geometry | accurate reconstruction |
| weak alignment        | strong edge consistency |

---

##  Key Advantages

* Enforces **geometry-aware reconstruction**
* Preserves **edges and symmetry**
* Reduces diffusion hallucination
* Improves results on **structured objects**

---

##  Limitations

* Strong dependence on edge quality
* Very large masks still challenging
* Fine details (carvings) may not fully recover

---

##  Summary

This pipeline enhances Stable Diffusion 2 by integrating **Canny-based structural conditioning**, transforming it from a purely generative model into a **structure-aware inpainting system**, particularly effective for monuments and architectural scenes.

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

 Shows strong **generalization beyond training dataset**

---

## 🏛️ Monument Results

| Mask % | Prompt ON | Prompt OFF |
| ------ | --------- | ---------- |
| 3%     | 30.19     | 27.81      |
| 10%    | 29.21     | 26.96      |
| 15%    | 28.26     | 26.62      |
| 20%    | 27.19     | 25.96      |

 Prompts significantly improve **structural reconstruction**

---

##  Observations

* Performance decreases as mask size increases (expected)
* Prompt guidance improves:

  * Semantic consistency
  * Structural recovery
* Faces show smoother degradation than monuments
* Monuments require stronger structural priors

---

## 🖼️ Sample Outputs

<img width="1536" height="512" alt="mask_10_rank1_PSNR_29 08_SSIM_0 8093" src="https://github.com/user-attachments/assets/08126449-1c0a-454e-a674-1db46d6cf6ee" />

<img width="1536" height="512" alt="mask_10_rank2_PSNR_28 40_SSIM_0 8063" src="https://github.com/user-attachments/assets/66812ec5-c352-4c3d-94c7-3cb390f155ae" />
<img width="1000" height="300" alt="mask_20_0" src="https://github.com/user-attachments/assets/8e8e6ce5-2098-48fd-ac6b-169cfa72f4f3" />
<img width="1000" height="300" alt="mask_20_1" src="https://github.com/user-attachments/assets/5baebed1-bda2-4c88-9237-afa70ccd42f5" />



* Original | Mask | Masked | Output comparisons
* Both prompt ON and OFF

---

##  Limitations

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


