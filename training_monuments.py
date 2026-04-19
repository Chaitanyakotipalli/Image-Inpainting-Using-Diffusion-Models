import os
import random
import gc
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from diffusers import (
    StableDiffusionControlNetInpaintPipeline,
    ControlNetModel,
    DDIMScheduler,
    AutoencoderKL,
    UNet2DConditionModel
)
from transformers import CLIPTokenizer, CLIPTextModel
from peft import LoraConfig, get_peft_model
class CFG:
    DATA = "/kaggle/input/datasets/chaitanyakotipalli18/final-data/mon_data/monuments"   # main dataset
    TEMPLE_DATA = "/kaggle/input/datasets/chaitanyakotipalli18/final-data/monuments_test"   # separate temple images for test
    OUT  = "/kaggle/working/results"
    LORA = "/kaggle/working/lora"

    SIZE = 512
    EPOCHS = 6
    BS = 1
    LR = 5e-6

    # PROMPT = "realistic architectural structure, symmetric building, consistent perspective"
    # NEG = "blurry, distorted, warped, artifacts"
    PROMPT=""
    NEG=""

    STEPS = 50
    GUIDANCE = 6.5
    SEED = 42

    SD = "sd2-community/stable-diffusion-2-inpainting"
    CN = "thibaud/controlnet-sd21-canny-diffusers"

cfg = CFG()
os.makedirs(cfg.OUT, exist_ok=True)
os.makedirs(cfg.LORA, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

def make_mask(size):
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)

    # small + medium only (no large)
    w = random.randint(size//10, size//3)
    h = random.randint(size//10, size//3)

    x = random.randint(0, size - w)
    y = random.randint(0, size - h)

    draw.rectangle([x, y, x+w, y+h], fill=255)
    return mask
def canny(img):
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    med = np.median(gray)
    low = int(max(0, 0.66 * med))
    high = int(min(255, 1.33 * med))
    edge = cv2.Canny(gray, low, high)
    return Image.fromarray(cv2.cvtColor(edge, cv2.COLOR_GRAY2RGB))

class DS(Dataset):
    def __init__(self, paths):
        self.paths = paths
        self.tf = transforms.Compose([
            transforms.Resize((cfg.SIZE, cfg.SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.1, 0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB").resize((cfg.SIZE, cfg.SIZE))
        mask = make_mask(cfg.SIZE)
        img_np = np.array(img)
        mask_np = np.array(mask)
        masked = img_np.copy()
        masked[mask_np > 127] = 0   # black fill in masked region
        return {
            "img": self.tf(img),
            "mask": transforms.ToTensor()(mask),
            "masked": self.tf(Image.fromarray(masked))
        }
@torch.no_grad()
def validate(unet, val_loader, vae, scheduler, text_emb):
    unet.eval()
    total_loss = 0
    for batch in val_loader:
        img = batch["img"].to(DEVICE, dtype=torch.float16)
        mask = batch["mask"].to(DEVICE, dtype=torch.float16)
        masked = batch["masked"].to(DEVICE, dtype=torch.float16)

        lat = vae.encode(img).latent_dist.sample() * vae.config.scaling_factor
        mlat = vae.encode(masked).latent_dist.sample() * vae.config.scaling_factor
        noise = torch.randn_like(lat)
        t = torch.randint(0, scheduler.config.num_train_timesteps, (lat.shape[0],), device=DEVICE).long()
        noisy = scheduler.add_noise(lat, noise, t)
        mask_ds = F.interpolate(mask, size=lat.shape[-2:], mode="nearest")
        inp = torch.cat([noisy, mask_ds, mlat], dim=1)
        pred = unet(inp, t, encoder_hidden_states=text_emb).sample
        loss = F.mse_loss(pred, noise)
        total_loss += loss.item()
    return total_loss / max(1, len(val_loader))

# ================= MODEL + TEXT SETUP =================

from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler

# Load tokenizer + text encoder (FP16)
tokenizer = CLIPTokenizer.from_pretrained(cfg.SD, subfolder="tokenizer")

text_enc = CLIPTextModel.from_pretrained(
    cfg.SD,
    subfolder="text_encoder",
    torch_dtype=torch.float16
).to(DEVICE)

# Precompute text embedding
text_input = tokenizer(
    cfg.PROMPT,
    return_tensors="pt",
    padding="max_length",
    max_length=77,
    truncation=True
)

with torch.no_grad():
    text_emb = text_enc(tokenizer("", return_tensors="pt", padding="max_length", max_length=77).input_ids.to(DEVICE))[0]

text_emb = text_emb.detach()  
print(" Text embedding ready")

# Load VAE + UNet (FP16 → IMPORTANT FIX)
vae = AutoencoderKL.from_pretrained(
    cfg.SD,
    subfolder="vae",
    torch_dtype=torch.float16
).to(DEVICE)

unet = UNet2DConditionModel.from_pretrained(
    cfg.SD,
    subfolder="unet",
    torch_dtype=torch.float16
).to(DEVICE)

scheduler = DDIMScheduler.from_pretrained(cfg.SD, subfolder="scheduler")

print(" Models loaded (FP16)")

def train(train_paths, val_paths, text_emb, resume_from=None):
    text_emb = text_emb.detach()

    vae.requires_grad_(False)
    text_enc.requires_grad_(False)
    unet.requires_grad_(False)

    from peft import LoraConfig, get_peft_model

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.05
    )

    lora_unet = get_peft_model(unet, lora)
    print(f"Trainable parameters: {lora_unet.num_parameters(only_trainable=True):,}")

    train_ds = DS(train_paths)
    val_ds   = DS(val_paths)

    train_dl = DataLoader(train_ds, batch_size=cfg.BS, shuffle=True, num_workers=0)
    val_dl   = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(lora_unet.parameters(), lr=cfg.LR)

    # ----- RESUME LOGIC -----
    start_epoch = 0
    best_val_loss = float("inf")
    if resume_from is not None and os.path.exists(resume_from):
        print(f" Resuming from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=DEVICE)
        lora_unet.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        print(f"   Resumed from epoch {start_epoch-1} | Best val loss so far: {best_val_loss:.4f}")
    else:
        print(" Starting training from scratch")

    print("\n Starting training...")

    for epoch in range(start_epoch, cfg.EPOCHS):
        lora_unet.train()
        total_loss = 0

        for batch in tqdm(train_dl, desc=f"Epoch {epoch+1}"):
            img = batch["img"].to(DEVICE, dtype=torch.float16)
            mask = batch["mask"].to(DEVICE, dtype=torch.float16)
            masked = batch["masked"].to(DEVICE, dtype=torch.float16)

            with torch.no_grad():
                lat = vae.encode(img).latent_dist.sample() * vae.config.scaling_factor
                mlat = vae.encode(masked).latent_dist.sample() * vae.config.scaling_factor

            noise = torch.randn_like(lat)
            t = torch.randint(0, scheduler.config.num_train_timesteps, (lat.shape[0],), device=DEVICE).long()

            noisy = scheduler.add_noise(lat, noise, t)

            mask_ds = F.interpolate(mask, size=lat.shape[-2:], mode="nearest")
            inp = torch.cat([noisy, mask_ds, mlat], dim=1)

            pred = lora_unet(inp, t, encoder_hidden_states=text_emb).sample
            loss = F.mse_loss(pred, noise)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item()

        avg_train = total_loss / len(train_dl)
        val_loss = validate(lora_unet, val_dl, vae, scheduler, text_emb)

        print(f"Epoch {epoch+1} | Train: {avg_train:.4f} | Val: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            lora_unet.save_pretrained(cfg.LORA)
            print(f" Best model saved (val loss: {val_loss:.4f})")
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": lora_unet.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
        }
        torch.save(checkpoint, f"{cfg.OUT}/checkpoint_epoch_{epoch+1}.pt")
        print(f" Checkpoint saved: {cfg.OUT}/checkpoint_epoch_{epoch+1}.pt")
        lora_unet.save_pretrained(cfg.LORA)
        gc.collect()
        torch.cuda.empty_cache()

    print("\nTraining completed")
    return lora_unet

from peft import PeftModel

def load_pipe():

    control = ControlNetModel.from_pretrained(
        "thibaud/controlnet-sd21-canny-diffusers", 
        torch_dtype=torch.float16
    )

    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        cfg.SD,
        controlnet=control,
        torch_dtype=torch.float16,
        safety_checker=None
    )

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(DEVICE)

    # LoRA load (correct)
    pipe.unet = PeftModel.from_pretrained(pipe.unet, cfg.LORA)

    return pipe

def blend(orig, out, mask):
    src = np.array(out)
    dst = np.array(orig)
    m = np.array(mask)
    M = cv2.moments(m)
    if M["m00"] == 0:
        return out
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    try:
        res = cv2.seamlessClone(src, dst, m, (cx, cy), cv2.NORMAL_CLONE)
        return Image.fromarray(res)
    except:
        return out

def infer(pipe, img, mask):
    edge = canny(img).resize((cfg.SIZE, cfg.SIZE))
    gen = torch.Generator(device=DEVICE).manual_seed(cfg.SEED)
    out = pipe(
        prompt=cfg.PROMPT,
        negative_prompt=cfg.NEG,
        image=img,
        mask_image=mask,
        control_image=edge,
        height=cfg.SIZE,
        width=cfg.SIZE,
        num_inference_steps=cfg.STEPS,
        guidance_scale=cfg.GUIDANCE,
        controlnet_conditioning_scale=1.2,
        generator=gen
    ).images[0]
    return blend(img, out, mask)
def masked_psnr(pred, gt, mask):
    mask_bin = (mask > 0.5).float()
    if mask_bin.sum() == 0:
        return float('inf')
    diff = (pred - gt) ** 2
    mse = (diff * mask_bin).sum() / mask_bin.sum()
    return -10 * torch.log10(mse + 1e-8)
def test_and_visualize(pipe, test_paths, num_samples=10):
    random.shuffle(test_paths)
    test_paths = test_paths[:num_samples]
    tf = transforms.Compose([
        transforms.Resize((cfg.SIZE, cfg.SIZE)),
        transforms.ToTensor()
    ])
    psnr_list = []
    psnr_mask=[]
    for i, p in enumerate(test_paths):
        img = Image.open(p).convert("RGB").resize((cfg.SIZE, cfg.SIZE))
        mask = make_mask(cfg.SIZE)
        out = infer(pipe, img, mask)

        gt = tf(img).to(DEVICE)
        pred = tf(out).to(DEVICE)
        if pred.shape != gt.shape:
            pred = F.interpolate(pred.unsqueeze(0), size=gt.shape[-2:], mode="bilinear").squeeze(0)
        mse = F.mse_loss(pred, gt)
        psnr = -10 * torch.log10(mse + 1e-8)
        psnr_list.append(psnr.item())
        mask_tensor = transforms.ToTensor()(mask).to(DEVICE)   # you already have mask
        psnr_msk = masked_psnr(pred, gt, mask_tensor)
        psnr_mask.append(psnr_msk.item())
        if i < 5:  # show first 5
            # masked image for display
            img_np = np.array(img)
            mask_np = np.array(mask)
            masked_disp = img_np.copy()
            masked_disp[mask_np > 127] = 0
            plt.figure(figsize=(12, 3))
            plt.subplot(1, 4, 1); plt.imshow(img); plt.title("Original"); plt.axis("off")
            plt.subplot(1, 4, 3); plt.imshow(masked_disp); plt.title("Masked Input"); plt.axis("off")
            plt.subplot(1, 4, 4); plt.imshow(out); plt.title(f"Output\nPSNR: {psnr:.2f} dB"); plt.axis("off")
            plt.tight_layout()
            plt.show()

    print(f"\n===== TEST RESULTS =====")
    print(f"Mean PSNR on {len(test_paths)} test images: {np.mean(psnr_list):.2f} dB")
    print(f"Mean PSNR Mask on {len(test_paths)} test images: {np.mean(psnr_mask):.2f} dB")
    return psnr_list,psnr_mask
all_paths = list(Path(cfg.DATA).rglob("*.jpg")) + list(Path(cfg.DATA).rglob("*.png"))
    # Load temple images (separate folder)
temple_paths = list(Path(cfg.TEMPLE_DATA).rglob("*.jpg")) if Path(cfg.TEMPLE_DATA).exists() else []

random.shuffle(all_paths)
random.shuffle(temple_paths)

    # Test set: first 50 temple images (completely unseen)
test_paths = temple_paths if temple_paths else []
    # Remaining temples go to training
remaining_temples = temple_paths[50:] if temple_paths else []
combined_train = all_paths 

    # Split combined training data into train (90%) and validation (10%)
train_paths, val_paths = train_test_split(combined_train, test_size=0.1, random_state=42)

print(f"Total training images: {len(train_paths)}")
print(f"Validation images: {len(val_paths)}")
print(f"Test images (unseen temples): {len(test_paths)}")
train(train_paths, val_paths,text_emb)
@torch.no_grad()
def test_prompt_comparison(pipe, test_paths, num_samples=10):

    random.shuffle(test_paths)
    test_paths = test_paths[:num_samples]

    tf = transforms.Compose([
        transforms.Resize((cfg.SIZE, cfg.SIZE)),
        transforms.ToTensor()
    ])
    modes = [
        {
            "name": "NO PROMPT",
            "prompt": "",
            "negative": "",
            "guidance": 1.0   # important
        },
        {
            "name": "WITH PROMPT",
            "prompt": "detailed temple architecture, realistic stone structure, symmetric building",
            "negative": "blurry, distorted, broken structure, artifacts",
            "guidance": 7.0
        }
    ]

    results = {}

    for mode in modes:

        print("\n========================")
        print(f"MODE: {mode['name']}")
        print("========================")

        psnr_list = []
        psnr_mask=[]
        for i, p in enumerate(test_paths):

            img = Image.open(p).convert("RGB").resize((cfg.SIZE, cfg.SIZE))
            mask = make_mask(cfg.SIZE)

            # masked display
            img_np = np.array(img)
            mask_np = np.array(mask)
            masked_disp = img_np.copy()
            masked_disp[mask_np > 127] = 0

            edge = canny(img).resize((cfg.SIZE, cfg.SIZE))
            out = pipe(
                prompt=mode["prompt"],
                negative_prompt=mode["negative"],
                image=img,
                mask_image=mask,
                control_image=edge,
                height=cfg.SIZE,
                width=cfg.SIZE,
                num_inference_steps=50,   # slightly better quality
                guidance_scale=mode["guidance"]
            ).images[0]

            # metrics
            gt = tf(img).to(DEVICE)
            pred = tf(out).to(DEVICE)

            if pred.shape != gt.shape:
                pred = F.interpolate(pred.unsqueeze(0), size=gt.shape[-2:], mode="bilinear").squeeze(0)

            mse = F.mse_loss(pred, gt)
            psnr = -10 * torch.log10(mse + 1e-8)

            psnr_list.append(psnr.item())
            mask_tensor = transforms.ToTensor()(mask).to(DEVICE)   # you already have mask
            psnr_msk = masked_psnr(pred, gt, mask_tensor)
            psnr_mask.append(psnr_msk.item())
            # visualize few
            if i < 3:
                plt.figure(figsize=(12,3))

                plt.subplot(1,4,1)
                plt.imshow(img)
                plt.title("Original")

                plt.subplot(1,4,3)
                plt.imshow(masked_disp)
                plt.title("Masked")

                plt.subplot(1,4,4)
                plt.imshow(out)
                plt.title(f"{mode['name']}\nPSNR: {psnr:.2f}")

                for ax in plt.gcf().axes:
                    ax.axis("off")

                plt.tight_layout()
                plt.show()

        mean_psnr = np.mean(psnr_list)
        results[mode["name"]] = mean_psnr
        mask_mean = np.mean(psnr_mask)
        results[mode["name"]+"mask"] = mask_mean
        print(f"\nMean PSNR: {mean_psnr:.2f} dB")
        print(f"\nMean PSNR MASK: {mask_mean:.2f} dB")

    # comparison
    print("\n========= FINAL COMPARISON =========")
    for k, v in results.items():
        print(f"{k}: {v:.2f} dB")

    diff = results["WITH PROMPT"] - results["NO PROMPT"]
    print(f"\nPrompt effect: {diff:+.2f} dB")

    return results
pipe = load_pipe()

test_prompt_comparison(pipe, test_paths, num_samples=5)