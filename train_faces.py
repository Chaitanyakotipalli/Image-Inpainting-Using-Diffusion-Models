import os
import random
import gc
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw
from tqdm import tqdm
import matplotlib.pyplot as plt
from pytorch_msssim import ssim as ssim_metric

from diffusers import StableDiffusionInpaintPipeline, DDPMScheduler
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model
from transformers import CLIPTokenizer, CLIPTextModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

MODEL_ID = "runwayml/stable-diffusion-inpainting"
IMG_SIZE = 256
BATCH_SIZE = 1
GRAD_ACCUM = 4
start_epoch = 5
EPOCHS = 6
LR = 1e-5
WARMUP_STEPS = 500
LR_SCHEDULER = "cosine"

INF_STEPS = 50
GUIDANCE = 3.0

DATA_PATH = "/kaggle/input/datasets/jessicali9530/celeba-dataset/img_align_celeba/img_align_celeba"
CHECKPOINT_DIR = "/kaggle/working"
SAVE_PATH = os.path.join(CHECKPOINT_DIR, "lora_best.pth")
FULL_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "full_checkpoint.pt")

PROMPT = ""
SEED = 42
# Dataset sizes
TRAIN_SIZE = 15000
VAL_SIZE = 500
TEST_SIZE = 500

# Mask configuration - IMPORTANT for generalization
MASK_TYPES = ["rectangle", "circle", "ellipse"]
MIN_MASK_SIZE = 0.1  # 10% of image
MAX_MASK_SIZE = 0.6  # 60% of image
HYBRID_MODE = True  # Cache latents, generate new masks each epoch
CACHE_IMAGES_ONLY = True  # Only cache original images, not masks

USE_FP16 = True
PIN_MEMORY = True
RESUME_TRAINING = False

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
FULL_CHECKPOINT_PATH = "/kaggle/working/full_checkpoint.pt"
SAVE_PATH = "/kaggle/working/best_model.pth"
TEMP_PATH = "/kaggle/working/temp_step.pth"
class MaskGenerator:
    def __init__(self, img_size, mask_types=None, min_size=0.1, max_size=0.6):
        self.img_size = img_size
        self.mask_types = mask_types or ["rectangle", "circle", "ellipse"]
        self.min_size = min_size
        self.max_size = max_size
    def generate(self):
        mask = Image.new("L", (self.img_size, self.img_size), 0)
        draw = ImageDraw.Draw(mask)
        min_pixels = int(self.img_size * self.min_size)
        max_pixels = int(self.img_size * self.max_size)
        w = random.randint(min_pixels, max_pixels)
        h = random.randint(min_pixels, max_pixels)
        x0 = random.randint(0, self.img_size - w)
        y0 = random.randint(0, self.img_size - h)
        x1 = x0 + w
        y1 = y0 + h
        shape = random.choice(self.mask_types)
        if shape == "rectangle":
            draw.rectangle([x0, y0, x1, y1], fill=255)
        elif shape == "circle":
            radius = min(w, h) // 2
            center_x = x0 + w // 2
            center_y = y0 + h // 2
            draw.ellipse([center_x - radius, center_y - radius, 
                         center_x + radius, center_y + radius], fill=255)
        elif shape == "ellipse":
            draw.ellipse([x0, y0, x1, y1], fill=255)
        
        return transforms.ToTensor()(mask)

class DatasetInpaintOptimal(Dataset):
    def __init__(self, paths, vae=None, cache_images=True):
        self.paths = paths
        self.cache_images = cache_images
        self.mask_generator = MaskGenerator(IMG_SIZE, MASK_TYPES, MIN_MASK_SIZE, MAX_MASK_SIZE)
        self.tf = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor()
        ])
        if cache_images:
            print(f"🔄 Caching {len(paths)} images (pixel space)...")
            self.images_cache = []
            for path in tqdm(paths, desc="Caching images"):
                img = Image.open(path).convert("RGB")
                img = self.tf(img)
                self.images_cache.append(img)
            print(f"✅ Cached {len(self.images_cache)} images")
        self.vae = vae if cache_images else None
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, i):
        if self.cache_images:
            img = self.images_cache[i]
        else:
            img = Image.open(self.paths[i]).convert("RGB")
            img = self.tf(img)
        mask = self.mask_generator.generate()
        masked = img * (1 - mask)
        if self.vae is not None:
            with torch.no_grad():
                img_latent = self.vae.encode(
                    img.unsqueeze(0).to(DEVICE, dtype=torch.float16) * 2 - 1
                ).latent_dist.sample() * 0.18215
                
                masked_latent = self.vae.encode(
                    masked.unsqueeze(0).to(DEVICE, dtype=torch.float16) * 2 - 1
                ).latent_dist.sample() * 0.18215
                return img_latent.cpu(), mask, masked_latent.cpu()
        
        return img, mask, masked

print("\n Loading dataset...")
all_imgs = sorted([os.path.join(DATA_PATH, f) for f in os.listdir(DATA_PATH) if f.endswith(".jpg")])
random.shuffle(all_imgs)

train_paths = all_imgs[:TRAIN_SIZE]
val_paths = all_imgs[TRAIN_SIZE:TRAIN_SIZE+VAL_SIZE]
test_paths = all_imgs[TRAIN_SIZE+VAL_SIZE:TRAIN_SIZE+VAL_SIZE+TEST_SIZE]

print(f"Training: {len(train_paths)} images")
print(f"Validation: {len(val_paths)} images")
print(f"Testing: {len(test_paths)} images")
print("\n🔧 Loading model...")
gc.collect()
torch.cuda.empty_cache()

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if USE_FP16 else torch.float32,
    safety_checker=None,
    low_cpu_mem_usage=True
)
pipe.enable_attention_slicing()
pipe.enable_vae_slicing()

unet = pipe.unet.to(DEVICE)
vae = pipe.vae.to(DEVICE)
scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

vae.requires_grad_(False)
unet.requires_grad_(False)


print("\n Applying LoRA...")
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    lora_dropout=0.05,
    bias="none"
)
unet = get_peft_model(unet, lora_config)
unet.print_trainable_parameters()

tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder").to(DEVICE)
text_encoder.requires_grad_(False)

def get_text_embedding(batch_size):
    text_input = tokenizer(PROMPT, padding="max_length", max_length=77, return_tensors="pt")
    emb = text_encoder(text_input.input_ids.to(DEVICE))[0].to(torch.float16)
    return emb.expand(batch_size, -1, -1)

print(f"\n Creating dataset with HYBRID mode (cache images, new masks each epoch)...")
train_dataset = DatasetInpaintOptimal(train_paths, vae=vae, cache_images=True)
val_dataset = DatasetInpaintOptimal(val_paths, vae=vae, cache_images=True)
test_dataset = DatasetInpaintOptimal(test_paths, vae=vae, cache_images=True)

def collate_fn(batch):
    """Handle both latent and pixel outputs"""
    if len(batch[0]) == 3 and isinstance(batch[0][0], torch.Tensor) and batch[0][0].dim() == 4:
        # Latent mode
        img_latents = torch.cat([b[0] for b in batch], dim=0)
        masks = torch.stack([b[1] for b in batch], dim=0)
        masked_latents = torch.cat([b[2] for b in batch], dim=0)
        return img_latents, masks, masked_latents
    else:
        # Pixel mode
        imgs = torch.stack([b[0] for b in batch], dim=0)
        masks = torch.stack([b[1] for b in batch], dim=0)
        maskeds = torch.stack([b[2] for b in batch], dim=0)
        return imgs, masks, maskeds

train_dl = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=0, 
    pin_memory=PIN_MEMORY,
    collate_fn=collate_fn
)
val_dl = DataLoader(
    val_dataset, 
    batch_size=1, 
    num_workers=0,
    collate_fn=collate_fn
)
test_dl = DataLoader(
    test_dataset, 
    batch_size=1, 
    num_workers=0,
    collate_fn=collate_fn
)

params = [p for p in unet.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(params, lr=LR)

total_steps = (len(train_dl) // GRAD_ACCUM) * EPOCHS
lr_scheduler = get_scheduler(
    LR_SCHEDULER,
    optimizer=optimizer,
    num_warmup_steps=WARMUP_STEPS,
    num_training_steps=total_steps
)
scaler = torch.amp.GradScaler("cuda")
start_epoch = 0
best_val_loss = float("inf")

if RESUME_TRAINING and os.path.exists(FULL_CHECKPOINT_PATH):
    print(f"\n Loading checkpoint...")
    checkpoint = torch.load(FULL_CHECKPOINT_PATH, map_location=DEVICE)
    unet.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    start_epoch = checkpoint['epoch']
    best_val_loss = checkpoint['best_val_loss']
    print(f" Resumed from epoch {start_epoch}")
else:
    print("\nStarting training from scratch")

@torch.no_grad()
def validate(unet, val_loader):
    unet.eval()
    total_loss = 0
    num_batches = 0
    
    for batch in tqdm(val_loader, desc="Validation"):
        if len(batch) != 3:
            continue
        img_latents, mask, masked_latents = batch
        img_latents = img_latents.to(DEVICE, dtype=torch.float16)
        mask = mask.to(DEVICE, dtype=torch.float16)
        masked_latents = masked_latents.to(DEVICE, dtype=torch.float16)
        
        B = img_latents.size(0)

        noise = torch.randn_like(img_latents)
        t = torch.randint(0, scheduler.config.num_train_timesteps, (B,), device=DEVICE).long()
        noisy = scheduler.add_noise(img_latents, noise, t)

        mask_ds = F.interpolate(mask, size=img_latents.shape[-2:], mode="nearest")
        inp = torch.cat([noisy, mask_ds, masked_latents], dim=1)
        text_emb = get_text_embedding(B)
        with torch.amp.autocast("cuda"):
            noise_pred = unet(inp, t, encoder_hidden_states=text_emb).sample
            loss = F.mse_loss(noise_pred, noise)

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(1, num_batches)


FULL_CHECKPOINT_PATH = "/kaggle/working/full_checkpoint.pt"
SAVE_PATH = "/kaggle/working/best_model.pth"
TEMP_PATH = "/kaggle/working/temp_step.pt"

import os
os.makedirs("/kaggle/working/", exist_ok=True)

start_epoch = 5
best_val_loss = float("inf")
if os.path.exists(TEMP_PATH):
    print(" Loading TEMP checkpoint...")
    checkpoint = torch.load(TEMP_PATH, map_location=DEVICE)

    unet.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])

    start_epoch = checkpoint['epoch']
    best_val_loss = checkpoint['best_val_loss']

    print(f" Resumed from epoch {start_epoch}")

elif os.path.exists(FULL_CHECKPOINT_PATH):
    print(" Loading FULL checkpoint...")
    checkpoint = torch.load(FULL_CHECKPOINT_PATH, map_location=DEVICE)

    unet.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])

    start_epoch = checkpoint['epoch']
    best_val_loss = checkpoint['best_val_loss']

    print(f" Resumed from epoch {start_epoch}")
print("\n" + "="*60)
print("STARTING TRAINING (SAFE HYBRID MODE)")
print("="*60)

for epoch in range(start_epoch, EPOCHS):
    unet.train()
    total_loss = 0
    optimizer.zero_grad()

    progress_bar = enumerate(train_dl)

    for step, batch in progress_bar:
        if step % 1000 == 0:
            print(f"Epoch {epoch+1} → Step {step}/{len(train_dl)}")
        if step % 1000 == 0 and step > 0:
            torch.save({
                'epoch': epoch,
                'step': step,
                'model_state_dict': unet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_val_loss': best_val_loss,
            }, TEMP_PATH)

            print(f" Temp checkpoint saved at step {step}")
        if len(batch) == 3:
            img_latents, mask, masked_latents = batch
        else:
            continue

        img_latents = img_latents.to(DEVICE, dtype=torch.float16)
        mask = mask.to(DEVICE, dtype=torch.float16)
        masked_latents = masked_latents.to(DEVICE, dtype=torch.float16)
        B = img_latents.size(0)
        noise = torch.randn_like(img_latents)
        t = torch.randint(0, scheduler.config.num_train_timesteps, (B,), device=DEVICE).long()
        noisy = scheduler.add_noise(img_latents, noise, t)

        mask_ds = F.interpolate(mask, size=img_latents.shape[-2:], mode="nearest")
        inp = torch.cat([noisy, mask_ds, masked_latents], dim=1)
        text_emb = get_text_embedding(B)
        with torch.amp.autocast("cuda"):
            noise_pred = unet(inp, t, encoder_hidden_states=text_emb).sample
            loss = F.mse_loss(noise_pred, noise) / GRAD_ACCUM

        scaler.scale(loss).backward()

        if (step + 1) % GRAD_ACCUM == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            lr_scheduler.step()

        total_loss += loss.item() * GRAD_ACCUM
    if (step + 1) % GRAD_ACCUM != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    avg_epoch_loss = total_loss / len(train_dl)
    print(f"\n Epoch {epoch+1} completed. Avg loss: {avg_epoch_loss:.4f}")
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': unet.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'best_val_loss': best_val_loss,
    }, FULL_CHECKPOINT_PATH)

    print("Full checkpoint saved (pre-validation)")
    try:
        val_loss = validate(unet, val_dl)
        print(f" Validation loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(unet.state_dict(), SAVE_PATH)
            print(f" Best model saved!")

    except Exception as e:
        print(f" Validation failed but training is SAFE: {e}")

print("\n" + "="*60)
print("TRAINING COMPLETED!")
print(f"Best validation loss: {best_val_loss:.4f}")
print("="*60)


@torch.no_grad()
def quick_evaluation():
    unet.load_state_dict(torch.load(SAVE_PATH))
    unet.eval()
    pipe.unet = unet
    pipe.to(DEVICE)
    
    to_pil = transforms.ToPILImage()
    
    print("\n" + "="*60)
    print("QUICK EVALUATION (Testing Generalization)")
    print("="*60)
    
    # Test different mask types
    mask_types = ["rectangle", "circle", "ellipse"]
    
    for mask_type in mask_types:
        print(f"\n Testing {mask_type} masks...")
        mask_gen = MaskGenerator(IMG_SIZE, [mask_type], MIN_MASK_SIZE, MAX_MASK_SIZE)
        
        for i in range(2):  # Test 2 images per mask type
            random.shuffle(test_paths)
            img_path = test_paths[i]
            img = Image.open(img_path).convert("RGB")
            tf = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor()
            ])
            
            img_tensor = tf(img)
            mask = mask_gen.generate()
            masked = img_tensor * (1 - mask)
            
            # Run inference
            out = pipe(
                prompt=PROMPT,
                image=to_pil(img_tensor),
                mask_image=to_pil(mask),
                num_inference_steps=30,  # Fewer steps for quick test
                guidance_scale=GUIDANCE
            ).images[0]
            
            # Display
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            axes[0].imshow(img_tensor.permute(1,2,0))
            axes[0].set_title("Original")
            axes[0].axis("off")
            axes[1].imshow(mask.squeeze(0), cmap="gray")
            axes[1].set_title(f"Mask ({mask_type})")
            axes[1].axis("off")
            axes[2].imshow(masked.permute(1,2,0))
            axes[2].set_title("Masked")
            axes[2].axis("off")
            axes[3].imshow(out)
            axes[3].set_title("Output")
            axes[3].axis("off")
            plt.tight_layout()
            plt.show()
# Run quick evaluation
quick_evaluation()

print("\n All tasks completed successfully!")
print(f" Best model saved at: {SAVE_PATH}")

@torch.no_grad()
def test_and_visualize(n_samples=10):
    print("\n" + "="*50)
    print("TEST SET EVALUATION (UNSEEN DATA)")
    print("="*50)

    unet.eval()
    pipe.unet = unet
    pipe.to(DEVICE)

    dataset = DatasetInpaintOptimal(test_paths)
    to_tensor = transforms.ToTensor()
    print(type(dataset))
    psnr_list = []
    ssim_list = []

    for i in range(n_samples):
        img, mask, masked = dataset[i+20] # chek this

        # Run inference
        out_pil = pipe(
            prompt=PROMPT,
            image=transforms.ToPILImage()(img),
            mask_image=transforms.ToPILImage()(mask.squeeze(0)),
            num_inference_steps=50,
            guidance_scale=3.0
        ).images[0]

        out = to_tensor(out_pil).to(DEVICE)
        gt = img.to(DEVICE)

        # Resize (important fix)
        out = F.interpolate(out.unsqueeze(0), size=gt.shape[-2:], mode="bilinear").squeeze(0)

        # Metrics
        mse = F.mse_loss(out, gt)
        psnr = -10 * torch.log10(mse + 1e-8)
        ssim = ssim_metric(out.unsqueeze(0), gt.unsqueeze(0), data_range=1.0)

        psnr_list.append(psnr.item())
        ssim_list.append(ssim.item())

        # ===== Visualization =====
        plt.figure(figsize=(10,3))

        plt.subplot(1,4,1)
        plt.imshow(img.permute(1,2,0))
        plt.title("Original")

        plt.subplot(1,4,2)
        plt.imshow(mask.squeeze(0), cmap="gray")
        plt.title("Mask")

        plt.subplot(1,4,3)
        plt.imshow(masked.permute(1,2,0))
        plt.title("Masked")

        plt.subplot(1,4,4)
        plt.imshow(out_pil)
        plt.title(f"PSNR={psnr:.1f}\nSSIM={ssim:.3f}")

        plt.show()

    print("\n===== SUMMARY =====")
    print(f"Mean PSNR: {np.mean(psnr_list):.2f} dB")
    print(f"Mean SSIM: {np.mean(ssim_list):.4f}")

test_and_visualize(5)