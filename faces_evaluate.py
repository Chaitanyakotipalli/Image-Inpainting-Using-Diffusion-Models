import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image, ImageDraw
import torch.nn.functional as F
from pytorch_msssim import ssim as ssim_metric

from diffusers import StableDiffusionInpaintPipeline
from peft import LoraConfig, get_peft_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_ID = "runwayml/stable-diffusion-inpainting"
IMG_SIZE = 256

SAVE_PATH = "/kaggle/input/datasets/chaitanyakotipalli18/testing-weights-faces/best_model (3).pth"  # upload here
DATA_PATH = "/kaggle/input/datasets/chaitanyakotipalli18/dfvdfvfvb"

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    safety_checker=None
)

unet = pipe.unet

# LoRA (same config)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    lora_dropout=0.05,
    bias="none"
)

unet = get_peft_model(unet, lora_config)

# Load trained weights
unet.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
unet.eval()

pipe.unet = unet
pipe.to(DEVICE)

def single_square_mask(size, percent):
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)

    area = int(size * size * percent)
    side = int(np.sqrt(area))

    x = random.randint(0, size - side)
    y = random.randint(0, size - side)

    draw.rectangle([x, y, x+side, y+side], fill=255)
    return transforms.ToTensor()(mask)
def multi_small_mask(size, percent=0.05):
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)

    num_masks = random.choice([2, 3])
    total_area = int(size * size * percent)
    area_each = total_area // num_masks
    side = int(np.sqrt(area_each))

    for _ in range(num_masks):
        x = random.randint(0, size - side)
        y = random.randint(0, size - side)
        draw.rectangle([x, y, x+side, y+side], fill=255)

    return transforms.ToTensor()(mask)

imgs = [f for f in os.listdir(DATA_PATH) if f.endswith(".jpg")]
random.shuffle(imgs)
imgs = imgs[:1]

n_samples = min(50, len(imgs))
@torch.no_grad()
def run_test(use_prompt=False):

    results = []  
    mode = "prompt_on" if use_prompt else "prompt_off"  
    os.makedirs(f"{SAVE_DIR}/{mode}", exist_ok=True)

    print("\n========================")
    print("PROMPT:", "ON" if use_prompt else "OFF")
    print("========================")

    tf = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor()
    ])

    mask_levels = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

    for m in mask_levels:

        psnr_list = []
        ssim_list = []

        print(f"\nMask {int(m*100)}%")

        for i in range(n_samples):

            img = Image.open(os.path.join(DATA_PATH, imgs[i])).convert("RGB")
            img = tf(img)

            # masks
            if m == 0.05:
                mask = multi_small_mask(256, 0.05)
            else:
                mask = single_square_mask(256, m)

            masked = img * (1 - mask)

            prompt = ""
            neg = ""

            if use_prompt:
                prompt = "a human face, photorealistic, natural features, clear, complete"
                neg = "blurry, distorted, incomplete, artifacts, cropped, bad quality, deformed"

            out_pil = pipe(
                prompt=prompt,
                negative_prompt=neg,
                image=transforms.ToPILImage()(img),
                mask_image=transforms.ToPILImage()(mask.squeeze(0)),
                num_inference_steps=50,
                guidance_scale=3.0
            ).images[0]

            out = transforms.ToTensor()(out_pil).to(DEVICE)
            gt = img.to(DEVICE)

            out = F.interpolate(out.unsqueeze(0), size=gt.shape[-2:], mode="bilinear").squeeze(0)

            mse = F.mse_loss(out, gt)
            psnr = (-10 * torch.log10(mse + 1e-8)).item()
            ssim = ssim_metric(out.unsqueeze(0), gt.unsqueeze(0), data_range=1.0).item()

            psnr_list.append(psnr)
            ssim_list.append(ssim)
            if i < 2:
                fig, ax = plt.subplots(1,4, figsize=(10,3))

                ax[0].imshow(img.permute(1,2,0))
                ax[0].set_title("Original")

                ax[1].imshow(mask.squeeze(0), cmap="gray")
                ax[1].set_title("Mask")

                ax[2].imshow(masked.permute(1,2,0))
                ax[2].set_title("Masked")

                ax[3].imshow(out.cpu().permute(1,2,0))
                ax[3].set_title("Output")

                for a in ax: a.axis("off")

                plt.savefig(f"{SAVE_DIR}/{mode}/mask_{int(m*100)}_{i}.png")
                plt.close()

        results.append({
            "mask_%": int(m*100),
            "PSNR_mean": np.mean(psnr_list),
            "SSIM_mean": np.mean(ssim_list),
            "mode": mode
        })

        print(f"PSNR: {np.mean(psnr_list):.2f}")
        print(f"SSIM: {np.mean(ssim_list):.3f}")

    return pd.DataFrame(results)

df_off = run_test(False)
df_on  = run_test(True)
final_df = pd.concat([df_off, df_on])
final_df.to_csv(f"{SAVE_DIR}/results.csv", index=False)

final_df

import glob
import os
import matplotlib.pyplot as plt
from PIL import Image

def visualize_saved_results(results_dir="results/prompt_off"):
    # Find all PNG files in the directory
    image_paths = glob.glob(os.path.join(results_dir, "*.png"))
    
    if not image_paths:
        print(f"No images found in {results_dir}. Did you run the test loop yet?")
        return

    # Sort images so they appear in order of Mask %
    image_paths.sort()

    # Calculate grid size (3 columns wide)
    num_images = len(image_paths)
    cols = 3
    rows = (num_images // cols) + (1 if num_images % cols != 0 else 0)

    fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows))
    axes = axes.flatten() # Make it easy to iterate

    print(f"Found {num_images} images. Visualizing...")

    for i in range(len(axes)):
        if i < num_images:
            img_path = image_paths[i]
            img = Image.open(img_path)
            axes[i].imshow(img)
            
            # Extract the filename to use as a title
            title = os.path.basename(img_path).replace(".png", "").replace("_", " ")
            axes[i].set_title(title, fontsize=10)
        
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

# Run it
visualize_saved_results()