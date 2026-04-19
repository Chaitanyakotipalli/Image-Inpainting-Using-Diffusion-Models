import os, random
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageFilter
from torchvision import transforms
import torch.nn.functional as F

from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel
from peft import PeftModel

from pathlib import Path
DEVICE = "cuda"

#DATA_PATH = "/kaggle/input/datasets/chaitanyakotipalli18/final-data/monuments_test" 
DATA_PATH="/kaggle/input/datasets/chaitanyakotipalli18/testing-data-mon/cheta"# 👈 monuments test images
LORA_PATH = "/kaggle/input/datasets/chaitanyakotipalli18/final-new"   # 👈 uploaded LoRA

# collect test images
test_paths = []
for ext in ["jpg", "png", "jpeg"]:
    test_paths += list(Path(DATA_PATH).rglob(f"*.{ext}"))

random.shuffle(test_paths)
test_paths = test_paths[:30]
print("Total test images:", len(test_paths))

BASE_MODEL = "sd2-community/stable-diffusion-2-inpainting"
CONTROLNET = "thibaud/controlnet-sd21-canny-diffusers"

controlnet = ControlNetModel.from_pretrained(CONTROLNET, torch_dtype=torch.float16)

pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    BASE_MODEL,
    controlnet=controlnet,
    torch_dtype=torch.float16,
    safety_checker=None
).to(DEVICE)

# load LoRA
pipe.unet = PeftModel.from_pretrained(pipe.unet, LORA_PATH)

pipe.enable_attention_slicing()
pipe.unet.eval()

print("Model loaded")
def make_mask(size, percent):
    num = 1
    if(percent==0.03): num = 2
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)

    area = int(size * size * percent)
    side = int(np.sqrt(area))

    for _ in range(num):
        x = random.randint(0, size - side)
        y = random.randint(0, size - side)
        draw.rectangle([x, y, x+side, y+side], fill=255)
    return mask

def canny(img):
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return Image.fromarray(edges).convert("RGB")
tf = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor()
])

def psnr(pred, gt):
    mse = F.mse_loss(pred, gt)
    return -10 * torch.log10(mse + 1e-8)
prompt = (
    "highly realistic architectural inpainting, "
    "seamless color and texture blending with surrounding region, "
    "match exact stone and brick color distribution, "
    "consistent illumination and shadow direction, "
    "preserve fine architectural details and carvings, "
    "continuous texture without visible boundaries, "
    "natural material appearance, no color shift, no patch artifacts"
)
neg_prompt = (
    "color mismatch, visible seams, patchy texture, blurry region, "
    "washed out colors, grey overlay, fog, haze, low contrast, "
    "plastic texture, artificial smoothness, overexposed, underexposed, "
    "distorted structure, inconsistent lighting"
)
all_image_paths = test_paths
def run_test_and_save_grids(paths):
    output_dir = "test_results/comparisons"
    os.makedirs(output_dir, exist_ok=True)

    mask_levels = [0.03,0.10,0.15,0.20]
    summary_data = []

    for m in mask_levels:
        level_results = []
        print(f"\nProcessing Mask {int(m*100)}%...")

        for i, path in enumerate(paths):
            # 1. Load
            img = Image.open(path).convert("RGB").resize((512, 512))
            mask = make_mask(512, m)

            img_np = np.array(img)
            mask_np = np.array(mask)

            masked_np = img_np.copy()
            masked_np[mask_np > 127] = 0
            masked_img = Image.fromarray(masked_np)

            # IMPORTANT: compute once and store
            edge = canny(img)

            # 2. Inference
            with torch.inference_mode():
                out = pipe(
                    prompt=prompt,
                    negative_prompt=neg_prompt,
                    image=masked_img,
                    mask_image=mask,
                    control_image=edge,
                    num_inference_steps=50,
                    controlnet_conditioning_scale=1.2,
                    guidance_scale=5.0,
                    height=512,
                    width=512
                ).images[0]

            # 3. Metrics (safe normalization)
            gt_tensor = transforms.ToTensor()(img).unsqueeze(0).to(DEVICE)
            pred_tensor = transforms.ToTensor()(out).unsqueeze(0).to(DEVICE)

            pred_tensor = torch.clamp(pred_tensor, 0, 1)

            p_val = psnr_metric(pred_tensor, gt_tensor).item()
            s_val = ssim_metric(pred_tensor, gt_tensor).item()

            # Combined score
            score = p_val + (s_val * 100)  # weight SSIM properly

            level_results.append({
                "path": path,
                "psnr": p_val,
                "ssim": s_val,
                "score": score,
                "out": out,
                "masked": masked_img,
                "orig": img,
                "edge": edge
            })

        # ✅ SORT by combined quality
        level_results.sort(key=lambda x: x['score'], reverse=True)

        # 4. Save top 2 correctly
        for rank in range(min(2, len(level_results))):
            res = level_results[rank]

            combined = Image.new('RGB', (1536, 512))

            # Correct order: ORIGINAL → MASKED → EDGE → OUTPUT
            combined.paste(res['orig'], (0, 0))
            combined.paste(res['masked'], (512, 0))
            combined.paste(res['out'], (1024, 0))

            fname = (
                f"mask_{int(m*100)}_rank{rank+1}"
                f"_PSNR_{res['psnr']:.2f}_SSIM_{res['ssim']:.4f}.png"
            )

            combined.save(os.path.join(output_dir, fname))

        # 5. Stats
        summary_data.append({
            "Mask %": int(m*100),
            "Mean PSNR": np.mean([x['psnr'] for x in level_results]),
            "Mean SSIM": np.mean([x['ssim'] for x in level_results])
        })

        print(f"Mean {int(m*100)}%: PSNR: {np.mean([x['psnr'] for x in level_results]):.2f}")

    pd.DataFrame(summary_data).to_csv("test_results/summary.csv", index=False)

    print("\n Done. Best comparisons saved correctly.")
results_df = run_test_and_save_grids(all_image_paths)
import glob
import os
import matplotlib.pyplot as plt
from PIL import Image

def visualize_saved_results(results_dir="test_results/comparisons"):
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
prompt =""
neg_prompt = ""
results_df = run_test_and_save_grids(all_image_paths)