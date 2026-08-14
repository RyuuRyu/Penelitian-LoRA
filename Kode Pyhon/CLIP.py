import os
import glob
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "openai/clip-vit-base-patch32"

print(f"Loading CLIP model ({model_id}) on {device}...")
model = CLIPModel.from_pretrained(model_id).to(device)
processor = CLIPProcessor.from_pretrained(model_id)
model.eval()


experiments = [
    {"lora": "R8_A4",   "folder": r"D:\PIctures\Generations\ComfyUI\Txt2Img\Megamendung\Rank-8-Alpha-4"},
    {"lora": "R8_A8",   "folder": r"D:\PIctures\Generations\ComfyUI\Txt2Img\Megamendung\Rank-8-Alpha-8"},
    {"lora": "R8_A16",  "folder": r"D:\PIctures\Generations\ComfyUI\Txt2Img\Megamendung\Rank-8-Alpha-16"},
    {"lora": "R16_A8",  "folder": r"D:\PIctures\Generations\ComfyUI\Txt2Img\Megamendung\Rank-16-Alpha-8"},
    {"lora": "R16_A16", "folder": r"D:\PIctures\Generations\ComfyUI\Txt2Img\Megamendung\Rank-16-Alpha-16"},
    {"lora": "R16_A32", "folder": r"D:\PIctures\Generations\ComfyUI\Txt2Img\Megamendung\Rank-16-Alpha-32"},
    {"lora": "R32_A16", "folder": r"D:\PIctures\Generations\ComfyUI\Txt2Img\Megamendung\Rank-32-Alpha-16"},
    {"lora": "R32_A32", "folder": r"D:\PIctures\Generations\ComfyUI\Txt2Img\Megamendung\Rank-32-Alpha-32"},
    {"lora": "R32_A64", "folder": r"D:\PIctures\Generations\ComfyUI\Txt2Img\Megamendung\Rank-32-Alpha-64"},
]


PROMPT_TEST = "mgmdg, masterpiece, ultra quality, 4K, intricate detail, fabric texture, textile design, traditional indonesian batik, megamendung, repeating pattern, detailed pattern, ornamental design, cloth surface, pattern focus, big pattern, blue theme,"

results = []
per_image_results = []

print("Starting CLIP Score calculation...")

base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_dir, "CSVs")
os.makedirs(output_dir, exist_ok=True)
summary_output_path = os.path.join(output_dir, "CLIP_Score.csv")
per_image_output_path = os.path.join(base_dir, "CLIP per Image", "CLIP_Scores_per_Image.csv")
os.makedirs(os.path.dirname(per_image_output_path), exist_ok=True)

for exp in experiments:
    lora_name = exp["lora"]
    img_paths = sorted(glob.glob(os.path.join(exp["folder"], "*.png"))) + \
                sorted(glob.glob(os.path.join(exp["folder"], "*.jpg")))

    if not img_paths:
        print(f"[Warning] No images found in {exp['folder']}")
        continue

    scores = []

    for img_path in img_paths:
        image = Image.open(img_path).convert("RGB")

        
        inputs = processor(
            text=[PROMPT_TEST],
            images=image,
            return_tensors="pt",
            padding=True
        ).to(device)

        
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image 
            clip_score = logits_per_image.item() / 100.0  

        scores.append(clip_score)
        per_image_results.append({
            "LoRA Configuration": lora_name,
            "Image File": os.path.basename(img_path),
            "Image Path": img_path,
            "CLIP Score": round(clip_score, 6),
        })

    
    avg_score = sum(scores) / len(scores)

    results.append({
        "LoRA Configuration": lora_name,
        "Sample Count": len(scores),
        "Mean CLIP Score": round(avg_score, 4),
        "Min Score": round(min(scores), 4),
        "Max Score": round(max(scores), 4)
    })

    print(f"[{lora_name}] Mean CLIP Score: {avg_score:.4f}")


summary_columns = [
    "LoRA Configuration",
    "Sample Count",
    "Mean CLIP Score",
    "Min Score",
    "Max Score",
]
summary_df = pd.DataFrame(results, columns=summary_columns)
summary_df.to_csv(summary_output_path, index=False)

per_image_columns = [
    "LoRA Configuration",
    "Image File",
    "Image Path",
    "CLIP Score",
]
per_image_df = pd.DataFrame(per_image_results, columns=per_image_columns)
per_image_df.to_csv(per_image_output_path, index=False)

print(f"\nCalculation Finished! Summary saved to {summary_output_path}")
print(f"Per-image CLIP scores saved to {per_image_output_path}")