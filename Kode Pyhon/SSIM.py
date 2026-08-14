import  os
import csv
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

REFERENCE_DIR = r"D:\Documents\Kuliah\Skripsi\Analisis_Lora\Analisis\SSIM Reference"
BASE_DIR = r"D:\PIctures\Generations\ComfyUI\Txt2Img\Megamendung"

MODELS = [
    "Rank-8-Alpha-4",
    "Rank-8-Alpha-8",
    "Rank-8-Alpha-16",
    "Rank-16-Alpha-8",
    "Rank-16-Alpha-16",
    "Rank-16-Alpha-32",
    "Rank-32-Alpha-16",
    "Rank-32-Alpha-32",
    "Rank-32-Alpha-64",
]

valid_ext = (".png", ".jpg", ".jpeg")

def load_output_image(path):
    """
    Load gambar output inferensi sebagai grayscale array.
    Resolusi gambar output digunakan sebagai target
    penyeragaman untuk seluruh gambar referensi.
    Returns: (array grayscale, (width, height))
    """
    img = Image.open(path).convert("L")
    size = img.size
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr, size
    
def load_reference_resized(path, target_size):
    """
    Load gambar referensi dan resize ke target_size
    menggunakan interpolasi Lanczos, kemudian konversi
    ke grayscale array ternormalisasi.
    target_size: (width, height) dari gambar output
    """
    img = Image.open(path).convert("L")
    img = img.resize(target_size, Image.LANCZOS)
    return np.array(img, dtype=np.float32) / 255.0

def compute_ssim_pair(output_arr, ref_arr):
    """
    Hitung SSIM antara satu pasang gambar output dan refernsi.
    Kedua array harus memiliki dimensi yang sama.
    """
    return float(ssim(
        output_arr,
        ref_arr,
        data_range=1.0
    ))

def get_ref_paths(ref_dir):
    """Dapatkan daftar path seluruh gambar referensi."""
    paths = []
    for f in sorted(os.listdir(ref_dir)):
        if f.lower().endswith(valid_ext):
            paths.append(os.path.join(ref_dir, f))
    return paths

ref_paths = get_ref_paths(REFERENCE_DIR)

print("=" *  15)
print("Subset Gambar Referensi")
print("=" *  15)
for p in ref_paths:
    img_size = Image.open(p).size
    print(f"  ✓ {os.path.basename(p):40s} {img_size[0]}×{img_size[1]}")
print(f"\nTotal referensi: {len(ref_paths)} gambar")
print("Catatan: Setiap referensi akan di-resize dinamis")
print("         menyesuaikan resolusi gambar output.\n")

if len(ref_paths) == 0:
    raise ValueError("Tidak ada gambar referensi ditemukan di folder '{REFERENCE_DIR}'."
                     "Pastikan folder berisi setidaknya satu gambar referensi dengan ekstensi .png, .jpg, atau .jpeg."
                     )

results = []
per_image_results = []

base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_dir, "evaluation")
os.makedirs(output_dir, exist_ok=True)
summary_output_path = os.path.join(output_dir, "SSIM_Score.csv")
per_image_output_path = os.path.join(base_dir, "SSIM per Image", "SSIM_Scores_per_Image.csv")
os.makedirs(os.path.dirname(per_image_output_path), exist_ok=True)

for model_name in MODELS:
    folder = os.path.join(BASE_DIR, model_name)

    if not os.path.exists(folder):
        print(f"[SKIP] Folder tidak ditemukan: {folder}\n")
        continue

    output_files = sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith(valid_ext)
    ])

    if len(output_files) == 0:
        print(f"[SKIP] Tidak ada gambar output di: {folder}\n")
        continue

    print("=" * 15)
    print(f"Model: {model_name}")
    print("=" * 15)

    per_image_mean_scores = []

    for out_filename in output_files:
        out_path = os.path.join(folder, out_filename)

        
        out_arr, out_size = load_output_image(out_path)

        
        pair_scores = []
        for ref_path in ref_paths:
            ref_arr = load_reference_resized(ref_path, out_size)
            score   = compute_ssim_pair(out_arr, ref_arr)
            pair_scores.append(score)

        
        mean_vs_refs = float(np.mean(pair_scores))
        per_image_mean_scores.append(mean_vs_refs)
        per_image_results.append({
            "LoRA Configuration": model_name,
            "Image File": out_filename,
            "Image Path": out_path,
            "SSIM Score": round(mean_vs_refs, 6),
        })

        print(
            f"  {out_filename:35s} → "
            f"resolusi {out_size[0]}×{out_size[1]} | "
            f"SSIM (mean vs refs): {mean_vs_refs:.4f}"
        )

    
    mean_ssim = float(np.mean(per_image_mean_scores))
    min_ssim  = float(np.min(per_image_mean_scores))
    max_ssim  = float(np.max(per_image_mean_scores))

    print(f"\n  → Mean SSIM : {mean_ssim:.4f}")
    print(f"     Min SSIM : {min_ssim:.4f}")
    print(f"     Max SSIM : {max_ssim:.4f}\n")

    results.append({
        "LoRA Configuration" : model_name,
        "Sample Count"       : len(per_image_mean_scores),
        "Mean SSIM"          : round(mean_ssim, 4),
        "Min Score"          : round(min_ssim,  4),
        "Max Score"          : round(max_ssim,  4),
    })


# SIMPAN CSV


summary_fieldnames = [
    "LoRA Configuration",
    "Sample Count",
    "Mean SSIM",
    "Min Score",
    "Max Score",
]

with open(summary_output_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
    writer.writeheader()
    writer.writerows(results)

per_image_fieldnames = [
    "LoRA Configuration",
    "Image File",
    "Image Path",
    "SSIM Score",
]

with open(per_image_output_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=per_image_fieldnames)
    writer.writeheader()
    writer.writerows(per_image_results)

print("=" * 15)
print(f"✓ Hasil SSIM disimpan ke: {summary_output_path}")
print(f"✓ SSIM per image disimpan ke: {per_image_output_path}")
print(f"  Total model dievaluasi: {len(results)}")
print("=" * 15)