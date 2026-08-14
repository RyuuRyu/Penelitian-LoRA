import csv
from pathlib import Path


def buat_csv_dataset(folder_dataset: str, output_csv: str):
    """
    Membaca folder dataset yang berisi file gambar dan file teks caption.
    Setiap gambar akan dicocokkan dengan file teks yang memiliki nama yang sama
    (contoh: image01.jpg -> image01.txt), lalu disimpan ke file CSV.

    Kolom CSV:
    1. nama_file_teks
    2. isi_teks
    3. gambar
    """
    folder = Path(folder_dataset)
    csv_path = Path(output_csv)

    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Folder dataset tidak ditemukan: {folder_dataset}")

    ekstensi_gambar = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}
    baris = []

    for file_gambar in sorted(folder.iterdir()):
        if not file_gambar.is_file() or file_gambar.suffix.lower() not in ekstensi_gambar:
            continue

        
        file_teks = file_gambar.with_suffix(".txt")
        if not file_teks.exists():
            
            alternatif = folder / f"{file_gambar.stem}.txt"
            file_teks = alternatif if alternatif.exists() else None

        if file_teks is None:
            isi_teks = ""
            nama_file_teks = f"{file_gambar.stem}.txt"
        else:
            isi_teks = file_teks.read_text(encoding="utf-8", errors="ignore").strip()
            nama_file_teks = file_teks.name

        baris.append(
            {
                "nama_file_teks": nama_file_teks,
                "isi_teks": isi_teks,
                "gambar": file_gambar.name,
            }
        )

    with csv_path.open("w", newline="", encoding="utf-8") as file_csv:
        writer = csv.DictWriter(
            file_csv,
            fieldnames=["nama_file_teks", "isi_teks", "gambar"],
        )
        writer.writeheader()
        writer.writerows(baris)

    print(f"CSV berhasil dibuat: {csv_path}")
    print(f"Jumlah data: {len(baris)}")


if __name__ == "__main__":
    
    folder_dataset = r"D:\Documents\Kuliah\Skripsi\Analisis_Lora\dataset\batik-megamendung\Lora\img\40_mgmdg batik pattern - Copy\thumbnails"
    output_csv = r"D:\Documents\Kuliah\Skripsi\Analisis_Lora\dataset\batik-megamendung\Lora\img\40_mgmdg batik pattern - Copy\thumbnails\dataset.csv"

    buat_csv_dataset(folder_dataset, output_csv)
