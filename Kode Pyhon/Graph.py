from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_clip_scores(csv_path: str | Path, output_path: str | Path) -> None:
    csv_path = Path(csv_path)
    output_path = Path(output_path)

    
    df = None
    for sep, decimal in [(",", "."), (";", ",")]:
        try:
            df = pd.read_csv(csv_path, sep=sep, decimal=decimal)
            if "LoRA Configuration" in df.columns:
                break
        except Exception:
            continue

    if df is None or "LoRA Configuration" not in df.columns:
        raise ValueError(f"CSV tidak memiliki kolom yang diharapkan: {csv_path}")

    labels = df["LoRA Configuration"].astype(str).tolist()
    mean_scores = df["Mean SSIM"].tolist()
    min_scores = df["Min Score"].tolist()
    max_scores = df["Max Score"].tolist()

    x = range(len(labels))

    plt.figure(figsize=(10, 6))
    plt.plot(x, mean_scores, marker="o", label="Mean", linewidth=2)
    plt.fill_between(x, min_scores, max_scores, alpha=0.2, label="Min-Max")

    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("SSIM")
    plt.title("Nilai SSIM per Konfigurasi LoRA")
    plt.grid(True, alpha=0.3)
    plt.legend()

    for xi, yi in zip(x, mean_scores):
        plt.annotate(
            f"{yi:.4f}",
            (xi, yi),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
            color="black",
        )

    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Graph saved to {output_path}")


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    plot_clip_scores(script_dir / "evaluation" / "SSIM_Score.csv", script_dir / "SSIM_graph.png")
