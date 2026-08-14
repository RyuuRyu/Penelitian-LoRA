from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def create_heatmap(csv_path: str | Path, output_path: str | Path) -> None:
    csv_path = Path(csv_path)
    output_path = Path(output_path)

    df = pd.read_csv(csv_path, sep=";", decimal=",")

    if "LoRA Configuration" not in df.columns:
        raise ValueError("CSV must contain a 'LoRA Configuration' column")

    metric_columns = [col for col in df.columns if col not in {"LoRA Configuration", "Sample Count"}]
    if not metric_columns:
        raise ValueError("No metric columns were found in the CSV")

    matrix = df[metric_columns].to_numpy(dtype=float)
    labels = df["LoRA Configuration"].tolist()

    fig, ax = plt.subplots(figsize=(max(8, len(metric_columns) * 2.2), max(6, len(labels) * 0.7)))
    heatmap = ax.imshow(matrix, cmap="viridis")

    ax.set_xticks(np.arange(len(metric_columns)))
    ax.set_xticklabels(metric_columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)

    for row_idx, row in enumerate(matrix):
        for col_idx, value in enumerate(row):
            ax.text(
                col_idx,
                row_idx,
                f"{value:.4f}",
                ha="center",
                va="center",
                color="white" if value > np.mean(matrix) else "black",
                fontsize=8,
            )

    ax.set_title("CLIP Score Heatmap")
    ax.set_xlabel("Metric")
    ax.set_ylabel("LoRA Configuration")
    fig.colorbar(heatmap, ax=ax, label="Score")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Heatmap saved to {output_path}")


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    create_heatmap(script_dir / "CLIP Score.csv", script_dir / "clip_heatmap.png")
