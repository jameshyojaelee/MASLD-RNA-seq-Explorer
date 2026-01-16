
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Ensure we can import from the parent directory
SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
sys.path.append(str(PARENT_DIR))

from discordance import annotate_discordance_status, plot_tug_of_war, plot_barcode_heatmap, plot_radar, DATASETS, MAGENTA, CYAN, GREY, TEXT_COLOR

def main():
    # Configuration (matching app defaults)
    PADJ_CUT = 0.1
    LFC_CUT = 0.8
    TPM_CUT = 1.0
    
    # Load Data
    master_path = PARENT_DIR / "master_ortholog_matrix.csv.gz"
    if not master_path.exists():
        print(f"Error: Master matrix not found at {master_path}")
        return

    print(f"Loading {master_path}...")
    df = pd.read_csv(master_path)
    
    # Annotate Status
    print("Annotating gene status...")
    df_annot = annotate_discordance_status(df, PADJ_CUT, LFC_CUT, TPM_CUT)
    
    # Classify Genes
    status_cols = [f"Status_{d}" for d in DATASETS]
    
    def classify(row):
        statuses = [row[c] for c in status_cols]
        has_up = "UP" in statuses
        has_down = "DOWN" in statuses
        is_active = has_up or has_down
        
        if not is_active:
            return "Inactive"
        if has_up and has_down:
            return "Contradictory"
        return "Consistent"

    df_annot["Category"] = df_annot.apply(classify, axis=1)
    
    # Calculate Cumulative Metrics
    # "Net Cumulative LFC": Sum of raw LFCs across ALL datasets (assuming 0 if NA? No, use raw column values)
    # Actually, for "Cumulative LFC" usually we sum the significant ones or all of them.
    # To be robust, let's sum LFCs where they are measured (not NA).
    # If a gene is not significant in a dataset, should its LFC count? 
    #   - If comparing "contradiction", we often care about the significant signals. 
    #   - But "cumulative LFC" might imply total perturbation.
    # Let's sum the LFCs of SIGNIFICANT datasets only?
    #   - If I sum only significant, a contradictory gene (+2, -2) sums to 0. 
    #   - A consistent gene (+2, +2) sums to +4.
    #   - This highlights the cancellation.
    # Let's try Sum of LFCs for SIGNIFICANT entries.
    
    def calc_metrics(row):
        net_lfc = 0
        abs_lfc = 0
        sig_count = 0
        for d in DATASETS:
            status = row[f"Status_{d}"]
            if status != "NS":
                val = row[f"{d}_lfc"]
                if pd.notna(val):
                    net_lfc += val
                    abs_lfc += abs(val)
                    sig_count += 1
        return pd.Series([net_lfc, abs_lfc, sig_count], index=["Net_LFC", "Abs_LFC", "Sig_Count"])

    metrics = df_annot.apply(calc_metrics, axis=1)
    df_final = pd.concat([df_annot, metrics], axis=1)
    
    # Filter for plotting
    plot_df = df_final[df_final["Category"].isin(["Contradictory", "Consistent"])].copy()
    
    print(f"Analysis Summary:")
    print(plot_df["Category"].value_counts())
    
    plots_dir = PARENT_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    sns.set_context("paper", font_scale=1.4) # Increase global font scale
    
    # 1. Violin Plot of Net Cumulative LFC
    plt.figure(figsize=(8, 6))
    sns.violinplot(data=plot_df, x="Category", y="Net_LFC", palette=[MAGENTA, CYAN], cut=0)
    plt.axhline(0, color='black', linestyle='--', alpha=0.5)
    plt.title("Net Cumulative LFC (Significant Datasets Only)", fontsize=18)
    plt.ylabel("Sum of LFCs (Significant)", fontsize=16)
    plt.xlabel("Category", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(plots_dir / "cumulative_lfc_violin.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # 2. KDE Plot of Net Cumulative LFC
    plt.figure(figsize=(8, 6))
    sns.kdeplot(data=plot_df, x="Net_LFC", hue="Category", fill=True, palette=[MAGENTA, CYAN], common_norm=False)
    plt.axvline(0, color='black', linestyle='--', alpha=0.5)
    plt.title("Distribution of Net Cumulative LFC", fontsize=18)
    plt.xlabel("Net Cumulative LFC", fontsize=16)
    plt.ylabel("Density", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # Fix legend fontsize if needed, but font_scale should handle it
    plt.savefig(plots_dir / "cumulative_lfc_density.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Scatter: Net vs Abs (The "Bowtie" or "V" plot)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=plot_df, x="Net_LFC", y="Abs_LFC", hue="Category", alpha=0.6, palette=[MAGENTA, CYAN], s=80) 
    plt.title("Activity (Abs LFC) vs Directionality (Net LFC)", fontsize=20)
    plt.xlabel("Net Cumulative LFC (Directionality)", fontsize=16)
    plt.ylabel("Absolute Cumulative LFC (Total Activity)", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(title="Category", title_fontsize=16, fontsize=14)
    plt.savefig(plots_dir / "cumulative_abs_vs_net_lfc.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Plots saved to {plots_dir}")

if __name__ == "__main__":
    main()
