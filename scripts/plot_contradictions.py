
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi

# --- Configuration ---
IN_FILE = "deg_contradictions_5datasets.csv"
OUT_DIR = "plots"
DATASETS = ["Hoang_et_al", "Govaere_et_al", "In_house_MCD", "External_MCD_1", "External_MCD_2"]

# Theme
plt.style.use("default") # White background
MAGENTA = "#D02090" # Slightly darker magenta for visibility on white
CYAN = "#008B8B" # Dark Cyan/Teal
GREY = "#D3D3D3" # Light Grey for elements
TEXT_COLOR = "black"
NOTE_TEXT = "Cutoffs: padj < 0.1, |log2FC| > 0.8, TPM > 1.0"

def load_data():
    return pd.read_csv(IN_FILE)

# --- Plot 1: Tug of War Scatter ---
def plot_tug_of_war(df):
    print("Generating Plot 1: Tug of War Scatter...")
    
    # Calculate Scores
    def calc_scores(r):
        pos_sum = 0
        neg_sum = 0
        for d in DATASETS:
            lfc = r[f"{d}_lfc"]
            if pd.notna(lfc) and r[f"Status_{d}"] != "NS":
                if lfc > 0: pos_sum += lfc
                else: neg_sum += abs(lfc)
        return pos_sum, neg_sum

    scores = df.apply(calc_scores, axis=1, result_type="expand")
    df["Pos_Score"] = scores[0]
    df["Neg_Score"] = scores[1]
    
    plt.figure(figsize=(10, 8))
    
    # Plot
    plt.scatter(df["Pos_Score"], df["Neg_Score"], 
                c=MAGENTA, alpha=0.6, edgecolors=TEXT_COLOR, s=60, linewidths=0.5)
    
    # Labels
    plt.title("Tug of War: Cumulative Upregulation vs Downregulation", 
              fontsize=16, color=MAGENTA, fontweight="bold", pad=20)
    plt.xlabel("Sum of Significant Positive LFCs (Pro-Target strength)", fontsize=12, color=TEXT_COLOR)
    plt.ylabel("Sum of Significant Negative LFCs (Anti-Target strength)", fontsize=12, color=TEXT_COLOR)
    
    # Diagonal Line (Equal contradiction)
    max_val = max(df["Pos_Score"].max(), df["Neg_Score"].max())
    plt.plot([0, max_val], [0, max_val], color="black", linestyle="--", linewidth=1)
    plt.text(max_val*0.8, max_val*0.85, "Equal Contradiction", color=TEXT_COLOR, rotation=45)
    
    # Annotate Top Contradictors
    df["Total_Mag"] = df["Pos_Score"] + df["Neg_Score"]
    
    # Filter for high magnitude - Show more genes (Top 45)
    top_conflict = df[df["Pos_Score"] > 0.5].sort_values("Total_Mag", ascending=False).head(45)
    
    for _, row in top_conflict.iterrows():
        plt.text(row["Pos_Score"]+0.05, row["Neg_Score"]+0.05, row["Symbol"], color=TEXT_COLOR, fontsize=7)

    plt.grid(True, linestyle=":", color=GREY, alpha=0.5)
    plt.figtext(0.5, 0.01, NOTE_TEXT, ha="center", fontsize=10, color=TEXT_COLOR)
    plt.savefig(f"{OUT_DIR}/contradiction_tug_of_war.png", dpi=300, bbox_inches="tight")
    plt.close()

# --- Plot 2: Discordance Barcode Heatmap ---
def plot_barcode_heatmap(df):
    print("Generating Plot 2: Barcode Heatmap...")
    
    # Filter interesting genes - Show more (Top 100)
    
    def count_sigs(r):
        return sum([1 for d in DATASETS if r[f"Status_{d}"] != "NS"])
    
    df["Sig_Count"] = df.apply(count_sigs, axis=1)
    
    # Sort by Count then by Total Magnitude to be interesting
    df["Total_Mag"] = df.apply(lambda r: sum([abs(r[f"{d}_lfc"]) for d in DATASETS if pd.notna(r[f"{d}_lfc"])]), axis=1)
    
    sub = df.sort_values(["Sig_Count", "Total_Mag"], ascending=[False, False]).head(100)
        
    # Prepare Matrix
    matrix = pd.DataFrame(index=sub["Symbol"])
    for d in DATASETS:
        # We want discrete values for heatmap: 1=UP, -1=DOWN, 0=NS
        col_data = []
        for _, r in sub.iterrows():
            s = r[f"Status_{d}"]
            if s == "UP": val = 1
            elif s == "DOWN": val = -1
            else: val = 0
            col_data.append(val)
        matrix[d] = col_data
        
    # Taller figure for 100 genes
    plt.figure(figsize=(8, 20))
    
    # Custom CMP
    from matplotlib.colors import ListedColormap
    # -1: Cyan, 0: Grey, 1: Magenta
    cmap = ListedColormap([CYAN, GREY, MAGENTA])
    
    ax = sns.heatmap(matrix, cmap=cmap, cbar=False, linewidths=0.5, linecolor="white")
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=MAGENTA, label='Upregulated'),
        Patch(facecolor=GREY, label='Not Significant'),
        Patch(facecolor=CYAN, label='Downregulated')
    ]
    plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.02), 
               ncol=3, frameon=False, labelcolor=TEXT_COLOR)
    
    plt.title("Discordance Barcode: Top 100 Active Genes", fontsize=16, color=MAGENTA, fontweight="bold", pad=40)
    plt.yticks(color=TEXT_COLOR, fontsize=8) # Smaller font for many genes
    plt.xticks(color=TEXT_COLOR, rotation=45)
    
    plt.figtext(0.5, 0.01, NOTE_TEXT, ha="center", fontsize=10, color=TEXT_COLOR)
    plt.savefig(f"{OUT_DIR}/contradiction_barcode.png", dpi=300, bbox_inches="tight")
    plt.close()

# --- Plot 3: Radar/Spider Chart ---
def plot_radar(df):
    print("Generating Plot 3: Radar Chart...")
    
    # Select specific genes with diverse patterns
    candidates = ["TYMP", "FASN", "TM7SF2", "ACSS2", "PPP1R3C"] # Top Human UP / Mouse DOWN genes
    # Add one valid random if candidates missing
    existent = df[df["Symbol"].isin(candidates)]
    if len(existent) < 3:
        existent = df.head(3)
        
    genes = existent
    
    # Setup Data
    categories = DATASETS
    N = len(categories)
    
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True)
    
    # Let's set Center = -3, Outer = +3
    
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    
    plt.xticks(angles[:-1], categories, color=TEXT_COLOR, size=12)
    
    ax.set_rlabel_position(0)
    plt.yticks([-2, -1, 0, 1, 2], ["-2", "-1", "0", "+1", "+2"], color="grey", size=10)
    plt.ylim(-3, 3)
    
    # Zero line
    values_zero = [0] * N
    values_zero += values_zero[:1]
    ax.plot(angles, values_zero, linewidth=1, linestyle="--", color="black", alpha=0.5)
    
    colors = [MAGENTA, "#32CD32", "#1E90FF", "#FFA500", "#9370DB"]
    
    for idx, (i, row) in enumerate(genes.iterrows()):
        values = []
        for d in categories:
            val = row[f"{d}_lfc"]
            if pd.isna(val): val = 0
            # Cap
            if val > 3: val = 3
            if val < -3: val = -3
            values.append(val)
            
        values += values[:1]
        c = colors[idx % len(colors)]
        
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=row["Symbol"], color=c)
        ax.fill(angles, values, color=c, alpha=0.1)

    plt.title("Radial LFC Profile of Discordant Genes", size=20, color=MAGENTA, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), labelcolor=TEXT_COLOR)
    
    plt.figtext(0.5, 0.01, NOTE_TEXT, ha="center", fontsize=10, color=TEXT_COLOR)
    plt.savefig(f"{OUT_DIR}/contradiction_radar.png", dpi=300, bbox_inches="tight")
    plt.close()

def main():
    df = load_data()
    plot_tug_of_war(df)
    plot_barcode_heatmap(df)
    plot_radar(df)
    print("Done generating plots.")

if __name__ == "__main__":
    main()
