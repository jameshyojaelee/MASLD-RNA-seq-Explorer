
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
import streamlit as st

# --- Constants ---
DATASETS = ["Hoang_et_al", "Govaere_et_al", "In_house_MCD", "External_MCD_1", "External_MCD_2"]
MAGENTA = "#D02090"
CYAN = "#008B8B"
GREY = "#D3D3D3"
TEXT_COLOR = "black"

def get_status(row, prefix, padj_cut, lfc_cut, tpm_cut):
    try:
        padj = row[f"{prefix}_padj"]
        lfc = row[f"{prefix}_lfc"]
        tpm = row[f"{prefix}_tpm"]
        
        if pd.isna(padj) or padj >= padj_cut:
            return "NS"
        if pd.isna(lfc) or abs(lfc) <= lfc_cut:
            return "NS"
        if pd.isna(tpm) or tpm < tpm_cut:
            return "NS"
            
        if lfc > 0:
            return "UP"
        if lfc < 0:
            return "DOWN"
        return "NS"
    except KeyError:
        return "NS"

def annotate_discordance_status(df, padj_cut, lfc_cut, tpm_cut):
    # We need to add "Status_Dataset" columns dynamically
    df_out = df.copy()
    for d in DATASETS:
        # Vectorized implementation for speed
        padj_mask = (df_out[f"{d}_padj"] < padj_cut)
        lfc_mag_mask = (df_out[f"{d}_lfc"].abs() > lfc_cut)
        tpm_mask = (df_out[f"{d}_tpm"] >= tpm_cut)
        
        sig_mask = padj_mask & lfc_mag_mask & tpm_mask
        
        # Initialize as NS
        df_out[f"Status_{d}"] = "NS"
        
        # Set UP
        up_mask = sig_mask & (df_out[f"{d}_lfc"] > 0)
        df_out.loc[up_mask, f"Status_{d}"] = "UP"
        
        # Set DOWN
        down_mask = sig_mask & (df_out[f"{d}_lfc"] < 0)
        df_out.loc[down_mask, f"Status_{d}"] = "DOWN"
        
    return df_out

def filter_master_matrix(df, padj_cut, lfc_cut, tpm_cut):
    df_out = annotate_discordance_status(df, padj_cut, lfc_cut, tpm_cut)
        
    # Check Contradiction
    # A gene is contradictory if it has at least one UP and at least one DOWN
    # row-wise check
    status_cols = [f"Status_{d}" for d in DATASETS]
    status_values = df_out[status_cols].values
    
    # Check for presence of 'UP' and 'DOWN' in each row
    has_up = (status_values == "UP").any(axis=1)
    has_down = (status_values == "DOWN").any(axis=1)
    
    contradiction_mask = has_up & has_down
    return df_out[contradiction_mask].copy()

def plot_tug_of_war(df, note_text):
    if df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No contradictory genes found with current cutoffs", ha='center')
        return fig
        
    def calc_scores(r):
        pos_sum = 0
        neg_sum = 0
        for d in DATASETS:
            lfc = r[f"{d}_lfc"]
            status = r[f"Status_{d}"]
            if pd.notna(lfc) and status != "NS":
                if lfc > 0: pos_sum += lfc
                else: neg_sum += abs(lfc)
        return pos_sum, neg_sum

    scores = df.apply(calc_scores, axis=1, result_type="expand")
    df["Pos_Score"] = scores[0]
    df["Neg_Score"] = scores[1]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(df["Pos_Score"], df["Neg_Score"], 
                c=MAGENTA, alpha=0.6, edgecolors=TEXT_COLOR, s=60, linewidths=0.5)
    
    ax.set_title("Tug of War: Cumulative Upregulation vs Downregulation", 
              fontsize=16, color=TEXT_COLOR, fontweight="bold", pad=20)
    ax.set_xlabel("Sum of Significant Positive LFCs (Pro-Target strength)", fontsize=12, color=TEXT_COLOR)
    ax.set_ylabel("Sum of Significant Negative LFCs (Anti-Target strength)", fontsize=12, color=TEXT_COLOR)
    
    max_val = max(df["Pos_Score"].max(), df["Neg_Score"].max()) if not df.empty else 1
    ax.plot([0, max_val], [0, max_val], color="black", linestyle="--", linewidth=1)
    ax.text(max_val*0.8, max_val*0.85, "Equal Contradiction", color=TEXT_COLOR, rotation=45)
    
    df["Total_Mag"] = df["Pos_Score"] + df["Neg_Score"]
    top_conflict = df[df["Pos_Score"] > 0.5].sort_values("Total_Mag", ascending=False).head(45)
    
    for _, row in top_conflict.iterrows():
        ax.text(row["Pos_Score"]+0.05, row["Neg_Score"]+0.05, row["Symbol"], color=TEXT_COLOR, fontsize=7)

    ax.grid(True, linestyle=":", color=GREY, alpha=0.5)
    plt.tight_layout()
    return fig

def plot_barcode_heatmap(df, note_text):
    if df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No genes", ha="center")
        return fig
        
    def count_sigs(r):
        return sum([1 for d in DATASETS if r[f"Status_{d}"] != "NS"])
    
    df["Sig_Count"] = df.apply(count_sigs, axis=1)
    df["Total_Mag"] = df.apply(lambda r: sum([abs(r[f"{d}_lfc"]) for d in DATASETS if pd.notna(r[f"{d}_lfc"])]), axis=1)
    sub = df.sort_values(["Sig_Count", "Total_Mag"], ascending=[False, False]).head(100)
    
    matrix = pd.DataFrame(index=sub["Symbol"])
    for d in DATASETS:
        col_data = []
        for _, r in sub.iterrows():
            s = r[f"Status_{d}"]
            if s == "UP": val = 1
            elif s == "DOWN": val = -1
            else: val = 0
            col_data.append(val)
        matrix[d] = col_data
    
    matrix = matrix.T
    fig, ax = plt.subplots(figsize=(20, 3))
    
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap([CYAN, GREY, MAGENTA])
    
    sns.heatmap(matrix, cmap=cmap, cbar=False, linewidths=0.5, linecolor="white", ax=ax, square=True)
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=MAGENTA, label='Upregulated'),
        Patch(facecolor=GREY, label='Not Significant'),
        Patch(facecolor=CYAN, label='Downregulated')
    ]
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.01, 0.5), 
               frameon=False, labelcolor=TEXT_COLOR)
    
    ax.set_title("Discordance Barcode: Top 100 Active Genes", fontsize=16, color=TEXT_COLOR, fontweight="bold", pad=20)
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=6, color=TEXT_COLOR)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=6, color=TEXT_COLOR)
    ax.tick_params(axis='y', pad=10)
    
    plt.subplots_adjust(bottom=0.3, right=0.85)
    fig.text(0.95, 0.02, note_text, ha="right", fontsize=12, color=TEXT_COLOR)
    return fig

def plot_radar(df, candidates, title_text):
    existent = df[df["Symbol"].isin(candidates)]
    if existent.empty:
        # If specific candidates not found, try to pick top relevant ones
        if title_text.count("Human UP") > 0:
             # Try to find some Human UP / Mouse DOWN
             # Heuristic check
             human_up = (df["Status_Hoang_et_al"]=="UP") | (df["Status_Govaere_et_al"]=="UP")
             mouse_down = (df["Status_In_house_MCD"]=="DOWN") | (df["Status_External_MCD_1"]=="DOWN")
             existent = df[human_up & mouse_down].head(5)
        elif title_text.count("Human DOWN") > 0:
             human_down = (df["Status_Hoang_et_al"]=="DOWN") | (df["Status_Govaere_et_al"]=="DOWN")
             mouse_up = (df["Status_In_house_MCD"]=="UP") | (df["Status_External_MCD_1"]=="UP")
             existent = df[human_down & mouse_up].head(5)
        
        if existent.empty:
            existent = df.head(3)

    genes = existent
    categories = DATASETS
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    
    plt.xticks(angles[:-1], categories, color=TEXT_COLOR, size=16)
    ax.tick_params(axis='x', pad=30)
    ax.set_rlabel_position(0)
    plt.yticks([-2, -1, 0, 1, 2], ["-2", "-1", "0", "+1", "+2"], color="grey", size=12)
    plt.ylim(-3, 3)
    
    values_zero = [0] * N
    values_zero += values_zero[:1]
    ax.plot(angles, values_zero, linewidth=1, linestyle="--", color="black", alpha=0.5)
    
    colors = [MAGENTA, "#32CD32", "#1E90FF", "#FFA500", "#9370DB"]
    
    for idx, (i, row) in enumerate(genes.iterrows()):
        values = []
        for d in categories:
            val = row[f"{d}_lfc"]
            if pd.isna(val): val = 0
            if val > 3: val = 3
            if val < -3: val = -3
            values.append(val)
        values += values[:1]
        c = colors[idx % len(colors)]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=row["Symbol"], color=c)
        ax.fill(angles, values, color=c, alpha=0.1)

    plt.title(f"Radial LFC Profile: {title_text}", size=24, color=TEXT_COLOR, y=1.2)
    plt.legend(loc='upper left', bbox_to_anchor=(1.1, 1.0), labelcolor=TEXT_COLOR, fontsize=12)
    return fig
