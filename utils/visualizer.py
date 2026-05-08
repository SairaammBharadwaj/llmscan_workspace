import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_causal_maps(token_scores, layer_scores):
    """
    Generates a two-panel heatmap visualization for Token and Layer causal effects.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={'height_ratios': [1, 1]})
    
    # We need to reshape the 1D arrays into 2D arrays (1 row, N columns) for a heatmap
    token_matrix = np.array(token_scores).reshape(1, -1)
    layer_matrix = np.array(layer_scores).reshape(1, -1)
    
    # --- Top Plot: Token Causal Effects (Heatmap) ---
    sns.heatmap(token_matrix, ax=ax1, cmap="Blues", cbar=True, 
                xticklabels=range(len(token_scores)), yticklabels=["Prompt CE"],
                cbar_kws={"label": "Causal Effect"})
    ax1.set_title("Token-Level Causal Map", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Token Sequence Index")
    
    # --- Bottom Plot: Layer Causal Effects (Heatmap) ---
    sns.heatmap(layer_matrix, ax=ax2, cmap="Blues", cbar=True,
                xticklabels=range(len(layer_scores)), yticklabels=["Layer CE"],
                cbar_kws={"label": "Causal Effect"})
    ax2.set_title("Layer-Level Causal Map", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Transformer Layer")
    
    plt.tight_layout()
    return fig