import matplotlib.pyplot as plt
import numpy as np


def normalize(values):
    arr = np.array(values, dtype=np.float32)
    norm = np.linalg.norm(arr)
    return arr / (norm + 1e-8)


def plot_token_importance(token_scores):
    token_scores = normalize(token_scores)
    plt.figure(figsize=(12, 4))
    plt.plot(token_scores, linewidth=2)
    plt.fill_between(range(len(token_scores)), token_scores, alpha=0.3)
    plt.title("Token-Level Causal Influence")
    plt.xlabel("Token Index")
    plt.ylabel("Normalized Influence")
    plt.grid(True)
    plt.tight_layout()
    return plt


def plot_layer_importance(layer_scores):
    layer_scores = normalize(layer_scores)
    plt.figure(figsize=(12, 4))
    plt.bar(range(len(layer_scores)), layer_scores)
    plt.title("Layer-Level Causal Influence")
    plt.xlabel("Transformer Layer")
    plt.ylabel("Normalized Influence")
    plt.tight_layout()
    return plt


def plot_causal_distribution(token_scores, layer_scores):
    token_scores = normalize(token_scores)
    layer_scores = normalize(layer_scores)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(token_scores, bins=15)
    axes[0].set_title("Token Influence Distribution")
    axes[1].hist(layer_scores, bins=15)
    axes[1].set_title("Layer Influence Distribution")
    plt.tight_layout()
    return plt


def plot_causal_heatmap(token_scores, layer_scores):
    token_scores = normalize(token_scores)
    layer_scores = normalize(layer_scores)
    heatmap = np.outer(token_scores, layer_scores)
    plt.figure(figsize=(12, 6))
    plt.imshow(heatmap, aspect="auto")
    plt.colorbar()
    plt.title("Token-Layer Causal Interaction Map")
    plt.xlabel("Layer Index")
    plt.ylabel("Token Index")
    plt.tight_layout()
    return plt


def plot_risk_summary(causal_risk, semantic_risk, final_risk):
    labels = ["Causal", "Semantic", "Final"]
    values = [causal_risk, semantic_risk, final_risk]
    plt.figure(figsize=(6, 4))
    plt.bar(labels, values)
    plt.ylim(0, 1)
    plt.title("Risk Fusion Summary")
    plt.ylabel("Risk Score")
    plt.tight_layout()
    return plt
