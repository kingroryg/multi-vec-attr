"""
Publication-Quality Plots for CDW Paper

Matplotlib-based visualizations following academic standards.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path

# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.figsize': (6, 4),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Color palette
COLORS = {
    'cdw': '#2E86AB',       # Blue
    'random': '#A23B72',    # Purple
    'ringid': '#F18F01',    # Orange
    'treering': '#C73E1D',  # Red
    'theory': '#3A3A3A',    # Dark gray
}


def plot_accuracy_vs_vendors(
    results: Dict,
    output_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot identification accuracy vs number of vendors.

    This is the main result figure (Figure 1 in paper).

    Args:
        results: Experiment results dict with 'runs' containing per-vendor data
        output_path: If provided, save figure to this path
        show: If True, display the figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    runs = results.get('runs', [])

    # Extract data
    vendor_counts = [r['num_vendors'] for r in runs]
    cdw_accs = [r['cdw']['accuracy_mean'] for r in runs]
    cdw_stds = [r['cdw']['accuracy_std'] for r in runs]
    random_accs = [r['random']['accuracy_mean'] for r in runs]
    random_stds = [r['random']['accuracy_std'] for r in runs]

    # Plot CDW
    ax.errorbar(
        vendor_counts, cdw_accs, yerr=cdw_stds,
        marker='o', capsize=3, capthick=1,
        color=COLORS['cdw'], label='CDW (Ours)',
        linewidth=2, markersize=6,
    )

    # Plot Random baseline
    ax.errorbar(
        vendor_counts, random_accs, yerr=random_stds,
        marker='s', capsize=3, capthick=1,
        color=COLORS['random'], label='Random Codes',
        linewidth=2, markersize=6, linestyle='--',
    )

    # Plot theoretical prediction (if available)
    if 'theoretical_accuracy' in runs[0]:
        theory_accs = [r.get('theoretical_accuracy', None) for r in runs]
        if all(t is not None for t in theory_accs):
            ax.plot(
                vendor_counts, theory_accs,
                color=COLORS['theory'], label='Theoretical',
                linewidth=1.5, linestyle=':',
            )

    ax.set_xlabel('Number of Vendors (N)')
    ax.set_ylabel('Identification Accuracy')
    ax.set_xscale('log', base=2)
    ax.set_xticks(vendor_counts)
    ax.set_xticklabels([str(v) for v in vendor_counts])
    ax.set_ylim([0, 1.05])
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_title('Vendor Identification Accuracy vs. Number of Vendors')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"Saved figure to: {output_path}")

    if show:
        plt.show()

    return fig


def plot_robustness_heatmap(
    results: Dict,
    output_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot robustness results as a heatmap.

    Args:
        results: Experiment results with attack accuracy data
        output_path: Save path
        show: Display figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Expected structure: results['attacks'][attack_name][param] = accuracy
    attacks = results.get('attacks', {})

    attack_names = list(attacks.keys())
    if not attack_names:
        ax.text(0.5, 0.5, 'No attack data available',
                ha='center', va='center', transform=ax.transAxes)
        return fig

    # Build matrix
    methods = ['cdw', 'random']
    data = []
    labels = []

    for attack, params in attacks.items():
        for param_val, acc_data in params.items():
            labels.append(f"{attack}\n({param_val})")
            row = [acc_data.get(m, 0) for m in methods]
            data.append(row)

    data = np.array(data)

    im = ax.imshow(data.T, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(['CDW (Ours)', 'Random'])

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Accuracy')

    # Add value annotations
    for i in range(len(methods)):
        for j in range(len(labels)):
            text = ax.text(j, i, f'{data[j, i]:.2f}',
                          ha='center', va='center', color='black', fontsize=8)

    ax.set_title('Watermark Robustness Under Various Attacks')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)

    if show:
        plt.show()

    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    vendor_labels: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot confusion matrix for vendor identification.

    Args:
        cm: Confusion matrix of shape (num_vendors, num_vendors)
        vendor_labels: Labels for each vendor
        output_path: Save path
        show: Display figure

    Returns:
        Matplotlib figure
    """
    num_vendors = cm.shape[0]

    if vendor_labels is None:
        vendor_labels = [str(i) for i in range(num_vendors)]

    # Normalize
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(8, 8))

    im = ax.imshow(cm_normalized, cmap='Blues', vmin=0, vmax=1)

    ax.set_xticks(range(num_vendors))
    ax.set_yticks(range(num_vendors))
    ax.set_xticklabels(vendor_labels, rotation=45, ha='right')
    ax.set_yticklabels(vendor_labels)

    ax.set_xlabel('Predicted Vendor')
    ax.set_ylabel('True Vendor')
    ax.set_title('Vendor Identification Confusion Matrix')

    plt.colorbar(im, ax=ax, label='Proportion')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)

    if show:
        plt.show()

    return fig


def plot_theory_vs_empirical(
    theory_predictions: List[float],
    empirical_results: List[float],
    output_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Scatter plot comparing theoretical predictions with empirical results.

    Args:
        theory_predictions: Predicted accuracy values
        empirical_results: Observed accuracy values
        output_path: Save path
        show: Display figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(5, 5))

    ax.scatter(theory_predictions, empirical_results,
               alpha=0.7, s=50, c=COLORS['cdw'])

    # Add y=x line
    lims = [0, 1]
    ax.plot(lims, lims, 'k--', alpha=0.5, label='y = x')

    # Compute R²
    correlation = np.corrcoef(theory_predictions, empirical_results)[0, 1]
    r_squared = correlation ** 2

    ax.text(0.05, 0.95, f'R² = {r_squared:.3f}',
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top')

    ax.set_xlabel('Theoretical Accuracy')
    ax.set_ylabel('Empirical Accuracy')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    ax.set_title('Theory vs. Empirical Validation')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)

    if show:
        plt.show()

    return fig


def plot_quality_tradeoff(
    strengths: List[float],
    accuracies: List[float],
    fids: List[float],
    output_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot quality-accuracy tradeoff curve.

    Args:
        strengths: Watermark strength values
        accuracies: Corresponding accuracy values
        fids: Corresponding FID values
        output_path: Save path
        show: Display figure

    Returns:
        Matplotlib figure
    """
    fig, ax1 = plt.subplots(figsize=(6, 4))

    # Accuracy on left y-axis
    color1 = COLORS['cdw']
    ax1.set_xlabel('Watermark Strength (α)')
    ax1.set_ylabel('Accuracy', color=color1)
    ax1.plot(strengths, accuracies, 'o-', color=color1,
             label='Accuracy', linewidth=2, markersize=6)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim([0, 1.05])

    # FID on right y-axis
    ax2 = ax1.twinx()
    color2 = COLORS['random']
    ax2.set_ylabel('FID Increase', color=color2)
    ax2.plot(strengths, fids, 's--', color=color2,
             label='FID', linewidth=2, markersize=6)
    ax2.tick_params(axis='y', labelcolor=color2)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')

    ax1.grid(True, alpha=0.3)
    ax1.set_title('Quality-Accuracy Tradeoff')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)

    if show:
        plt.show()

    return fig


def generate_all_figures(
    results_dir: str,
    output_dir: str,
):
    """
    Generate all paper figures from experiment results.

    Args:
        results_dir: Directory containing experiment results
        output_dir: Directory to save figures
    """
    results_path = Path(results_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Figure 1: Vendor scaling
    exp1_path = list(results_path.glob("exp1_*/*/results.json"))
    if exp1_path:
        with open(exp1_path[0]) as f:
            exp1_results = json.load(f)
        plot_accuracy_vs_vendors(
            exp1_results,
            output_path=str(output_path / "fig1_vendor_scaling.pdf"),
            show=False
        )
        print("Generated Figure 1: Vendor Scaling")

    print(f"\nAll figures saved to: {output_dir}")
