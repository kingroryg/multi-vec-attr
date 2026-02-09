"""
Analysis and Visualization

Tools for analyzing experiment results and generating publication-quality figures.
"""

from .plots import (
    plot_accuracy_vs_vendors,
    plot_robustness_heatmap,
    plot_confusion_matrix,
    plot_theory_vs_empirical,
    plot_quality_tradeoff,
)

from .tables import (
    generate_main_results_table,
    generate_robustness_table,
    generate_ablation_table,
    generate_all_tables,
)

from .paper_figures import generate_all_figures

__all__ = [
    "plot_accuracy_vs_vendors",
    "plot_robustness_heatmap",
    "plot_confusion_matrix",
    "plot_theory_vs_empirical",
    "plot_quality_tradeoff",
    "generate_main_results_table",
    "generate_robustness_table",
    "generate_ablation_table",
    "generate_all_tables",
    "generate_all_figures",
]
