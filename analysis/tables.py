"""
LaTeX Table Generation for CDW Paper

Generate publication-ready tables from experiment results.
"""

import numpy as np
from typing import Dict, List, Optional


def generate_main_results_table(
    results: Dict,
    caption: str = "Main Results",
) -> str:
    """
    Generate LaTeX table for main results (Table 1).

    Args:
        results: Experiment results dict
        caption: Table caption

    Returns:
        LaTeX table string
    """
    runs = results.get('runs', [])

    # Header
    latex = r"""
\begin{table}[t]
\centering
\caption{""" + caption + r"""}
\label{tab:main_results}
\begin{tabular}{lccccc}
\toprule
Method & $N=4$ & $N=8$ & $N=16$ & $N=32$ & $N=64$ \\
\midrule
"""

    # Extract data by vendor count
    vendor_counts = [4, 8, 16, 32, 64]
    cdw_accs = {}
    random_accs = {}

    for run in runs:
        n = run['num_vendors']
        cdw_accs[n] = run['cdw']['accuracy_mean']
        random_accs[n] = run['random']['accuracy_mean']

    # CDW row
    cdw_row = "CDW (Ours) "
    for n in vendor_counts:
        acc = cdw_accs.get(n, 0)
        cdw_row += f"& {acc*100:.1f}\\% "
    cdw_row += r"\\"
    latex += cdw_row + "\n"

    # Random row
    random_row = "Random Codes "
    for n in vendor_counts:
        acc = random_accs.get(n, 0)
        random_row += f"& {acc*100:.1f}\\% "
    random_row += r"\\"
    latex += random_row + "\n"

    # Footer
    latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
    return latex


def generate_robustness_table(
    results: Dict,
    caption: str = "Robustness Results",
) -> str:
    """
    Generate LaTeX table for robustness results (Table 2).

    Args:
        results: Experiment results dict
        caption: Table caption

    Returns:
        LaTeX table string
    """
    attacks = results.get('attacks', {})

    latex = r"""
\begin{table}[t]
\centering
\caption{""" + caption + r"""}
\label{tab:robustness}
\begin{tabular}{lcc}
\toprule
Attack & CDW (Ours) & Random \\
\midrule
"""

    for attack_name, params in attacks.items():
        for param_val, acc_data in params.items():
            cdw_acc = acc_data.get('cdw', 0) * 100
            random_acc = acc_data.get('random', 0) * 100
            label = f"{attack_name} ({param_val})"
            latex += f"{label} & {cdw_acc:.1f}\\% & {random_acc:.1f}\\% \\\\\n"

    latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
    return latex


def generate_ablation_table(
    results: Dict,
    caption: str = "Ablation Study",
) -> str:
    """
    Generate LaTeX table for ablation study.

    Args:
        results: Experiment results dict
        caption: Table caption

    Returns:
        LaTeX table string
    """
    latex = r"""
\begin{table}[t]
\centering
\caption{""" + caption + r"""}
\label{tab:ablation}
\begin{tabular}{lccc}
\toprule
Configuration & Accuracy & FID & LPIPS \\
\midrule
"""

    ablations = results.get('ablations', [])
    for ablation in ablations:
        name = ablation.get('name', 'Unknown')
        acc = ablation.get('accuracy', 0) * 100
        fid = ablation.get('fid', 0)
        lpips = ablation.get('lpips', 0)
        latex += f"{name} & {acc:.1f}\\% & {fid:.2f} & {lpips:.4f} \\\\\n"

    latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
    return latex


def generate_all_tables(
    results: Dict,
    output_dir: str,
):
    """
    Generate all paper tables.

    Args:
        results: Combined experiment results
        output_dir: Directory to save tables
    """
    from pathlib import Path

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Table 1
    if 'exp1' in results:
        table1 = generate_main_results_table(results['exp1'])
        with open(output_path / "table1_main_results.tex", 'w') as f:
            f.write(table1)
        print("Generated Table 1: Main Results")

    # Table 2
    if 'exp2' in results:
        table2 = generate_robustness_table(results['exp2'])
        with open(output_path / "table2_robustness.tex", 'w') as f:
            f.write(table2)
        print("Generated Table 2: Robustness")

    print(f"\nAll tables saved to: {output_dir}")
