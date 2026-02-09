#!/usr/bin/env python
"""
Generate all paper figures from experiment results.

This script reads experiment results and generates publication-quality
figures for the CDW paper.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

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


def load_experiment_results(results_dir: Path) -> Dict[str, Any]:
    """Load all experiment results from directory."""
    results = {}

    # Find latest run for each experiment
    for exp_dir in results_dir.iterdir():
        if not exp_dir.is_dir():
            continue

        exp_name = exp_dir.name.split('_')[0]  # e.g., "exp1" from "exp1_vendor_scaling"

        # Find latest run
        run_dirs = sorted(exp_dir.iterdir(), reverse=True)
        for run_dir in run_dirs:
            results_file = run_dir / "results.json"
            if results_file.exists():
                with open(results_file) as f:
                    results[exp_name] = json.load(f)
                print(f"Loaded {exp_name} from {run_dir}")
                break

    return results


def generate_figure1_vendor_scaling(
    results: Dict[str, Any],
    output_dir: Path,
):
    """Figure 1: Vendor identification accuracy vs number of vendors."""
    if "exp1" not in results:
        print("Skipping Figure 1: exp1 results not found")
        return

    output_path = output_dir / "fig1_vendor_scaling.pdf"
    plot_accuracy_vs_vendors(
        results["exp1"],
        output_path=str(output_path),
        show=False,
    )
    print(f"Generated: {output_path}")


def generate_figure2_robustness(
    results: Dict[str, Any],
    output_dir: Path,
):
    """Figure 2: Robustness heatmap under various attacks."""
    if "exp2" not in results:
        print("Skipping Figure 2: exp2 results not found")
        return

    output_path = output_dir / "fig2_robustness_heatmap.pdf"
    plot_robustness_heatmap(
        results["exp2"],
        output_path=str(output_path),
        show=False,
    )
    print(f"Generated: {output_path}")


def generate_figure3_quality_tradeoff(
    results: Dict[str, Any],
    output_dir: Path,
):
    """Figure 3: Quality-accuracy tradeoff."""
    if "exp3" not in results:
        print("Skipping Figure 3: exp3 results not found")
        return

    # Extract data for plot
    strengths_data = results["exp3"].get("strengths", {})
    strengths = []
    accuracies = []
    fids = []

    for strength_str, data in sorted(strengths_data.items(), key=lambda x: float(x[0])):
        strengths.append(data["strength"])
        accuracies.append(data["accuracy"])
        fids.append(data["fid"])

    if strengths:
        output_path = output_dir / "fig3_quality_tradeoff.pdf"
        plot_quality_tradeoff(
            strengths=strengths,
            accuracies=accuracies,
            fids=fids,
            output_path=str(output_path),
            show=False,
        )
        print(f"Generated: {output_path}")


def generate_figure4_theory_validation(
    results: Dict[str, Any],
    output_dir: Path,
):
    """Figure 4: Theory vs empirical scatter plot."""
    if "exp6" not in results:
        print("Skipping Figure 4: exp6 results not found")
        return

    scatter_data = results["exp6"].get("scatter_data", {})
    empirical = scatter_data.get("empirical", [])
    theoretical = scatter_data.get("theoretical", [])

    if empirical and theoretical:
        output_path = output_dir / "fig4_theory_validation.pdf"
        plot_theory_vs_empirical(
            theory_predictions=theoretical,
            empirical_results=empirical,
            output_path=str(output_path),
            show=False,
        )
        print(f"Generated: {output_path}")


def generate_figure5_code_length(
    results: Dict[str, Any],
    output_dir: Path,
):
    """Figure 5: Accuracy vs code length."""
    if "exp4" not in results:
        print("Skipping Figure 5: exp4 results not found")
        return

    import matplotlib.pyplot as plt
    import numpy as np

    code_lengths_data = results["exp4"].get("code_lengths", {})

    code_lengths = []
    empirical_accs = []
    theoretical_accs = []

    for length_str, data in sorted(code_lengths_data.items(), key=lambda x: int(x[0])):
        code_lengths.append(data["code_length"])
        empirical_accs.append(data["empirical_accuracy"])
        theoretical_accs.append(data["theoretical_accuracy"])

    if code_lengths:
        fig, ax = plt.subplots(figsize=(6, 4))

        ax.plot(code_lengths, empirical_accs, 'o-', label='Empirical', linewidth=2, markersize=6)
        ax.plot(code_lengths, theoretical_accs, 's--', label='Theoretical', linewidth=2, markersize=6)

        ax.set_xlabel('Code Length (L)')
        ax.set_ylabel('Identification Accuracy')
        ax.set_xscale('log', base=2)
        ax.set_ylim([0, 1.05])
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title('Accuracy vs Code Length')

        output_path = output_dir / "fig5_code_length.pdf"
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Generated: {output_path}")


def generate_figure6_code_families(
    results: Dict[str, Any],
    output_dir: Path,
):
    """Figure 6: Code family comparison bar chart."""
    if "exp5" not in results:
        print("Skipping Figure 6: exp5 results not found")
        return

    import matplotlib.pyplot as plt
    import numpy as np

    families_data = results["exp5"].get("families", {})

    # Group by vendor count
    vendor_counts = [16, 32, 64]
    family_names = ["walsh_hadamard", "gold", "random"]
    display_names = ["Walsh-Hadamard", "Gold", "Random"]

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(vendor_counts))
    width = 0.25

    for i, (family, display) in enumerate(zip(family_names, display_names)):
        accs = []
        for n in vendor_counts:
            if family in families_data and str(n) in families_data[family]:
                accs.append(families_data[family][str(n)]["accuracy"])
            else:
                accs.append(0)

        bars = ax.bar(x + i * width, accs, width, label=display)

    ax.set_xlabel('Number of Vendors')
    ax.set_ylabel('Identification Accuracy')
    ax.set_xticks(x + width)
    ax.set_xticklabels([str(n) for n in vendor_counts])
    ax.set_ylim([0, 1.05])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_title('Code Family Comparison')

    output_path = output_dir / "fig6_code_families.pdf"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Generated: {output_path}")


def generate_all_figures(results_dir: str, output_dir: str):
    """Generate all paper figures."""
    results_path = Path(results_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from: {results_path}")
    print(f"Saving figures to: {output_path}")
    print()

    # Load all results
    results = load_experiment_results(results_path)

    if not results:
        print("No experiment results found!")
        return

    # Generate each figure
    generate_figure1_vendor_scaling(results, output_path)
    generate_figure2_robustness(results, output_path)
    generate_figure3_quality_tradeoff(results, output_path)
    generate_figure4_theory_validation(results, output_path)
    generate_figure5_code_length(results, output_path)
    generate_figure6_code_families(results, output_path)

    # Generate tables
    tables_dir = output_path / "tables"
    tables_dir.mkdir(exist_ok=True)

    print("\nGenerating LaTeX tables...")
    generate_all_tables(results, str(tables_dir))

    print(f"\nAll figures and tables saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory containing experiment results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/figures",
        help="Directory to save figures",
    )
    args = parser.parse_args()

    generate_all_figures(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
