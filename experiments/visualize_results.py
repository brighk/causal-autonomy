"""
Visualization Script for CAF Experiment Results
================================================
Creates publication-ready plots comparing CAF with baseline methods.

Usage:
    python -m experiments.visualize_results [path/to/metrics.json]
"""

import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, Any


def load_metrics(json_path: str) -> Dict[str, Any]:
    """Load metrics from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def plot_primary_metrics(metrics_data: Dict[str, Any], output_dir: Path):
    """
    Create bar chart comparing primary metrics across all methods.
    """
    methods_data = metrics_data['metrics_by_method']
    methods = list(methods_data.keys())

    # Extract metrics
    inference_depths = [methods_data[m]['primary_metrics']['mean_inference_depth'] for m in methods]
    contradiction_rates = [methods_data[m]['primary_metrics']['contradiction_rate_percent'] for m in methods]
    entailment_accs = [methods_data[m]['primary_metrics']['entailment_accuracy'] for m in methods]

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Colors: CAF in blue, baselines in gray
    colors = ['#2E86AB' if m == 'CAF' else '#A9A9A9' for m in methods]

    # Plot 1: Inference Depth (lower is worse)
    axes[0].bar(methods, inference_depths, color=colors, edgecolor='black', linewidth=1.5)
    axes[0].set_ylabel('Inference Depth', fontsize=12, fontweight='bold')
    axes[0].set_title('(a) Inference Depth', fontsize=13, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(axis='y', alpha=0.3, linestyle='--')

    # Plot 2: Contradiction Rate (higher = better detection)
    axes[1].bar(methods, contradiction_rates, color=colors, edgecolor='black', linewidth=1.5)
    axes[1].set_ylabel('Contradiction Rate (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('(b) Contradiction Detection Rate', fontsize=13, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(axis='y', alpha=0.3, linestyle='--')

    # Plot 3: Entailment Accuracy (higher is better) - MOST IMPORTANT
    bars = axes[2].bar(methods, entailment_accs, color=colors, edgecolor='black', linewidth=1.5)
    axes[2].set_ylabel('Entailment Accuracy', fontsize=12, fontweight='bold')
    axes[2].set_title('(c) Entailment Accuracy ★', fontsize=13, fontweight='bold')
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].grid(axis='y', alpha=0.3, linestyle='--')
    axes[2].set_ylim(0, 1.0)

    # Add value labels on bars for entailment accuracy
    for bar in bars:
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()

    # Save figure
    output_path = output_dir / 'primary_metrics_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")

    output_path_pdf = output_dir / 'primary_metrics_comparison.pdf'
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"✓ Saved: {output_path_pdf}")

    plt.close()


def plot_improvements(metrics_data: Dict[str, Any], output_dir: Path):
    """
    Create bar chart showing CAF improvements over each baseline.
    """
    if 'comparisons_to_caf' not in metrics_data:
        print("No comparison data available")
        return

    comparisons = metrics_data['comparisons_to_caf']
    baselines = list(comparisons.keys())

    # Extract improvement percentages for entailment accuracy
    entailment_improvements = [
        comparisons[b]['entailment_accuracy']['improvement_pct']
        for b in baselines
    ]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create bars
    bars = ax.barh(baselines, entailment_improvements, color='#06A77D',
                   edgecolor='black', linewidth=1.5)

    ax.set_xlabel('Improvement over Baseline (%)', fontsize=13, fontweight='bold')
    ax.set_title('CAF Entailment Accuracy Improvement', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2.,
                f'+{width:.1f}%',
                ha='left', va='center', fontsize=11, fontweight='bold')

    plt.tight_layout()

    # Save figure
    output_path = output_dir / 'caf_improvements.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")

    output_path_pdf = output_dir / 'caf_improvements.pdf'
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"✓ Saved: {output_path_pdf}")

    plt.close()


def plot_per_domain_breakdown(metrics_data: Dict[str, Any], output_dir: Path):
    """
    Create grouped bar chart showing CAF performance across domains.
    """
    caf_data = metrics_data['metrics_by_method'].get('CAF')
    if not caf_data or 'by_domain' not in caf_data:
        print("No per-domain data available")
        return

    domain_data = caf_data['by_domain']
    domains = list(domain_data.keys())

    # Extract metrics per domain
    depths = [domain_data[d]['mean_depth'] for d in domains]
    contradictions = [domain_data[d]['contradiction_rate'] for d in domains]
    accuracies = [domain_data[d]['entailment_accuracy'] for d in domains]

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Inference Depth by Domain
    axes[0].bar(domains, depths, color='#E63946', edgecolor='black', linewidth=1.5)
    axes[0].set_ylabel('Mean Depth', fontsize=12, fontweight='bold')
    axes[0].set_title('(a) Inference Depth by Domain', fontsize=13, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(axis='y', alpha=0.3, linestyle='--')

    # Plot 2: Contradiction Rate by Domain
    axes[1].bar(domains, contradictions, color='#F4A261', edgecolor='black', linewidth=1.5)
    axes[1].set_ylabel('Contradiction Rate (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('(b) Contradiction Detection by Domain', fontsize=13, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(axis='y', alpha=0.3, linestyle='--')

    # Plot 3: Entailment Accuracy by Domain
    bars = axes[2].bar(domains, accuracies, color='#2A9D8F', edgecolor='black', linewidth=1.5)
    axes[2].set_ylabel('Entailment Accuracy', fontsize=12, fontweight='bold')
    axes[2].set_title('(c) Entailment Accuracy by Domain', fontsize=13, fontweight='bold')
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].grid(axis='y', alpha=0.3, linestyle='--')
    axes[2].set_ylim(0, 1.0)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()

    # Save figure
    output_path = output_dir / 'per_domain_breakdown.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")

    output_path_pdf = output_dir / 'per_domain_breakdown.pdf'
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"✓ Saved: {output_path_pdf}")

    plt.close()


def plot_semantic_invariance(metrics_data: Dict[str, Any], output_dir: Path):
    """
    Create bar chart showing semantic invariance (CAF vs baselines).
    """
    methods_data = metrics_data['metrics_by_method']
    methods = list(methods_data.keys())

    # Extract semantic invariance
    invariances = [
        methods_data[m]['secondary_metrics']['semantic_invariance_mean']
        for m in methods
    ]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Colors: CAF in blue, baselines in gray
    colors = ['#2E86AB' if m == 'CAF' else '#A9A9A9' for m in methods]

    bars = ax.bar(methods, invariances, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Semantic Invariance', fontsize=13, fontweight='bold')
    ax.set_title('Semantic Invariance Across Prompt Perturbations', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()

    # Save figure
    output_path = output_dir / 'semantic_invariance.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")

    output_path_pdf = output_dir / 'semantic_invariance.pdf'
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"✓ Saved: {output_path_pdf}")

    plt.close()


def create_summary_figure(metrics_data: Dict[str, Any], output_dir: Path):
    """
    Create a comprehensive summary figure with all key metrics.
    """
    methods_data = metrics_data['metrics_by_method']
    methods = list(methods_data.keys())

    # Extract all metrics
    metrics_dict = {
        'Inference\nDepth': [methods_data[m]['primary_metrics']['mean_inference_depth'] for m in methods],
        'Contradiction\nRate (%)': [methods_data[m]['primary_metrics']['contradiction_rate_percent'] for m in methods],
        'Entailment\nAccuracy': [methods_data[m]['primary_metrics']['entailment_accuracy'] for m in methods],
        'Semantic\nInvariance': [methods_data[m]['secondary_metrics']['semantic_invariance_mean'] for m in methods]
    }

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(methods))
    width = 0.2
    multiplier = 0

    colors = ['#E63946', '#F4A261', '#2A9D8F', '#457B9D']

    for i, (metric_name, values) in enumerate(metrics_dict.items()):
        # Normalize values to 0-1 range for comparison
        if 'Contradiction' in metric_name:
            normalized = np.array(values) / 100.0
        elif 'Depth' in metric_name:
            normalized = np.array(values) / max(values) if max(values) > 0 else values
        else:
            normalized = values

        offset = width * multiplier
        bars = ax.bar(x + offset, normalized, width, label=metric_name,
                     color=colors[i], edgecolor='black', linewidth=1)
        multiplier += 1

    ax.set_ylabel('Normalized Score', fontsize=13, fontweight='bold')
    ax.set_title('CAF vs Baselines: All Metrics (Normalized)', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(methods)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()

    # Save figure
    output_path = output_dir / 'summary_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")

    output_path_pdf = output_dir / 'summary_comparison.pdf'
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"✓ Saved: {output_path_pdf}")

    plt.close()


def main():
    """Main visualization pipeline."""
    parser = argparse.ArgumentParser(
        description='Visualize CAF experiment results'
    )
    parser.add_argument(
        'metrics_json',
        nargs='?',
        help='Path to experiment metrics JSON file'
    )
    parser.add_argument(
        '--output-dir',
        default='experiments/results/figures',
        help='Directory to save figures'
    )

    args = parser.parse_args()

    # Find most recent metrics file if not specified
    if args.metrics_json is None:
        results_dir = Path('experiments/results')
        metrics_files = sorted(results_dir.glob('experiment_metrics_*.json'))
        if not metrics_files:
            print("Error: No metrics JSON files found in experiments/results/")
            return
        args.metrics_json = str(metrics_files[-1])
        print(f"Using most recent metrics: {args.metrics_json}")

    # Load metrics
    print(f"\nLoading metrics from: {args.metrics_json}")
    metrics_data = load_metrics(args.metrics_json)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate all visualizations
    print("\nGenerating visualizations...")
    print("=" * 60)

    plot_primary_metrics(metrics_data, output_dir)
    plot_improvements(metrics_data, output_dir)
    plot_per_domain_breakdown(metrics_data, output_dir)
    plot_semantic_invariance(metrics_data, output_dir)
    create_summary_figure(metrics_data, output_dir)

    print("=" * 60)
    print(f"\n✓ All figures saved to: {output_dir}")
    print("\nGenerated figures:")
    print("  1. primary_metrics_comparison.png/pdf - Main comparison across methods")
    print("  2. caf_improvements.png/pdf - CAF improvement percentages")
    print("  3. per_domain_breakdown.png/pdf - Performance by domain")
    print("  4. semantic_invariance.png/pdf - Consistency across perturbations")
    print("  5. summary_comparison.png/pdf - Normalized all-metrics view")


if __name__ == '__main__':
    # Set style for publication-ready figures
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 11

    main()
