"""
Visualization Module for Sensitivity Analysis Comparison (once)

This module provides the 'ResultsVisualizer' class, which is designed to
take the results of a sensitivity analysis and generate professional,
publication-ready comparison plots for both sensitivity indices and ranks.

"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

class ResultsVisualizer:
    """
    Handles the visualization of sensitivity analysis results.
    
    This class is initialized with analysis results and a configuration object,
    and it provides methods to generate comparison plots. It now defines its
    own plotting styles internally.
    """
    def __init__(self, results, config):
        """
        Initializes the ResultsVisualizer.

        Args:
            results (dict): A dictionary containing the analysis results.
            config (AnalysisConfig): The configuration object, used for labels
                                     and variable counts.
        """
        if not results:
            raise ValueError("Results data cannot be empty.")
        self.results = results
        self.config = config
        
        # Define plotting styles directly within the visualizer
        self.colors = {
            'rf_pvi': 'skyblue', 'sobol_toi': 'plum',
            'true_analytic': 'black', 'grid': '#BDC3C7',
            'edge': 'gray', 'text': 'black'
        }
        self.plot_params = {
            'font.size': 12, 'font.family': 'serif',
            'axes.labelsize': 13, 'axes.titlesize': 15,
            'xtick.labelsize': 12, 'ytick.labelsize': 12,
            'legend.fontsize': 11, 'figure.titlesize': 17,
            'axes.linewidth': 1.2, 'grid.linewidth': 0.8,
            'lines.linewidth': 2.5, 'lines.markersize': 8
        }
        
        # Apply plotting style and parameters
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update(self.plot_params)

    def plot_index_comparison(self):
        """Generates and displays the sensitivity index comparison plot."""
        print("Generating sensitivity index comparison plot...")
        self._plot_comparison(mode='index')

    def plot_rank_comparison(self):
        """Generates and displays the feature rank comparison plot."""
        print("Generating rank comparison plot...")
        self._plot_comparison(mode='rank')
        
    def _plot_comparison(self, mode):
        """Internal plotting function to create comparison bar charts."""
        is_rank_mode = mode == 'rank'
        
        true_vals = self.results['true_rank'] if is_rank_mode else self.results['true_index']
        method1_vals = self.results['rank_rf_pvi'] if is_rank_mode else self.results['rf_pvi']
        method2_vals = self.results['rank_sobol_toi'] if is_rank_mode else self.results['sobol_toi']
        
        ylabel = 'Importance Rank' if is_rank_mode else 'Total-Order Index (TOI)'
        title = 'RF-PVI vs. Sobol-TOI Rank Comparison' if is_rank_mode else 'RF-PVI vs. Sobol-TOI Sensitivity Indices'
        true_label = 'True Rank' if is_rank_mode else 'True (Analytic)'

        fig, ax = plt.subplots(figsize=(12, 7))
        x = np.arange(len(self.config.variables))
        width = 0.35

        ax.bar(x - width/2, method1_vals, width, label='RF-PVI', color=self.colors['rf_pvi'], alpha=0.9, edgecolor=self.colors['edge'])
        ax.bar(x + width/2, method2_vals, width, label='Sobol-TOI', color=self.colors['sobol_toi'], alpha=0.9, edgecolor=self.colors['edge'])
        ax.plot(x, true_vals, label=true_label, color=self.colors['true_analytic'], marker='D', markersize=7, linestyle='--', zorder=5)

        for i, val in enumerate(true_vals):
            txt = f'{val}' if is_rank_mode else f'{val:.2f}'
            offset = -0.2 if is_rank_mode else (max(true_vals) * 0.05)
            ax.annotate(txt, (x[i], val + offset), ha='center', fontsize=12, color=self.colors['text'], fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(self.config.variables, fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.set_title(title, fontweight='bold', pad=20)
        
        if is_rank_mode:
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{int(y)}'))
            ax.invert_yaxis()
            ax.set_ylim(self.config.n_variables + 0.5, 0.5)
        else:
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.2f}'))
            ax.set_ylim(0, max(ax.get_ylim()) * 1.1)

        legend = ax.legend(title='Method', title_fontsize='12', loc='best', frameon=True)
        legend.get_frame().set_edgecolor(self.colors['edge'])
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

        plt.tight_layout()
        plt.savefig(f'figure {ylabel}.jpg',dpi=300,bbox_inches='tight')
        plt.show()