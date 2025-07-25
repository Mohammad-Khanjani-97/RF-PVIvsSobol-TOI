"""
Visualization Module for Multi-Sample Sensitivity Analysis with Replications

This module provides the 'ResultsVisualizer' class, designed to generate plots
showing confidence intervals for sensitivity indices and correctness rates for
parameter rankings, based on replicated multi-samples analyses.

"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, PercentFormatter
import math

class ResultsVisualizer:
    """Handles all visualization for the multi-sample, replicated sensitivity analysis."""
    
    def __init__(self, analyzer):
        """
        Initializes the visualizer with the completed analyzer object.

        Args:
            analyzer (SensitivityAnalyzer): The analyzer instance after
                                            run_analysis() has been called.
        """
        self.analyzer = analyzer
        
        # Define plotting styles
        self.colors = {
            'rf_pvi': 'darkblue', 'sobol': 'mediumvioletred', 'true': '#2C3E50',
            'grid': '#BDC3C7', 'spine': '#34495E', 'face': '#FAFAFA'
        }
        self.plot_params = {
            'font.size': 11, 'font.family': 'serif',
            'axes.labelsize': 12, 'axes.titlesize': 14,
            'xtick.labelsize': 10, 'ytick.labelsize': 10,
            'legend.fontsize': 10, 'figure.titlesize': 16,
            'axes.linewidth': 1.2, 'grid.linewidth': 0.8,
            'lines.linewidth': 2.5, 'lines.markersize': 7
        }
        plt.rcParams.update(self.plot_params)

    def create_confidence_interval_plot(self):
        """Creates the sensitivity indices plot with 95% confidence intervals."""
        fig = self._create_figure_grid('Total-Order Sensitivity Index (with 95% CI)')
        x_axis = self.analyzer.config.number_of_runs

        for i in range(self.analyzer.config.n_variables):
            ax = fig.add_subplot(fig.gs[i // fig.cols, i % fig.cols])
            
            # Plot Sobol-TOI mean and confidence interval
            ax.plot(x_axis, self.analyzer.sobol_toi_mean[:, i], label='Sobol-TOI', color=self.colors['sobol'], marker='s', markerfacecolor='white', markeredgewidth=2)
            ax.fill_between(x_axis, self.analyzer.sobol_toi_ci_lower[:, i], self.analyzer.sobol_toi_ci_upper[:, i], color=self.colors['sobol'], alpha=0.35)

            # Plot RF-PVI mean and confidence interval
            ax.plot(x_axis, self.analyzer.rf_pvi_mean[:, i], label='RF-PVI', color=self.colors['rf_pvi'], marker='o', markerfacecolor='white', markeredgewidth=2)
            ax.fill_between(x_axis, self.analyzer.rf_pvi_ci_lower[:, i], self.analyzer.rf_pvi_ci_upper[:, i], color=self.colors['rf_pvi'], alpha=0.35)
            
            # Plot analytical true value
            ax.axhline(y=self.analyzer.true_indices[i], label='True', color=self.colors['true'], linestyle='--')
            
            self._format_ci_axes(ax, i)
            
        # fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('TO multi sample with replications.jpg',dpi=300,bbox_inches='tight')                
        plt.show()
    
    def create_rank_correctness_plot(self):
        """
        Creates a plot comparing the average rank correctness rate, grouped by
        parameter sensitivity (High, Moderate, Low).
        """
        # 1. Get necessary data from the analyzer object
        true_indices = self.analyzer.true_indices
        rank_rf = self.analyzer.rank_correctness_rf
        rank_sobol = self.analyzer.rank_correctness_sobol
        x_axis = self.analyzer.config.number_of_runs

        # 2. Categorize parameters into High, Moderate, and Low sensitivity groups
        if len(true_indices) < 3:
            print("Cannot generate grouped rank plot: requires at least 3 variables for categorization.")
            return

        boundaries = np.quantile(true_indices, [1/3, 2/3])
        low_b, high_b = boundaries[0], boundaries[1]

        high_indices = np.where(true_indices > high_b)[0]
        low_indices = np.where(true_indices <= low_b)[0]
        
        all_indices = np.arange(len(true_indices))
        moderate_indices = np.setdiff1d(all_indices, np.concatenate([low_indices, high_indices]))
        
        categories = {
            "High Sensitivity": high_indices,
            "Moderate Sensitivity": moderate_indices,
            "Low Sensitivity": low_indices,
        }

        # 3. Create figure layout
        plot_categories = {name: indices for name, indices in categories.items() if len(indices) > 0}
        n_plots = len(plot_categories)
        if n_plots == 0:
            print("Could not form parameter sensitivity groups to plot.")
            return

        fig, axes = plt.subplots(n_plots, 1, figsize=(9, 4 * n_plots), sharex=True, squeeze=False)
        axes = axes.flatten()
        fig.suptitle('Rank Success(%) by Sensitivity Groups', fontsize=18, fontweight='bold')

        # 4. Loop through categories, calculate average correctness, and plot
        for i, (name, indices) in enumerate(plot_categories.items()):
            ax = axes[i]
            
            # Calculate the average rank correctness for all parameters in the group
            avg_rank_rf = np.mean(rank_rf[:, indices], axis=1)
            avg_rank_sobol = np.mean(rank_sobol[:, indices], axis=1)

            # Plot the averaged data
            ax.plot(x_axis, avg_rank_rf, label='RF-PVI', color=self.colors['rf_pvi'], marker='o', markersize=10, markerfacecolor='white', markeredgewidth=4,linewidth=4)
            ax.plot(x_axis, avg_rank_sobol, label='Sobol-TOI', color=self.colors['sobol'], marker='s', markersize=10, markerfacecolor='white', markeredgewidth=4,linewidth=4)
            
            # 5. Format each subplot
            ax.set_ylim(-0.05, 1.05)
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
            ax.set_title(f'{name} Parameters', fontweight='bold', pad=15)
            ax.set_ylabel('Success (%)')
            self._apply_common_styling(ax, i, legend_loc='lower right')

            if i < n_plots - 1:
                ax.tick_params(labelbottom=False)

        axes[-1].set_xlabel('Number of Runs')
        # fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('Rank multi sample with replications.jpg',dpi=300,bbox_inches='tight')        
        plt.show()

    def _create_figure_grid(self, title):
        """Helper to create a standard figure and GridSpec layout."""
        n_vars = self.analyzer.config.n_variables
        cols = math.ceil(math.sqrt(n_vars))
        rows = math.ceil(n_vars / cols)
        
        fig = plt.figure(figsize=(cols * 4.5, rows * 3.5))
        # fig.gs = fig.add_gridspec(rows, cols, hspace=0.7, wspace=0.4, top=0.92, bottom=0.1, left=0.08, right=0.97)
        fig.gs = fig.add_gridspec(rows, cols, hspace=0.7, wspace=0.4)
        fig.suptitle(title, fontsize=18, fontweight='bold')
        fig.cols = cols
        return fig

    def _format_ci_axes(self, ax, param_index):
        """Formats axes for the confidence interval plots."""
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.3f}'))
        ax.set_title(f'Parameter {self.analyzer.variables[param_index]}', fontweight='bold', pad=15)
        ax.set_xlabel('Number of Runs')
        ax.set_ylabel('Total-Order (TO)')
        self._apply_common_styling(ax, param_index, legend_loc='upper right')
    
    def _apply_common_styling(self, ax, plot_index, legend_loc='upper right'):
        """Applies common styling to subplots."""
        ax.grid(True, linestyle=':', alpha=0.6, color=self.colors['grid'])
        ax.set_axisbelow(True)
        
        ax.set_xticks(self.analyzer.config.number_of_runs)
        ax.set_xticklabels([f'{x:,}' for x in self.analyzer.config.number_of_runs], rotation=45, ha='right')
        
        for spine in ax.spines.values():
            spine.set_color(self.colors['spine'])

        # Show legend only on the first subplot to avoid redundancy
        if plot_index == 0:
            legend = ax.legend(loc=legend_loc, frameon=True, fancybox=True, shadow=True)
            legend.get_frame().set_alpha(0.95)
