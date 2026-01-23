#!/usr/bin/env python3
"""
Visualization module for worker counting results.

Generates charts and graphs for worker presence data.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

from aggregator import DailyAggregation, TimeSlot


class WorkerVisualizer:
    """
    Generates visualizations for worker counting data.

    Supports:
    - Time-series line charts
    - Hourly bar charts
    - Daily summary charts
    - Multi-day comparison
    """

    def __init__(self, style: str = "default", figsize: Tuple[int, int] = (12, 6)):
        """
        Initialize visualizer.

        Args:
            style: Matplotlib style (default, seaborn, ggplot, etc.)
            figsize: Default figure size (width, height)
        """
        self.figsize = figsize
        if style != "default":
            try:
                plt.style.use(style)
            except Exception:
                pass

        self.colors = {
            "primary": "#2196F3",
            "secondary": "#4CAF50",
            "accent": "#FF9800",
            "danger": "#F44336",
            "neutral": "#9E9E9E"
        }

    def plot_hourly_counts(
        self,
        aggregation: DailyAggregation,
        output_path: str = None,
        title: str = None,
        show: bool = False
    ) -> Optional[str]:
        """
        Plot hourly worker counts as a line chart.

        Args:
            aggregation: DailyAggregation result
            output_path: Path to save the chart (optional)
            title: Chart title (optional)
            show: Whether to display the chart

        Returns:
            Path to saved file if output_path provided
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        hours = sorted(aggregation.hourly_counts.keys())
        counts = [aggregation.hourly_counts[h] for h in hours]

        x_positions = range(len(hours))

        ax.plot(x_positions, counts, marker='o', linewidth=2,
                color=self.colors["primary"], markersize=8)
        ax.fill_between(x_positions, counts, alpha=0.3, color=self.colors["primary"])

        peak_idx = counts.index(max(counts))
        ax.scatter([peak_idx], [max(counts)], color=self.colors["accent"],
                   s=150, zorder=5, label=f'Peak: {max(counts)} workers')

        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Number of Workers', fontsize=12)

        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        else:
            ax.set_title(f'Worker Count - {aggregation.site_id} ({aggregation.date})',
                        fontsize=14, fontweight='bold')

        ax.set_xticks(x_positions)
        ax.set_xticklabels(hours, rotation=45, ha='right')

        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='upper right')

        stats_text = f'Total Unique: {aggregation.total_unique_workers}\nMan-hours: {aggregation.man_hours:.1f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved chart to: {output_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return output_path

    def plot_hourly_bar(
        self,
        aggregation: DailyAggregation,
        output_path: str = None,
        title: str = None,
        show: bool = False
    ) -> Optional[str]:
        """
        Plot hourly worker counts as a bar chart.

        Args:
            aggregation: DailyAggregation result
            output_path: Path to save the chart
            title: Chart title
            show: Whether to display

        Returns:
            Path to saved file
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        hours = sorted(aggregation.hourly_counts.keys())
        counts = [aggregation.hourly_counts[h] for h in hours]

        colors = [self.colors["accent"] if c == max(counts)
                  else self.colors["primary"] for c in counts]

        bars = ax.bar(hours, counts, color=colors, edgecolor='white', linewidth=1.2)

        for bar, count in zip(bars, counts):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       str(count), ha='center', va='bottom', fontsize=10)

        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Number of Workers', fontsize=12)

        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        else:
            ax.set_title(f'Hourly Worker Distribution - {aggregation.site_id}',
                        fontsize=14, fontweight='bold')

        ax.set_ylim(0, max(counts) * 1.2 if counts else 10)
        plt.xticks(rotation=45, ha='right')

        ax.axhline(y=np.mean(counts) if counts else 0, color=self.colors["secondary"],
                   linestyle='--', label=f'Average: {np.mean(counts):.1f}')
        ax.legend()

        plt.tight_layout()

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()

        return output_path

    def plot_daily_summary(
        self,
        aggregation: DailyAggregation,
        output_path: str = None,
        show: bool = False
    ) -> Optional[str]:
        """
        Create a summary dashboard for daily data.

        Args:
            aggregation: DailyAggregation result
            output_path: Path to save
            show: Whether to display

        Returns:
            Path to saved file
        """
        fig = plt.figure(figsize=(14, 8))

        ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=2)
        hours = sorted(aggregation.hourly_counts.keys())
        counts = [aggregation.hourly_counts[h] for h in hours]
        ax1.fill_between(range(len(hours)), counts, alpha=0.4, color=self.colors["primary"])
        ax1.plot(range(len(hours)), counts, marker='o', color=self.colors["primary"], linewidth=2)
        ax1.set_title('Worker Timeline', fontsize=12, fontweight='bold')
        ax1.set_xticks(range(len(hours)))
        ax1.set_xticklabels(hours, rotation=45, ha='right')
        ax1.set_ylabel('Workers')
        ax1.grid(True, alpha=0.3)

        ax2 = plt.subplot2grid((2, 3), (0, 2))
        metrics = ['Unique\nWorkers', 'Man-\nhours', 'Peak\nCount']
        values = [aggregation.total_unique_workers, aggregation.man_hours, aggregation.peak_count]
        colors_list = [self.colors["primary"], self.colors["secondary"], self.colors["accent"]]
        bars = ax2.bar(metrics, values, color=colors_list)
        for bar, val in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}' if isinstance(val, float) else str(val),
                    ha='center', fontsize=11, fontweight='bold')
        ax2.set_title('Key Metrics', fontsize=12, fontweight='bold')
        ax2.set_ylim(0, max(values) * 1.3)

        ax3 = plt.subplot2grid((2, 3), (1, 0), colspan=3)
        if counts:
            non_zero = [c for c in counts if c > 0]
            stats = {
                'Average': np.mean(non_zero) if non_zero else 0,
                'Max': max(counts),
                'Min (non-zero)': min(non_zero) if non_zero else 0,
                'Std Dev': np.std(non_zero) if non_zero else 0,
                'Active Hours': len(non_zero)
            }
            ax3.axis('off')
            table_data = [[k, f'{v:.1f}' if isinstance(v, float) else str(v)]
                         for k, v in stats.items()]
            table = ax3.table(cellText=table_data,
                             colLabels=['Metric', 'Value'],
                             loc='center',
                             cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(11)
            table.scale(0.8, 1.5)

        fig.suptitle(f'Daily Summary - {aggregation.site_id} ({aggregation.date})',
                    fontsize=14, fontweight='bold', y=1.02)

        plt.tight_layout()

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()

        return output_path

    def plot_multi_day_comparison(
        self,
        aggregations: List[DailyAggregation],
        output_path: str = None,
        show: bool = False
    ) -> Optional[str]:
        """
        Compare worker counts across multiple days.

        Args:
            aggregations: List of DailyAggregation results
            output_path: Path to save
            show: Whether to display

        Returns:
            Path to saved file
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        ax1 = axes[0]
        for agg in aggregations:
            hours = sorted(agg.hourly_counts.keys())
            counts = [agg.hourly_counts[h] for h in hours]
            ax1.plot(range(len(hours)), counts, marker='o', label=agg.date, linewidth=2)

        if aggregations:
            hours = sorted(aggregations[0].hourly_counts.keys())
            ax1.set_xticks(range(len(hours)))
            ax1.set_xticklabels(hours, rotation=45, ha='right')

        ax1.set_title('Daily Comparison - Worker Timeline', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Workers')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        ax2 = axes[1]
        dates = [a.date for a in aggregations]
        unique_workers = [a.total_unique_workers for a in aggregations]
        man_hours = [a.man_hours for a in aggregations]

        x = np.arange(len(dates))
        width = 0.35

        bars1 = ax2.bar(x - width/2, unique_workers, width, label='Unique Workers',
                        color=self.colors["primary"])
        bars2 = ax2.bar(x + width/2, man_hours, width, label='Man-hours',
                        color=self.colors["secondary"])

        ax2.set_xlabel('Date')
        ax2.set_ylabel('Count')
        ax2.set_title('Daily Totals Comparison', fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(dates)
        ax2.legend()

        plt.tight_layout()

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()

        return output_path

    def plot_worker_presence_heatmap(
        self,
        presence_data: Dict[str, List[str]],
        output_path: str = None,
        show: bool = False
    ) -> Optional[str]:
        """
        Create a heatmap showing worker presence over time.

        Args:
            presence_data: Dict mapping person_id to list of timestamps
            output_path: Path to save
            show: Whether to display

        Returns:
            Path to saved file
        """
        if not presence_data:
            print("No presence data to visualize")
            return None

        all_times = set()
        for times in presence_data.values():
            all_times.update(times)
        all_times = sorted(all_times)

        if len(all_times) > 24:
            all_times = all_times[::len(all_times)//24 + 1]

        workers = sorted(presence_data.keys())
        matrix = np.zeros((len(workers), len(all_times)))

        for i, worker in enumerate(workers):
            worker_times = set(presence_data[worker])
            for j, time in enumerate(all_times):
                if time in worker_times:
                    matrix[i, j] = 1

        fig, ax = plt.subplots(figsize=(max(12, len(all_times) * 0.5), max(6, len(workers) * 0.4)))

        cmap = plt.cm.Blues
        im = ax.imshow(matrix, aspect='auto', cmap=cmap)

        ax.set_yticks(range(len(workers)))
        ax.set_yticklabels(workers)

        time_labels = [t.split('T')[1][:5] if 'T' in t else t[:5] for t in all_times]
        ax.set_xticks(range(len(all_times)))
        ax.set_xticklabels(time_labels, rotation=45, ha='right')

        ax.set_xlabel('Time')
        ax.set_ylabel('Worker ID')
        ax.set_title('Worker Presence Heatmap', fontsize=14, fontweight='bold')

        plt.tight_layout()

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()

        return output_path


def demo_visualization():
    """Generate demo visualizations."""
    from aggregator import create_test_aggregation

    aggregation = create_test_aggregation()

    viz = WorkerVisualizer()

    output_dir = Path("reports/graphs")
    output_dir.mkdir(parents=True, exist_ok=True)

    viz.plot_hourly_counts(aggregation, str(output_dir / "hourly_line.png"))
    viz.plot_hourly_bar(aggregation, str(output_dir / "hourly_bar.png"))
    viz.plot_daily_summary(aggregation, str(output_dir / "daily_summary.png"))

    print(f"\nDemo charts saved to: {output_dir}")


if __name__ == "__main__":
    demo_visualization()
