"""
DraftVisualizer: Visualization tools for diagnosing agent behavior
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
from pathlib import Path

from ..environment.draft_env import Position


class DraftVisualizer:
    """
    Visualization tools for fantasy football draft agents.

    Includes:
    - Draft heatmaps (position distribution)
    - Confidence over time
    - Reward progression
    - Win rate comparisons
    """

    def __init__(self, output_dir: str = "experiments/diagnostics"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)

    def plot_draft_heatmap(
        self,
        draft_history: List[Dict],
        title: str = "Draft Heatmap",
        save_name: str = "draft_heatmap.png"
    ):
        """
        Create a heatmap showing when positions were drafted.

        A "smart" agent shows diverse spread across rounds.
        A "dumb" agent over-drafts one position.

        Args:
            draft_history: List of picks with {'round', 'position'}
            title: Plot title
            save_name: Filename to save
        """
        # Create matrix: rounds x positions
        positions = [Position.QB, Position.RB, Position.WR, Position.TE, Position.DEF, Position.K]
        rounds = max(pick['round'] for pick in draft_history)

        matrix = np.zeros((rounds, len(positions)))

        for pick in draft_history:
            round_idx = pick['round'] - 1
            try:
                pos_idx = positions.index(pick['position'])
                matrix[round_idx, pos_idx] += 1
            except ValueError:
                continue

        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(
            matrix,
            annot=True,
            fmt='.0f',
            cmap='YlOrRd',
            xticklabels=[p.value for p in positions],
            yticklabels=[f"Round {i+1}" for i in range(rounds)],
            ax=ax,
            cbar_kws={'label': 'Number of Picks'}
        )

        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Position', fontsize=12)
        ax.set_ylabel('Round', fontsize=12)

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"✓ Draft heatmap saved to {save_path}")

    def plot_confidence_over_time(
        self,
        confidences: List[float],
        title: str = "Model Confidence Over Training",
        save_name: str = "confidence.png"
    ):
        """
        Plot model confidence (from log probs) over time.

        Increasing confidence = model is learning and becoming more certain.

        Args:
            confidences: List of confidence scores
            title: Plot title
            save_name: Filename to save
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        episodes = list(range(1, len(confidences) + 1))

        # Plot raw confidence
        ax.plot(episodes, confidences, alpha=0.3, color='blue', label='Raw')

        # Plot moving average
        window = min(10, len(confidences) // 5)
        if window > 1:
            moving_avg = np.convolve(confidences, np.ones(window)/window, mode='valid')
            ax.plot(episodes[window-1:], moving_avg, color='blue', linewidth=2, label=f'MA ({window})')

        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Confidence Score', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"✓ Confidence plot saved to {save_path}")

    def plot_reward_progression(
        self,
        rewards: List[float],
        title: str = "Reward Progression",
        save_name: str = "rewards.png"
    ):
        """
        Plot average reward over training episodes.

        Args:
            rewards: List of average rewards per episode
            title: Plot title
            save_name: Filename to save
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        episodes = list(range(1, len(rewards) + 1))

        # Plot raw rewards
        ax.plot(episodes, rewards, alpha=0.3, color='green', label='Raw')

        # Plot moving average
        window = min(10, len(rewards) // 5)
        if window > 1:
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax.plot(episodes[window-1:], moving_avg, color='green', linewidth=2, label=f'MA ({window})')

        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Average Reward', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"✓ Reward plot saved to {save_path}")

    def plot_win_rate_comparison(
        self,
        agent_names: List[str],
        win_rates: List[float],
        title: str = "Win Rate Comparison",
        save_name: str = "win_rates.png"
    ):
        """
        Bar chart comparing win rates of different agents.

        Args:
            agent_names: List of agent names
            win_rates: List of win rates (0-1)
            title: Plot title
            save_name: Filename to save
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        colors = ['red' if name == 'Greedy Bot' else 'blue' for name in agent_names]

        bars = ax.bar(agent_names, [wr * 100 for wr in win_rates], color=colors, alpha=0.7)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'{height:.1f}%',
                ha='center',
                va='bottom',
                fontsize=12,
                fontweight='bold'
            )

        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_ylabel('Win Rate (%)', fontsize=12)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"✓ Win rate comparison saved to {save_path}")

    def plot_position_distribution(
        self,
        draft_history: List[Dict],
        title: str = "Positional Distribution",
        save_name: str = "position_dist.png"
    ):
        """
        Pie chart showing distribution of positions drafted.

        Args:
            draft_history: List of picks with {'position'}
            title: Plot title
            save_name: Filename to save
        """
        position_counts = {}

        for pick in draft_history:
            pos = pick['position'].value if hasattr(pick['position'], 'value') else pick['position']
            position_counts[pos] = position_counts.get(pos, 0) + 1

        fig, ax = plt.subplots(figsize=(10, 8))

        colors = sns.color_palette('Set2', len(position_counts))

        wedges, texts, autotexts = ax.pie(
            position_counts.values(),
            labels=position_counts.keys(),
            autopct='%1.1f%%',
            colors=colors,
            startangle=90
        )

        # Style percentage text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(12)
            autotext.set_fontweight('bold')

        ax.set_title(title, fontsize=16, fontweight='bold')

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"✓ Position distribution saved to {save_path}")

    def plot_training_summary(
        self,
        metrics: Dict[str, List[float]],
        save_name: str = "training_summary.png"
    ):
        """
        Create a summary dashboard with multiple metrics.

        Args:
            metrics: Dict with keys 'rewards', 'confidences', 'win_rates'
            save_name: Filename to save
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Reward progression
        if 'rewards' in metrics:
            ax = axes[0, 0]
            episodes = list(range(1, len(metrics['rewards']) + 1))
            ax.plot(episodes, metrics['rewards'], color='green', linewidth=2)
            ax.set_title('Reward Progression', fontsize=14, fontweight='bold')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Avg Reward')
            ax.grid(True, alpha=0.3)

        # Confidence progression
        if 'confidences' in metrics:
            ax = axes[0, 1]
            episodes = list(range(1, len(metrics['confidences']) + 1))
            ax.plot(episodes, metrics['confidences'], color='blue', linewidth=2)
            ax.set_title('Confidence Progression', fontsize=14, fontweight='bold')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Confidence')
            ax.grid(True, alpha=0.3)

        # Win rate over time
        if 'win_rates' in metrics:
            ax = axes[1, 0]
            episodes = list(range(1, len(metrics['win_rates']) + 1))
            ax.plot(episodes, [wr * 100 for wr in metrics['win_rates']], color='purple', linewidth=2)
            ax.set_title('Win Rate Over Time', fontsize=14, fontweight='bold')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Win Rate (%)')
            ax.set_ylim(0, 100)
            ax.grid(True, alpha=0.3)

        # Summary statistics
        ax = axes[1, 1]
        ax.axis('off')

        summary_text = "Training Summary\n\n"
        if 'rewards' in metrics:
            summary_text += f"Final Avg Reward: {metrics['rewards'][-1]:.3f}\n"
            summary_text += f"Best Reward: {max(metrics['rewards']):.3f}\n\n"

        if 'confidences' in metrics:
            summary_text += f"Final Confidence: {metrics['confidences'][-1]:.3f}\n"
            summary_text += f"Avg Confidence: {np.mean(metrics['confidences']):.3f}\n\n"

        if 'win_rates' in metrics:
            summary_text += f"Final Win Rate: {metrics['win_rates'][-1]:.1%}\n"

        ax.text(0.5, 0.5, summary_text, fontsize=14, ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.suptitle('GRPO Training Dashboard', fontsize=18, fontweight='bold')
        plt.tight_layout()

        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"✓ Training summary dashboard saved to {save_path}")
