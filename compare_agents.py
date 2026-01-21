"""
compare_agents.py: Compare RL Agent vs. Greedy Bot baseline

This is the ultimate test from Week 4 of the roadmap.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import argparse
import json

from src.environment import DraftEnv
from src.data import PlayerLoader
from src.models import DraftAgent
from src.diagnostics import DraftAnalyzer, DraftVisualizer
from tournament import TournamentSimulator


class GreedyAgent:
    """Simple greedy agent that always picks highest VOR"""

    def pick_player(self, scenario, available_players):
        # Just pick first (assuming they're sorted by VOR)
        return available_players[0], "Picking highest VOR player", None

    def calculate_confidence(self, logprobs):
        return 1.0  # Always confident


def main():
    parser = argparse.ArgumentParser(description="Compare RL agent to baseline")
    parser.add_argument('--rl-checkpoint', type=str, required=True,
                        help='Path to RL agent checkpoint')
    parser.add_argument('--leagues', type=int, default=1000,
                        help='Number of leagues to simulate per agent')
    parser.add_argument('--output', type=str, default='experiments/comparison',
                        help='Output directory')

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("AGENT COMPARISON: RL Agent vs. Greedy Baseline")
    print("="*60 + "\n")

    # Load player pool
    print("Loading player pool...")
    loader = PlayerLoader()
    players = loader.load_players(num_players=300)

    # Initialize agents
    print(f"Loading RL agent from {args.rl_checkpoint}...")
    rl_agent = DraftAgent(model_name=args.rl_checkpoint)

    print("Initializing Greedy baseline...")
    greedy_agent = GreedyAgent()

    # Run tournaments
    agents = {
        'RL Agent': rl_agent,
        'Greedy Bot': greedy_agent
    }

    tournament = TournamentSimulator(
        agent=rl_agent,  # Will be swapped for each agent
        player_pool=players
    )

    all_results = tournament.compare_agents(
        agents=agents,
        num_leagues=args.leagues
    )

    # Detailed comparison
    print("\n" + "="*60)
    print("DETAILED COMPARISON")
    print("="*60 + "\n")

    rl_results = all_results['RL Agent']
    greedy_results = all_results['Greedy Bot']

    comparison = DraftAnalyzer.compare_to_baseline(
        agent_results={
            'win_rate': rl_results['win_rate'],
            'avg_agent_score': rl_results['avg_agent_score']
        },
        baseline_results={
            'win_rate': greedy_results['win_rate'],
            'avg_agent_score': greedy_results['avg_agent_score']
        }
    )

    print(json.dumps(comparison['improvements'], indent=2))

    # Save results
    results_file = output_dir / "comparison_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'rl_agent': {
                'win_rate': float(rl_results['win_rate']),
                'avg_score': float(rl_results['avg_agent_score'])
            },
            'greedy_bot': {
                'win_rate': float(greedy_results['win_rate']),
                'avg_score': float(greedy_results['avg_agent_score'])
            },
            'comparison': comparison
        }, f, indent=2)

    print(f"\n✓ Comparison results saved to {results_file}")

    # Final verdict
    print("\n" + "="*60)
    print("HYPOTHESIS TEST RESULT")
    print("="*60)

    if comparison['success']:
        print("\n✓✓✓ HYPOTHESIS VALIDATED! ✓✓✓")
        print("\nThe RL Agent trained on draft scenarios successfully prioritizes")
        print("long-term roster balance over greedy \"highest-projected-player\" strategies.")
        print(f"\nWin rate improvement: {comparison['improvements']['win_rate']['pct_change']:.1f}%")
    else:
        print("\n✗✗✗ HYPOTHESIS REJECTED ✗✗✗")
        print("\nThe RL Agent did not outperform the greedy baseline.")
        print("Possible reasons:")
        print("  - Insufficient training episodes")
        print("  - Reward function needs tuning")
        print("  - Model capacity limitations")
        print("  - Training instability")

    print("="*60 + "\n")


if __name__ == "__main__":
    main()
