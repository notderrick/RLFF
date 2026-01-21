"""
Tournament Simulator: Run 1,000+ league simulations to compare agents

This is the final test: RL Agent vs. Greedy Bots
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
from tqdm import tqdm
from typing import List, Dict
import json

from src.environment import DraftEnv
from src.data import PlayerLoader
from src.models import DraftAgent
from src.diagnostics import DraftVisualizer


class TournamentSimulator:
    """
    Run large-scale tournament simulations.

    Test: RL Agent vs. N Greedy Bots
    Metric: Championship probability (win rate)
    """

    def __init__(
        self,
        agent: DraftAgent,
        player_pool: List,
        num_teams: int = 12,
        rounds: int = 15
    ):
        self.agent = agent
        self.player_pool = player_pool
        self.num_teams = num_teams
        self.rounds = rounds

    def run_tournament(
        self,
        num_leagues: int = 1000,
        agent_position: int = 1,
        save_results: bool = True
    ) -> Dict:
        """
        Run tournament simulation.

        Args:
            num_leagues: Number of leagues to simulate
            agent_position: Draft position for agent
            save_results: Whether to save results to file

        Returns:
            Tournament results
        """
        print(f"\n{'='*60}")
        print(f"TOURNAMENT SIMULATION")
        print(f"{'='*60}")
        print(f"Leagues: {num_leagues}")
        print(f"Teams per league: {self.num_teams}")
        print(f"Agent position: {agent_position}")
        print(f"{'='*60}\n")

        wins = 0
        agent_scores = []
        opponent_avg_scores = []
        agent_picks = []

        for league_idx in tqdm(range(num_leagues), desc="Simulating leagues"):
            # Create fresh environment
            env = DraftEnv(
                player_pool=self.player_pool.copy(),
                num_teams=self.num_teams,
                rounds=self.rounds,
                agent_draft_position=agent_position,
                greedy_opponents=True
            )

            obs, info = env.reset()

            # Run draft
            league_picks = []
            while not info['draft_complete']:
                agent_roster = env.rosters[env.agent_draft_position - 1]
                available_players = env.available_players[:20]

                from src.data import ScenarioGenerator
                scenario = ScenarioGenerator.generate_scenario(
                    round_num=info['round'],
                    pick_num=info['pick'],
                    roster=agent_roster,
                    available_players=available_players,
                    top_n=8,
                    include_reasoning_prompt=False  # Faster for tournament
                )

                # Agent picks
                player_name, _, _ = self.agent.pick_player(
                    scenario,
                    [p.name for p in available_players]
                )

                # Find and execute
                player = next((p for p in env.available_players if p.name == player_name), None)

                if player:
                    action = env.available_players.index(player)
                    obs, reward, terminated, truncated, info = env.step(action)

                    league_picks.append({
                        'round': info['round'],
                        'position': player.position,
                        'player': player.name,
                        'vor': player.vor
                    })

                    if terminated or truncated:
                        break
                else:
                    # Fallback: greedy
                    player = max(env.available_players[:20], key=lambda p: p.vor)
                    action = env.available_players.index(player)
                    obs, reward, terminated, truncated, info = env.step(action)

                    if terminated or truncated:
                        break

            # Evaluate rosters
            agent_roster = env.rosters[env.agent_draft_position - 1]
            agent_score = env.evaluate_roster(agent_roster)
            agent_scores.append(agent_score)

            # Opponent scores
            opponent_scores = []
            for i, roster in enumerate(env.rosters):
                if i != env.agent_draft_position - 1:
                    score = env.evaluate_roster(roster)
                    opponent_scores.append(score)

            avg_opponent = np.mean(opponent_scores)
            opponent_avg_scores.append(avg_opponent)

            # Did agent win?
            if agent_score > max(opponent_scores):
                wins += 1

            agent_picks.extend(league_picks)

        # Calculate metrics
        win_rate = wins / num_leagues
        avg_score = np.mean(agent_scores)
        avg_opponent = np.mean(opponent_avg_scores)
        score_advantage = avg_score - avg_opponent

        results = {
            'num_leagues': num_leagues,
            'wins': wins,
            'win_rate': win_rate,
            'avg_agent_score': avg_score,
            'avg_opponent_score': avg_opponent,
            'score_advantage': score_advantage,
            'agent_scores': agent_scores,
            'opponent_scores': opponent_avg_scores,
            'agent_picks': agent_picks
        }

        # Print results
        self._print_results(results)

        # Save results
        if save_results:
            self._save_results(results)

        return results

    def compare_agents(
        self,
        agents: Dict[str, DraftAgent],
        num_leagues: int = 1000
    ) -> Dict:
        """
        Compare multiple agents in tournament.

        Args:
            agents: Dict of {name: agent}
            num_leagues: Leagues per agent

        Returns:
            Comparison results
        """
        print(f"\n{'='*60}")
        print(f"AGENT COMPARISON TOURNAMENT")
        print(f"{'='*60}\n")

        all_results = {}

        for agent_name, agent in agents.items():
            print(f"\nTesting {agent_name}...")

            self.agent = agent
            results = self.run_tournament(
                num_leagues=num_leagues,
                save_results=False
            )

            all_results[agent_name] = results

        # Visualize comparison
        self._visualize_comparison(all_results)

        return all_results

    def _print_results(self, results: Dict):
        """Print tournament results"""
        print(f"\n{'='*60}")
        print(f"TOURNAMENT RESULTS")
        print(f"{'='*60}")
        print(f"Leagues Simulated: {results['num_leagues']}")
        print(f"Wins: {results['wins']}")
        print(f"Win Rate: {results['win_rate']:.2%}")
        print(f"Avg Agent Score: {results['avg_agent_score']:.2f}")
        print(f"Avg Opponent Score: {results['avg_opponent_score']:.2f}")
        print(f"Score Advantage: {results['score_advantage']:.2f}")
        print(f"{'='*60}\n")

        # Verdict
        if results['win_rate'] > 0.15:  # Above 1/12 random chance
            print("✓ HYPOTHESIS VALIDATED: Agent beats greedy baseline!")
        else:
            print("✗ HYPOTHESIS REJECTED: Agent does not beat baseline")

    def _save_results(self, results: Dict):
        """Save tournament results to file"""
        output_dir = Path("experiments/tournament")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save JSON
        results_file = output_dir / "tournament_results.json"

        # Convert numpy types to Python types
        serializable_results = {
            'num_leagues': results['num_leagues'],
            'wins': results['wins'],
            'win_rate': float(results['win_rate']),
            'avg_agent_score': float(results['avg_agent_score']),
            'avg_opponent_score': float(results['avg_opponent_score']),
            'score_advantage': float(results['score_advantage']),
            'agent_scores': [float(x) for x in results['agent_scores']],
            'opponent_scores': [float(x) for x in results['opponent_scores']]
        }

        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        print(f"✓ Results saved to {results_file}")

        # Create visualizations
        viz = DraftVisualizer(output_dir="experiments/tournament")

        # Draft heatmap
        viz.plot_draft_heatmap(
            results['agent_picks'],
            title="RL Agent Draft Heatmap",
            save_name="agent_heatmap.png"
        )

        # Position distribution
        viz.plot_position_distribution(
            results['agent_picks'],
            title="RL Agent Position Distribution",
            save_name="agent_positions.png"
        )

    def _visualize_comparison(self, all_results: Dict):
        """Create comparison visualizations"""
        viz = DraftVisualizer(output_dir="experiments/tournament")

        agent_names = list(all_results.keys())
        win_rates = [results['win_rate'] for results in all_results.values()]

        viz.plot_win_rate_comparison(
            agent_names,
            win_rates,
            title="Tournament Win Rate Comparison",
            save_name="win_rate_comparison.png"
        )


def main():
    """Run tournament"""
    import argparse

    parser = argparse.ArgumentParser(description="Run fantasy football draft tournament")
    parser.add_argument('--model', type=str, default='microsoft/Phi-3.5-mini-instruct',
                        help='Model to use')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to trained model checkpoint')
    parser.add_argument('--leagues', type=int, default=100,
                        help='Number of leagues to simulate')
    parser.add_argument('--position', type=int, default=5,
                        help='Agent draft position (1-12)')

    args = parser.parse_args()

    # Load players
    print("Loading player pool...")
    loader = PlayerLoader()
    players = loader.load_players(num_players=300)

    # Initialize agent
    print(f"Loading agent: {args.model}")
    agent = DraftAgent(model_name=args.model)

    # Load checkpoint if provided
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        agent.load(args.checkpoint)

    # Create tournament
    tournament = TournamentSimulator(
        agent=agent,
        player_pool=players
    )

    # Run tournament
    results = tournament.run_tournament(
        num_leagues=args.leagues,
        agent_position=args.position
    )

    print("\n✓ Tournament complete!")


if __name__ == "__main__":
    main()
