"""
train_grpo.py: GRPO (Group Relative Policy Optimization) training

This is Week 3 of the roadmap: Teach the model strategic reasoning.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import argparse

from src.environment import DraftEnv
from src.data import PlayerLoader
from src.models import DraftAgent
from src.training import GRPOTrainer
from src.diagnostics import DraftVisualizer


def main():
    parser = argparse.ArgumentParser(description="GRPO training for fantasy football drafting")
    parser.add_argument('--model', type=str, default='microsoft/Phi-3.5-mini-instruct',
                        help='Base model (or path to SFT checkpoint)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to SFT checkpoint to continue from')
    parser.add_argument('--output', type=str, default='experiments/grpo',
                        help='Output directory')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of training episodes')
    parser.add_argument('--candidates', type=int, default=8,
                        help='Number of candidate picks per scenario')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate')
    parser.add_argument('--num-players', type=int, default=300,
                        help='Number of players in pool')

    args = parser.parse_args()

    print("\n" + "="*60)
    print("GRPO TRAINING")
    print("="*60)
    print(f"Model: {args.checkpoint or args.model}")
    print(f"Output: {args.output}")
    print(f"Episodes: {args.episodes}")
    print(f"Candidates per scenario: {args.candidates}")
    print(f"Learning rate: {args.lr}")
    print("="*60 + "\n")

    # Load player pool
    print("Loading player pool...")
    loader = PlayerLoader()
    players = loader.load_players(num_players=args.num_players)

    # Initialize agent
    model_path = args.checkpoint if args.checkpoint else args.model
    print(f"Loading agent: {model_path}")
    agent = DraftAgent(model_name=model_path)

    # Create environment
    print("Creating draft environment...")
    env = DraftEnv(
        player_pool=players.copy(),
        num_teams=12,
        rounds=15,
        agent_draft_position=5,
        greedy_opponents=True
    )

    # Initialize trainer
    print("Initializing GRPO trainer...")
    trainer = GRPOTrainer(
        agent=agent,
        env=env,
        output_dir=args.output,
        num_candidates=args.candidates,
        learning_rate=args.lr
    )

    # Train
    print("\nStarting GRPO training...\n")
    trainer.train(
        num_episodes=args.episodes,
        log_interval=10,
        save_interval=25
    )

    # Evaluate
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60 + "\n")

    eval_results = trainer.evaluate(num_episodes=100)

    # Visualize training metrics
    print("\nGenerating visualizations...")
    viz = DraftVisualizer(output_dir=args.output)

    viz.plot_training_summary(
        metrics={
            'rewards': trainer.training_history['avg_rewards'],
            'confidences': trainer.training_history['avg_confidence']
        }
    )

    print("\n✓ GRPO training complete!")
    print(f"\nNext steps:")
    print(f"  1. Run full tournament: python tournament.py --checkpoint {args.output}/checkpoint_final/model --leagues 1000")
    print(f"  2. Compare to baseline: python compare_agents.py")


if __name__ == "__main__":
    main()
