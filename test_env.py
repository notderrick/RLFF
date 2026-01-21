"""
Test script to verify DraftEnv is working correctly.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.environment import DraftEnv
from src.data import PlayerLoader, ScenarioGenerator


def test_player_loader():
    """Test player data generation"""
    print("=" * 60)
    print("TEST 1: PlayerLoader")
    print("=" * 60)

    loader = PlayerLoader(data_dir="data/raw")
    players = loader.load_players(year=2024, num_players=200, use_cache=False)

    print(f"✓ Loaded {len(players)} players")

    # Show top 10 by VOR
    top_vor = sorted(players, key=lambda p: p.vor, reverse=True)[:10]
    print("\nTop 10 Players by VOR:")
    for i, p in enumerate(top_vor, 1):
        print(f"  {i}. {p.name} ({p.position.value}) - VOR: {p.vor:.1f}, Proj: {p.projected_points:.1f}")

    return players


def test_draft_env(players):
    """Test DraftEnv basic functionality"""
    print("\n" + "=" * 60)
    print("TEST 2: DraftEnv")
    print("=" * 60)

    env = DraftEnv(
        player_pool=players,
        num_teams=12,
        rounds=15,
        agent_draft_position=5,
        greedy_opponents=True
    )

    # Reset environment
    obs, info = env.reset()
    print("✓ Environment reset successful")
    print(f"\nInitial observation:\n{obs}\n")

    # Take a few steps
    print("=" * 60)
    print("Taking 3 draft picks...")
    print("=" * 60)

    for step_num in range(3):
        valid_actions = env.get_valid_actions()
        action = valid_actions[0]  # Always pick first available (greedy)

        obs, reward, terminated, truncated, info = env.step(action)

        print(f"\n--- Step {step_num + 1} ---")
        print(f"Picked: {env.drafted_players[-1].name}")
        print(f"Reward: {reward:.2f}")
        print(f"Round: {info['round']}, Pick: {info['pick']}")
        print(f"Roster size: {info['roster_size']}")

        if terminated or truncated:
            print("Draft ended!")
            break

        print(f"\nNext observation:\n{obs}\n")

    print("✓ DraftEnv step() working correctly")

    return env


def test_scenario_generator(players):
    """Test natural language scenario generation"""
    print("\n" + "=" * 60)
    print("TEST 3: ScenarioGenerator")
    print("=" * 60)

    from src.environment.draft_env import Roster

    # Create a partially-filled roster
    roster = Roster()
    roster.add_player(players[0])  # Add first player
    roster.add_player(players[5])  # Add another

    # Generate scenario
    scenario = ScenarioGenerator.generate_scenario(
        round_num=3,
        pick_num=5,
        roster=roster,
        available_players=players[10:25],
        top_n=8,
        include_reasoning_prompt=True
    )

    print(scenario)
    print("\n✓ Scenario generation working")

    # Test training example generation
    print("\n" + "=" * 60)
    print("TEST 4: Training Example Generation")
    print("=" * 60)

    training_example = ScenarioGenerator.generate_training_example(
        round_num=3,
        pick_num=5,
        roster=roster,
        available_players=players[10:25],
        correct_pick=players[12],
        reasoning="This player fills a positional need and has high VOR."
    )

    print("PROMPT:")
    print(training_example['prompt'])
    print("\nCOMPLETION:")
    print(training_example['completion'])

    print("\n✓ Training example generation working")


def test_roster_evaluation(env):
    """Test roster quality evaluation"""
    print("\n" + "=" * 60)
    print("TEST 5: Roster Evaluation")
    print("=" * 60)

    agent_roster = env.rosters[env.agent_draft_position - 1]
    score = env.evaluate_roster(agent_roster)

    print(f"Agent roster score: {score:.2f}")
    print(f"Roster: {agent_roster.to_string()}")

    print("\n✓ Roster evaluation working")


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("RLFF Environment Test Suite")
    print("=" * 60 + "\n")

    try:
        # Test 1: Load players
        players = test_player_loader()

        # Test 2: DraftEnv
        env = test_draft_env(players)

        # Test 3: Scenario generator
        test_scenario_generator(players)

        # Test 4: Roster evaluation
        test_roster_evaluation(env)

        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60 + "\n")

        print("Next steps:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Run SFT training script (coming next)")
        print("  3. Implement GRPO training loop")
        print("  4. Build tournament simulator")

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
