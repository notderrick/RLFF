"""
GRPOTrainer: Group Relative Policy Optimization for fantasy football drafting

This is the core RL component that teaches strategic reasoning.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from pathlib import Path
import json

from ..environment import DraftEnv
from ..data import PlayerLoader, ScenarioGenerator
from ..models import DraftAgent


class GRPOTrainer:
    """
    Group Relative Policy Optimization trainer.

    Algorithm:
    1. For a given draft scenario, generate K different picks
    2. Evaluate each pick with the reward function
    3. Update model to favor high-reward picks
    4. Repeat for many scenarios

    This teaches the model long-term strategic thinking.
    """

    def __init__(
        self,
        agent: DraftAgent,
        env: DraftEnv,
        output_dir: str = "experiments/grpo",
        num_candidates: int = 8,
        learning_rate: float = 1e-5
    ):
        """
        Initialize GRPO trainer.

        Args:
            agent: DraftAgent to train
            env: DraftEnv for simulation
            output_dir: Directory to save checkpoints
            num_candidates: Number of picks to generate per scenario
            learning_rate: Learning rate for policy updates
        """
        self.agent = agent
        self.env = env
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_candidates = num_candidates
        self.learning_rate = learning_rate

        # Optimizer (only if using PyTorch)
        if not agent.use_mlx and hasattr(agent.model, 'parameters'):
            self.optimizer = torch.optim.AdamW(
                agent.model.parameters(),
                lr=learning_rate
            )
        else:
            self.optimizer = None

        # Metrics
        self.training_history = {
            'episodes': [],
            'avg_rewards': [],
            'avg_confidence': [],
            'win_rates': []
        }

    def train(
        self,
        num_episodes: int = 100,
        log_interval: int = 10,
        save_interval: int = 25
    ):
        """
        Train agent with GRPO.

        Args:
            num_episodes: Number of training episodes (drafts)
            log_interval: Episodes between logging
            save_interval: Episodes between checkpoints
        """
        print(f"\nStarting GRPO training")
        print(f"Episodes: {num_episodes}")
        print(f"Candidates per scenario: {self.num_candidates}")
        print(f"Learning rate: {self.learning_rate}")

        for episode in tqdm(range(num_episodes), desc="Training"):
            # Run one episode (full draft)
            episode_metrics = self._run_episode()

            # Update training history
            self.training_history['episodes'].append(episode)
            self.training_history['avg_rewards'].append(episode_metrics['avg_reward'])
            self.training_history['avg_confidence'].append(episode_metrics['avg_confidence'])

            # Log progress
            if (episode + 1) % log_interval == 0:
                self._log_progress(episode + 1)

            # Save checkpoint
            if (episode + 1) % save_interval == 0:
                self._save_checkpoint(episode + 1)

        # Final save
        self._save_checkpoint('final')
        self._save_metrics()

        print("\n✓ GRPO training complete")

    def _run_episode(self) -> Dict[str, float]:
        """
        Run one training episode (complete draft).

        Returns:
            Episode metrics
        """
        obs, info = self.env.reset()

        episode_rewards = []
        episode_confidences = []

        while not info['draft_complete']:
            # Generate scenario
            agent_roster = self.env.rosters[self.env.agent_draft_position - 1]
            available_players = self.env.available_players[:20]  # Top 20

            scenario = ScenarioGenerator.generate_scenario(
                round_num=info['round'],
                pick_num=info['pick'],
                roster=agent_roster,
                available_players=available_players,
                top_n=self.num_candidates,
                include_reasoning_prompt=True
            )

            # Generate K candidate picks
            candidates = self._generate_candidates(scenario, available_players)

            # Evaluate candidates with reward function
            rewards = []
            for player_name, reasoning, logprobs in candidates:
                # Find player in available pool
                player = next((p for p in self.env.available_players if p.name == player_name), None)

                if player is None:
                    # Model hallucinated - heavy penalty
                    rewards.append(-10.0)
                else:
                    # Simulate the pick and get reward
                    action = self.env.available_players.index(player)
                    # We need to peek at the reward without actually taking the step
                    # For now, calculate reward directly
                    reward = self.env._calculate_reward(player, agent_roster)
                    rewards.append(reward)

            # Select best candidate
            best_idx = np.argmax(rewards)
            best_player_name = candidates[best_idx][0]
            best_logprobs = candidates[best_idx][2]
            best_reward = rewards[best_idx]

            # Take action
            player = next((p for p in self.env.available_players if p.name == best_player_name), None)
            if player:
                action = self.env.available_players.index(player)
                obs, reward, terminated, truncated, info = self.env.step(action)

                episode_rewards.append(best_reward)

                # Calculate confidence
                if best_logprobs:
                    confidence = self.agent.calculate_confidence(best_logprobs)
                    episode_confidences.append(confidence)

                # Update policy (simplified - real GRPO would use group statistics)
                if self.optimizer and len(rewards) > 1:
                    self._update_policy(candidates, rewards)

                if terminated or truncated:
                    break
            else:
                # Hallucination - skip this turn
                break

        # Calculate episode metrics
        avg_reward = np.mean(episode_rewards) if episode_rewards else 0.0
        avg_confidence = np.mean(episode_confidences) if episode_confidences else 0.0

        return {
            'avg_reward': avg_reward,
            'avg_confidence': avg_confidence
        }

    def _generate_candidates(
        self,
        scenario: str,
        available_players: List,
        max_attempts: int = 20
    ) -> List[Tuple[str, Optional[str], Optional[List[float]]]]:
        """
        Generate K candidate picks for a scenario.

        Args:
            scenario: Natural language scenario
            available_players: List of available Player objects
            max_attempts: Maximum attempts to get valid candidates

        Returns:
            List of (player_name, reasoning, logprobs) tuples
        """
        candidates = []
        attempts = 0
        player_names = [p.name for p in available_players]

        while len(candidates) < self.num_candidates and attempts < max_attempts:
            # Generate with temperature for diversity
            from ..models.draft_agent import GenerationConfig

            config = GenerationConfig(
                temperature=0.8 + (attempts * 0.1),  # Increase temp for diversity
                top_p=0.9,
                max_tokens=200
            )

            player_name, reasoning, logprobs = self.agent.pick_player(
                scenario,
                player_names,
                config
            )

            # Validate pick is in available players
            if player_name and player_name in player_names:
                # Check if not duplicate
                if player_name not in [c[0] for c in candidates]:
                    candidates.append((player_name, reasoning, logprobs))

            attempts += 1

        # If we don't have enough candidates, add greedy picks
        while len(candidates) < self.num_candidates and len(player_names) > len(candidates):
            # Pick from available players we haven't chosen
            chosen_names = [c[0] for c in candidates]
            remaining = [name for name in player_names if name not in chosen_names]
            if remaining:
                candidates.append((remaining[0], None, None))

        return candidates

    def _update_policy(self, candidates: List[Tuple], rewards: List[float]):
        """
        Update policy based on candidate rewards.

        This is a simplified GRPO update. Full implementation would:
        1. Normalize rewards across group
        2. Calculate advantages
        3. Compute policy gradient with group baseline
        4. Update model parameters

        Args:
            candidates: List of (player_name, reasoning, logprobs)
            rewards: List of rewards for each candidate
        """
        if not self.optimizer:
            return  # Can't update with MLX yet

        # Normalize rewards
        rewards = np.array(rewards)
        if len(rewards) > 1:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # TODO: Implement full GRPO update
        # For now, this is a placeholder
        # Real implementation would compute policy gradient loss

    def evaluate(
        self,
        num_episodes: int = 100,
        opponent_type: str = "greedy"
    ) -> Dict[str, float]:
        """
        Evaluate agent performance.

        Args:
            num_episodes: Number of evaluation episodes
            opponent_type: Type of opponents ('greedy', 'random')

        Returns:
            Evaluation metrics
        """
        print(f"\nEvaluating agent over {num_episodes} episodes...")

        agent_scores = []
        opponent_avg_scores = []
        win_count = 0

        for episode in tqdm(range(num_episodes), desc="Evaluating"):
            # Run full draft
            obs, info = self.env.reset()

            while not info['draft_complete']:
                agent_roster = self.env.rosters[self.env.agent_draft_position - 1]
                available_players = self.env.available_players[:20]

                scenario = ScenarioGenerator.generate_scenario(
                    round_num=info['round'],
                    pick_num=info['pick'],
                    roster=agent_roster,
                    available_players=available_players,
                    top_n=8,
                    include_reasoning_prompt=True
                )

                # Agent picks (greedy selection for evaluation)
                player_name, _, _ = self.agent.pick_player(
                    scenario,
                    [p.name for p in available_players]
                )

                # Take action
                player = next((p for p in self.env.available_players if p.name == player_name), None)
                if player:
                    action = self.env.available_players.index(player)
                    obs, reward, terminated, truncated, info = self.env.step(action)

                    if terminated or truncated:
                        break
                else:
                    break

            # Evaluate final rosters
            agent_roster = self.env.rosters[self.env.agent_draft_position - 1]
            agent_score = self.env.evaluate_roster(agent_roster)
            agent_scores.append(agent_score)

            # Opponent scores
            opponent_scores = []
            for i, roster in enumerate(self.env.rosters):
                if i != self.env.agent_draft_position - 1:
                    score = self.env.evaluate_roster(roster)
                    opponent_scores.append(score)

            avg_opponent = np.mean(opponent_scores)
            opponent_avg_scores.append(avg_opponent)

            # Check if agent won
            if agent_score > avg_opponent:
                win_count += 1

        # Calculate metrics
        win_rate = win_count / num_episodes
        avg_score = np.mean(agent_scores)
        avg_opponent_score = np.mean(opponent_avg_scores)

        metrics = {
            'win_rate': win_rate,
            'avg_agent_score': avg_score,
            'avg_opponent_score': avg_opponent_score,
            'score_advantage': avg_score - avg_opponent_score
        }

        print(f"\n=== Evaluation Results ===")
        print(f"Win Rate: {win_rate:.2%}")
        print(f"Avg Agent Score: {avg_score:.2f}")
        print(f"Avg Opponent Score: {avg_opponent_score:.2f}")
        print(f"Score Advantage: {metrics['score_advantage']:.2f}")

        return metrics

    def _log_progress(self, episode: int):
        """Log training progress"""
        recent_rewards = self.training_history['avg_rewards'][-10:]
        recent_confidence = self.training_history['avg_confidence'][-10:]

        avg_reward = np.mean(recent_rewards) if recent_rewards else 0.0
        avg_conf = np.mean(recent_confidence) if recent_confidence else 0.0

        print(f"\nEpisode {episode}:")
        print(f"  Avg Reward (last 10): {avg_reward:.3f}")
        print(f"  Avg Confidence (last 10): {avg_conf:.3f}")

    def _save_checkpoint(self, episode):
        """Save training checkpoint"""
        checkpoint_dir = self.output_dir / f"checkpoint_{episode}"
        checkpoint_dir.mkdir(exist_ok=True)

        # Save agent
        self.agent.save(str(checkpoint_dir / "model"))

        print(f"✓ Checkpoint saved to {checkpoint_dir}")

    def _save_metrics(self):
        """Save training metrics"""
        metrics_file = self.output_dir / "training_metrics.json"

        with open(metrics_file, 'w') as f:
            json.dump(self.training_history, f, indent=2)

        print(f"✓ Metrics saved to {metrics_file}")
