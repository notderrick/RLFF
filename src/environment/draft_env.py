"""
DraftEnv: Gymnasium-compatible Fantasy Football Draft Simulator
Optimized for Apple Silicon (MPS support)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


class Position(Enum):
    """Fantasy football positions"""
    QB = "QB"
    RB = "RB"
    WR = "WR"
    TE = "TE"
    FLEX = "FLEX"
    DEF = "DEF"
    K = "K"


@dataclass
class Player:
    """Represents a fantasy football player"""
    id: str
    name: str
    position: Position
    adp: float  # Average Draft Position
    projected_points: float
    expert_rank: float
    bye_week: int
    team: str
    vor: float = 0.0  # Value Over Replacement (calculated)

    def __repr__(self):
        return f"{self.name} ({self.position.value}) - Proj: {self.projected_points:.1f}, ADP: {self.adp:.1f}"


@dataclass
class Roster:
    """Tracks a team's roster"""
    qb: List[Player] = field(default_factory=list)
    rb: List[Player] = field(default_factory=list)
    wr: List[Player] = field(default_factory=list)
    te: List[Player] = field(default_factory=list)
    flex: List[Player] = field(default_factory=list)
    dst: List[Player] = field(default_factory=list)
    k: List[Player] = field(default_factory=list)
    bench: List[Player] = field(default_factory=list)

    # Roster limits (standard league)
    MAX_QB = 2
    MAX_RB = 4
    MAX_WR = 4
    MAX_TE = 2
    MAX_DST = 1
    MAX_K = 1
    TOTAL_ROSTER = 15

    def add_player(self, player: Player) -> bool:
        """Add player to roster, returns False if roster constraints violated"""
        pos = player.position

        # Check position limits
        if pos == Position.QB and len(self.qb) >= self.MAX_QB:
            return False
        elif pos == Position.RB and len(self.rb) >= self.MAX_RB:
            return False
        elif pos == Position.WR and len(self.wr) >= self.MAX_WR:
            return False
        elif pos == Position.TE and len(self.te) >= self.MAX_TE:
            return False
        elif pos == Position.DEF and len(self.dst) >= self.MAX_DST:
            return False
        elif pos == Position.K and len(self.k) >= self.MAX_K:
            return False

        # Check total roster size
        if self.size() >= self.TOTAL_ROSTER:
            return False

        # Add to position list
        if pos == Position.QB:
            self.qb.append(player)
        elif pos == Position.RB:
            self.rb.append(player)
        elif pos == Position.WR:
            self.wr.append(player)
        elif pos == Position.TE:
            self.te.append(player)
        elif pos == Position.DEF:
            self.dst.append(player)
        elif pos == Position.K:
            self.k.append(player)

        return True

    def size(self) -> int:
        """Total roster size"""
        return (len(self.qb) + len(self.rb) + len(self.wr) +
                len(self.te) + len(self.dst) + len(self.k) + len(self.bench))

    def get_all_players(self) -> List[Player]:
        """Return all players on roster"""
        return (self.qb + self.rb + self.wr + self.te +
                self.dst + self.k + self.bench)

    def needs_position(self, position: Position) -> bool:
        """Check if roster needs more players at position"""
        if position == Position.QB:
            return len(self.qb) < 1  # At least 1 starter
        elif position == Position.RB:
            return len(self.rb) < 2  # At least 2 starters
        elif position == Position.WR:
            return len(self.wr) < 2  # At least 2 starters
        elif position == Position.TE:
            return len(self.te) < 1  # At least 1 starter
        return False

    def to_string(self) -> str:
        """Natural language representation of roster"""
        parts = []
        if self.qb:
            parts.append(f"QB: {', '.join(p.name for p in self.qb)}")
        if self.rb:
            parts.append(f"RB: {', '.join(p.name for p in self.rb)}")
        if self.wr:
            parts.append(f"WR: {', '.join(p.name for p in self.wr)}")
        if self.te:
            parts.append(f"TE: {', '.join(p.name for p in self.te)}")
        if self.dst:
            parts.append(f"DST: {', '.join(p.name for p in self.dst)}")
        if self.k:
            parts.append(f"K: {', '.join(p.name for p in self.k)}")

        if not parts:
            return "Empty roster"
        return " | ".join(parts)


class DraftEnv(gym.Env):
    """
    Gymnasium environment for fantasy football drafting.

    Observation: Natural language string describing draft state
    Action: Integer representing player index from available players
    Reward: VOR-based scoring with positional tier bonuses
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        player_pool: List[Player],
        num_teams: int = 12,
        rounds: int = 15,
        agent_draft_position: int = 1,
        greedy_opponents: bool = True,
        render_mode: Optional[str] = None
    ):
        super().__init__()

        self.player_pool = player_pool
        self.num_teams = num_teams
        self.rounds = rounds
        self.agent_draft_position = agent_draft_position
        self.greedy_opponents = greedy_opponents
        self.render_mode = render_mode

        # Draft state
        self.current_round = 1
        self.current_pick = 1
        self.snake_draft = True  # Snake vs linear
        self.available_players = []
        self.drafted_players = []

        # Rosters for all teams
        self.rosters = [Roster() for _ in range(num_teams)]

        # Action space: select from available players (dynamic size)
        # We'll use a large fixed size and mask invalid actions
        self.max_action_space = 500  # Max players we expect to choose from
        self.action_space = spaces.Discrete(self.max_action_space)

        # Observation space: we'll use text, but define a dummy space
        # In practice, the LLM will process the text directly
        self.observation_space = spaces.Text(max_length=2000)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[str, dict]:
        """Reset the draft environment"""
        super().reset(seed=seed)

        # Reset draft state
        self.current_round = 1
        self.current_pick = 1
        self.available_players = self.player_pool.copy()
        self.drafted_players = []
        self.rosters = [Roster() for _ in range(self.num_teams)]

        # Simulate picks until it's the agent's turn
        while not self._is_agent_turn():
            self._opponent_pick()

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action: int) -> Tuple[str, float, bool, bool, dict]:
        """
        Execute one draft pick.

        Args:
            action: Index of player to draft from available_players

        Returns:
            observation: Natural language draft state
            reward: VOR-based reward
            terminated: Draft is complete
            truncated: Error occurred
            info: Additional metadata
        """
        # Validate action
        if action < 0 or action >= len(self.available_players):
            # Invalid action - high penalty
            return self._get_observation(), -10.0, False, True, {"error": "Invalid action"}

        # Agent makes pick
        player = self.available_players[action]
        agent_roster = self.rosters[self.agent_draft_position - 1]

        # Check if pick violates roster constraints
        if not agent_roster.add_player(player):
            # Roster constraint violation - high penalty
            return self._get_observation(), -5.0, False, True, {"error": "Roster constraint violation"}

        # Remove player from available pool
        self.available_players.pop(action)
        self.drafted_players.append(player)

        # Calculate reward
        reward = self._calculate_reward(player, agent_roster)

        # Advance draft
        self._advance_pick()

        # Simulate opponent picks until it's agent's turn again or draft ends
        while not self._is_agent_turn() and not self._is_draft_complete():
            self._opponent_pick()

        # Check if draft is complete
        terminated = self._is_draft_complete()

        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, terminated, False, info

    def _is_agent_turn(self) -> bool:
        """Check if it's the agent's turn to pick"""
        if self.snake_draft:
            # Snake draft: odd rounds go 1->12, even rounds go 12->1
            if self.current_round % 2 == 1:
                return self.current_pick == self.agent_draft_position
            else:
                return self.current_pick == (self.num_teams - self.agent_draft_position + 1)
        else:
            # Linear draft
            return (self.current_pick - 1) % self.num_teams == (self.agent_draft_position - 1)

    def _advance_pick(self):
        """Advance to next pick"""
        self.current_pick += 1

        if self.current_pick > self.num_teams:
            self.current_round += 1
            self.current_pick = 1

    def _is_draft_complete(self) -> bool:
        """Check if draft is finished"""
        return self.current_round > self.rounds

    def _opponent_pick(self):
        """Simulate an opponent's pick (greedy by default)"""
        if not self.available_players:
            return

        if self.greedy_opponents:
            # Greedy: pick highest projected points
            best_idx = 0
            best_score = -1

            for idx, player in enumerate(self.available_players):
                if player.projected_points > best_score:
                    best_score = player.projected_points
                    best_idx = idx

            player = self.available_players.pop(best_idx)
        else:
            # Random pick
            player = self.available_players.pop(0)

        # Determine which team is picking
        team_idx = self._get_current_team_index()
        self.rosters[team_idx].add_player(player)
        self.drafted_players.append(player)

        self._advance_pick()

    def _get_current_team_index(self) -> int:
        """Get the index of the currently picking team"""
        if self.snake_draft:
            if self.current_round % 2 == 1:
                return self.current_pick - 1
            else:
                return self.num_teams - self.current_pick
        else:
            return (self.current_pick - 1) % self.num_teams

    def _calculate_reward(self, player: Player, roster: Roster) -> float:
        """
        Calculate VOR-based reward for a draft pick.

        Reward components:
        1. Base VOR score
        2. Positional tier bonus
        3. Positional need bonus
        """
        reward = 0.0

        # Component 1: Base VOR
        reward += player.vor * 0.1  # Scale down to reasonable range

        # Component 2: Positional tier bonus
        # Check if player is top-tier for their position
        position_players = [p for p in self.player_pool if p.position == player.position]
        position_rank = sum(1 for p in position_players if p.projected_points > player.projected_points) + 1

        if position_rank <= 5:
            reward += 1.0  # Top 5 at position
        elif position_rank <= 12:
            reward += 0.5  # Top 12 at position

        # Component 3: Positional need bonus
        if roster.needs_position(player.position):
            reward += 0.5

        # Component 4: Draft value (beating ADP)
        overall_pick = (self.current_round - 1) * self.num_teams + self.current_pick
        if player.adp > overall_pick + 5:  # Drafted 5+ spots before ADP
            reward += 0.3

        return reward

    def _get_observation(self) -> str:
        """
        Generate natural language observation of current draft state.

        This is what the LLM will see and reason about.
        """
        if self._is_draft_complete():
            return "Draft is complete."

        overall_pick = (self.current_round - 1) * self.num_teams + self.current_pick
        agent_roster = self.rosters[self.agent_draft_position - 1]

        # Top available players (limit to 8 for context)
        top_players = sorted(self.available_players, key=lambda p: p.projected_points, reverse=True)[:8]

        obs_parts = [
            f"Draft State: Round {self.current_round}, Pick {self.current_pick} (Overall: {overall_pick})",
            f"Current Roster: {agent_roster.to_string()}",
            f"Roster Size: {agent_roster.size()}/{self.rounds}",
            "\nTop Available Players:"
        ]

        for i, player in enumerate(top_players):
            obs_parts.append(
                f"{i}. {player.name} ({player.position.value}) - "
                f"Proj: {player.projected_points:.1f}, ADP: {player.adp:.1f}, "
                f"VOR: {player.vor:.1f}"
            )

        # Positional needs
        needs = []
        if agent_roster.needs_position(Position.QB):
            needs.append("QB")
        if agent_roster.needs_position(Position.RB):
            needs.append("RB")
        if agent_roster.needs_position(Position.WR):
            needs.append("WR")
        if agent_roster.needs_position(Position.TE):
            needs.append("TE")

        if needs:
            obs_parts.append(f"\nPositional Needs: {', '.join(needs)}")

        obs_parts.append("\nWhat is your pick?")

        return "\n".join(obs_parts)

    def _get_info(self) -> Dict[str, Any]:
        """Return additional debug information"""
        return {
            "round": self.current_round,
            "pick": self.current_pick,
            "roster_size": self.rosters[self.agent_draft_position - 1].size(),
            "available_players": len(self.available_players),
            "draft_complete": self._is_draft_complete()
        }

    def render(self):
        """Render the environment (human-readable)"""
        if self.render_mode == "human":
            print(self._get_observation())

    def get_valid_actions(self) -> List[int]:
        """Return list of valid action indices"""
        return list(range(len(self.available_players)))

    def evaluate_roster(self, roster: Roster) -> float:
        """
        Evaluate final roster quality for championship probability.

        Simple heuristic: sum of VOR for starters
        """
        score = 0.0

        # Best lineup
        if roster.qb:
            score += max(p.vor for p in roster.qb)
        if len(roster.rb) >= 2:
            score += sum(sorted([p.vor for p in roster.rb], reverse=True)[:2])
        if len(roster.wr) >= 2:
            score += sum(sorted([p.vor for p in roster.wr], reverse=True)[:2])
        if roster.te:
            score += max(p.vor for p in roster.te)

        return score
