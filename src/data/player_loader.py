"""
PlayerLoader: Fetches and processes fantasy football data
Sources: nfl_data_py + Sleeper API
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
import json

try:
    import nfl_data_py as nfl
except ImportError:
    nfl = None

from ..environment.draft_env import Player, Position


class PlayerLoader:
    """
    Loads and processes player data for fantasy football drafting.

    Data sources:
    1. nfl_data_py: Historical stats, projections
    2. Sleeper API: ADP data, expert rankings
    """

    def __init__(self, data_dir: str = "data/raw", cache: bool = True):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache = cache

        # Position replacement levels (for VOR calculation)
        # Number of starters at each position in a 12-team league
        self.replacement_levels = {
            Position.QB: 12,  # 1 QB per team
            Position.RB: 24,  # 2 RB per team
            Position.WR: 24,  # 2 WR per team
            Position.TE: 12,  # 1 TE per team
            Position.DEF: 12,
            Position.K: 12,
        }

    def load_players(
        self,
        year: int = 2024,
        num_players: int = 300,
        use_cache: bool = True
    ) -> List[Player]:
        """
        Load player pool with projections and ADP.

        Args:
            year: Season year
            num_players: Number of top players to include
            use_cache: Use cached data if available

        Returns:
            List of Player objects with projections and VOR calculated
        """
        cache_file = self.data_dir / f"players_{year}_{num_players}.json"

        # Check cache
        if use_cache and cache_file.exists():
            print(f"Loading cached players from {cache_file}")
            return self._load_from_cache(cache_file)

        # Load data from sources
        print(f"Fetching player data for {year}...")

        # Load from nfl_data_py
        if nfl is not None:
            players = self._load_from_nfl_data_py(year, num_players)
        else:
            # Fallback: generate synthetic data
            print("nfl_data_py not available, generating synthetic data")
            players = self._generate_synthetic_players(num_players)

        # Calculate VOR
        self._calculate_vor(players)

        # Cache results
        if self.cache:
            self._save_to_cache(players, cache_file)

        print(f"Loaded {len(players)} players")
        return players

    def _load_from_nfl_data_py(self, year: int, num_players: int) -> List[Player]:
        """Load real player data from nfl_data_py"""
        # Import latest weekly data
        weekly_data = nfl.import_weekly_data([year])

        # Aggregate season stats
        season_stats = weekly_data.groupby(['player_id', 'player_name', 'position', 'recent_team']).agg({
            'fantasy_points': 'sum',
            'fantasy_points_ppr': 'sum'
        }).reset_index()

        # Filter relevant positions
        valid_positions = ['QB', 'RB', 'WR', 'TE']
        season_stats = season_stats[season_stats['position'].isin(valid_positions)]

        # Sort by PPR points
        season_stats = season_stats.sort_values('fantasy_points_ppr', ascending=False)

        # Take top N players
        season_stats = season_stats.head(num_players)

        # Convert to Player objects
        players = []
        for idx, row in season_stats.iterrows():
            # Generate synthetic ADP based on projection rank
            adp = idx + 1 + np.random.normal(0, 3)  # Add noise

            player = Player(
                id=str(row['player_id']),
                name=row['player_name'],
                position=Position(row['position']),
                adp=max(1, adp),
                projected_points=row['fantasy_points_ppr'],
                expert_rank=idx + 1,
                bye_week=np.random.randint(5, 15),  # Random bye week
                team=row['recent_team'] if pd.notna(row['recent_team']) else 'FA',
                vor=0.0  # Calculated later
            )
            players.append(player)

        return players

    def _generate_synthetic_players(self, num_players: int = 300) -> List[Player]:
        """
        Generate synthetic player data for testing.

        Distribution:
        - QB: 30 players
        - RB: 80 players
        - WR: 120 players
        - TE: 40 players
        - DST: 20 players
        - K: 10 players
        """
        players = []
        player_id = 1

        # QB distribution (elite to replacement)
        qb_projections = np.concatenate([
            np.random.uniform(350, 400, 5),   # Elite
            np.random.uniform(300, 350, 10),  # Good
            np.random.uniform(250, 300, 10),  # Mediocre
            np.random.uniform(200, 250, 5),   # Replacement
        ])

        for i, proj in enumerate(qb_projections):
            players.append(Player(
                id=f"QB{player_id}",
                name=f"QB Player {player_id}",
                position=Position.QB,
                adp=i + 1 + np.random.normal(0, 2),
                projected_points=proj,
                expert_rank=i + 1,
                bye_week=np.random.randint(5, 15),
                team=f"TEAM{np.random.randint(1, 33)}",
                vor=0.0
            ))
            player_id += 1

        # RB distribution
        rb_projections = np.concatenate([
            np.random.uniform(280, 330, 10),  # Elite
            np.random.uniform(220, 280, 20),  # Good
            np.random.uniform(160, 220, 25),  # Flex-worthy
            np.random.uniform(100, 160, 25),  # Replacement
        ])

        for i, proj in enumerate(rb_projections):
            players.append(Player(
                id=f"RB{player_id}",
                name=f"RB Player {player_id}",
                position=Position.RB,
                adp=i + 1 + np.random.normal(0, 3),
                projected_points=proj,
                expert_rank=i + 1,
                bye_week=np.random.randint(5, 15),
                team=f"TEAM{np.random.randint(1, 33)}",
                vor=0.0
            ))
            player_id += 1

        # WR distribution
        wr_projections = np.concatenate([
            np.random.uniform(260, 310, 12),  # Elite
            np.random.uniform(200, 260, 25),  # Good
            np.random.uniform(150, 200, 35),  # Flex-worthy
            np.random.uniform(100, 150, 48),  # Replacement
        ])

        for i, proj in enumerate(wr_projections):
            players.append(Player(
                id=f"WR{player_id}",
                name=f"WR Player {player_id}",
                position=Position.WR,
                adp=i + 1 + np.random.normal(0, 3),
                projected_points=proj,
                expert_rank=i + 1,
                bye_week=np.random.randint(5, 15),
                team=f"TEAM{np.random.randint(1, 33)}",
                vor=0.0
            ))
            player_id += 1

        # TE distribution
        te_projections = np.concatenate([
            np.random.uniform(200, 250, 5),   # Elite
            np.random.uniform(140, 200, 10),  # Good
            np.random.uniform(100, 140, 15),  # Mediocre
            np.random.uniform(60, 100, 10),   # Replacement
        ])

        for i, proj in enumerate(te_projections):
            players.append(Player(
                id=f"TE{player_id}",
                name=f"TE Player {player_id}",
                position=Position.TE,
                adp=i + 1 + np.random.normal(0, 2),
                projected_points=proj,
                expert_rank=i + 1,
                bye_week=np.random.randint(5, 15),
                team=f"TEAM{np.random.randint(1, 33)}",
                vor=0.0
            ))
            player_id += 1

        # DST distribution
        dst_projections = np.random.uniform(80, 140, 20)
        for i, proj in enumerate(dst_projections):
            players.append(Player(
                id=f"DST{player_id}",
                name=f"DST {player_id}",
                position=Position.DEF,
                adp=i + 1 + np.random.normal(0, 2),
                projected_points=proj,
                expert_rank=i + 1,
                bye_week=np.random.randint(5, 15),
                team=f"TEAM{np.random.randint(1, 33)}",
                vor=0.0
            ))
            player_id += 1

        # K distribution
        k_projections = np.random.uniform(90, 130, 10)
        for i, proj in enumerate(k_projections):
            players.append(Player(
                id=f"K{player_id}",
                name=f"K Player {player_id}",
                position=Position.K,
                adp=i + 1 + np.random.normal(0, 2),
                projected_points=proj,
                expert_rank=i + 1,
                bye_week=np.random.randint(5, 15),
                team=f"TEAM{np.random.randint(1, 33)}",
                vor=0.0
            ))
            player_id += 1

        # Shuffle to mix positions
        np.random.shuffle(players)

        return players[:num_players]

    def _calculate_vor(self, players: List[Player]):
        """
        Calculate Value Over Replacement (VOR) for all players.

        VOR = Player's projected points - Replacement-level points at position
        """
        # Group by position
        by_position: Dict[Position, List[Player]] = {}
        for player in players:
            if player.position not in by_position:
                by_position[player.position] = []
            by_position[player.position].append(player)

        # Calculate replacement level for each position
        for position, pos_players in by_position.items():
            # Sort by projected points
            pos_players.sort(key=lambda p: p.projected_points, reverse=True)

            # Get replacement level index
            replacement_idx = self.replacement_levels.get(position, 12)

            # Handle case where we have fewer players than replacement level
            if len(pos_players) <= replacement_idx:
                replacement_points = pos_players[-1].projected_points
            else:
                replacement_points = pos_players[replacement_idx].projected_points

            # Calculate VOR for each player
            for player in pos_players:
                player.vor = max(0, player.projected_points - replacement_points)

    def _save_to_cache(self, players: List[Player], cache_file: Path):
        """Save players to JSON cache"""
        player_dicts = []
        for player in players:
            player_dicts.append({
                'id': player.id,
                'name': player.name,
                'position': player.position.value,
                'adp': player.adp,
                'projected_points': player.projected_points,
                'expert_rank': player.expert_rank,
                'bye_week': player.bye_week,
                'team': player.team,
                'vor': player.vor
            })

        with open(cache_file, 'w') as f:
            json.dump(player_dicts, f, indent=2)

    def _load_from_cache(self, cache_file: Path) -> List[Player]:
        """Load players from JSON cache"""
        with open(cache_file, 'r') as f:
            player_dicts = json.load(f)

        players = []
        for pd in player_dicts:
            players.append(Player(
                id=pd['id'],
                name=pd['name'],
                position=Position(pd['position']),
                adp=pd['adp'],
                projected_points=pd['projected_points'],
                expert_rank=pd['expert_rank'],
                bye_week=pd['bye_week'],
                team=pd['team'],
                vor=pd['vor']
            ))

        return players

    def get_draft_history(self, league_id: str, year: int) -> Optional[List[Dict]]:
        """
        Fetch draft history from Sleeper API.

        Args:
            league_id: Sleeper league ID
            year: Season year

        Returns:
            List of draft picks with metadata
        """
        # TODO: Implement Sleeper API integration
        # For now, return None
        return None
