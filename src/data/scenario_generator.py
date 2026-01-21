"""
ScenarioGenerator: Converts draft states into natural language scenarios
for LLM training and inference.
"""

from typing import List, Dict, Optional
from ..environment.draft_env import Player, Roster, Position


class ScenarioGenerator:
    """
    Generates natural language draft scenarios from structured data.

    This is critical for teaching the LLM to reason about drafts.
    """

    @staticmethod
    def generate_scenario(
        round_num: int,
        pick_num: int,
        roster: Roster,
        available_players: List[Player],
        top_n: int = 5,
        include_reasoning_prompt: bool = True
    ) -> str:
        """
        Generate a natural language draft scenario.

        Args:
            round_num: Current round number
            pick_num: Current pick number
            roster: Agent's current roster
            available_players: List of available players
            top_n: Number of top players to show
            include_reasoning_prompt: Whether to add <think> tags

        Returns:
            Natural language scenario string
        """
        overall_pick = (round_num - 1) * 12 + pick_num  # Assume 12-team league

        # Sort players by VOR (best strategy) or projected points
        top_players = sorted(available_players, key=lambda p: p.vor, reverse=True)[:top_n]

        # Build scenario
        parts = []

        # Header
        parts.append(f"=== DRAFT SCENARIO ===")
        parts.append(f"Round: {round_num} | Pick: {pick_num} | Overall: {overall_pick}")
        parts.append("")

        # Current roster
        parts.append("YOUR CURRENT ROSTER:")
        if roster.size() == 0:
            parts.append("  (Empty - first pick)")
        else:
            if roster.qb:
                parts.append(f"  QB: {', '.join(p.name for p in roster.qb)}")
            if roster.rb:
                parts.append(f"  RB: {', '.join(p.name for p in roster.rb)}")
            if roster.wr:
                parts.append(f"  WR: {', '.join(p.name for p in roster.wr)}")
            if roster.te:
                parts.append(f"  TE: {', '.join(p.name for p in roster.te)}")

        parts.append(f"  Total: {roster.size()} players")
        parts.append("")

        # Positional needs
        needs = []
        if roster.needs_position(Position.QB):
            needs.append("QB (CRITICAL)")
        if roster.needs_position(Position.RB):
            needs.append("RB (CRITICAL)")
        if roster.needs_position(Position.WR):
            needs.append("WR (CRITICAL)")
        if roster.needs_position(Position.TE):
            needs.append("TE (CRITICAL)")

        if needs:
            parts.append("POSITIONAL NEEDS: " + ", ".join(needs))
            parts.append("")

        # Available players
        parts.append("TOP AVAILABLE PLAYERS:")
        for i, player in enumerate(top_players, 1):
            adp_delta = overall_pick - player.adp
            value_indicator = ""
            if adp_delta > 10:
                value_indicator = " [REACH]"
            elif adp_delta < -10:
                value_indicator = " [VALUE]"

            parts.append(
                f"  {i}. {player.name} ({player.position.value}) - "
                f"Projected: {player.projected_points:.1f} pts | "
                f"ADP: {player.adp:.1f} | "
                f"VOR: {player.vor:.1f}{value_indicator}"
            )

        parts.append("")

        # Reasoning prompt
        if include_reasoning_prompt:
            parts.append("INSTRUCTIONS:")
            parts.append("Analyze the situation and choose the best player.")
            parts.append("Use <think> tags to explain your reasoning:")
            parts.append("  - What positions do you need most?")
            parts.append("  - Which player provides the best VOR?")
            parts.append("  - Are there any value/reach considerations?")
            parts.append("  - What is the next tier drop-off?")
            parts.append("")
            parts.append("Format: <think>Your reasoning</think>")
            parts.append("Then respond with: PICK: [Player Name]")
        else:
            parts.append("PICK: ")

        return "\n".join(parts)

    @staticmethod
    def generate_training_example(
        round_num: int,
        pick_num: int,
        roster: Roster,
        available_players: List[Player],
        correct_pick: Player,
        reasoning: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Generate a training example (scenario + correct answer).

        Used for supervised fine-tuning.

        Args:
            round_num: Current round number
            pick_num: Current pick number
            roster: Agent's current roster
            available_players: List of available players
            correct_pick: The correct player to pick
            reasoning: Optional reasoning for the pick

        Returns:
            Dictionary with 'prompt' and 'completion' keys
        """
        scenario = ScenarioGenerator.generate_scenario(
            round_num=round_num,
            pick_num=pick_num,
            roster=roster,
            available_players=available_players,
            top_n=8,
            include_reasoning_prompt=True
        )

        # Generate completion
        if reasoning:
            completion = f"<think>{reasoning}</think>\nPICK: {correct_pick.name}"
        else:
            # Auto-generate basic reasoning
            reasoning_parts = []

            # Check if pick fills a need
            if roster.needs_position(correct_pick.position):
                reasoning_parts.append(f"We have a critical need at {correct_pick.position.value}.")

            # Check if pick is high VOR
            top_vor = max(p.vor for p in available_players[:10])
            if correct_pick.vor >= top_vor * 0.9:
                reasoning_parts.append(f"{correct_pick.name} has excellent VOR of {correct_pick.vor:.1f}.")

            # Check draft value
            overall_pick = (round_num - 1) * 12 + pick_num
            if correct_pick.adp > overall_pick + 5:
                reasoning_parts.append(f"This is great value - ADP is {correct_pick.adp:.1f}.")

            if not reasoning_parts:
                reasoning_parts.append(f"{correct_pick.name} is the best player available.")

            reasoning = " ".join(reasoning_parts)
            completion = f"<think>{reasoning}</think>\nPICK: {correct_pick.name}"

        return {
            "prompt": scenario,
            "completion": completion
        }

    @staticmethod
    def parse_llm_response(response: str) -> Optional[str]:
        """
        Parse LLM response to extract the picked player name.

        Args:
            response: Raw LLM output

        Returns:
            Player name or None if parsing failed
        """
        # Look for "PICK: Player Name" pattern
        if "PICK:" in response:
            pick_line = [line for line in response.split('\n') if 'PICK:' in line]
            if pick_line:
                player_name = pick_line[0].split('PICK:')[-1].strip()
                return player_name

        # Fallback: return last line
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        if lines:
            return lines[-1]

        return None

    @staticmethod
    def extract_reasoning(response: str) -> Optional[str]:
        """
        Extract reasoning from <think> tags.

        Args:
            response: Raw LLM output

        Returns:
            Reasoning text or None
        """
        if '<think>' in response and '</think>' in response:
            start = response.index('<think>') + len('<think>')
            end = response.index('</think>')
            return response[start:end].strip()

        return None
