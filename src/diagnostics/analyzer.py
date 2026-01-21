"""
DraftAnalyzer: Diagnostic tools for analyzing agent reasoning
"""

from typing import List, Dict, Optional
import numpy as np
from collections import defaultdict

from ..environment.draft_env import Position, Roster


class DraftAnalyzer:
    """
    Tools for analyzing agent draft decisions and reasoning.

    Diagnostic tests:
    1. Token log probs: Is confidence increasing?
    2. Reasoning traces: Is the model thinking strategically?
    3. Positional balance: Is it drafting a balanced roster?
    4. Stress tests: Does it handle edge cases?
    """

    @staticmethod
    def analyze_confidence_trend(logprobs_history: List[List[float]]) -> Dict[str, float]:
        """
        Analyze if model confidence is increasing over training.

        Args:
            logprobs_history: List of log probability lists per episode

        Returns:
            Dict with trend metrics
        """
        # Calculate average confidence per episode
        confidences = []
        for logprobs in logprobs_history:
            if logprobs:
                avg_logprob = np.mean(logprobs)
                confidence = np.exp(avg_logprob)
                confidences.append(confidence)

        if len(confidences) < 2:
            return {'trend': 0.0, 'initial': 0.0, 'final': 0.0}

        # Calculate trend (linear regression slope)
        x = np.arange(len(confidences))
        y = np.array(confidences)

        # Simple linear regression
        slope = np.polyfit(x, y, 1)[0]

        return {
            'trend': slope,
            'initial': confidences[0],
            'final': confidences[-1],
            'mean': np.mean(confidences),
            'std': np.std(confidences)
        }

    @staticmethod
    def analyze_reasoning_quality(reasoning_traces: List[str]) -> Dict[str, any]:
        """
        Analyze quality of reasoning in <think> tags.

        Checks for strategic keywords:
        - Positional need mentions
        - VOR/value mentions
        - Tier/scarcity mentions
        - ADP/value considerations

        Args:
            reasoning_traces: List of reasoning strings

        Returns:
            Dict with quality metrics
        """
        if not reasoning_traces:
            return {'has_reasoning': False}

        strategic_keywords = {
            'positional_need': ['need', 'critical', 'must', 'shortage'],
            'value': ['VOR', 'value', 'worth', 'best available'],
            'scarcity': ['tier', 'drop-off', 'scarce', 'run on', 'next best'],
            'draft_value': ['ADP', 'reach', 'steal', 'value pick']
        }

        keyword_counts = defaultdict(int)
        total_traces = len(reasoning_traces)

        for trace in reasoning_traces:
            trace_lower = trace.lower()

            for category, keywords in strategic_keywords.items():
                if any(kw.lower() in trace_lower for kw in keywords):
                    keyword_counts[category] += 1

        # Calculate percentages
        percentages = {
            category: (count / total_traces) * 100
            for category, count in keyword_counts.items()
        }

        # Overall quality score (0-100)
        quality_score = np.mean(list(percentages.values()))

        return {
            'has_reasoning': True,
            'total_traces': total_traces,
            'keyword_percentages': percentages,
            'quality_score': quality_score
        }

    @staticmethod
    def analyze_positional_balance(draft_history: List[Dict]) -> Dict[str, any]:
        """
        Analyze if agent is drafting a balanced roster.

        Args:
            draft_history: List of picks with {'round', 'position', 'player'}

        Returns:
            Dict with balance metrics
        """
        position_counts = defaultdict(int)
        round_positions = defaultdict(list)

        for pick in draft_history:
            pos = pick['position']
            round_num = pick['round']

            if hasattr(pos, 'value'):
                pos = pos.value

            position_counts[pos] += 1
            round_positions[round_num].append(pos)

        # Calculate entropy (higher = more balanced)
        counts = np.array(list(position_counts.values()))
        probs = counts / counts.sum()
        entropy = -np.sum(probs * np.log(probs + 1e-10))

        # Max entropy for uniform distribution
        max_entropy = np.log(len(position_counts))
        balance_score = (entropy / max_entropy) * 100  # 0-100

        # Check for over-drafting (>40% of picks in one position)
        max_position = max(position_counts.items(), key=lambda x: x[1])
        is_overloaded = (max_position[1] / len(draft_history)) > 0.4

        return {
            'position_counts': dict(position_counts),
            'balance_score': balance_score,
            'entropy': entropy,
            'is_overloaded': is_overloaded,
            'overloaded_position': max_position[0] if is_overloaded else None,
            'round_diversity': {r: len(set(positions)) for r, positions in round_positions.items()}
        }

    @staticmethod
    def stress_test_broken_board(
        agent,
        low_value_players: List,
        scenario_template: str
    ) -> Dict[str, any]:
        """
        Stress test: Give agent only bad players and see if it chooses rationally.

        A smart agent picks the "least bad" option logically.
        A dumb agent might fail to pick at all or pick randomly.

        Args:
            agent: DraftAgent instance
            low_value_players: List of low-value Player objects
            scenario_template: Draft scenario template

        Returns:
            Dict with test results
        """
        from ..data.scenario_generator import ScenarioGenerator

        # Sort by VOR (even bad players have relative value)
        sorted_players = sorted(low_value_players, key=lambda p: p.vor, reverse=True)

        # Generate scenario
        from ..environment.draft_env import Roster
        roster = Roster()

        scenario = ScenarioGenerator.generate_scenario(
            round_num=10,
            pick_num=5,
            roster=roster,
            available_players=sorted_players[:8],
            top_n=8,
            include_reasoning_prompt=True
        )

        # Get agent's pick
        player_name, reasoning, logprobs = agent.pick_player(
            scenario,
            [p.name for p in sorted_players[:8]]
        )

        # Check if agent picked the best available
        best_player = sorted_players[0]
        picked_correct = (player_name == best_player.name)

        # Check reasoning quality
        has_reasoning = reasoning is not None and len(reasoning) > 10

        return {
            'scenario': 'broken_board',
            'picked_player': player_name,
            'best_player': best_player.name,
            'picked_correct': picked_correct,
            'has_reasoning': has_reasoning,
            'reasoning': reasoning
        }

    @staticmethod
    def compare_to_baseline(
        agent_results: Dict[str, float],
        baseline_results: Dict[str, float]
    ) -> Dict[str, any]:
        """
        Compare agent performance to baseline (greedy bot).

        Args:
            agent_results: Agent's evaluation metrics
            baseline_results: Baseline's evaluation metrics

        Returns:
            Comparison metrics
        """
        improvements = {}

        for key in agent_results:
            if key in baseline_results:
                agent_val = agent_results[key]
                baseline_val = baseline_results[key]

                if baseline_val != 0:
                    pct_change = ((agent_val - baseline_val) / baseline_val) * 100
                else:
                    pct_change = 0.0

                improvements[key] = {
                    'agent': agent_val,
                    'baseline': baseline_val,
                    'improvement': agent_val - baseline_val,
                    'pct_change': pct_change
                }

        # Overall verdict
        win_rate_better = improvements.get('win_rate', {}).get('improvement', 0) > 0
        score_better = improvements.get('avg_agent_score', {}).get('improvement', 0) > 0

        success = win_rate_better and score_better

        return {
            'improvements': improvements,
            'success': success,
            'verdict': 'BETTER' if success else 'WORSE/EQUAL'
        }

    @staticmethod
    def generate_diagnostic_report(
        agent_name: str,
        confidence_metrics: Dict,
        reasoning_metrics: Dict,
        balance_metrics: Dict,
        stress_test_results: Dict,
        comparison: Dict
    ) -> str:
        """
        Generate a comprehensive diagnostic report.

        Args:
            agent_name: Name of the agent
            confidence_metrics: From analyze_confidence_trend()
            reasoning_metrics: From analyze_reasoning_quality()
            balance_metrics: From analyze_positional_balance()
            stress_test_results: From stress_test_broken_board()
            comparison: From compare_to_baseline()

        Returns:
            Formatted diagnostic report
        """
        report = []
        report.append("=" * 60)
        report.append(f"DIAGNOSTIC REPORT: {agent_name}")
        report.append("=" * 60)
        report.append("")

        # 1. Confidence Analysis
        report.append("1. CONFIDENCE TREND")
        report.append("-" * 60)
        trend = confidence_metrics.get('trend', 0)
        trend_direction = "INCREASING ✓" if trend > 0 else "DECREASING ✗"
        report.append(f"   Trend: {trend_direction} ({trend:.6f})")
        report.append(f"   Initial: {confidence_metrics.get('initial', 0):.3f}")
        report.append(f"   Final: {confidence_metrics.get('final', 0):.3f}")
        report.append("")

        # 2. Reasoning Quality
        report.append("2. REASONING QUALITY")
        report.append("-" * 60)
        if reasoning_metrics.get('has_reasoning'):
            score = reasoning_metrics.get('quality_score', 0)
            status = "GOOD ✓" if score > 50 else "POOR ✗"
            report.append(f"   Quality Score: {score:.1f}/100 ({status})")
            report.append("   Strategic Keywords:")
            for category, pct in reasoning_metrics.get('keyword_percentages', {}).items():
                report.append(f"      - {category}: {pct:.1f}%")
        else:
            report.append("   No reasoning traces found ✗")
        report.append("")

        # 3. Positional Balance
        report.append("3. POSITIONAL BALANCE")
        report.append("-" * 60)
        balance = balance_metrics.get('balance_score', 0)
        balance_status = "BALANCED ✓" if balance > 70 else "UNBALANCED ✗"
        report.append(f"   Balance Score: {balance:.1f}/100 ({balance_status})")
        report.append("   Position Counts:")
        for pos, count in balance_metrics.get('position_counts', {}).items():
            report.append(f"      - {pos}: {count}")
        if balance_metrics.get('is_overloaded'):
            report.append(f"   ⚠️  Overloaded position: {balance_metrics['overloaded_position']}")
        report.append("")

        # 4. Stress Test
        report.append("4. STRESS TEST (Broken Board)")
        report.append("-" * 60)
        picked_correct = stress_test_results.get('picked_correct', False)
        test_status = "PASSED ✓" if picked_correct else "FAILED ✗"
        report.append(f"   Status: {test_status}")
        report.append(f"   Picked: {stress_test_results.get('picked_player', 'None')}")
        report.append(f"   Best Available: {stress_test_results.get('best_player', 'None')}")
        if stress_test_results.get('reasoning'):
            report.append(f"   Reasoning: {stress_test_results['reasoning'][:100]}...")
        report.append("")

        # 5. Baseline Comparison
        report.append("5. BASELINE COMPARISON")
        report.append("-" * 60)
        verdict = comparison.get('verdict', 'UNKNOWN')
        report.append(f"   Verdict: {verdict}")
        for metric, data in comparison.get('improvements', {}).items():
            report.append(f"   {metric}:")
            report.append(f"      Agent: {data['agent']:.3f}")
            report.append(f"      Baseline: {data['baseline']:.3f}")
            report.append(f"      Improvement: {data['improvement']:.3f} ({data['pct_change']:.1f}%)")
        report.append("")

        # Summary
        report.append("=" * 60)
        report.append("SUMMARY")
        report.append("=" * 60)

        checks = [
            ("Confidence increasing", trend > 0),
            ("Strategic reasoning", reasoning_metrics.get('quality_score', 0) > 50),
            ("Balanced roster", balance > 70),
            ("Stress test passed", picked_correct),
            ("Beats baseline", comparison.get('success', False))
        ]

        passed = sum(1 for _, check in checks if check)
        report.append(f"Tests Passed: {passed}/{len(checks)}")
        report.append("")

        for check_name, check_result in checks:
            status = "✓" if check_result else "✗"
            report.append(f"   {status} {check_name}")

        report.append("")
        report.append("=" * 60)

        return "\n".join(report)
