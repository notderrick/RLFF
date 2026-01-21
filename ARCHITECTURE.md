# RLFF Architecture Deep Dive

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         RLFF System Architecture                     │
└─────────────────────────────────────────────────────────────────────┘

                              USER INPUT
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        Training Pipeline                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐        │
│  │   Phase 1   │  →   │   Phase 2   │  →   │   Phase 3   │        │
│  │             │      │             │      │             │        │
│  │    DATA     │      │     SFT     │      │    GRPO     │        │
│  │ GENERATION  │      │  (LoRA FT)  │      │  (RL TRAIN) │        │
│  └─────────────┘      └─────────────┘      └─────────────┘        │
│         │                     │                     │               │
│         ▼                     ▼                     ▼               │
│  Player Pool         Model learns          Model learns            │
│  + VOR calc         pick grammar         strategy/reasoning        │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Tournament Evaluation                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│   RL Agent  vs.  Greedy Bots  (1,000 leagues)                       │
│                                                                       │
│   Metrics: Win Rate | Avg Score | Positional Balance                │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
                         DIAGNOSTIC REPORTS
```

---

## Component Interaction Flow

### 1. Draft Environment (DraftEnv)

```
┌───────────────────────────────────────────────────────────┐
│                      DraftEnv                             │
├───────────────────────────────────────────────────────────┤
│                                                           │
│  State:                                                   │
│    • Player pool (300 players with VOR)                  │
│    • 12 team rosters                                     │
│    • Current round/pick                                  │
│    • Available players                                   │
│                                                           │
│  Observation → Natural Language:                         │
│    "Round 5, Pick 52. Roster: QB (Allen), WR (Jefferson) │
│     Available: Mixon (RB, VOR=89), Pitts (TE, VOR=72)"  │
│                                                           │
│  Action → Player Index:                                  │
│    Agent selects index 0-N from available players        │
│                                                           │
│  Reward → VOR-based:                                     │
│    Base VOR (0.1x) + Tier bonus (1.0) + Need (0.5)     │
│    + Draft value (0.3)                                   │
│                                                           │
└───────────────────────────────────────────────────────────┘
```

### 2. Agent Decision Process

```
┌─────────────────────────────────────────────────────────────┐
│                   Agent Decision Flow                        │
└─────────────────────────────────────────────────────────────┘

1. Observe Draft State (Natural Language)
   ↓
   "Round 3, Pick 5. Need RB/TE. Top available:
    Mixon (RB, VOR=89), Pitts (TE, VOR=72)..."

2. Generate Reasoning (<think> tags)
   ↓
   <think>
   We desperately need RB depth. Mixon has elite VOR
   and the next RB tier drops significantly. Pitts is
   good but TEs are deep this year. Taking Mixon.
   </think>

3. Output Pick
   ↓
   PICK: Joe Mixon

4. Receive Reward
   ↓
   Reward = VOR(89)*0.1 + Need(RB)*0.5 + Value*0.3
         = 8.9 + 0.5 + 0.3 = 9.7

5. Update Policy (GRPO)
   ↓
   Increase probability of similar reasoning patterns
```

### 3. Training Data Generation

```
┌─────────────────────────────────────────────────────────────┐
│              Training Example Generation                     │
└─────────────────────────────────────────────────────────────┘

FOR each of 2,000 examples:

  1. Simulate Draft to Random State
     ↓
     Round 7, Pick 5 (various roster states)

  2. Get Top 8 Available Players
     ↓
     Sorted by VOR

  3. Greedy Oracle Picks Best
     ↓
     Highest VOR considering positional need

  4. Generate Training Pair
     ↓
     Prompt: Draft scenario in natural language
     Completion: <think>reasoning</think> PICK: PlayerName

  5. Save to Dataset
     ↓
     JSON format for SFT training
```

---

## Module Breakdown

### Environment Module (`src/environment/`)

**draft_env.py** (462 lines)

```python
class DraftEnv(gym.Env):
    """
    Core Features:
    - Snake draft (1→12, 12→1, repeat)
    - Roster constraint validation (2 QB max, 4 RB max, etc.)
    - VOR-based reward calculation
    - Natural language observation generation
    - Opponent simulation (greedy by default)
    """

    def step(action) → (obs, reward, done, truncated, info):
        """
        1. Validate action (player exists, roster allows)
        2. Update roster
        3. Calculate reward (VOR + bonuses)
        4. Simulate opponent picks until agent turn
        5. Return new observation
        """

    def _calculate_reward(player, roster) → float:
        """
        Reward Components:
        - Base VOR: player.vor * 0.1
        - Tier bonus: +1.0 if top-5, +0.5 if top-12
        - Need bonus: +0.5 if critical position
        - Value bonus: +0.3 if beating ADP by 5+
        """
```

**Key Design Decisions**:
- Natural language obs → allows LLM reasoning
- Modular reward → easy to tune weights
- Snake draft → realistic simulation

---

### Data Module (`src/data/`)

**player_loader.py** (290 lines)

```python
class PlayerLoader:
    """
    Responsibilities:
    1. Generate synthetic player pool (or load real data)
    2. Calculate VOR for each player
    3. Cache results for fast loading
    """

    def load_players(year, num_players) → List[Player]:
        """
        1. Generate projections (QB: 200-400 pts, RB: 100-330, etc.)
        2. Assign ADPs with noise
        3. Calculate VOR = Proj - Replacement Level
        4. Return sorted player pool
        """

    def _calculate_vor(players):
        """
        VOR Calculation:
        - Find replacement level (12th QB, 24th RB, etc.)
        - For each player: VOR = max(0, proj - replacement)
        - Result: Elite QBs have VOR~100, replacement~0
        """
```

**scenario_generator.py** (178 lines)

```python
class ScenarioGenerator:
    """
    Converts structured draft state → natural language
    """

    @staticmethod
    def generate_scenario(round, pick, roster, available) → str:
        """
        Output Format:
        ───────────────────────────────────────
        Draft State: Round 5, Pick 52
        Current Roster: QB (Allen), WR (Jefferson)
        Positional Needs: RB (CRITICAL), TE (CRITICAL)

        Top Available Players:
          1. Mixon (RB) - Proj: 245 | VOR: 89 [VALUE]
          2. Pitts (TE) - Proj: 178 | VOR: 72
        ───────────────────────────────────────
        """
```

---

### Model Module (`src/models/`)

**draft_agent.py** (206 lines)

```python
class DraftAgent:
    """
    LLM wrapper with MLX/PyTorch backends
    """

    def __init__(model_name, use_mlx=True):
        """
        1. Detect device (MPS/CUDA/CPU)
        2. Load tokenizer + model
        3. Apply LoRA if training
        """

    def generate(prompt, config) → (response, logprobs):
        """
        Generation with log probability tracking:
        1. Tokenize prompt
        2. Generate with sampling (temp, top_p, top_k)
        3. Track token log probs for confidence
        4. Decode response
        """

    def pick_player(scenario, available) → (name, reasoning, logprobs):
        """
        1. Generate response from scenario
        2. Parse "PICK: PlayerName" from output
        3. Extract <think>reasoning</think>
        4. Return structured result
        """
```

**MLX vs PyTorch**:
```
MLX (Apple Silicon):
  ✓ 2-3x faster on M1/M2/M3
  ✓ Lower memory usage
  ✗ Fewer features (no easy log probs)

PyTorch:
  ✓ Full feature support
  ✓ Token log probabilities
  ✗ Slower on Mac
```

---

### Training Module (`src/training/`)

**sft_trainer.py** (289 lines)

```python
class SFTTrainer:
    """
    Supervised Fine-Tuning with LoRA
    """

    def train(training_examples, epochs=3):
        """
        1. Create dataset from examples
        2. Apply LoRA (r=16, alpha=32)
        3. Train with causal LM objective
        4. Save checkpoint every 100 steps
        """

    def generate_training_data(num_examples=2000):
        """
        1. Run simulated drafts
        2. At each agent pick, record state
        3. Get greedy oracle pick (highest VOR)
        4. Generate (scenario, correct_pick) pair
        5. Save to JSON
        """
```

**grpo_trainer.py** (361 lines)

```python
class GRPOTrainer:
    """
    Group Relative Policy Optimization
    """

    def train(num_episodes=100):
        """
        FOR each episode (full draft):
          1. Reset environment
          2. FOR each agent turn:
             a. Generate K=8 candidate picks
             b. Evaluate each with reward function
             c. Select best candidate
             d. Update policy (favor high-reward)
          3. Calculate episode metrics
          4. Save checkpoint every 25 episodes
        """

    def _generate_candidates(scenario, available, K=8):
        """
        Generate diverse picks:
        1. Sample with temperature (0.8-1.2)
        2. Validate picks are in available pool
        3. Ensure no duplicates
        4. Fallback to greedy if needed
        """
```

**GRPO vs PPO**:
```
GRPO:
  ✓ Simpler (no value network)
  ✓ Group normalization (stable)
  ✓ Efficient (batch candidates)

PPO:
  ✗ More complex
  ✗ Needs value network
  ✗ Harder to tune
```

---

### Diagnostics Module (`src/diagnostics/`)

**visualizer.py** (303 lines)

Creates plots:
- Draft heatmaps (position × round)
- Confidence over time
- Reward progression
- Win rate comparisons
- Position distribution (pie charts)

**analyzer.py** (387 lines)

Analysis tools:
```python
analyze_confidence_trend(logprobs_history)
  → trend, initial, final confidence

analyze_reasoning_quality(reasoning_traces)
  → % mentions of VOR, needs, tiers, ADP

analyze_positional_balance(draft_history)
  → entropy score, overloaded positions

stress_test_broken_board(agent, bad_players)
  → can agent pick "least bad" rationally?

compare_to_baseline(agent_results, baseline)
  → % improvements, verdict (BETTER/WORSE)
```

---

## Data Flow Example

### Complete Pick Cycle

```
1. ENVIRONMENT STATE
   ├─ Round: 5
   ├─ Pick: 52
   ├─ Agent Roster: [Allen (QB), Jefferson (WR), Ekeler (RB)]
   ├─ Available: [Mixon, Pitts, Flowers, ...]
   └─ Needs: [RB, TE]

                ↓

2. SCENARIO GENERATION
   ├─ Convert to natural language
   ├─ List top 8 available players
   ├─ Highlight positional needs
   └─ Format for LLM

                ↓

3. AGENT INFERENCE (GRPO: Generate 8 candidates)
   ├─ Candidate 1: "Mixon" (temp=0.8)
   ├─ Candidate 2: "Pitts" (temp=0.9)
   ├─ Candidate 3: "Flowers" (temp=1.0)
   ├─ ...
   └─ Candidate 8: "Mixon" (temp=1.2) [duplicate, rejected]

                ↓

4. REWARD EVALUATION
   ├─ Mixon:   VOR=89 → reward = 8.9 + 0.5 (need) + 0.3 (value) = 9.7
   ├─ Pitts:   VOR=72 → reward = 7.2 + 0.5 (need) = 7.7
   ├─ Flowers: VOR=66 → reward = 6.6 + 0.0 = 6.6
   └─ ...

                ↓

5. SELECTION
   ├─ Best: Mixon (reward=9.7)
   └─ Reasoning: "Need RB depth, Mixon has elite VOR,
                  next RB tier drops significantly"

                ↓

6. POLICY UPDATE (simplified)
   ├─ Increase prob of "need RB" reasoning
   ├─ Increase prob of selecting high-VOR when needed
   └─ Decrease prob of reaches (low-VOR picks)

                ↓

7. ENVIRONMENT UPDATE
   ├─ Add Mixon to agent roster
   ├─ Remove from available pool
   ├─ Simulate 11 opponent picks (greedy)
   └─ Return to agent's next turn
```

---

## Reward Function Deep Dive

```python
def _calculate_reward(player: Player, roster: Roster) → float:
    """
    Reward = f(VOR, Tier, Need, Value)

    Goal: Teach agent to maximize long-term roster value
    """

    reward = 0.0

    # Component 1: Base VOR (70% weight)
    # ─────────────────────────────────
    # Raw value over replacement
    # Elite: 80-100 VOR → 8-10 reward
    # Good: 40-60 VOR → 4-6 reward
    # Replacement: 0-20 VOR → 0-2 reward
    reward += player.vor * 0.1

    # Component 2: Positional Tier (20% weight)
    # ──────────────────────────────────────────
    # Bonus for drafting tier-break players
    position_rank = get_rank(player, position)
    if position_rank <= 5:
        reward += 1.0  # Top 5: elite tier
    elif position_rank <= 12:
        reward += 0.5  # Top 12: starter tier

    # Component 3: Positional Need (10% weight)
    # ──────────────────────────────────────────
    # Critical if starter slots unfilled
    if roster.needs_position(player.position):
        reward += 0.5

    # Component 4: Draft Value (bonus/penalty)
    # ─────────────────────────────────────────
    # Reward beating ADP (value picks)
    # Penalize reaching (bad value)
    overall_pick = current_pick_number
    adp_diff = player.adp - overall_pick

    if adp_diff > 5:
        reward += 0.3  # Great value!
    elif adp_diff < -10:
        reward -= 0.3  # Reach!

    return reward
```

**Why This Works**:
1. **Base VOR**: Ensures fundamentally good picks
2. **Tier Bonus**: Teaches "draft before tier drop"
3. **Need Bonus**: Balances roster construction
4. **Value Bonus**: Punishes reaches, rewards steals

**Tuning Guide**:
- Increase VOR weight (0.1 → 0.2): More greedy
- Increase need weight (0.5 → 1.0): More balanced rosters
- Add tier drop-off penalty: Even more strategic

---

## Performance Optimization

### MLX Backend (Apple Silicon)

```python
# Traditional PyTorch (CPU/MPS)
generate_time = 2.5 seconds per pick
memory_usage = 8GB

# With MLX
generate_time = 0.9 seconds per pick  # 2.8x faster
memory_usage = 4GB                     # 2x more efficient
```

### Caching Strategy

```python
# Player pool cached to disk
cache_file = "data/raw/players_2024_300.json"

# First load: 30 seconds (generate + calculate VOR)
# Cached loads: 0.5 seconds (JSON read)
```

### Batch Processing

```python
# SFT Training
batch_size = 4
gradient_accumulation = 2
effective_batch = 8  # Good for 32GB RAM

# GRPO Training
num_candidates = 8
parallel_generation = True  # Generate all 8 at once
```

---

## Extensibility

### Adding New Features

1. **Bye Week Awareness**
```python
# In reward function
if len(roster.bye_weeks) > 0:
    if player.bye_week in roster.bye_weeks:
        reward -= 0.2  # Penalize bye week stacking
```

2. **Injury Risk**
```python
# In Player dataclass
@dataclass
class Player:
    injury_risk: float  # 0-1 scale

# In reward
reward -= player.injury_risk * 0.5
```

3. **Opponent Modeling**
```python
# Track opponent tendencies
opponent_preferences = analyze_draft_history(opponents)

# Adjust strategy
if opponent_likes_rbs_early:
    boost_rb_priority()
```

---

## Testing Strategy

```
Unit Tests:
├─ test_environment.py
│  ├─ test_roster_constraints()
│  ├─ test_vor_calculation()
│  ├─ test_reward_function()
│  └─ test_snake_draft()
│
├─ test_scenario_generation.py
│  ├─ test_natural_language_format()
│  └─ test_parsing()
│
└─ test_agent.py
   ├─ test_pick_parsing()
   └─ test_reasoning_extraction()

Integration Tests:
├─ test_sft_pipeline.py
├─ test_grpo_pipeline.py
└─ test_tournament.py

Stress Tests:
├─ test_broken_board()  # Only bad players available
├─ test_duplicate_picks()  # Model tries to draft twice
└─ test_invalid_picks()  # Model hallucinates players
```

---

## Conclusion

This architecture balances:
- **Simplicity**: Easy to understand and modify
- **Efficiency**: Runs on consumer hardware (M1 Max)
- **Interpretability**: Can inspect reasoning traces
- **Extensibility**: Modular design for new features

**Key Innovation**: Using natural language as the interface between RL environment and LLM agent, enabling interpretable strategic reasoning.
