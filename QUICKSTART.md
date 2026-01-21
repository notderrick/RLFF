# RLFF Quick Start Guide

## Installation

### 1. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Mac/Linux
```

### 2. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Install MLX (for Apple Silicon)
```bash
pip install mlx mlx-lm
```

## Week-by-Week Execution Plan

### Week 1: Test the Environment

**Goal**: Verify the simulator works correctly.

```bash
# Test the environment
python test_env.py
```

**Expected Output**:
- ✓ Player loader working
- ✓ DraftEnv step() working
- ✓ Scenario generation working
- ✓ Roster evaluation working

**What to Look For**:
- Do scenarios make sense in natural language?
- Are VOR calculations reasonable?
- Does the draft flow correctly?

---

### Week 2: Supervised Fine-Tuning (SFT)

**Goal**: Teach the model basic drafting grammar.

```bash
# Generate training data and fine-tune
python train_sft.py \
  --model microsoft/Phi-3.5-mini-instruct \
  --num-examples 2000 \
  --epochs 3 \
  --batch-size 4 \
  --use-lora
```

**What This Does**:
1. Generates 2,000 training examples from simulated drafts
2. Fine-tunes the model with LoRA (efficient training)
3. Saves checkpoint to `experiments/sft/final/`

**Time**: ~2-3 hours on M1 Max

**What to Look For**:
- Training loss decreasing
- Model can format picks correctly ("PICK: Player Name")
- No crashes or OOM errors

**Quick Test**:
```bash
# Test the SFT model
python -c "
from src.models import DraftAgent
agent = DraftAgent('experiments/sft/final')
response, _ = agent.generate('Pick the best QB available: Patrick Mahomes or Joe Burrow?')
print(response)
"
```

---

### Week 3: GRPO Training (Reinforcement Learning)

**Goal**: Teach strategic reasoning (VOR, positional scarcity).

```bash
# Run GRPO training
python train_grpo.py \
  --checkpoint experiments/sft/final \
  --episodes 100 \
  --candidates 8 \
  --lr 1e-5
```

**What This Does**:
1. Loads the SFT checkpoint
2. Runs 100 full draft simulations
3. For each pick, generates 8 candidate picks
4. Updates model to favor high-reward picks
5. Saves checkpoints every 25 episodes

**Time**: ~4-6 hours on M1 Max

**What to Look For**:
- Average reward increasing over episodes
- Confidence scores increasing
- No hallucinations (picking unavailable players)

**Checkpoints Saved**:
- `experiments/grpo/checkpoint_25/`
- `experiments/grpo/checkpoint_50/`
- `experiments/grpo/checkpoint_75/`
- `experiments/grpo/checkpoint_final/`

---

### Week 4: The Tournament Test

**Goal**: Run 1,000 simulated leagues to test championship probability.

```bash
# Compare RL agent to greedy baseline
python compare_agents.py \
  --rl-checkpoint experiments/grpo/checkpoint_final/model \
  --leagues 1000
```

**What This Does**:
1. Runs 1,000 leagues with RL agent
2. Runs 1,000 leagues with Greedy Bot
3. Compares win rates and scores
4. Generates visualizations

**Time**: ~1-2 hours

**Expected Output**:
```
=== HYPOTHESIS TEST RESULT ===
✓ HYPOTHESIS VALIDATED!

Win Rate:
  RL Agent:    18.5%
  Greedy Bot:  12.1%
  Improvement: +52.9%
```

**Success Criteria**:
- RL Agent win rate > 15% (above 1/12 random chance)
- RL Agent win rate > Greedy Bot win rate
- Average roster score higher than baseline

---

## Diagnostics: Peering into the Brain

### 1. Draft Heatmap
```bash
# Visualize positional diversity
python -c "
from src.diagnostics import DraftVisualizer
viz = DraftVisualizer()
# Will be generated automatically during tournament
"
```

**What to Look For**:
- **Smart Agent**: Diverse spread across positions and rounds
- **Dumb Agent**: Over-concentration in one position

### 2. Confidence Over Time
```bash
# Check if model is becoming more confident
ls experiments/grpo/confidence.png
```

**What to Look For**:
- Increasing trend = model is learning
- Flat or decreasing = model is confused

### 3. Reasoning Traces

Look at the `<think>` tags during evaluation:

```bash
# Run a single draft with reasoning
python -c "
from src.models import DraftAgent
from src.data import ScenarioGenerator, PlayerLoader

agent = DraftAgent('experiments/grpo/checkpoint_final/model')
loader = PlayerLoader()
players = loader.load_players()

scenario = ScenarioGenerator.generate_scenario(
    round_num=3, pick_num=5,
    roster=None, available_players=players[:10],
    include_reasoning_prompt=True
)

print(scenario)
player, reasoning, _ = agent.pick_player(scenario, [p.name for p in players[:10]])
print(f'\nPick: {player}')
print(f'Reasoning: {reasoning}')
"
```

**What to Look For**:
- Does it mention positional needs?
- Does it talk about VOR or value?
- Does it consider tier drop-offs?

---

## Troubleshooting

### Out of Memory (OOM)
**Solution**: Reduce batch size or use gradient accumulation
```bash
python train_sft.py --batch-size 2
```

### Model Hallucinating (Picking Non-Existent Players)
**Solution**: More SFT training or stricter validation
```bash
python train_sft.py --num-examples 5000 --epochs 5
```

### GRPO Not Learning (Flat Rewards)
**Possible Issues**:
1. Reward function too sparse → Add more shaping
2. Learning rate too high → Reduce to 5e-6
3. Model too small → Try Phi-4-mini instead of SmolLM

### MLX Not Working
**Fallback to PyTorch**:
```bash
python train_sft.py --model microsoft/Phi-3.5-mini-instruct
# Will automatically use PyTorch + MPS
```

---

## Advanced: Tuning the Reward Function

Edit `src/environment/draft_env.py:_calculate_reward()`:

```python
def _calculate_reward(self, player: Player, roster: Roster) -> float:
    reward = 0.0

    # 1. Base VOR (adjust weight)
    reward += player.vor * 0.1  # Try 0.05 or 0.2

    # 2. Positional tier bonus
    if position_rank <= 5:
        reward += 1.0  # Try 2.0 for more emphasis

    # 3. Positional need (critical!)
    if roster.needs_position(player.position):
        reward += 0.5  # Try 1.0 or 2.0

    # 4. Draft value (beating ADP)
    if player.adp > overall_pick + 5:
        reward += 0.3  # Try 0.5

    return reward
```

**Experiment**:
- Higher VOR weight → More emphasis on projections
- Higher need weight → More emphasis on roster balance
- Add penalties for duplicate positions

---

## Expected Results

### Baseline (Greedy Bot)
- Win Rate: ~10-12% (close to random 1/12)
- Strategy: Always pick highest projected points
- Weakness: Over-drafts WRs, ignores scarcity

### RL Agent (After Training)
- Win Rate: ~15-20% (significant improvement)
- Strategy: Balances VOR with positional need
- Strength: Recognizes tier drop-offs, avoids reaches

### "God-Tier" Performance
- Win Rate: >25%
- Strategy: Perfect VOR optimization + bye week management
- This would require:
  - More training episodes (500+)
  - Better reward shaping
  - Larger model (7B parameters)

---

## Next Steps After Week 4

### If Hypothesis Validated ✓
1. **Scale Up**: Run 10,000 league tournament
2. **Try Different Models**: GPT-2, SmolLM-1.7B, Phi-4-mini
3. **Add Features**: Bye week consideration, injury risk
4. **Publish Results**: Write a blog post or paper

### If Hypothesis Rejected ✗
1. **Diagnose**: Run all diagnostic tools
2. **Tune Rewards**: Adjust reward function weights
3. **More Training**: Increase SFT examples to 5,000
4. **Simplify Task**: Start with 8-round drafts instead of 15

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `test_env.py` | Test environment setup |
| `train_sft.py` | Week 2: Supervised fine-tuning |
| `train_grpo.py` | Week 3: RL training |
| `tournament.py` | Week 4: Large-scale evaluation |
| `compare_agents.py` | Final comparison |
| `src/environment/draft_env.py` | Core draft simulator |
| `src/data/scenario_generator.py` | Natural language conversion |
| `src/training/grpo_trainer.py` | RL training loop |
| `src/diagnostics/analyzer.py` | Diagnostic tools |

---

## Questions?

Check `README.md` for architecture details.

Good luck! 🏈📊🤖
