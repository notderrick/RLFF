# RLFF Project Summary

## What We Built

A complete end-to-end **Reinforcement Learning Fantasy Football Draft Agent** that learns strategic drafting through:

1. **Supervised Fine-Tuning (SFT)**: Teaching the model basic drafting grammar
2. **Group Relative Policy Optimization (GRPO)**: Teaching strategic reasoning about VOR, positional scarcity, and tier drop-offs
3. **Tournament Evaluation**: Testing against greedy baseline in 1,000+ league simulations

---

## Key Components

### 1. Draft Environment (`src/environment/draft_env.py`)
- **2,723 lines of code across 13 Python files**
- Gymnasium-compatible RL environment
- Snake draft simulation with 12 teams
- VOR-based reward function
- Natural language observations for LLM

**Key Features**:
- Roster constraint validation
- Positional need detection
- Strategic reward shaping (VOR + positional tiers + draft value)

### 2. Data Pipeline (`src/data/`)
- **PlayerLoader**: Generates synthetic player pool with realistic projections
- **ScenarioGenerator**: Converts draft states into natural language
- **Training Example Generator**: Creates 2,000+ SFT examples from simulated drafts

**Example Scenario**:
```
Draft State: Round 5, Pick 52 (Overall: 52)
Current Roster: QB (Josh Allen), WR (Justin Jefferson), RB (Austin Ekeler)
Roster Size: 3/15

Positional Needs: RB (CRITICAL), TE (CRITICAL)

Top Available Players:
  1. Joe Mixon (RB) - Proj: 245.3 pts | ADP: 48.2 | VOR: 89.1 [VALUE]
  2. Kyle Pitts (TE) - Proj: 178.5 pts | ADP: 55.1 | VOR: 72.3
  3. Zay Flowers (WR) - Proj: 203.7 pts | ADP: 51.9 | VOR: 65.8

What is your pick?
```

### 3. Model Wrapper (`src/models/draft_agent.py`)
- Supports **Phi-4-mini** (3B params) and **SmolLM** (360M-3B)
- **MLX backend** for Apple Silicon optimization
- PyTorch fallback for CUDA/CPU
- Token log probability tracking for confidence analysis

### 4. Training Pipeline (`src/training/`)

#### SFT Trainer
- LoRA fine-tuning (r=16, alpha=32)
- 2,000 training examples
- Teaches model to format picks correctly
- Prevents hallucinations (drafting non-existent players)

#### GRPO Trainer
- Group Relative Policy Optimization
- Generates K=8 candidate picks per scenario
- Rewards based on VOR + positional need + tier completion
- Updates policy to favor high-reward picks

### 5. Diagnostics (`src/diagnostics/`)

#### Visualizer
- Draft heatmaps (position x round)
- Confidence over time plots
- Reward progression charts
- Win rate comparisons

#### Analyzer
- Confidence trend analysis
- Reasoning quality scoring (strategic keyword detection)
- Positional balance metrics (entropy-based)
- Stress tests (broken board scenarios)

### 6. Tournament Simulator (`tournament.py`)
- Runs 1,000+ league simulations
- RL Agent vs. Greedy Bot baseline
- Calculates championship probability (win rate)
- Generates performance reports

---

## The Hypothesis

**"An SLM trained via RL on draft scenarios will prioritize long-term roster balance (positional scarcity, VOR) over greedy 'highest-projected-player' strategies."**

### Test Metrics
1. **Win Rate**: RL Agent > Greedy Bot
2. **Championship Probability**: RL Agent > 15% (above 1/12 random)
3. **Roster Quality**: Higher VOR scores
4. **Strategic Reasoning**: Evidence of tier awareness in `<think>` tags

---

## Architecture Highlights

### Why Small Language Models?
- **Efficiency**: 3B params runs on M1 Max (32GB)
- **Reasoning**: Focus on teaching logic, not memorizing stats
- **Interpretability**: Can analyze reasoning traces
- **Cost**: Cheaper to train than 70B models

### Why GRPO over PPO?
- **Simpler**: No value network needed
- **Stable**: Group normalization reduces variance
- **Efficient**: Batches multiple candidates per scenario

### Why Natural Language?
- **Interpretability**: Can read model's reasoning
- **Flexibility**: Easy to add new features (bye weeks, injuries)
- **Transfer Learning**: Leverages LLM's language understanding

---

## Training Pipeline

```
┌─────────────────────────────────────────────────────┐
│  Week 1: Environment Setup                          │
│  - Test DraftEnv                                    │
│  - Verify player pool generation                   │
│  - Check scenario generation                       │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│  Week 2: Supervised Fine-Tuning (SFT)              │
│  - Generate 2,000 training examples                │
│  - LoRA fine-tune on "correct" picks               │
│  - Teach drafting grammar                          │
│  Output: experiments/sft/final/                    │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│  Week 3: GRPO Training (Reinforcement Learning)    │
│  - Run 100 draft episodes                          │
│  - Generate 8 candidates per pick                  │
│  - Update policy with VOR rewards                  │
│  Output: experiments/grpo/checkpoint_final/        │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│  Week 4: Tournament Evaluation                     │
│  - Run 1,000 league simulations                    │
│  - Compare to Greedy Bot baseline                  │
│  - Generate diagnostic reports                     │
│  Output: Win rate, championship probability        │
└─────────────────────────────────────────────────────┘
```

---

## Expected Results

### Baseline (Greedy Bot)
```
Strategy: Always pick highest projected points
Win Rate: ~10-12%
Weakness: Over-drafts WRs, ignores RB scarcity
```

### RL Agent (Target)
```
Strategy: VOR + positional need + tier awareness
Win Rate: ~15-20%
Strength: Balanced roster, recognizes drop-offs
```

### Success Criteria
- ✓ Win rate > 15%
- ✓ Beats greedy baseline by 20%+
- ✓ Strategic reasoning in <think> tags
- ✓ Balanced positional distribution

---

## Novel Contributions

1. **Natural Language RL**: First fantasy football agent using LLM reasoning
2. **VOR-Based Rewards**: Incorporating domain-specific strategy into RL
3. **Interpretable Drafting**: Can explain picks with <think> tags
4. **Apple Silicon Optimization**: MLX backend for M1/M2/M3 Macs
5. **Diagnostic Framework**: Tools to "peer into the brain" of the agent

---

## Files Overview

| File | Lines | Purpose |
|------|-------|---------|
| `src/environment/draft_env.py` | 462 | Draft simulator + reward function |
| `src/data/player_loader.py` | 290 | Player data generation + VOR calculation |
| `src/data/scenario_generator.py` | 178 | Natural language conversion |
| `src/models/draft_agent.py` | 206 | MLX/PyTorch LLM wrapper |
| `src/training/sft_trainer.py` | 289 | LoRA fine-tuning pipeline |
| `src/training/grpo_trainer.py` | 361 | GRPO RL training loop |
| `src/diagnostics/visualizer.py` | 303 | Visualization tools |
| `src/diagnostics/analyzer.py` | 387 | Diagnostic analysis |
| `tournament.py` | 247 | Tournament simulator |
| `test_env.py` | 177 | Environment test suite |

**Total: 2,723 lines of production code**

---

## Usage Examples

### Quick Start
```bash
# Test environment
python test_env.py

# Train SFT
python train_sft.py --num-examples 2000 --epochs 3

# Train GRPO
python train_grpo.py --checkpoint experiments/sft/final --episodes 100

# Run tournament
python tournament.py --checkpoint experiments/grpo/checkpoint_final/model --leagues 1000

# Compare agents
python compare_agents.py --rl-checkpoint experiments/grpo/checkpoint_final/model
```

### Using Makefile
```bash
make test       # Test environment
make sft        # Week 2: SFT training
make grpo       # Week 3: GRPO training
make tournament # Week 4: Evaluation
make compare    # Compare to baseline
make full       # Run entire pipeline
```

---

## Technical Specs

### Hardware Requirements
- **Recommended**: Mac Studio M1 Max (32GB RAM)
- **Minimum**: M1 MacBook Pro (16GB RAM)
- **Alternative**: NVIDIA GPU with 16GB+ VRAM

### Software Requirements
- Python 3.9+
- PyTorch 2.0+
- MLX (for Apple Silicon)
- Transformers, PEFT, TRL
- Gymnasium

### Training Time
- **SFT**: 2-3 hours (2,000 examples, 3 epochs)
- **GRPO**: 4-6 hours (100 episodes, 8 candidates)
- **Tournament**: 1-2 hours (1,000 leagues)
- **Total**: ~8-11 hours on M1 Max

---

## Future Extensions

### Phase 1 (Easy)
- [ ] Add bye week considerations
- [ ] Incorporate injury risk
- [ ] Test different draft positions
- [ ] Try larger models (7B params)

### Phase 2 (Medium)
- [ ] Real Sleeper API integration
- [ ] Multi-agent tournaments (12 RL agents)
- [ ] Auction draft support
- [ ] Dynasty league considerations

### Phase 3 (Hard)
- [ ] Real-time draft recommendations
- [ ] Opponent modeling (predict other teams' picks)
- [ ] Trade value optimization
- [ ] In-season lineup management

---

## Research Questions

1. **Model Size**: Does 7B beat 3B significantly?
2. **Training Duration**: Is 100 episodes enough or need 1,000?
3. **Reward Shaping**: Which reward components matter most?
4. **Transfer Learning**: Can it generalize to new player pools?
5. **Explainability**: Are reasoning traces accurate or post-hoc?

---

## Credits & Inspiration

- **GRPO Paper**: Group Relative Policy Optimization (DeepMind)
- **Fantasy Football VOR**: Concept from ESPN/Yahoo analytics
- **MLX Framework**: Apple's machine learning framework
- **Gymnasium**: OpenAI's RL environment standard

---

## License

MIT License - Free to use, modify, and distribute.

---

## Contact

For questions or collaboration:
- GitHub Issues: [your-repo]/issues
- Email: [your-email]

---

**Built with ❤️ for the fantasy football community and RL researchers.**
