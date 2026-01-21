# RLFF: Reinforcement Learning Fantasy Football Draft Agent

## Project Overview
An RL-powered fantasy football draft agent that learns strategic drafting through reinforcement learning, optimized for Apple Silicon (M1 Max).

## Hypothesis
An SLM trained via RL on draft scenarios will prioritize long-term roster balance (positional scarcity, value-over-replacement) over greedy "highest-projected-player" strategies.

## Architecture

### Phase A: Data Engine
- **Tools**: nfl_data_py + Sleeper API
- **Output**: Natural language draft scenarios

### Phase B: Supervised Fine-Tuning (SFT)
- **Model**: Phi-4-mini or SmolLM-3B
- **Method**: LoRA fine-tuning on ~2,000 high-win draft picks
- **Purpose**: Teach drafting grammar

### Phase C: Reinforcement Learning (GRPO)
- **Framework**: MLX (Apple Silicon optimized)
- **Strategy**: Group Relative Policy Optimization
- **Reward**: VOR-based + positional tier completion

## Project Structure
```
RLFF/
├── data/               # Raw and processed data
├── src/
│   ├── environment/    # DraftEnv (Gymnasium)
│   ├── data/          # Data pipelines
│   ├── models/        # SLM wrappers
│   ├── training/      # SFT and RL training loops
│   └── diagnostics/   # Evaluation and visualization
├── configs/           # Hyperparameters
└── experiments/       # Training runs and logs
```

## Week-by-Week Roadmap
- **Week 1**: Data & Environment (DraftEnv with random picks)
- **Week 2**: SFT (Fine-tune 3B model)
- **Week 3**: Reward Loop (VOR-based GRPO for 100 episodes)
- **Week 4**: Tournament (1,000 simulated leagues vs Greedy Bots)

## Diagnostics
- Token log probabilities (confidence tracking)
- Reasoning traces (strategic thinking)
- Draft heatmaps (positional diversity)
- Stress tests (suboptimal scenarios)
