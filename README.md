# 🏈 RLFF: Reinforcement Learning Fantasy Football

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![MLX](https://img.shields.io/badge/MLX-Apple%20Silicon-orange.svg)](https://github.com/ml-explore/mlx)

**An RL-powered fantasy football draft agent that learns strategic drafting through reinforcement learning.**

> 🎯 **Hypothesis**: An SLM trained via RL on draft scenarios will prioritize long-term roster balance (VOR, positional scarcity) over greedy "highest-projected-player" strategies.

## ✨ Key Features

- 🤖 **Natural Language RL**: LLM-based agent that reasons about draft decisions
- 📊 **VOR-Based Strategy**: Optimizes for Value Over Replacement, not just raw projections
- 🧠 **Interpretable Reasoning**: Inspect `<think>` tags to understand agent's logic
- ⚡ **Apple Silicon Optimized**: MLX backend for 2-3x speedup on M1/M2/M3
- 📈 **Comprehensive Diagnostics**: Heatmaps, confidence plots, reasoning analysis
- 🎮 **Tournament Testing**: 1,000+ league simulations for robust evaluation

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

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Test environment
python test_env.py

# 3. Train with SFT (Week 2)
python train_sft.py --num-examples 2000 --epochs 3

# 4. Train with GRPO (Week 3)
python train_grpo.py --checkpoint experiments/sft/final --episodes 100

# 5. Run tournament (Week 4)
python tournament.py --checkpoint experiments/grpo/checkpoint_final/model --leagues 1000

# Or use Makefile shortcuts
make test && make sft && make grpo && make tournament
```

## 📊 Expected Results

| Agent | Strategy | Win Rate | Notes |
|-------|----------|----------|-------|
| **Greedy Bot** | Highest projected points | ~10-12% | Over-drafts WRs, ignores scarcity |
| **RL Agent** | VOR + positional balance | **~15-20%** | Strategic tier awareness |

**Success Criteria**: ✓ Win rate > 15% | ✓ Beats baseline by 20%+ | ✓ Strategic reasoning

## 🔬 Diagnostics

- **Token Log Probs**: Confidence tracking over training
- **Reasoning Traces**: Strategic thinking in `<think>` tags
- **Draft Heatmaps**: Positional diversity visualization
- **Stress Tests**: Performance on suboptimal scenarios

## 📖 Documentation

- **[QUICKSTART.md](QUICKSTART.md)**: Step-by-step execution guide
- **[ARCHITECTURE.md](ARCHITECTURE.md)**: Deep technical dive
- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)**: Comprehensive overview

## 🛠️ Tech Stack

- **RL Environment**: Gymnasium
- **Models**: Phi-4-mini (3B) / SmolLM (360M-3B)
- **Training**: LoRA fine-tuning + GRPO
- **Backend**: MLX (Apple Silicon) / PyTorch
- **Visualization**: Matplotlib, Seaborn

## 📝 License

MIT License - Free to use, modify, and distribute.

## 🙏 Acknowledgments

- **GRPO**: Group Relative Policy Optimization (DeepMind)
- **VOR Concept**: ESPN/Yahoo fantasy analytics
- **MLX**: Apple's ML framework

---

**Built with ❤️ for fantasy football and RL research**
