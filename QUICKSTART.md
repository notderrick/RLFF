# 🚀 RLFF Quick Start Guide

## Installation Complete ✅

All dependencies installed, model downloaded, tests passing. You're ready to start training!

---

## Start the Dashboard (Recommended)

```bash
./start_dashboard.sh
```

Then open: **http://localhost:5001**

---

## Week 2: Run Your First Training

### Option 1: Via Dashboard (Easiest)
1. Start dashboard with `./start_dashboard.sh`
2. Open http://localhost:5001
3. Click **"START SFT"** button
4. Watch real-time logs in terminal window
5. Wait ~30-60 minutes for completion
6. Checkpoint saved to `experiments/sft/`

### Option 2: Via Command Line
```bash
source venv/bin/activate
python -m src.training.run_sft --config configs/default.yaml
```

---

## What Happens During SFT?

1. **Loads Model**: Phi-3.5-mini-instruct (~7.5GB)
2. **Generates Data**: 2,000 draft scenarios with expert picks
3. **Applies LoRA**: Adds trainable adapters (r=16, alpha=32)
4. **Trains**: 3 epochs, batch_size=4
5. **Saves Checkpoint**: `experiments/sft/checkpoint-final/` (~500MB)

**Expected Duration**: 30-60 minutes
**Memory Usage**: ~14-15GB RAM

---

## Configuration

All settings in `configs/default.yaml`:

```yaml
model:
  name: "microsoft/Phi-3.5-mini-instruct"
  use_mlx: true  # Apple Silicon optimization

sft:
  num_examples: 2000
  epochs: 3
  batch_size: 4
  learning_rate: 2e-4
  lora_r: 16
  lora_alpha: 32
```

**Quick Mode** (for testing):
- Change `num_examples: 2000` → `num_examples: 100`
- Change `epochs: 3` → `epochs: 1`
- Duration: ~5 minutes

---

## Commands Cheat Sheet

```bash
# Start dashboard
./start_dashboard.sh

# Activate virtual environment
source venv/bin/activate

# Run tests
python test_env.py

# Manual SFT training
python -m src.training.run_sft --config configs/default.yaml

# Kill dashboard
lsof -i :5001
kill $(lsof -t -i :5001)
```

---

**You're all set!** 🎉

Next step: Start the dashboard and click **"START SFT"**

```bash
./start_dashboard.sh
```

Then navigate to: http://localhost:5001
