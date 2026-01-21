# Makefile for RLFF project
# Convenience commands for training and evaluation

.PHONY: help install test sft grpo tournament compare clean

help:
	@echo "RLFF: Reinforcement Learning Fantasy Football"
	@echo ""
	@echo "Available commands:"
	@echo "  make install     - Install dependencies"
	@echo "  make test        - Test environment setup"
	@echo "  make sft         - Run supervised fine-tuning (Week 2)"
	@echo "  make grpo        - Run GRPO training (Week 3)"
	@echo "  make tournament  - Run tournament evaluation (Week 4)"
	@echo "  make compare     - Compare RL agent vs baseline"
	@echo "  make full        - Run complete pipeline (SFT -> GRPO -> Tournament)"
	@echo "  make clean       - Clean generated files"

install:
	@echo "Installing dependencies..."
	pip install --upgrade pip
	pip install -r requirements.txt
	@echo "✓ Installation complete"

test:
	@echo "Testing environment..."
	python test_env.py

sft:
	@echo "Running SFT training..."
	python train_sft.py \
		--model microsoft/Phi-3.5-mini-instruct \
		--num-examples 2000 \
		--epochs 3 \
		--batch-size 4 \
		--use-lora

grpo:
	@echo "Running GRPO training..."
	python train_grpo.py \
		--checkpoint experiments/sft/final \
		--episodes 100 \
		--candidates 8 \
		--lr 1e-5

tournament:
	@echo "Running tournament..."
	python tournament.py \
		--checkpoint experiments/grpo/checkpoint_final/model \
		--leagues 1000

compare:
	@echo "Comparing agents..."
	python compare_agents.py \
		--rl-checkpoint experiments/grpo/checkpoint_final/model \
		--leagues 1000

full: test sft grpo tournament compare
	@echo "✓ Full pipeline complete!"

clean:
	@echo "Cleaning generated files..."
	rm -rf experiments/*/
	rm -rf data/raw/*.json
	rm -rf __pycache__ src/**/__pycache__
	@echo "✓ Clean complete"
