"""
train_sft.py: Supervised Fine-Tuning script

This is Week 2 of the roadmap: Teach the model basic drafting grammar.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import argparse

from src.training import SFTTrainer


def main():
    parser = argparse.ArgumentParser(description="SFT training for fantasy football drafting")
    parser.add_argument('--model', type=str, default='microsoft/Phi-3.5-mini-instruct',
                        help='Base model to fine-tune')
    parser.add_argument('--output', type=str, default='experiments/sft',
                        help='Output directory')
    parser.add_argument('--num-examples', type=int, default=2000,
                        help='Number of training examples to generate')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='Learning rate')
    parser.add_argument('--use-lora', action='store_true', default=True,
                        help='Use LoRA for efficient fine-tuning')
    parser.add_argument('--load-data', type=str, default=None,
                        help='Load training data from file instead of generating')

    args = parser.parse_args()

    print("\n" + "="*60)
    print("SUPERVISED FINE-TUNING (SFT)")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Output: {args.output}")
    print(f"Training examples: {args.num_examples}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Use LoRA: {args.use_lora}")
    print("="*60 + "\n")

    # Initialize trainer
    trainer = SFTTrainer(
        model_name=args.model,
        output_dir=args.output,
        use_lora=args.use_lora
    )

    # Load or generate training data
    if args.load_data:
        print(f"Loading training data from {args.load_data}...")
        training_examples = trainer.load_training_data(args.load_data)
    else:
        print("Generating training data...")
        training_examples = trainer.generate_training_data(
            num_examples=args.num_examples,
            output_file=f"{args.output}/training_data.json"
        )

    # Split into train/eval
    split_idx = int(len(training_examples) * 0.9)
    train_data = training_examples[:split_idx]
    eval_data = training_examples[split_idx:]

    print(f"\nTrain set: {len(train_data)} examples")
    print(f"Eval set: {len(eval_data)} examples")

    # Train
    trainer.train(
        training_examples=train_data,
        eval_examples=eval_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )

    print("\n✓ SFT training complete!")
    print(f"\nNext steps:")
    print(f"  1. Run GRPO training: python train_grpo.py --checkpoint {args.output}/final")
    print(f"  2. Evaluate: python tournament.py --checkpoint {args.output}/final")


if __name__ == "__main__":
    main()
