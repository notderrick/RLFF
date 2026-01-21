"""
SFTTrainer: Supervised Fine-Tuning for fantasy football drafting
Uses LoRA for efficient fine-tuning on Apple Silicon
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional
from tqdm import tqdm
import json
from pathlib import Path

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset as HFDataset


class DraftDataset(Dataset):
    """Dataset for draft training examples"""

    def __init__(self, examples: List[Dict[str, str]], tokenizer, max_length: int = 1024):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        # Combine prompt and completion
        text = f"{example['prompt']}\n\n{example['completion']}"

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze()
        }


class SFTTrainer:
    """
    Supervised Fine-Tuning trainer with LoRA.

    This teaches the model the "grammar" of drafting so it doesn't
    try to draft players who aren't available.
    """

    def __init__(
        self,
        model_name: str = "microsoft/Phi-3.5-mini-instruct",
        output_dir: str = "experiments/sft",
        use_lora: bool = True,
        device: str = "auto"
    ):
        """
        Initialize SFT trainer.

        Args:
            model_name: Base model to fine-tune
            output_dir: Directory to save checkpoints
            use_lora: Whether to use LoRA
            device: Device to use
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_lora = use_lora

        if device == "auto":
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        print(f"Initializing SFTTrainer")
        print(f"Base model: {model_name}")
        print(f"Device: {self.device}")
        print(f"Using LoRA: {use_lora}")

        # Load tokenizer and model
        self._load_model()

    def _load_model(self):
        """Load model and tokenizer"""
        print("Loading tokenizer and model...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            device_map=self.device
        )

        # Apply LoRA if requested
        if self.use_lora:
            print("Applying LoRA configuration...")
            lora_config = LoraConfig(
                r=16,  # Rank
                lora_alpha=32,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Common attention modules
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )

            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

        print("✓ Model loaded successfully")

    def train(
        self,
        training_examples: List[Dict[str, str]],
        eval_examples: Optional[List[Dict[str, str]]] = None,
        epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-4,
        save_steps: int = 100,
        eval_steps: int = 100
    ):
        """
        Train the model with supervised fine-tuning.

        Args:
            training_examples: List of {'prompt': ..., 'completion': ...}
            eval_examples: Optional evaluation set
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            save_steps: Steps between checkpoints
            eval_steps: Steps between evaluations
        """
        print(f"\nStarting SFT training")
        print(f"Training examples: {len(training_examples)}")
        if eval_examples:
            print(f"Evaluation examples: {len(eval_examples)}")
        print(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {learning_rate}")

        # Create datasets
        train_dataset = DraftDataset(training_examples, self.tokenizer)

        eval_dataset = None
        if eval_examples:
            eval_dataset = DraftDataset(eval_examples, self.tokenizer)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=50,
            logging_steps=10,
            save_steps=save_steps,
            eval_steps=eval_steps if eval_dataset else None,
            evaluation_strategy="steps" if eval_dataset else "no",
            save_total_limit=3,
            fp16=self.device != "cpu",
            gradient_accumulation_steps=2,
            report_to=["none"],  # Disable wandb for now
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )

        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

        # Train
        print("\nTraining...")
        trainer.train()

        # Save final model
        final_dir = self.output_dir / "final"
        self.model.save_pretrained(final_dir)
        self.tokenizer.save_pretrained(final_dir)

        print(f"\n✓ Training complete. Model saved to {final_dir}")

    def generate_training_data(
        self,
        num_examples: int = 2000,
        output_file: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Generate synthetic training data from simulated drafts.

        Args:
            num_examples: Number of training examples to generate
            output_file: Optional file to save examples

        Returns:
            List of training examples
        """
        print(f"\nGenerating {num_examples} training examples...")

        from ..environment import DraftEnv
        from ..data import PlayerLoader, ScenarioGenerator

        # Load players
        loader = PlayerLoader()
        players = loader.load_players(num_players=300)

        examples = []

        # Run multiple drafts
        num_drafts = num_examples // 15  # ~15 picks per draft
        print(f"Running {num_drafts} simulated drafts...")

        for draft_idx in tqdm(range(num_drafts)):
            # Create environment
            env = DraftEnv(
                player_pool=players.copy(),
                num_teams=12,
                rounds=15,
                agent_draft_position=(draft_idx % 12) + 1,
                greedy_opponents=True
            )

            obs, info = env.reset()

            # Simulate draft
            while not info['draft_complete']:
                # Get agent roster
                agent_roster = env.rosters[env.agent_draft_position - 1]

                # Get available players
                available = env.available_players[:20]  # Top 20

                # Greedy pick: highest VOR
                best_player = max(available, key=lambda p: p.vor)

                # Generate training example
                example = ScenarioGenerator.generate_training_example(
                    round_num=info['round'],
                    pick_num=info['pick'],
                    roster=agent_roster,
                    available_players=available,
                    correct_pick=best_player
                )

                examples.append(example)

                # Take the action
                action = env.available_players.index(best_player)
                obs, reward, terminated, truncated, info = env.step(action)

                if terminated or truncated:
                    break

                if len(examples) >= num_examples:
                    break

            if len(examples) >= num_examples:
                break

        examples = examples[:num_examples]

        # Save if requested
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w') as f:
                json.dump(examples, f, indent=2)

            print(f"✓ Saved {len(examples)} examples to {output_file}")

        print(f"✓ Generated {len(examples)} training examples")
        return examples

    def load_training_data(self, file_path: str) -> List[Dict[str, str]]:
        """Load training data from JSON file"""
        with open(file_path, 'r') as f:
            examples = json.load(f)

        print(f"✓ Loaded {len(examples)} examples from {file_path}")
        return examples
