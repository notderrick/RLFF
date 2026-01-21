"""
DraftAgent: Wrapper for SLM (Phi-4-mini or SmolLM) optimized for Apple Silicon
"""

import torch
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
import numpy as np

try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load, generate
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    print("Warning: MLX not available, falling back to PyTorch")

from transformers import AutoTokenizer, AutoModelForCausalLM


@dataclass
class GenerationConfig:
    """Configuration for LLM generation"""
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1


class DraftAgent:
    """
    Wrapper for Small Language Model for fantasy football drafting.

    Supports:
    - MLX (Apple Silicon optimized)
    - PyTorch (fallback)
    - LoRA fine-tuning
    - Token log probability tracking
    """

    def __init__(
        self,
        model_name: str = "microsoft/Phi-3.5-mini-instruct",
        use_mlx: bool = True,
        device: str = "auto"
    ):
        """
        Initialize the draft agent.

        Args:
            model_name: HuggingFace model identifier
            use_mlx: Use MLX if available (for Apple Silicon)
            device: Device to use ('auto', 'mps', 'cuda', 'cpu')
        """
        self.model_name = model_name
        self.use_mlx = use_mlx and MLX_AVAILABLE

        if device == "auto":
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        print(f"Initializing DraftAgent with {model_name}")
        print(f"Using device: {self.device}")
        print(f"MLX available: {MLX_AVAILABLE}, Using MLX: {self.use_mlx}")

        # Load model and tokenizer
        self._load_model()

    def _load_model(self):
        """Load model and tokenizer"""
        if self.use_mlx:
            self._load_mlx()
        else:
            self._load_pytorch()

    def _load_mlx(self):
        """Load model with MLX (Apple Silicon optimized)"""
        print("Loading model with MLX...")
        try:
            self.model, self.tokenizer = load(self.model_name)
            print("✓ MLX model loaded successfully")
        except Exception as e:
            print(f"✗ Failed to load with MLX: {e}")
            print("Falling back to PyTorch...")
            self.use_mlx = False
            self._load_pytorch()

    def _load_pytorch(self):
        """Load model with PyTorch"""
        print("Loading model with PyTorch...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            device_map=self.device
        )
        print("✓ PyTorch model loaded successfully")

    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        return_logprobs: bool = False
    ) -> Tuple[str, Optional[List[float]]]:
        """
        Generate response from the model.

        Args:
            prompt: Input prompt
            config: Generation configuration
            return_logprobs: Whether to return token log probabilities

        Returns:
            Tuple of (generated_text, log_probs)
        """
        if config is None:
            config = GenerationConfig()

        if self.use_mlx:
            return self._generate_mlx(prompt, config, return_logprobs)
        else:
            return self._generate_pytorch(prompt, config, return_logprobs)

    def _generate_mlx(
        self,
        prompt: str,
        config: GenerationConfig,
        return_logprobs: bool
    ) -> Tuple[str, Optional[List[float]]]:
        """Generate with MLX"""
        response = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=config.max_tokens,
            temp=config.temperature,
            top_p=config.top_p,
        )

        # MLX doesn't easily expose log probs, so we return None
        return response, None

    def _generate_pytorch(
        self,
        prompt: str,
        config: GenerationConfig,
        return_logprobs: bool
    ) -> Tuple[str, Optional[List[float]]]:
        """Generate with PyTorch"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Generate with log probs if requested
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                repetition_penalty=config.repetition_penalty,
                do_sample=True,
                output_scores=return_logprobs,
                return_dict_in_generate=return_logprobs
            )

        if return_logprobs:
            # Extract log probabilities
            scores = outputs.scores  # List of tensors (vocab_size,)
            logprobs = []
            for score in scores:
                probs = torch.softmax(score, dim=-1)
                max_prob = probs.max().item()
                logprobs.append(np.log(max_prob))

            generated_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
            response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            return response, logprobs
        else:
            response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            return response, None

    def pick_player(
        self,
        scenario: str,
        available_players: List[str],
        config: Optional[GenerationConfig] = None
    ) -> Tuple[str, Optional[str], Optional[List[float]]]:
        """
        Pick a player given a draft scenario.

        Args:
            scenario: Natural language draft scenario
            available_players: List of available player names
            config: Generation configuration

        Returns:
            Tuple of (player_name, reasoning, log_probs)
        """
        response, logprobs = self.generate(scenario, config, return_logprobs=True)

        # Parse response
        from ..data.scenario_generator import ScenarioGenerator
        player_name = ScenarioGenerator.parse_llm_response(response)
        reasoning = ScenarioGenerator.extract_reasoning(response)

        return player_name, reasoning, logprobs

    def calculate_confidence(self, logprobs: List[float]) -> float:
        """
        Calculate confidence score from log probabilities.

        Returns:
            Confidence score between 0 and 1
        """
        if not logprobs:
            return 0.0

        # Average log prob, converted to probability
        avg_logprob = np.mean(logprobs)
        confidence = np.exp(avg_logprob)

        return confidence

    def save(self, output_dir: str):
        """Save model checkpoint"""
        if self.use_mlx:
            # TODO: Implement MLX model saving
            print("Warning: MLX model saving not yet implemented")
        else:
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            print(f"✓ Model saved to {output_dir}")

    def load(self, checkpoint_dir: str):
        """Load model checkpoint"""
        self.model_name = checkpoint_dir
        self._load_model()
        print(f"✓ Model loaded from {checkpoint_dir}")
