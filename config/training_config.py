from dataclasses import dataclass, field
from typing import List, Optional
import torch
import os

@dataclass
class TrainingConfig:
    """Configuration for LoRA fine-tuning of Stable Diffusion."""
    
    # Model configuration
    base_model_path: str = "CompVis/stable-diffusion-v1-4"
    
    # LoRA configuration
    lora_rank: int = 16
    lora_alpha: int = 32  # Typically 2x the rank
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "to_q", "to_k", "to_v", "to_out.0",  # Attention modules
        "conv1", "conv2"  # Convolutional layers
    ])
    
    # Training configuration
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    num_epochs: int = 10
    
    # Progressive training
    progressive_training: bool = True
    image_sizes: List[int] = field(default_factory=lambda: [128, 256, 384, 512])
    
    # Dataset configuration
    dataset_name: str = "lambdalabs/naruto-blip-captions"
    
    # Output configuration
    output_dir: str = "./lora_output"
    
    # Sampling configuration
    sample_prompts: List[str] = field(default_factory=lambda: [
        "Naruto Uzumaki using Rasengan, high quality anime",
        "Sasuke Uchiha with Sharingan, detailed anime style"
    ])
    sampling_steps: int = 200
    
    # Runtime configuration
    seed: int = 42
    device: torch.device = None
    
    def __post_init__(self):
        """Initialize dependent parameters after main initialization."""
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create output directories
        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        self.sample_dir = os.path.join(self.output_dir, "samples")
        self.epoch_dir = os.path.join(self.output_dir, "epochs")
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.sample_dir, exist_ok=True)
        os.makedirs(self.epoch_dir, exist_ok=True)