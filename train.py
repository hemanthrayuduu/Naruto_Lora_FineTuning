import os
import torch
import random
import numpy as np
from datasets import load_dataset

from config.training_config import TrainingConfig
from models.lora import load_models, add_lora_to_unet
from data.dataset import AnimeDataset, create_dataloader
from data.transforms import get_transforms
from training.trainer import LoRATrainer
from utils.memory import free_memory


def setup_environment():
    """Set up the training environment."""
    # Set environment variables for memory management
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def set_seeds(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_data_loader_factory(config):
    """Create a factory function for data loaders."""
    def data_loader_factory(dataset_raw, tokenizer, image_size):
        # Create transform for current size
        transform = get_transforms(image_size)
        
        # Create dataset
        dataset = AnimeDataset(dataset_raw["train"], transform=transform)
        
        # Check for distributed training
        is_distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
        
        # Create dataloader
        dataloader = create_dataloader(
            dataset, 
            tokenizer, 
            batch_size=config.batch_size,
            is_distributed=is_distributed
        )
        
        return dataloader
    
    return data_loader_factory


def main():
    """Main training function."""
    # Setup
    setup_environment()
    
    # Create configuration
    config = TrainingConfig()
    
    # Set random seeds
    set_seeds(config.seed)
    
    # Check for distributed training
    is_distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if is_distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")
        is_main_process = (int(os.environ.get("RANK", "0")) == 0)
    else:
        is_main_process = True
    
    # Print training details
    if is_main_process:
        print(f"Starting LoRA fine-tuning on {config.device}")
        print(f"Dataset: {config.dataset_name}")
        print(f"LoRA rank: {config.lora_rank}, alpha: {config.lora_alpha}")
        print(f"Batch size: {config.batch_size}, Gradient accumulation: {config.gradient_accumulation_steps}")
        if config.progressive_training:
            print(f"Progressive training with sizes: {config.image_sizes}")
        print(f"Output directory: {config.output_dir}")
    
    # Clear memory before loading models
    free_memory()
    
    # Load models
    if is_main_process:
        print("Loading models...")
    
    tokenizer, text_encoder, vae, unet, noise_scheduler = load_models(
        config.base_model_path, 
        config.device
    )
    
    # Add LoRA to UNet
    if is_main_process:
        print("Setting up LoRA adapters...")
        
    unet = add_lora_to_unet(
        unet,
        rank=config.lora_rank,
        alpha=config.lora_alpha,
        dropout=config.lora_dropout,
        target_modules=config.lora_target_modules
    )
    
    # Create model dictionary
    models = {
        "tokenizer": tokenizer,
        "text_encoder": text_encoder,
        "vae": vae,
        "unet": unet,
        "noise_scheduler": noise_scheduler
    }
    
    # Create data loader factory
    data_loader_factory = create_data_loader_factory(config)
    
    # Create trainer
    trainer = LoRATrainer(config, models, is_distributed=is_distributed)
    
    # Train the model
    trainer.train(data_loader_factory)


if __name__ == "__main__":
    main()