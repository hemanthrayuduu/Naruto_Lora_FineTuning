import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
from typing import Dict, Tuple


def load_models(model_path, device):
    """Load all required models for Stable Diffusion."""
    # Load tokenizer
    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    
    # Load text encoder
    text_encoder = CLIPTextModel.from_pretrained(
        model_path, 
        subfolder="text_encoder"
    ).to(device)
    
    # Load VAE
    vae = AutoencoderKL.from_pretrained(
        model_path, 
        subfolder="vae"
    ).to(device)
    
    # Load UNet and enable gradient checkpointing to save memory
    unet = UNet2DConditionModel.from_pretrained(
        model_path, 
        subfolder="unet"
    ).to(device)
    unet.enable_gradient_checkpointing()
    
    # Load scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")
    
    # Freeze models that don't need to be trained
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    
    return tokenizer, text_encoder, vae, unet, noise_scheduler


def add_lora_to_unet(unet, rank=16, alpha=32, dropout=0.05, target_modules=None):
    """Add LoRA adapters to the UNet model."""
    if target_modules is None:
        target_modules = [
            "to_q", "to_k", "to_v", "to_out.0",  # Attention modules
            "conv1", "conv2"  # Convolutional layers
        ]
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none"
    )
    
    # Apply LoRA to UNet
    unet = get_peft_model(unet, lora_config)
    
    return unet


def get_trainable_params_stats(model):
    """Get statistics about trainable parameters."""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    percentage = 100 * trainable_params / all_params
    
    return {
        "trainable": trainable_params,
        "total": all_params,
        "percentage": percentage
    }


def load_lora_weights(base_model_path, lora_weights_path, device):
    """Load and merge LoRA weights with base model for inference."""
    # Load base UNet
    unet = UNet2DConditionModel.from_pretrained(
        base_model_path,
        subfolder="unet"
    )
    
    # Load and apply LoRA weights
    unet = PeftModel.from_pretrained(unet, lora_weights_path)
    
    # Merge weights for faster inference
    unet = unet.merge_and_unload()
    
    # Move to device
    unet = unet.to(device)
    
    return unet