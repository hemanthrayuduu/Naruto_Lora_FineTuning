import os
import torch
from tqdm import tqdm
from bitsandbytes.optim import AdamW8bit

from ..utils.memory import free_memory
from ..utils.visualization import generate_sample_images
from ..models.lora import get_trainable_params_stats
from .scheduler import create_scheduler


class LoRATrainer:
    """Trainer for LoRA fine-tuning of Stable Diffusion."""
    
    def __init__(self, config, models, is_distributed=False):
        """
        Initialize the trainer.
        
        Args:
            config: Training configuration
            models: Dictionary containing model components
            is_distributed: Whether to use distributed training
        """
        self.config = config
        self.device = config.device
        self.is_distributed = is_distributed
        
        # Extract models
        self.tokenizer = models["tokenizer"]
        self.text_encoder = models["text_encoder"]
        self.vae = models["vae"]
        self.unet = models["unet"]
        self.noise_scheduler = models["noise_scheduler"]
        
        # Set distributed training info
        if is_distributed:
            self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            self.rank = int(os.environ.get("RANK", "0"))
            self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
            self.is_main_process = (self.rank == 0)
        else:
            self.is_main_process = True
        
        # Print trainable parameters info
        if self.is_main_process:
            param_stats = get_trainable_params_stats(self.unet)
            print(f"LoRA trainable parameters: {param_stats['trainable']:,} / "
                  f"{param_stats['total']:,} ({param_stats['percentage']:.2f}%)")
            
        # Wrap model for distributed training
        if self.is_distributed:
            self.unet = torch.nn.parallel.DistributedDataParallel(
                self.unet,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=True
            )
            
        # Create size schedule for progressive training
        if self.config.progressive_training:
            self.size_schedule = self._create_size_schedule()
        else:
            self.size_schedule = [self.config.image_sizes[0]] * self.config.num_epochs
    
    def _create_size_schedule(self):
        """Create a schedule of image sizes for progressive training."""
        image_sizes = self.config.image_sizes
        num_epochs = self.config.num_epochs
        
        # Distribute epochs across different sizes with emphasis on higher resolutions
        if len(image_sizes) == 4:  # Typical 4-step progression
            weights = [0.2, 0.3, 0.2, 0.3]  # 20%, 30%, 20%, 30%
        else:
            # Equal distribution for custom size lists
            weights = [1/len(image_sizes)] * len(image_sizes)
        
        # Calculate epochs per size
        size_schedule = []
        remaining_epochs = num_epochs
        
        for i, (size, weight) in enumerate(zip(image_sizes, weights)):
            if i < len(image_sizes) - 1:
                size_epochs = max(1, int(num_epochs * weight))
                size_schedule.extend([size] * size_epochs)
                remaining_epochs -= size_epochs
            else:
                # Last size gets all remaining epochs
                size_schedule.extend([size] * remaining_epochs)
        
        return size_schedule
    
    def _get_optimizer(self, parameters):
        """Create an optimizer for training."""
        return AdamW8bit(
            parameters,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
    
    def train_step(self, batch):
        """Perform a single training step."""
        # Move batch to device
        pixel_values = batch["pixel_values"].to(self.device)
        input_ids = batch["input_ids"].to(self.device)
        
        with torch.no_grad():
            # Get text embeddings
            text_embeddings = self.text_encoder(input_ids)[0]
            
            # Encode images to latent space
            latents = self.vae.encode(pixel_values).latent_dist.sample() * 0.18215
            
            # Add noise to latents
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, 
                self.noise_scheduler.config.num_train_timesteps, 
                (latents.shape[0],), 
                device=self.device
            )
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Predict noise with UNet
        if self.is_distributed:
            noise_pred = self.unet.module(noisy_latents, timesteps, text_embeddings).sample
        else:
            noise_pred = self.unet(noisy_latents, timesteps, text_embeddings).sample
        
        # Calculate loss (scaled by gradient accumulation steps)
        loss = torch.nn.functional.mse_loss(noise_pred, noise) / self.config.gradient_accumulation_steps
        
        return loss
    
    def save_checkpoint(self, path):
        """Save a checkpoint of the model."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        if self.is_distributed:
            unet_to_save = self.unet.module
        else:
            unet_to_save = self.unet
            
        unet_to_save.save_pretrained(path)
        print(f"Saved LoRA model to {path}")
    
    def generate_samples(self, step_name):
        """Generate sample images to monitor training progress."""
        # Set UNet to eval mode
        if self.is_distributed:
            unet_eval = self.unet.module
        else:
            unet_eval = self.unet
        
        unet_eval.eval()
        
        # Generate images
        output_dir = os.path.join(self.config.sample_dir, f"{step_name}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate samples
        generate_sample_images(
            unet_eval,
            self.vae,
            self.text_encoder,
            self.tokenizer,
            self.noise_scheduler,
            self.device,
            self.config.sample_prompts,
            output_dir,
            step_name
        )
        
        # Set UNet back to training mode
        unet_eval.train()
    
    def train(self, dataloader_fn):
        """Run the training loop."""
        from datasets import load_dataset
        
        # Load dataset
        dataset_raw = load_dataset(self.config.dataset_name)
        
        # Initialize tracking variables
        global_step = 0
        
        # Training loop through epochs
        for epoch in range(self.config.num_epochs):
            # Get current image size for this epoch
            current_image_size = self.size_schedule[epoch]
            
            # Log current epoch settings
            if self.is_main_process:
                print(f"\nEpoch {epoch+1}/{self.config.num_epochs} - Image size: {current_image_size}x{current_image_size}")
            
            # Get dataloader for current image size
            dataloader = dataloader_fn(dataset_raw, self.tokenizer, current_image_size)
            
            # Calculate steps per epoch
            num_update_steps_per_epoch = len(dataloader) // self.config.gradient_accumulation_steps
            if len(dataloader) % self.config.gradient_accumulation_steps != 0:
                num_update_steps_per_epoch += 1
            
            # Create optimizer
            if self.is_distributed:
                optimizer = self._get_optimizer(
                    filter(lambda p: p.requires_grad, self.unet.module.parameters())
                )
            else:
                optimizer = self._get_optimizer(
                    filter(lambda p: p.requires_grad, self.unet.parameters())
                )
            
            # Create learning rate scheduler
            total_training_steps = self.config.num_epochs * num_update_steps_per_epoch
            lr_scheduler = create_scheduler(
                optimizer=optimizer,
                num_warmup_steps=int(0.05 * total_training_steps),
                num_training_steps=total_training_steps
            )
            
            # Clear memory before training
            free_memory()
            
            # Setup progress tracking
            if self.is_main_process:
                progress_bar = tqdm(
                    total=num_update_steps_per_epoch, 
                    desc=f"Epoch {epoch+1}/{self.config.num_epochs}"
                )
            
            # Reset optimizer
            optimizer.zero_grad()
            
            # Training tracking
            epoch_loss = 0.0
            steps_this_epoch = 0
            
            # Set distributed sampler epoch if using distributed training
            if self.is_distributed and hasattr(dataloader, 'sampler') and hasattr(dataloader.sampler, 'set_epoch'):
                dataloader.sampler.set_epoch(epoch)
            
            # Iterate through batches
            for step, batch in enumerate(dataloader):
                # Free memory periodically
                if step % 20 == 0:
                    free_memory()
                
                # Forward and backward pass
                loss = self.train_step(batch)
                loss.backward()
                
                # Track loss
                epoch_loss += loss.item() * self.config.gradient_accumulation_steps
                steps_this_epoch += 1
                
                # Update weights after accumulation steps
                if (step + 1) % self.config.gradient_accumulation_steps == 0 or step == len(dataloader) - 1:
                    # Clip gradients
                    if self.is_distributed:
                        torch.nn.utils.clip_grad_norm_(self.unet.module.parameters(), max_norm=1.0)
                    else:
                        torch.nn.utils.clip_grad_norm_(self.unet.parameters(), max_norm=1.0)
                    
                    # Update weights
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    
                    # Synchronize (distributed training)
                    if self.is_distributed:
                        torch.distributed.barrier()
                    
                    # Update progress
                    if self.is_main_process:
                        progress_bar.update(1)
                        progress_bar.set_postfix(loss=loss.detach().item() * self.config.gradient_accumulation_steps)
                    
                    # Update global step
                    global_step += 1
                    
                    # Generate samples periodically
                    if self.is_main_process and global_step % self.config.sampling_steps == 0:
                        checkpoint_path = os.path.join(self.config.checkpoint_dir, f"step_{global_step}")
                        self.save_checkpoint(checkpoint_path)
                        self.generate_samples(f"step_{global_step}")
            
            # End of epoch processing
            if steps_this_epoch > 0:
                avg_epoch_loss = epoch_loss / steps_this_epoch
                if self.is_main_process:
                    print(f"Epoch {epoch+1} completed with average loss: {avg_epoch_loss:.6f}")
            
            # Save epoch checkpoint
            if self.is_main_process:
                epoch_path = os.path.join(self.config.epoch_dir, f"epoch_{epoch+1}")
                self.save_checkpoint(epoch_path)
                self.generate_samples(f"epoch_{epoch+1}")
        
        # Save final model
        if self.is_main_process:
            final_path = os.path.join(self.config.output_dir, "final_model")
            self.save_checkpoint(final_path)
            
            # Generate final examples with more diverse prompts
            extended_prompts = [
                "Naruto Uzumaki using Rasengan, detailed anime style",
                "Sasuke Uchiha with Sharingan, high quality anime art",
                "Kakashi Hatake using Chidori, vibrant anime style",
                "Hinata Hyuga with Byakugan activated, detailed anime art",
                "Gaara controlling sand, official Naruto art style"
            ]
            
            # Store original prompts
            original_prompts = self.config.sample_prompts
            
            # Use extended prompts for final samples
            self.config.sample_prompts = extended_prompts
            self.generate_samples("final")
            
            # Restore original prompts
            self.config.sample_prompts = original_prompts
        
        # Clean up distributed environment
        if self.is_distributed:
            torch.distributed.destroy_process_group()
        
        print("Training completed successfully!")