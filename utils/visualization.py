import os
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

def generate_sample_images(
    unet, 
    vae, 
    text_encoder, 
    tokenizer, 
    scheduler, 
    device, 
    prompts, 
    output_dir, 
    prefix
):
    """Generate sample images during training for progress visualization."""
    # Create pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16
    )
    
    # Replace components with our trained ones
    pipe.unet = unet
    pipe.vae = vae
    pipe.text_encoder = text_encoder
    pipe.tokenizer = tokenizer
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Negative prompt
    negative_prompt = "lowres, bad anatomy, bad hands, cropped, worst quality, low quality, blurry"
    
    # Generate images
    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            try:
                # Generate image
                image = pipe(
                    prompt, 
                    negative_prompt=negative_prompt,
                    num_inference_steps=50,
                    guidance_scale=7.5
                ).images[0]
                
                # Save image
                save_path = os.path.join(output_dir, f"{prefix}_{i}.png")
                image.save(save_path)
                print(f"Generated sample image: {save_path}")
            except Exception as e:
                print(f"Error generating image for prompt '{prompt}': {e}")