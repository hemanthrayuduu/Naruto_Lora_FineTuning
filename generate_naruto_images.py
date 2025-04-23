# generate_naruto_images.py
import os
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import argparse


def load_trained_model(base_model_path, lora_model_path, device):
    """Load a previously trained LoRA model for image generation."""
    # Print loading status
    print(f"Loading LoRA weights from: {lora_model_path}")
    
    # Load base model
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_path, 
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False
    )
    
    # Load LoRA weights
    from peft import PeftModel
    pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_model_path)
    
    # Merge weights for faster inference
    print("Merging LoRA weights with base model...")
    pipe.unet = pipe.unet.merge_and_unload()
    
    # Use faster scheduler
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    
    # Move to GPU
    pipe = pipe.to(device)
    
    return pipe


def generate_images(
    pipe,
    prompt,
    output_dir="generated_images",
    negative_prompt="lowres, bad anatomy, bad hands, cropped, worst quality, low quality, blurry",
    num_images=1,
    height=512,
    width=512,
    guidance_scale=7.5,
    num_steps=50,
    seed=None
):
    """Generate images from a user prompt using the trained LoRA model."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set seed for reproducibility if provided
    if seed is not None:
        torch.manual_seed(seed)
        print(f"Using seed: {seed}")
    
    print(f"Generating {num_images} image(s) for prompt: \"{prompt}\"")
    print(f"Image size: {width}x{height}, Steps: {num_steps}, Guidance scale: {guidance_scale}")
    
    # Generate images
    for i in range(num_images):
        # Generate image
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale
        ).images[0]
        
        # Save image
        filename = f"{output_dir}/image_{i+1}.png"
        image.save(filename)
        print(f"Saved image to {filename}")
    
    print(f"Successfully generated {num_images} image(s)!")
    return True


def interactive_mode():
    """Run in interactive mode where user can generate multiple images."""
    # Configure paths
    base_model_path = "CompVis/stable-diffusion-v1-4"
    lora_model_path = input("Enter path to trained LoRA model: ")
    output_dir = input("Enter output directory (default: 'generated_images'): ") or "generated_images"
    
    # Check if model path exists
    if not os.path.exists(lora_model_path):
        print(f"Error: Model path '{lora_model_path}' does not exist!")
        return False
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    try:
        pipe = load_trained_model(base_model_path, lora_model_path, device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return False
    
    # Interactive loop
    while True:
        # Get prompt
        prompt = input("\nEnter your prompt (or 'quit' to exit): ")
        if prompt.lower() in ['quit', 'exit', 'q']:
            break
        
        # Get generation parameters
        try:
            num_images = int(input("Number of images to generate (default: 1): ") or 1)
            width = int(input("Image width (default: 512): ") or 512)
            height = int(input("Image height (default: 512): ") or 512)
            steps = int(input("Number of inference steps (default: 50): ") or 50)
            guidance = float(input("Guidance scale (default: 7.5): ") or 7.5)
            seed_input = input("Random seed (default: random): ")
            seed = int(seed_input) if seed_input else None
            
            negative_prompt = input("Negative prompt (press Enter for default): ")
            if not negative_prompt:
                negative_prompt = "lowres, bad anatomy, bad hands, cropped, worst quality, low quality, blurry"
        except ValueError as e:
            print(f"Invalid input: {e}")
            continue
        
        # Generate images
        subfolder = os.path.join(output_dir, prompt.replace(" ", "_")[:30])
        generate_images(
            pipe=pipe,
            prompt=prompt,
            output_dir=subfolder,
            negative_prompt=negative_prompt,
            num_images=num_images,
            height=height,
            width=width,
            guidance_scale=guidance,
            num_steps=steps,
            seed=seed
        )
    
    print("Thank you for using the Naruto image generator!")
    return True


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description="Generate anime-style images using LoRA-finetuned Stable Diffusion")
    
    # Add arguments
    parser.add_argument("--model", type=str, help="Path to trained LoRA model")
    parser.add_argument("--prompt", type=str, help="Text prompt for image generation")
    parser.add_argument("--output", type=str, default="generated_images", help="Output directory")
    parser.add_argument("--num", type=int, default=1, help="Number of images to generate")
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument("--height", type=int, default=512, help="Image height")
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--guidance", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    
    # Parse arguments
    args = parser.parse_args()
    
    # If interactive mode requested, run that instead
    if args.interactive:
        return interactive_mode()
    
    # Check required arguments
    if not args.model:
        print("Error: Missing required argument --model")
        return False
    
    if not args.prompt:
        print("Error: Missing required argument --prompt")
        return False
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    base_model_path = "CompVis/stable-diffusion-v1-4"
    try:
        pipe = load_trained_model(base_model_path, args.model, device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return False
    
    # Generate images
    return generate_images(
        pipe=pipe,
        prompt=args.prompt,
        output_dir=args.output,
        num_images=args.num,
        height=args.height,
        width=args.width,
        guidance_scale=args.guidance,
        num_steps=args.steps,
        seed=args.seed
    )


if __name__ == "__main__":
    # Run the script
    success = main()
    if not success:
        print("\nImage generation failed. Please check the error messages above.")