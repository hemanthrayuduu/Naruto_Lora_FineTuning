# Naruto LoRA Fine-tuning

A project to fine-tune Stable Diffusion using Low-Rank Adaptation (LoRA) for generating Naruto-style anime images.

## 🚀 Project Overview

This repository contains a complete workflow for fine-tuning Stable Diffusion models using LoRA to generate high-quality Naruto-style anime images. The project includes training scripts, configuration files, data handling utilities, and an image generation script.


The final fine-tuned model is available in the `fine-tuned_model` directory.

## 🛠️ Features

- Fine-tune Stable Diffusion models using efficient LoRA techniques
- Progressive training with multiple image resolutions
- Interactive image generation mode 
- Configurable generation parameters
- Built-in data handling utilities
- Memory optimization for training

## 📋 Requirements

The following dependencies are required:

```
torch>=2.0.0
transformers>=4.30.0
diffusers>=0.21.0
accelerate>=0.21.0
peft>=0.5.0
bitsandbytes>=0.41.0
datasets>=2.12.0
Pillow>=9.5.0
tqdm>=4.65.0
numpy>=1.24.0
tensorboard>=2.13.0
```

For interactive notebooks (optional):
```
ipywidgets>=8.0.0
matplotlib>=3.7.0
```

## 🚀 Installation

1. Clone this repository:
```bash
git clone https://github.com/hemanthrayudu/naruto_lora_finetuning.git
cd naruto_lora_finetuning
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## 📦 Project Structure

```
├── config/                     # Configuration files
│   └── training_config.py      # Training configuration parameters
├── data/                       # Data handling utilities
│   ├── dataset.py              # Custom datasets and dataloaders
│   └── transforms.py           # Image transformations
├── fine-tuned_model/           # Directory containing the final fine-tuned model
├── models/                     # Model definitions
│   └── lora.py                 # LoRA implementation for Stable Diffusion
├── training/                   # Training utilities
│   └── trainer.py              # LoRA training implementation
├── utils/                      # Utility functions
│   ├── memory.py               # Memory optimization utilities
│   └── visualization.py        # Visualization utilities
├── finetuning.ipynb            # Interactive notebook for fine-tuning
├── generating-imgs.ipynb       # Interactive notebook for image generation
├── generate_naruto_images.py   # Script for generating images
├── requirements.txt            # Project dependencies
└── train.py                    # Main training script
```

## 🔧 Training

To train the model, run:

```bash
python train.py
```

The training configuration can be modified in `config/training_config.py`, including:
- Base model path
- LoRA parameters (rank, alpha, dropout)
- Training hyperparameters (batch size, learning rate, epochs) 
- Dataset configuration
- Progressive training settings

For a detailed walkthrough of the fine-tuning process, refer to this notebook:
[finetuning.ipynb](https://github.com/hemanthrayudu/naruto_lora_finetuning/blob/master/finetuning.ipynb)

## 🖼️ Image Generation

After training, you can generate images using the trained model:

### Interactive Mode

```bash
python generate_naruto_images.py --interactive
```

### Command Line Mode

```bash
python generate_naruto_images.py --model path/to/lora/model --prompt "Your prompt here"
```

Additional parameters:
- `--output`: Output directory
- `--num`: Number of images to generate
- `--width`: Image width
- `--height`: Image height
- `--steps`: Number of inference steps
- `--guidance`: Guidance scale
- `--seed`: Random seed for reproducibility

For a guided example of image generation with the fine-tuned model, refer to this notebook:
[generating-imgs.ipynb](https://github.com/hemanthrayudu/naruto_lora_finetuning/blob/master/generating-imgs.ipynb)

## 🔍 Web Application

A full stack web application has been deployed to showcase this fine-tuned model. The web app allows users to generate custom Naruto-style images by providing text prompts. The application features an intuitive interface for adjusting generation parameters and viewing the resulting images.

The full stack web application repository is available at: [Naruto-Image-Generator](https://github.com/hemanthrayuduu/Naruto-Image-Generator)

## 📝 License

This project is released under the MIT License.

## 🙏 Acknowledgements

- The project uses the [Diffusers](https://github.com/huggingface/diffusers) library by Hugging Face
- Training data provided by the LambdaLabs Naruto BLIP captions dataset
- Special thanks to the Stable Diffusion community for their valuable resources and insights 