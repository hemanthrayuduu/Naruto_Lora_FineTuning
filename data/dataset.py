import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from typing import Dict, Any, Callable, Optional


class AnimeDataset(Dataset):
    """Dataset for anime images with captions."""
    
    def __init__(self, dataset, transform: Optional[Callable] = None):
        self.dataset = dataset
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.dataset[idx]
        
        # Extract image and caption
        image = item["image"].convert("RGB")
        caption = item["text"]
        
        # Apply image transformations if specified
        if self.transform:
            image = self.transform(image)
        
        return {
            "pixel_values": image,
            "caption": caption
        }


def create_dataloader(
    dataset, 
    tokenizer, 
    batch_size=2, 
    is_distributed=False, 
    num_workers=0
):
    """Create a DataLoader for the anime dataset."""
    # Set up distributed sampler if needed
    if is_distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        shuffle = False
    else:
        sampler = None
        shuffle = True
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle,
        collate_fn=lambda examples: collate_fn(examples, tokenizer),
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader


def batch_prepare_samples(examples, tokenizer):
    """Prepare and batch samples together with tokenization."""
    # Stack images
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    
    # Get captions
    captions = [example["caption"] for example in examples]
    
    # Tokenize captions
    inputs = tokenizer(
        captions, 
        padding="max_length", 
        max_length=77, 
        truncation=True, 
        return_tensors="pt"
    )
    
    return {
        "pixel_values": pixel_values,
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
    }