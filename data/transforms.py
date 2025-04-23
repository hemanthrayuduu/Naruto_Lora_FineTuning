from torchvision import transforms


def get_transforms(image_size):
    """Get standard image transformations for a given size."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])