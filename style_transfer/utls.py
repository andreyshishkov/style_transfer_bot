import torch
import torchvision.transforms as transforms
from PIL import Image


def get_loader(image_size: int = 336) -> transforms.Compose:
    loader = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )
    return loader


def load_image(image_path: str, image_size: int = 336) -> torch.Tensor:
    loader = get_loader(image_size)

    image = Image.open(image_path)
    image = loader(image).unsqueeze(0)
    return image
