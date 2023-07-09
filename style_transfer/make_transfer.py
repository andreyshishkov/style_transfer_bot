import torch
import torch.optim as optim
from torchvision.utils import save_image
import os
from style_transfer.model import VGG
from style_transfer.utls import load_image
import time


def transfer_style(original_path: str, style_path: str, model: VGG, device, save_dir='generated_images') -> str:
    original_image = load_image(original_path).to(device)
    style_image = load_image(style_path).to(device)
    generated = original_image.clone().requires_grad_(True)

    total_steps = 150
    lr = 0.001
    alpha = 1
    beta = 0.04

    optimizer = optim.Adam([generated], lr=lr)

    for step in range(total_steps):
        generated_features = model(generated)
        original_img_features = model(original_image)
        style_features = model(style_image)

        style_loss = original_loss = 0
        for gen_feature, orig_feature, style_feature in zip(
            generated_features,
            original_img_features,
            style_features
        ):
            batch_size, channel, height, width = gen_feature.shape
            original_loss += torch.mean((gen_feature - orig_feature) ** 2)

            # compute Gr5am matrix
            G = gen_feature.view(channel, height * width).mm(
                gen_feature.view(channel, height * width).t()
            )

            A = style_feature.view(channel, height * width).mm(
                style_feature.view(channel, height * width).t()
            )

            style_loss += torch.mean((G - A) ** 2)

        total_loss = alpha * original_loss + beta * style_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    orig_name = os.path.basename(original_path)
    result_name = os.path.join(save_dir, orig_name)
    save_image(generated, result_name)

    return result_name


