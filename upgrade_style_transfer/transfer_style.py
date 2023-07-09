import torch
import os
from torch.autograd import Variable
from upgrade_style_transfer.model import Net
from upgrade_style_transfer.utils import tensor_load_rgbimage, preprocess_batch, tensor_save_bgrimage


def transfer_style(original_path: str, style_path: str, style_model: Net, save_dir: str = 'generated_images') -> str:
    content_image = tensor_load_rgbimage(
        original_path,
        size=512,
        keep_asp=True).unsqueeze(0)
    style = tensor_load_rgbimage(
        style_path,
        size=512).unsqueeze(0)
    style = preprocess_batch(style)

    style_v = Variable(style)
    content_image = Variable(preprocess_batch(content_image))
    style_model.setTarget(style_v)
    output = style_model(content_image)

    orig_name = os.path.basename(original_path)
    result_name = os.path.join(save_dir, orig_name)

    tensor_save_bgrimage(output.data[0], result_name, False)

    return result_name
