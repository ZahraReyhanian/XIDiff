import torch
import os
import cv2
from tqdm import tqdm
import numpy as np
from PIL import Image
from utils import sample_visual


def generate_image(pl_module, datamodule, batch_size=1, save_root='./', seed=42):
    os.makedirs(save_root, exist_ok=True)
    print(save_root)

    it = 0
    dataloader = datamodule.test_dataloader()
    print(datamodule.data_test.idx_to_label)
    for batch in tqdm(dataloader, total=len(dataloader), desc='Generating Images: '):
        print('target_label:', batch['target_label'])

        original_image = Image.open(batch['src_path'][0]).convert("RGB")
        target_image = Image.open(batch['target_path'][0]).convert("RGB")

        plotting_images = sample_batch(batch, pl_module, seed=seed)

        for i, image in enumerate(plotting_images):
            save_name = f"img_{i}_{it}.jpg"
            # save image
            save_path = os.path.join(save_root, save_name)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, image)

            save_path = os.path.join(save_root, f"img_{i}_{it}_{batch['target_label'].item()}-exp.jpg")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            target_image.save(save_path)

            save_path = os.path.join(save_root, f"img_{i}_{it}-id.jpg")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            original_image.save(save_path)

            it += 1

        if it > 500:
            break


def sample_batch(batch, pl_module, seed=None):

    if seed is not None:
        generator = torch.manual_seed(seed)
    else:
        generator = None
    pred_images = sample_visual.render_condition(batch, pl_module, between_zero_and_one=True,
                                   show_progress=False, generator=generator, mixing_batch=None,
                                   return_x0_intermediates=False)

    # select which time to plot
    plotting_images = pred_images * 255
    return plotting_images[:, :, :, ::-1]


