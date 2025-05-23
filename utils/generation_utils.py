import torch
from diffusers.pipelines.ddim.pipeline_ddim_cond import DDIMPipeline
from models.conditioner import mix_hidden_states
import os
import cv2
from tqdm import tqdm
import numpy as np
from typing import Optional, Tuple, Union

def to_real_img(img):
    np.multiply(img, 255.0, out=img)

    np.clip(img, 0, 255, out=img)

    img = img.astype(np.uint8, copy=False)

    return img

def tensor_to_uint8_img(tensor):
    img = tensor.squeeze().detach().cpu().numpy()

    if img.ndim == 3 and img.shape[0] in [1, 3]:  # (C, H, W)
        img = np.moveaxis(img, 0, -1)

    return to_real_img(img)


def generate_image(pl_module, dataloader, device, batch_size=1, num_workers=0, save_root='./',
                   num_partition=1, partition_idx=0, seed=42):
    os.makedirs(save_root, exist_ok=True)
    print(save_root)

    it = 0
    for batch in tqdm(dataloader, total=len(dataloader), desc='Generating Dataset: '):
        print('source_label:', batch['src_label'])
        print('target_label:', batch['target_label'])

        plotting_images = sample_batch(batch, pl_module, seed=seed)

        for i, image in enumerate(plotting_images):
            save_name = f"img_{i}_{it}.jpg"
            # save image
            save_path = os.path.join(save_root, save_name)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, image)

            exp_image = tensor_to_uint8_img(batch['exp_img'])
            save_path = os.path.join(save_root, f"img_{i}_{it}-exp.jpg")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, exp_image)

            id_img = tensor_to_uint8_img(batch['id_img'])
            save_path = os.path.join(save_root, f"img_{i}_{it}-id.jpg")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, id_img)

            it += 1

        if it > 10:
            break


def sample_batch(batch, pl_module, seed=None):
    # batch = {'style_image': style_images,
    #          'class_label': torch.arange(len(style_images)),
    #          'id_image': id_images}

    if seed is not None:
        generator = torch.manual_seed(seed)
    else:
        generator = None
    pred_images = render_condition(batch, pl_module, between_zero_and_one=True,
                                   show_progress=False, generator=generator, mixing_batch=None,
                                   return_x0_intermediates=False)

    # select which time to plot
    plotting_images = to_real_img(pred_images)
    return plotting_images


@torch.no_grad()
def render_condition(batch, pl_module, batch_size=1, between_zero_and_one=True, show_progress=False,
                     generator=None, mixing_batch=None, mixing_method='label_interpolate', source_alpha=0.0,
                     return_x0_intermediates=False):
    if generator is None:
        generator = torch.manual_seed(0)

    # for key, val in batch.items():
    #     if isinstance(val, torch.Tensor):
    #         batch[key] = val.to(pl_module.device)

    encoder_hidden_states = pl_module.get_encoder_hidden_states(batch, batch_size)
    if mixing_batch is not None:
        for key, val in mixing_batch.items():
            if isinstance(val, torch.Tensor):
                mixing_batch[key] = val.to(pl_module.device)
        mixing_hidden_states = pl_module.get_encoder_hidden_states(mixing_batch, batch_size)
        encoder_hidden_states = mix_hidden_states(encoder_hidden_states, mixing_hidden_states,
                                                  mixing_method=mixing_method,
                                                  source_alpha=source_alpha,
                                                  pl_module=pl_module)

    # print("uuusiinng emaaaaaa: ", pl_module.use_ema)
    pipeline = DDIMPipeline(
        unet=pl_module.ema_model.averaged_model if pl_module.use_ema else pl_module.model,
        scheduler=pl_module.noise_scheduler_ddim)
    pipeline.set_progress_bar_config(disable=not show_progress)

    # add random noise to image
    # clean_images = batch['id_img']
    # bsz = clean_images.shape[0]
    # noise = torch.randn(clean_images.shape).to(clean_images.device)
    # use_cuda = torch.cuda.is_available()
    # device = torch.device("cuda" if use_cuda else "cpu")
    # timesteps = torch.randint(0, pl_module.n_steps, (bsz,)).long().to(device)
    #
    # noisy_images = pl_module.noise_scheduler_ddim.add_noise(clean_images, noise, timesteps)

    pred_result = pipeline(generator=generator, batch_size=batch_size, output_type="numpy",
                           num_inference_steps=50, eta=1.0, use_clipped_model_output=False,
                           encoder_hidden_states=encoder_hidden_states,
                           return_x0_intermediates=return_x0_intermediates)

    pred_images = pred_result.images
    pred_images = np.clip(pred_images, 0, 1)
    if not between_zero_and_one:
        # between -1 and 1
        pred_images = (pred_images - 0.5) / 0.5

    if return_x0_intermediates:
        x0_intermediates = pred_result.x0_intermediates
        return pred_images, x0_intermediates

    return pred_images
