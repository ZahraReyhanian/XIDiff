import torch
from diffusers.pipelines.ddim.pipeline_ddim_cond import DDIMPipeline
from diffusers.pipelines.ddpm.pipeline_ddpm_cond import DDPMPipeline
from diffusers.pipelines.ddim.pipeline_ddim_guide import DDIMGuidedPipeline
from models.conditioner import mix_hidden_states
from visualizations.resizer import Resizer
import os
import cv2
from tqdm import tqdm
import numpy as np
from PIL import Image


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

        if it > 50:
            break


def sample_batch(batch, pl_module, seed=None):

    if seed is not None:
        generator = torch.manual_seed(seed)
    else:
        generator = None
    pred_images = render_condition(batch, pl_module, between_zero_and_one=True,
                                   show_progress=False, generator=generator, mixing_batch=None,
                                   return_x0_intermediates=False)

    # select which time to plot
    plotting_images = pred_images * 255
    return plotting_images[:, :, :, ::-1]


@torch.no_grad()
def render_condition(batch, pl_module, sampler='ddim', between_zero_and_one=True, show_progress=False,
                     generator=None, mixing_batch=None, mixing_method='label_interpolate', source_alpha=0.0,
                     return_x0_intermediates=False):
    if generator is None:
        generator = torch.manual_seed(0)

    batch_size = len(batch['id_img'])
    for key, val in batch.items():
        if isinstance(val, torch.Tensor):
            batch[key] = val.to(pl_module.device)

    encoder_hidden_states = pl_module.get_encoder_hidden_states(batch, batch_size)
    if mixing_batch is not None:
        for key, val in mixing_batch.items():
            if isinstance(val, torch.Tensor):
                mixing_batch[key] = val.to(pl_module.device)
        mixing_hidden_states = pl_module.get_encoder_hidden_states(mixing_batch, batch_size)
        encoder_hidden_states = mix_hidden_states(encoder_hidden_states, mixing_hidden_states,
                                                  condition_type=pl_module.unet_config['params']['condition_type'],
                                                  condition_source=pl_module.unet_config['params']['condition_source'],
                                                  mixing_method=mixing_method,
                                                  source_alpha=source_alpha,
                                                  pl_module=pl_module)
    if sampler == 'ddpm':
        pipeline = DDPMPipeline(
            unet=pl_module.ema_model.averaged_model if pl_module.hparams.use_ema else pl_module.model,
            scheduler=pl_module.noise_scheduler)
        pipeline.set_progress_bar_config(disable=not show_progress)
        pred_result = pipeline(generator=generator, batch_size=batch_size, output_type="numpy",
                               encoder_hidden_states=encoder_hidden_states)
    elif sampler == 'ddim':
        pipeline = DDIMPipeline(
            unet=pl_module.ema_model.averaged_model if pl_module.hparams.use_ema else pl_module.model,
            scheduler=pl_module.noise_scheduler_ddim)
        pipeline.set_progress_bar_config(disable=not show_progress)
        pred_result = pipeline(generator=generator, batch_size=batch_size, output_type="numpy",
                               num_inference_steps=50, eta=1.0, use_clipped_model_output=False,
                               encoder_hidden_states=encoder_hidden_states,
                               return_x0_intermediates=return_x0_intermediates)
    elif sampler == 'ddim_ilvr':
        pipeline = DDIMPipeline(
            unet=pl_module.ema_model.averaged_model if pl_module.hparams.use_ema else pl_module.model,
            scheduler=pl_module.noise_scheduler_ddim)
        pipeline.set_progress_bar_config(disable=not show_progress)
        down_N = 8
        range_t = 100
        shape = batch['image'].shape
        shape_d = (shape[0], 3, int(shape[2] /down_N), int(shape[3] /down_N))
        down = Resizer(shape, 1 / down_N).to(pl_module.device)
        up = Resizer(shape_d, down_N).to(pl_module.device)
        ilvr_params = [down, up, range_t, batch['image']]
        pred_result = pipeline(generator=generator, batch_size=batch_size, output_type="numpy",
                               num_inference_steps=50, eta=1.0, use_clipped_model_output=False,
                               encoder_hidden_states=encoder_hidden_states,
                               return_x0_intermediates=return_x0_intermediates,
                               ilvr=ilvr_params)
    elif sampler == 'ddim_guided':
        pl_module.recognition_model.device = pl_module.device
        pipeline = DDIMGuidedPipeline(
            unet=pl_module.ema_model.averaged_model if pl_module.hparams.use_ema else pl_module.model,
            recognition_model=pl_module.recognition_model,
            scheduler=pl_module.noise_scheduler_ddim)
        pipeline.set_progress_bar_config(disable=not show_progress)
        reference_recognition_feature = encoder_hidden_states['center_emb']
        pred_result = pipeline(generator=generator, batch_size=batch_size, output_type="numpy",
                               num_inference_steps=50, eta=1.0, use_clipped_model_output=False,
                               encoder_hidden_states=encoder_hidden_states,
                               reference_recognition_feature=reference_recognition_feature,
                               return_x0_intermediates=return_x0_intermediates)
    else:
        raise ValueError('')
    pred_images = pred_result.images
    pred_images = np.clip(pred_images, 0, 1)
    if not between_zero_and_one:
        # between -1 and 1
        pred_images = (pred_images - 0.5) / 0.5

    if return_x0_intermediates:
        x0_intermediates = pred_result.x0_intermediates
        return pred_images, x0_intermediates

    return pred_images
