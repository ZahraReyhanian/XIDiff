import torch
import numpy as np
from PIL import Image
from diffusers.pipelines.ddim.pipeline_ddim_cond import DDIMPipeline
from diffusers.pipelines.ddpm.pipeline_ddpm_cond import DDPMPipeline
from diffusers.pipelines.ddim.pipeline_ddim_guide import DDIMGuidedPipeline
from models.conditioner import mix_hidden_states
from visualizations.resizer import Resizer


@torch.no_grad()
def render_condition(batch, pl_module, sampler='ddim', between_zero_and_one=True, show_progress=False,
                     generator=None, mixing_batch=None, mixing_method='label_interpolate', source_alpha=1.0,
                     return_x0_intermediates=False):
    if generator is None:
        generator = torch.manual_seed(0)

    batch_size = len(batch['id_img'])
    for key, val in batch.items():
        if isinstance(val, torch.Tensor):
            batch[key] = val.to(pl_module.device)

    encoder_hidden_states = pl_module.get_encoder_hidden_states(batch, batch_size, alpha=source_alpha)
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


def to_image_npy_uint8(pred_images_npy):
    pred_images = np.clip(pred_images_npy*255+0.5, 0, 255).astype(np.uint8)
    return pred_images


def save_uint8(pred_uint8_image, path):
    im = Image.fromarray(pred_uint8_image)
    im.save(path)
