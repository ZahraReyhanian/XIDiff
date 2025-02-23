import copy
import os
import random

import numpy as np
import torch
import torch.nn as nn
from functools import partial
from pytorch_lightning.utilities import rank_zero_only
from torchvision.utils import save_image
from tqdm import tqdm
import warnings
from typing import Optional, Tuple, Union

torch.cuda.empty_cache()
# Set up some parameters
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


@rank_zero_only
def log_hyperparameters(object_dict: dict) -> None:
    """Controls which config parts are saved by lightning loggers.

    Additionally saves:
    - Number of model parameters
    """

    hparams = {}

    cfg = object_dict["cfg"]
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        return

    hparams["model"] = cfg["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["datamodule"] = cfg["datamodule"]
    hparams["trainer"] = cfg["trainer"]

    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")

    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)

def gather(consts: torch.Tensor, t: torch.Tensor):
    """Gather consts for $t$ and reshape to feature map shape"""
    c = consts.gather(-1, t.to(device))
    return c.reshape(-1, 1, 1, 1)


def disabled_train(mode=True, self=None):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

def enable_full_determinism(seed: int):
    """
    Helper function for reproducible behavior during distributed training. See
    - https://pytorch.org/docs/stable/notes/randomness.html for pytorch
    """
    # set seed first
    set_seed(seed)

    #  Enable PyTorch deterministic mode. This potentially requires either the environment
    #  variable 'CUDA_LAUNCH_BLOCKING' or 'CUBLAS_WORKSPACE_CONFIG' to be set,
    # depending on the CUDA version, so we set them both here
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True)

    # Enable CUDNN deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_seed(seed: int):
    """
    Args:
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # ^^ safe to call this function even if cuda is not available


class EMAModel(nn.Module):
    """
    Exponential Moving Average of models weights
    """

    def __init__(
        self,
        model,
        update_after_step=0,
        inv_gamma=1.0,
        power=2 / 3,
        min_value=0.0,
        max_value=0.9999,
        device=None,
    ):
        """
        @crowsonkb's notes on EMA Warmup:
            If gamma=1 and power=1, implements a simple average. gamma=1, power=2/3 are good values for models you plan
            to train for a million or more steps (reaches decay factor 0.999 at 31.6K steps, 0.9999 at 1M steps),
            gamma=1, power=3/4 for models you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999
            at 215.4k steps).
        Args:
            inv_gamma (float): Inverse multiplicative factor of EMA warmup. Default: 1.
            power (float): Exponential factor of EMA warmup. Default: 2/3.
            min_value (float): The minimum EMA decay rate. Default: 0.
        """
        super(EMAModel, self).__init__()

        self.averaged_model = copy.deepcopy(model).eval()
        self.averaged_model.requires_grad_(False)

        self.update_after_step = update_after_step
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value
        self.max_value = max_value

        if device is not None:
            self.averaged_model = self.averaged_model.to(device=device)

        self.decay = 0.0
        self.optimization_step = 0
        self.train = partial(disabled_train, self=self)

    def get_decay(self, optimization_step):
        """
        Compute the decay factor for the exponential moving average.
        """
        step = max(0, optimization_step - self.update_after_step - 1)
        value = 1 - (1 + step / self.inv_gamma) ** -self.power

        if step <= 0:
            return 0.0

        return max(self.min_value, min(value, self.max_value))

    @torch.no_grad()
    def step(self, new_model):
        ema_state_dict = {}
        ema_params = self.averaged_model.state_dict()

        self.decay = self.get_decay(self.optimization_step)

        for key, param in new_model.named_parameters():
            if isinstance(param, dict):
                continue
            try:
                ema_param = ema_params[key]
            except KeyError:
                ema_param = param.float().clone() if param.ndim == 1 else copy.deepcopy(param)
                ema_params[key] = ema_param

            ema_param = ema_param.to(param.data.device)
            if not param.requires_grad:
                ema_params[key].copy_(param.to(dtype=ema_param.dtype).data)
                ema_param = ema_params[key]
            else:
                ema_param.mul_(self.decay)
                ema_param.add_(param.data.to(dtype=ema_param.dtype), alpha=1 - self.decay)

            ema_state_dict[key] = ema_param

        for key, param in new_model.named_buffers():
            ema_state_dict[key] = param

        self.averaged_model.load_state_dict(ema_state_dict, strict=False)
        self.optimization_step += 1

# def generate_image(model, fake_image_path, im_size, dataloader, batch_size, n_steps, device, generator: Optional[torch.Generator] = None,):
#
#     with torch.no_grad():
#         it = 0
#         for batch in dataloader:
#             print("image: ", it)
#             noise = torch.randn(batch[0].shape).to(device)
#             timesteps = torch.randint(
#                 0, n_steps, (batch_size,), device=device
#             ).long().to(device)
#             clean_image = batch[0].to(device)
#             image = model.noise_scheduler.add_noise(clean_image, noise, timesteps)
#             image = image.to(device)
#
#             # set step values
#             model.noise_scheduler.set_timesteps(n_steps)
#
#             for t in tqdm(model.noise_scheduler.timesteps):
#                 # 1. predict noise model_output
#                 t = t.to(device)
#                 encoder_hidden_states = model.get_encoder_hidden_states(batch, batch_size=None)
#                 model_output = model.model(image, t, encoder_hidden_states=encoder_hidden_states).sample
#
#                 # 2. compute previous image: x_t -> x_t-1
#                 image = model.noise_scheduler.step(model_output, t, image, generator=generator).prev_sample
#
#             image = (image / 2 + 0.5).clamp(0, 1)
#             image = image.cpu()
#             save_image(tensor=image, fp=f'{fake_image_path}/img_{it}.png')
#             it+=1
#
