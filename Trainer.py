import os
from typing import Any, List
import pytorch_lightning as pl

from torch.nn import functional as F
from models.conditioner import make_condition
from models import model_helper
import torchmetrics
from utils.training_utils import EMAModel
from utils.training_utils import q_xt_x0
from losses.consistency_loss import calc_identity_consistency_loss
import torch
from recognition.recognition_helper import make_id_extractor, RecognitionModel, make_recognition_model
# from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

class Trainer(pl.LightningModule):
    """main class"""

    def __init__(self,
                 unet_config=None,
                 id_ext_config=None,
                 output_dir=None,
                 ckpt_path=None,
                 mse_loss_lambda=1,
                 identity_consistency_loss_lambda=0.05,
                 optimizer=torch.optim.AdamW,
                 *args, **kwargs
                 ):

        super(Trainer, self).__init__()

        self.optimizer = optimizer
        self.unet_config = unet_config
        self.output_dir = output_dir
        self.mse_loss_lambda = mse_loss_lambda
        self.identity_consistency_loss_lambda = identity_consistency_loss_lambda

        # self.noise_scheduler = DDPMScheduler(num_train_timesteps=sampler['num_train_timesteps'],
        #                                      beta_start=sampler['beta_start'],
        #                                      beta_end=sampler['beta_end'],
        #                                      variance_type=sampler['variance_type'],
        #                                      tensor_format="pt")

        self.model = model_helper.make_unet(unet_config)

        self.ema_model = EMAModel(self.model, inv_gamma=1.0, power=3 / 4, max_value=0.9999)
        if 'gradient_checkpointing' in unet_config and unet_config['gradient_checkpointing']:
            self.model.enable_gradient_checkpointing()

        self.valid_loss_metric = torchmetrics.MeanMetric()
        self.id_extractor = make_id_extractor(id_ext_config, unet_config)

        # disabled training
        self.recognition_model: RecognitionModel = make_recognition_model(id_ext_config["recognition_config"])
        self.recognition_model_eval = self.recognition_model


        if ckpt_path is not None:
            print('loading checkpoint in initalization from ', ckpt_path)
            ckpt = torch.load(ckpt_path, map_location='cpu')['state_dict']
            model_statedict = {key[6:]: val for key, val in ckpt.items() if key.startswith('model.')}
            self.model.load_state_dict(model_statedict)

    def get_parameters(self):
        if self.unet_config['freeze_unet']:
            print('freeze unet skip optim params')
            params = []
        else:
            params = list(self.model.parameters())
        if self.id_extractor is not None:
            params = params + list(self.id_extractor.parameters())
        return params

    def configure_optimizers(self):
        opt = self.optimizer(params=self.get_parameters())
        return [opt], []

    def load_state_dict(self, state_dict: 'OrderedDict[str, Tensor]', strict: bool = True):
        result = super(Trainer, self).load_state_dict(state_dict, strict=False)
        if result.missing_keys:
            print('\n\n\n\nMissing Keys during Loading statedict')
            print(result.missing_keys)
        if result.unexpected_keys:
            print('\n\n\n\nunexpected_keys Keys during Loading statedict')
            print(result.unexpected_keys)
        return result

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        if self.current_epoch == 0:
            # one time copy of project files
            os.makedirs(self.output_dir, exist_ok=True)

    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx=0):
        pass

    def shared_step(self, batch, stage='train', optimizer_idx=0, n_steps=1000, *args, **kwargs):
        clean_images = batch[0]

        bsz = clean_images.shape[0]

        # Sample a random timestep for each image
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        timesteps = torch.randint(
            0, n_steps, (bsz,), device=clean_images.device
        ).long().to(device)

        loss_dict = {}
        total_loss = 0.0

        if optimizer_idx == 0:
            noisy_images, noise = q_xt_x0(clean_images, timesteps)
            encoder_hidden_states = self.get_encoder_hidden_states(batch, batch_size=None)
            noise_pred = self.model(noisy_images, timesteps, encoder_hidden_states=encoder_hidden_states).sample
            mse_loss = F.mse_loss(noise_pred, noise)

            total_loss = total_loss + mse_loss * self.mse_loss_lambda
            loss_dict[f'{stage}/mse_loss'] = mse_loss
            if stage != 'train':
                loss_dict[f'{stage}/total_loss'] = total_loss
                return total_loss, loss_dict



            #TODO extra identity_consistency_loss_lambda
            # if self.identity_consistency_loss_lambda > 0:
            #     id_loss, spatial_loss = calc_identity_consistency_loss(eps=noise_pred, timesteps=timesteps,
            #                                                            noisy_images=noisy_images, batch=batch,
            #                                                            pl_module=self)
            #     total_loss = total_loss + id_loss * self.identity_consistency_loss_lambda
            #
            #     loss_dict[f'{stage}/id_loss'] = id_loss

            loss_dict[f'{stage}/total_loss'] = total_loss


        return total_loss, loss_dict

    def get_encoder_hidden_states(self, batch, batch_size=None):
        encoder_hidden_states = make_condition(pl_module=self,
                                               batch=batch
                                               )
        if batch_size is not None and encoder_hidden_states is not None:
            for key, val in encoder_hidden_states.items():
                if val is not None:
                    encoder_hidden_states[key] = val[:batch_size]

        return encoder_hidden_states

    def forward(self, x, c, *args, **kwargs):
        raise ValueError('should not be here. Not Implemented')

    def training_step(self, batch, batch_idx):
        # import cv2
        # from src.general_utils.img_utils import tensor_to_numpy
        # cv2.imwrite('/mckim/temp/temp3.png',tensor_to_numpy(batch['image'].cpu()[10])) # this is in rgb. so wrong color saved

        loss, loss_dict = self.shared_step(batch, stage='train')
        if self.ema_model.averaged_model.device != self.device:
            self.ema_model.averaged_model.to(self.device)
        self.ema_model.step(self.model)
        # self.log("ema_decay", self.ema_model.decay, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        #
        # # self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        # self.log_dict(loss_dict, prog_bar=True)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, stage='val'):
        _, loss_dict = self.shared_step(batch, stage=stage)
        self.valid_loss_metric.update(loss_dict[f'{stage}/mse_loss'])

    # def validation_epoch_end(self, outputs, stage='val', *args, **kwargs):
    #     # self.log('num_samples', self.num_samples)
    #     self.log('epoch', self.current_epoch)
    #     self.log('global_step', self.global_step)
    #     self.log(f'{stage}/mse_loss', self.valid_loss_metric.compute())
    #     self.valid_loss_metric.reset()

    def on_train_batch_end(self, *args, **kwargs):
        pass

    def test_step(self, batch: Any, batch_idx: int):
        return self.validation_step(batch, batch_idx, stage='test')

    def test_epoch_end(self, outputs: List[Any]):
        return self.validation_epoch_end(outputs, stage='test')

    @property
    def x_T_size(self):
        in_channels = self.unet_config.in_channels
        encoded_size = self.unet_config.image_size
        x_T_size = [in_channels, encoded_size, encoded_size]
        return x_T_size