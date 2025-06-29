import os
from typing import Any, List
import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from torch.nn import functional as F

from models.conditioner import make_condition
from recognition.external_mapping import make_external_mapping
from models import model_helper
import torchmetrics
from utils.training_utils import EMAModel
from losses.consistency_loss import calc_identity_consistency_loss
import torch
from recognition.recognition_helper import RecognitionModel, make_recognition_model, same_config
from recognition.recognition_helper import disabled_train
from recognition.label_mapping import make_label_mapping
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from functools import partial
from losses.perceptual_loss import perceptual_loss
from models.unet import AttentionBlock

class Trainer(pl.LightningModule):
    """main class"""

    def __init__(self,
                 unet_config=None,
                 recognition=None,
                 recognition_eval=None,
                 label_mapping=None,
                 external_mapping=None,
                 output_dir=None,
                 lr=0.001,
                 mse_loss_lambda=1,
                 perceptual_loss_lambda=0.05, #Todo should be tested
                 identity_consistency_loss_lambda=0.05,
                 identity_consistency_loss_weight_start_bias=0.0,
                 identity_consistency_loss_time_cut=0.0,
                 identity_consistency_loss_source="mix",
                 identity_consistency_mix_loss_version="polynomial_1",
                 identity_consistency_loss_center_source="id_image",
                 spatial_consistency_loss_lambda=0.0,
                 identity_consistency_loss_version= "simple_mean",
                 optimizer=torch.optim.AdamW,
                 sampler=None,
                 use_ema=True,
                 use_pretrained=False,
                 pretrained_style_path=None,
                 perceptual_loss_weight=[],
                 freeze_label_mapping=True,
                 only_attention_finetuning=True,
                 image_size=112,
                 root='',
                 *args, **kwargs
                 ):

        super(Trainer, self).__init__()

        self.optimizer = optimizer
        self.lr = lr
        self.unet_config = unet_config
        self.output_dir = output_dir
        self.mse_loss_lambda = mse_loss_lambda
        self.perceptual_loss_lambda = perceptual_loss_lambda
        self.identity_consistency_loss_lambda = identity_consistency_loss_lambda
        self.identity_consistency_loss_source = identity_consistency_loss_source
        self.identity_consistency_mix_loss_version = identity_consistency_mix_loss_version
        self.identity_consistency_loss_center_source = identity_consistency_loss_center_source
        self.identity_consistency_loss_weight_start_bias = identity_consistency_loss_weight_start_bias
        self.identity_consistency_loss_time_cut = identity_consistency_loss_time_cut

        self.spatial_consistency_loss_lambda = spatial_consistency_loss_lambda
        self.identity_consistency_loss_version = identity_consistency_loss_version
        self.sampler = sampler
        self.n_steps = sampler['num_train_timesteps']
        self.use_ema = use_ema
        self.perceptual_loss_weight = perceptual_loss_weight
        self.freeze_label_mapping = freeze_label_mapping
        self.only_attention_finetuning = only_attention_finetuning

        recognition['ckpt_path'] = os.path.join(root, recognition['ckpt_path'])
        recognition['center_path'] = os.path.join(root, recognition['center_path'])
        recognition_eval['center_path'] = os.path.join(root, recognition_eval['center_path'])

        self.model = model_helper.make_unet(unet_config)
        self.ema_model = EMAModel(self.model, inv_gamma=1.0, power=3 / 4, max_value=0.9999)
        # if 'gradient_checkpointing' in unet_config and unet_config['gradient_checkpointing']:
        #     self.model.enable_gradient_checkpointing()

        self.noise_scheduler = DDPMScheduler(num_train_timesteps=sampler['num_train_timesteps'],
                                             beta_start=sampler['beta_start'],
                                             beta_end=sampler['beta_end'],
                                             variance_type=sampler['variance_type'],
                                             tensor_format="pt")
        self.noise_scheduler_ddim = DDIMScheduler(num_train_timesteps=sampler['num_train_timesteps'],
                                                  beta_start=sampler['beta_start'],
                                                  beta_end=sampler['beta_end'],
                                                  tensor_format="pt")


        # enable training
        print("make recognition model")
        print(recognition)
        self.recognition_model: RecognitionModel = make_recognition_model(recognition, root, enable_training=False)
        if same_config(recognition, recognition_eval,
                       skip_keys=['return_spatial', 'center_path']):
            self.recognition_model_eval = self.recognition_model
        else:
            self.recognition_model_eval: RecognitionModel = make_recognition_model(recognition_eval, root)

        self.valid_loss_metric = torchmetrics.MeanMetric()

        self.label_mapping = make_label_mapping(label_mapping, unet_config, root)

        # self.expression_encoder = create_expression_encoder(device, root=root)

        self.external_mapping = make_external_mapping(external_mapping, unet_config)

        if pretrained_style_path:
            pretrained_style_path = os.path.join(root, pretrained_style_path)
            self.external_mapping.load_state_dict(torch.load(pretrained_style_path, weights_only=True))


        if use_pretrained: #use pretrained unet
            ckpt_path = os.path.join(root, unet_config['params']['pretrained_model_path'])

            statedict = torch.load(ckpt_path, map_location='cpu', weights_only=True)
            self.model.load_state_dict(statedict, strict=True)


        if unet_config['freeze_unet']:
            print('freeze unet')
            self.model = self.model.eval()
            self.model.train = partial(disabled_train, self=self.model)
            for param in self.model.parameters():
                param.requires_grad = False

        # freeze identity encoder
        if freeze_label_mapping:
            for param in self.label_mapping.parameters():
                param.requires_grad = False

        # Just fine-tune AttentionBlock and others should be freez
        if only_attention_finetuning:
            for module_name, module in self.model.named_modules():
                if not isinstance(module, AttentionBlock):
                    for param in module.parameters():
                        param.requires_grad = False

        self.save_hyperparameters()

    def get_parameters(self):
        if self.unet_config['freeze_unet']:
            print('freeze unet skip optim params')
            params = []
        elif self.only_attention_finetuning:
            params = []
            for param in self.model.parameters():
                if param.requires_grad:
                    params.append(param)
        else:
            params = list(self.model.parameters())
        if self.label_mapping is not None and not self.freeze_label_mapping:
            params = params + list(self.label_mapping.parameters())
        if self.external_mapping is not None:
            params = params + list(self.external_mapping.parameters())
        return params

    def configure_optimizers(self):
        optimizer = self.optimizer(params=self.get_parameters(), lr=self.lr)
        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8),
            'name': 'my_logging_name'
        }
        return [optimizer], [lr_scheduler]

    def load_state_dict(self, state_dict: 'OrderedDict[str, Tensor]', strict: bool = True):
        result = super(Trainer, self).load_state_dict(state_dict, strict=False)
        if result.missing_keys:
            print('\n\n\Missing Keys during Loading statedict')
            # print(result.missing_keys)
        if result.unexpected_keys:
            print('unexpected_keys Keys during Loading statedict')
            # print(result.unexpected_keys)
        return result

    @rank_zero_only
    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        if self.current_epoch == 0:
            # one time copy of project files
            os.makedirs(self.output_dir, exist_ok=True)

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx=0):
        pass

    def shared_step(self, batch, stage='train', optimizer_idx=0, *args, **kwargs):
        clean_images = batch["exp_img"]

        bsz = clean_images.shape[0]
        noise = torch.randn(clean_images.shape).to(self.device)

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (bsz,)
        ).long().to(self.device)

        loss_dict = {}
        total_loss = 0.0

        if optimizer_idx == 0:
            noisy_images = self.noise_scheduler.add_noise(clean_images, noise, timesteps)
            encoder_hidden_states = self.get_encoder_hidden_states(batch, batch_size=None)
            noise_pred = self.model(noisy_images, timesteps, encoder_hidden_states=encoder_hidden_states).sample
            mse_loss = F.mse_loss(noise_pred, noise)

            total_loss = total_loss + mse_loss * self.mse_loss_lambda
            loss_dict[f'{stage}/mse_loss'] = mse_loss
            if stage == 'test':
                loss_dict[f'{stage}/total_loss'] = total_loss
                return total_loss, loss_dict


            if self.identity_consistency_loss_lambda > 0 or \
                    self.spatial_consistency_loss_lambda > 0:
                id_loss, spatial_loss = calc_identity_consistency_loss(eps=noise_pred, timesteps=timesteps,
                                                                       noisy_images=noisy_images, batch=batch,
                                                                       pl_module=self)
                total_loss = total_loss + id_loss * self.identity_consistency_loss_lambda
                if spatial_loss is not None:
                    total_loss = total_loss + spatial_loss * self.spatial_consistency_loss_lambda
                    loss_dict[f'{stage}/spatial_loss'] = spatial_loss
                loss_dict[f'{stage}/id_loss'] = id_loss

            if self.perceptual_loss_lambda > 0:
                perc_loss = perceptual_loss(self.perceptual_loss_weight,
                                            eps=noise_pred,
                                            timesteps=timesteps,
                                            noisy_images=noisy_images,
                                            pl_module=self,
                                            target_pixels=batch["exp_img"])
                total_loss = total_loss + perc_loss*self.perceptual_loss_lambda
                loss_dict[f'{stage}/perc_loss'] = perc_loss

            loss_dict[f'{stage}/total_loss'] = total_loss


        return total_loss, loss_dict


    # def on_before_optimizer_step(self, optimizer):
    #     # Compute the 2-norm for each layer
    #     # If using mixed precision, the gradients are already unscaled here
    #     norms = grad_norm(self.layer, norm_type=2)
    #     self.log_dict(norms)

    def get_encoder_hidden_states(self, batch, batch_size=None):
        for key, val in batch.items():
            if isinstance(val, torch.Tensor):
                batch[key] = val.to(self.device)

        encoder_hidden_states = make_condition(pl_module=self,
                                               condition_type=self.unet_config['params']['condition_type'],
                                               condition_source=self.unet_config['params']['condition_source'],
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
        loss, loss_dict = self.shared_step(batch, stage='train')
        if self.ema_model.averaged_model.device != self.device:
            self.ema_model.averaged_model.to(self.device)
        self.ema_model.step(self.model)
        self.log("ema_decay", self.ema_model.decay, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        self.log_dict(loss_dict, prog_bar=True)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, stage='val'):
        loss, loss_dict = self.shared_step(batch, stage=stage)
        self.valid_loss_metric.update(loss_dict[f'{stage}/mse_loss'])
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def on_validation_epoch_end(self, stage = 'val', *args, **kwargs):
        self.log('epoch', self.current_epoch, sync_dist=True)
        self.log('global_step', self.global_step, sync_dist=True)
        self.log(f'{stage}/mse_loss', self.valid_loss_metric.compute(), sync_dist=True)
        self.valid_loss_metric.reset()

    def on_train_batch_end(self, *args, **kwargs):
        pass

    def on_train_epoch_end(self, *args, **kwargs):
        lr = self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]
        self.log('learning_rate', lr, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def test_step(self, batch: Any, batch_idx: int):
        return self.validation_step(batch, batch_idx, stage='test')

    def on_test_epoch_end(self, outputs: List[Any]):
        return self.on_validation_epoch_end(stage='test')

    @property
    def x_T_size(self):
        in_channels = self.unet_config['params']["in_channels"]
        encoded_size = self.unet_config['params']["image_size"]
        x_T_size = [in_channels, encoded_size, encoded_size]
        return x_T_size