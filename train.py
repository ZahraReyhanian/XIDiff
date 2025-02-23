import os
import json
import torch

from typing import Tuple
import pytorch_lightning as pl
from pytorch_lightning.strategies.ddp import DDPStrategy
# from pytorch_lightning.loggers.wandb import WandbLogger
# from utils.training_utils import log_hyperparameters
from torch.utils.data import DataLoader
from Trainer import Trainer as MyModelTrainer
from utils import os_utils
from utils.callbacks import create_list_of_callbacks
from datamodules.face_datamodule import FaceDataModule
from utils.generation_utils import generate_image
from torchvision import datasets, transforms

epochs=40
n_steps=100

unet_config = {
    "image_size": 112,
    "num_channels": 128,
    "num_res_blocks": 1,
    "channel_mult": '',
    "learn_sigma": True,
    "class_cond": False,
    "use_checkpoint": False,
    "attention_resolutions": "16",
    "num_heads": 4,
    "num_head_channels": 64,
    "num_heads_upsample": -1,
    "use_scale_shift_norm": True,
    "dropout": 0.0,
    "resblock_updown": True,
    "use_fp16": False,
    "use_new_attention_order": False,
    "freeze_unet": False
}

id_ext_config = {
    "version": "v4",
    "out_channel": 256,
    "num_latent": 8,
    "recognition_config":{
        "dataset": "webface4m",
        "loss_fn": "adaface",
        "normalize_feature": False,
        "return_spatial": [2],
        "head_name": 'none',
        "backbone": "ir_50",
        "ckpt_path": '/pretrained_models/adaface_ir50_webface4m.ckpt',
        # "center_path": '/pretrained_models/center_ir_50_adaface_webface4m_faces_webface_112x112.pth'
    }
}

sampler= {
    "num_train_timesteps": n_steps,
    "beta_start": 0.0001,
    "beta_end": 0.02,
    "variance_type": "fixed_small"
}

def training(cfg):
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    # set seed for random number generators in pytorch, numpy and python.random
    pl.seed_everything(cfg["seed"], workers=True)
    path = cfg["dataset_path"]

    datamodule = FaceDataModule(dataset_path=path)

    model = MyModelTrainer(unet_config=unet_config,
                           ckpt_path=cfg["ckpt_path"],
                           lr=cfg['lr'],
                           id_ext_config= id_ext_config,
                           output_dir=cfg["output_dir"],
                           mse_loss_lambda=cfg["mse_loss_lambda"],
                           identity_consistency_loss_lambda=cfg["identity_consistency_loss_lambda"],
                           sampler=sampler)

    print("Instantiating callbacks...")
    callbacks = create_list_of_callbacks(cfg["ckpt_path"])

    # print("Instantiating loggers...")
    # logger = WandbLogger(project=cfg["project_task"], log_model='all', id= cfg["id"], save_dir=cfg["output_dir"],)
    print("before train.....................................................................")
    strategy = DDPStrategy(find_unused_parameters=False)
    trainer = pl.Trainer(accelerator="gpu", callbacks=callbacks, strategy=strategy, max_epochs=epochs)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        # "logger": logger,
        "trainer": trainer,
    }


    if cfg["training"]:
        print("Starting training...")
        if cfg["ckpt_path"]:
            print('continuing from ', cfg["ckpt_path"])

        trainer.fit(model=model, datamodule=datamodule)
        trainer.save_checkpoint(f"{cfg['ckpt_path']}/final.ckpt")

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        print("Starting testing!")
        if cfg.get("ckpt_path") and not cfg.get("train"):
            print("Using predefined ckpt_path", cfg.get('ckpt_path'))
            ckpt_path = cfg.get("ckpt_path")
        elif cfg.get('trainer')['ckpt_path'] and not cfg.get("train"):
            print('Model weight will be loaded during Making the Model')
            ckpt_path = None
        else:
            ckpt_path = trainer.checkpoint_callback.best_model_path

        if ckpt_path == "":
            print("Best ckpt not found! Using current weights for testing...")
            raise ValueError('')
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        print(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    transform = transforms.Compose([transforms.Resize((48,48)),
                                    transforms.ToTensor()])

    # load dataset
    print("cudaaaaaaaaaaaaa is available?!")
    print(torch.cuda.is_available())
    model.eval()
    model.model = model.model.cuda()
    data_test = datasets.ImageFolder(f'{path}test', transform=transform)
    bs = 1
    test_loader = DataLoader(data_test, batch_size=1)
    device = torch.device("cuda")

    generate_image(pl_module=model,
                   save_root="generated_images",
                   batch_size=bs,
                   dataloader=test_loader,
                   device=device)

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


def main():
    with open('config/config.json') as f:
        cfg = json.load(f)

    print(f"tmp directory : {cfg['output_dir']}")
    #TODO: if exits remove
    # if 'experiments' in cfg.output_dir:
    #     print(f"removing tmp directory : {cfg.output_dir}")
    #     shutil.rmtree(cfg.paths.output_dir, ignore_errors=True)
    run_name = os_utils.make_runname(cfg["prefix"])
    task = os.path.basename(cfg["project_task"])
    exp_root = os.path.dirname(cfg["output_dir"])
    output_dir = os_utils.make_output_dir(exp_root, task, run_name)
    os.makedirs(output_dir, exist_ok=True)

    wandb_name = os.path.basename(cfg["output_dir"])

    print(f"Current working directory : {os.getcwd()}")
    print(f"Saving Directory          : {cfg['output_dir']}")
    available_gpus = torch.cuda.device_count()
    print("available_gpus------------", available_gpus)
    # cfg.datamodule.batch_size = int(cfg.datamodule.total_gpu_batch_size / available_gpus)
    # print('Per GPU batchsize:', cfg.datamodule.batch_size)
    # time.sleep(1)

    # train the model
    metric_dict, _ = training(cfg)
    print(metric_dict)


if __name__ == "__main__":
    main()