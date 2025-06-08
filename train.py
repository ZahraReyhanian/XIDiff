import os
import json
import torch
import time

from typing import Tuple
import pytorch_lightning as pl
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger

from Trainer import Trainer as MyModelTrainer
from utils import os_utils
from utils.callbacks import create_list_of_callbacks
from datamodules.face_datamodule import FaceDataModule
from utils.os_utils import get_latest_file

epochs = 30
use_pretrained = True  # True: Finetune unet from main dcface, False: Train from 0 unet
continue_training = False # Training of Trainer


def training(cfg, general_cfg):
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    Args:
        cfg : Configuration.
        general_cfg : Other Configuration.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    root = cfg['root']

    # set seed for random number generators in pytorch, numpy and python.random
    pl.seed_everything(cfg["seed"], workers=True)
    json_path = os.path.join(root, cfg["json_path"])

    datamodule = FaceDataModule(json_path=json_path, img_size=(cfg["image_size"], cfg["image_size"]),
                                batch_size=cfg["batch_size"])

    modelTrainer_path = torch.load(os.path.join(root, 'pretrained_models/dcface_3x3.ckpt'))

    model = MyModelTrainer(unet_config=general_cfg['unet_config'],
                           use_pretrained=use_pretrained,
                           lr=cfg['lr'],
                           recognition=general_cfg['recognition'],
                           recognition_eval=general_cfg['recognition_eval'],
                           label_mapping=general_cfg['label_mapping'],
                           external_mapping=general_cfg['external_mapping'],
                           # pretrained_style_path=cfg['style_ckpt_path']+"/"+cfg['name_style_ckpt'],
                           output_dir=cfg["output_dir"],
                           mse_loss_lambda=cfg["mse_loss_lambda"],
                           identity_consistency_loss_lambda=cfg["identity_consistency_loss_lambda"],
                           perceptual_loss_lambda=cfg['perceptual_loss_lambda'],
                           perceptual_loss_weight=cfg['perceptual_loss_weight'],
                           sampler=general_cfg['sampler'],
                           root=root)
    model.load_state_dict(modelTrainer_path['state_dict'], strict=True)

    print("Instantiating callbacks...")
    model_ckpt_path = root + cfg["ckpt_path"]
    callbacks = create_list_of_callbacks(model_ckpt_path)

    # print("Instantiating loggers...")
    # logger = WandbLogger(project=cfg["project"], log_model='all', id=cfg["id"], save_dir=cfg["log_dir"], )

    logger = TensorBoardLogger("lightning_logs", name="my_model")
    strategy = DDPStrategy(find_unused_parameters=False)
    trainer = pl.Trainer(accelerator="gpu",
                         callbacks=callbacks,
                         strategy=strategy,
                         max_epochs=epochs,
                         logger=logger
                         )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if cfg["training"]:
        print("Starting training...")
        # use pretrain MyModelTrainer
        if continue_training:
            print('loading checkpoint in initalization from ', model_ckpt_path, '...............')
            name = get_latest_file(model_ckpt_path)
            print(name)
            model_ckpt_path_name = os.path.join(model_ckpt_path, name)
            trainer.fit(model=model, datamodule=datamodule, ckpt_path=model_ckpt_path_name)
        else:
            trainer.fit(model=model, datamodule=datamodule)
        trainer.save_checkpoint(f"{model_ckpt_path}/final.ckpt")

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        print("Starting testing!")
        model_ckpt_path = trainer.checkpoint_callback.best_model_path

        if model_ckpt_path == "":
            print("Best ckpt not found! Using current weights for testing...")
            raise ValueError('')

        trainer.test(model=model, datamodule=datamodule, ckpt_path=model_ckpt_path)
        print(f"Best ckpt path: {model_ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


def main():
    with open('config/config.json') as f:
        cfg = json.load(f)

    with open('config/general.json') as f:
        general_cfg = json.load(f)

    print(f"tmp directory : {cfg['output_dir']}")
    # TODO: if exits remove
    # if 'experiments' in cfg.output_dir:
    #     print(f"removing tmp directory : {cfg.output_dir}")
    #     shutil.rmtree(cfg.paths.output_dir, ignore_errors=True)
    run_name = os_utils.make_runname(cfg["prefix"])
    task = os.path.basename(cfg["project"])
    exp_root = os.path.dirname(cfg["output_dir"])
    output_dir = os_utils.make_output_dir(exp_root, task, run_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Current working directory : {os.getcwd()}")
    print(f"Saving Directory          : {cfg['output_dir']}")
    available_gpus = torch.cuda.device_count()
    print("available_gpus------------", available_gpus)

    print('Per GPU batchsize:', cfg['batch_size'])
    time.sleep(1)

    # train the model
    metric_dict, _ = training(cfg, general_cfg)
    print(metric_dict)


if __name__ == "__main__":
    main()
