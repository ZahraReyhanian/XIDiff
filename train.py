import os
import json
import torch
# root = pyrootutils.setup_root(
#     search_from=__file__,
#     indicator=[".git", "pyproject.toml"],
#     pythonpath=True,
#     dotenv=True,
# )
# dotenv.load_dotenv(dotenv_path=root.parent.parent / '.env', override=True)
# assert os.getenv('DATA_ROOT')
# assert os.path.isdir(os.getenv('DATA_ROOT'))
# import time
#
# LOG_ROOT = str(root.parent / 'experiments')
# os.environ.update({'LOG_ROOT': LOG_ROOT})
# os.environ.update({'PROJECT_TASK': root.stem})
# os.environ.update({'REPO_ROOT': str(root.parent)})

from typing import Tuple
import pytorch_lightning as pl
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.loggers.wandb import WandbLogger
from Trainer import Trainer as MyModelTrainer
from utils import os_utils
from utils.training_utils import log_hyperparameters
from utils.callbacks import create_list_of_callbacks
from datamodules.face_datamodule import FaceDataModule

unet_config = {
    "image_size": (48, 48),
    "num_channels": 128,
    "num_res_blocks": 1,
    "channel_mult": '',
    "learn_sigma": True,
    "class_cond": False,
    "use_checkpoint": False,
    "attention_resolutions": '16',
    "num_heads": 4,
    "num_head_channels": 64,
    "num_heads_upsample": -1,
    "use_scale_shift_norm": True,
    "dropout": 0.0,
    "resblock_updown": True,
    "use_fp16": False,
    "use_new_attention_order": False
}

def train(cfg):
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
    pl.seed_everything(cfg.seed, workers=True)

    datamodule = FaceDataModule(dataset_path=cfg.dataset_path)

    model = MyModelTrainer(datamodule=datamodule,
                           unet_config=unet_config,
                           output_dir=cfg.output_dir,
                           ckpt_path=cfg.ckpt_path,
                           mse_loss_lambda=cfg.mse_loss_lambda,
                           identity_consistency_loss_lambda=cfg.identity_consistency_loss_lambda)

    print("Instantiating callbacks...")
    callbacks = create_list_of_callbacks(cfg.ckpt_path)

    print("Instantiating loggers...")
    id = ""
    logger = WandbLogger(project=cfg.project_task, log_model='all', id= id, save_dir=cfg.output_dir,)

    strategy = DDPStrategy(find_unused_parameters=False)
    trainer = Trainer(callbacks=callbacks, logger=logger, strategy=strategy)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        print("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        print("Starting training!")
        if cfg.get("ckpt_path"):
            print('continuing from ', cfg.get("ckpt_path"))
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    #TODO test
    # if cfg.get("test"):
    #     print("Starting testing!")
    #     if cfg.get("ckpt_path") and not cfg.get("train"):
    #         print("Using predefined ckpt_path", cfg.get('ckpt_path'))
    #         ckpt_path = cfg.get("ckpt_path")
    #     elif cfg.get('trainer')['ckpt_path'] and not cfg.get("train"):
    #         print('Model weight will be loaded during Making the Model')
    #         ckpt_path = None
    #     else:
    #         ckpt_path = trainer.checkpoint_callback.best_model_path
    #
    #     if ckpt_path == "":
    #         print("Best ckpt not found! Using current weights for testing...")
    #         raise ValueError('')
    #     trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
    #     print(f"Best ckpt path: {ckpt_path}")
    #
    # test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics}
        # , **test_metrics}

    return metric_dict, object_dict


def main():
    with open('data.json') as f:
        cfg = json.load(f)

    print(f"tmp directory : {cfg.output_dir}")
    #TODO: if exits remove
    # if 'experiments' in cfg.output_dir:
    #     print(f"removing tmp directory : {cfg.output_dir}")
    #     shutil.rmtree(cfg.paths.output_dir, ignore_errors=True)
    run_name = os_utils.make_runname(cfg.prefix)
    task = os.path.basename(cfg.project_task)
    exp_root = os.path.dirname(cfg.output_dir)
    output_dir = os_utils.make_output_dir(exp_root, task, run_name)
    os.makedirs(output_dir, exist_ok=True)
    cfg.output_dir = output_dir
    cfg.log_dir = output_dir

    wandb_name = os.path.basename(cfg.output_dir)

    print(f"Current working directory : {os.getcwd()}")
    print(f"Saving Directory          : {cfg.output_dir}")
    available_gpus = torch.cuda.device_count()
    print("available_gpus------------", available_gpus)
    # cfg.datamodule.batch_size = int(cfg.datamodule.total_gpu_batch_size / available_gpus)
    # print('Per GPU batchsize:', cfg.datamodule.batch_size)
    # time.sleep(1)
    #
    # cfg = option_parsing.post_process(cfg)

    # train the model
    metric_dict, _ = train(cfg)
    print(metric_dict)


if __name__ == "__main__":
    main()