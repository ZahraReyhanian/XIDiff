from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor

def create_list_of_callbacks(path):
    lr_monitor = LearningRateMonitor(logging_interval='step')
    model_checkpoint = ModelCheckpoint(
        dirpath=path,
        filename= "{epoch}-{step}",
        monitor= "train/total_loss",
        every_n_train_steps=10000,
        save_top_k=-1,
        mode='min',
        save_last=True,
        save_weights_only= False,)

    callbacks = [lr_monitor, model_checkpoint]
    return callbacks
