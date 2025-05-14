from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor

def create_list_of_callbacks(path):
    lr_monitor = LearningRateMonitor(logging_interval='step')
    model_checkpoint = ModelCheckpoint(
        dirpath=path,
        filename= "epoch_{epoch:03d}",
        monitor= "val/mse_loss",
        save_last= True,
        every_n_epochs=5,
        save_top_k=1,
        mode='min',
        save_weights_only= False,)

    callbacks = [lr_monitor, model_checkpoint]
    return callbacks
