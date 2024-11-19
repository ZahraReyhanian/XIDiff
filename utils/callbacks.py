from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor

def create_list_of_callbacks(path):
    lr_monitor = LearningRateMonitor(logging_interval='step')
    model_checkpoint = ModelCheckpoint(path)

    callbacks = [lr_monitor, model_checkpoint]
    return callbacks
