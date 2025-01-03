from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor

def create_list_of_callbacks(path):
    lr_monitor = LearningRateMonitor(logging_interval='step')
    model_checkpoint = ModelCheckpoint(
        monitor='val_loss',
        dirpath=path,
        filename='cddpm-{epoch:02d}-{val_loss:.2f}',
        mode='min',)

    callbacks = [lr_monitor, model_checkpoint]
    return callbacks
