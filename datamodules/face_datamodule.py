from typing import Optional
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from keras.utils import image_dataset_from_directory

class FaceDataModule(LightningDataModule):

    def __init__(
            self,
            dataset_path,
            img_size=(48, 48),
            seed=42,
            batch_size=32
    ):
        super().__init__()

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.dataset_path = dataset_path
        self.img_size = img_size
        self.seed = seed
        self.batch_size = batch_size

    @property
    def num_classes(self):
        return len(self.data_train.class_names)

    def setup(self, stage: Optional[str] = None):
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = image_dataset_from_directory(self.dataset_path+'/train',
                                                           labels="inferred",
                                                           label_mode="categorical",
                                                           color_mode="rgb",
                                                           image_size=self.img_size,
                                                           seed=self.seed,
                                                           batch_size=self.batch_size,
                                                           shuffle=True
                                                        )
            self.data_val = image_dataset_from_directory(self.dataset_path+'/valid',
                                                         labels="inferred",
                                                         label_mode="categorical",
                                                         color_mode="rgb",
                                                         image_size=self.img_size,
                                                         seed=self.seed,
                                                         batch_size=self.batch_size,
                                                         shuffle=False)
            self.data_test = image_dataset_from_directory(self.dataset_path+'/test',
                                                          labels="inferred",
                                                          label_mode="categorical",
                                                          color_mode="rgb",
                                                          image_size=self.img_size,
                                                          seed=self.seed,
                                                          batch_size=self.batch_size,
                                                          shuffle=False)
            print('train data:', len(self.data_train))
            print('val data:', len(self.data_val))
            print('test_data:', len(self.data_test))


    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            shuffle=False,
        )