from typing import Optional
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision import datasets, transforms
from datamodules.wrapper_dataset import WrapperDataset

class FaceDataModule(LightningDataModule):

    def __init__(
            self,
            dataset_path,
            img_size=(48, 48),
            seed=42,
            batch_size=32,
            transforms_setting=None
    ):
        super().__init__()

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.dataset_path = dataset_path
        self.img_size = img_size
        self.seed = seed
        self.batch_size = batch_size
        if transforms_setting is None:
            self.transform = transforms.Compose([transforms.Resize(self.img_size),
                                            transforms.ToTensor()])
        else:
            self.transform = transforms_setting

    @property
    def num_classes(self):
        return len(self.data_train.class_names)

    def setup(self, stage: Optional[str] = None):
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:


            # load dataset
            train_base = datasets.ImageFolder(f'{self.dataset_path}train', transform=self.transform)
            val_base = datasets.ImageFolder(f'{self.dataset_path}valid', transform=self.transform)
            test_base = datasets.ImageFolder(f'{self.dataset_path}test', transform=self.transform)

            self.data_train = WrapperDataset(train_base)
            self.data_val = WrapperDataset(val_base)
            self.data_test = WrapperDataset(test_base)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=11,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=11,
            drop_last=True
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True
        )