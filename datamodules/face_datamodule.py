from typing import Optional
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision import datasets, transforms

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
            transform = transforms.Compose([transforms.Resize(self.img_size),
                                            transforms.ToTensor()])

            # load dataset
            self.data_train = datasets.ImageFolder(f'{self.dataset_path}train', transform=transform)
            self.data_val = datasets.ImageFolder(f'{self.dataset_path}test', transform=transform)

            print(self.data_train)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=11
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=11
        )

    # def test_dataloader(self):
    #     return DataLoader(
    #         dataset=self.data_test,
    #         batch_size=self.batch_size,
    #         shuffle=False,
    #     )