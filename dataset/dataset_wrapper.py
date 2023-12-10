import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data.sampler import SubsetRandomSampler
from torch_geometric.data import DataLoader


class MolDataModule(LightningDataModule):
    def __init__(self, batch_size, data_path, aug='node', num_workers=12, valid_size=0.05):
        super().__init__()
        self.aug = aug
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.data_path = data_path

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            if self.aug == 'node':
                from .dataset import MolNodeAugDataset
                ds = MolNodeAugDataset
            elif self.aug == 'subgraph':
                from .dataset_subgraph import MolSubGraphAugDataset
                ds = MolSubGraphAugDataset
            elif self.aug == 'mix':
                from .dataset_mix import MolMixAugDataset
                ds = MolMixAugDataset
            self.train_dataset = ds(self.data_path)
            self.train_loader, self.valid_loader = self.get_train_validation_data_loaders(self.train_dataset)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.valid_loader

    def get_train_validation_data_loaders(self, train_dataset):
        # obtain training indices that will be used for validation
        num_train = len(train_dataset)
        indices = list(range(num_train))
        np.random.shuffle(indices)

        split = int(np.floor(self.valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]

        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler,
                                  num_workers=self.num_workers, drop_last=True)
        valid_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
                                  num_workers=self.num_workers, drop_last=True)

        return train_loader, valid_loader
