from torch.utils.data import Dataset
import numpy as np
import torch
import os
import lightning as L
from torch.utils.data import DataLoader

class EMGDataset(Dataset):

    def __init__(self, root_dir):
        super(EMGDataset, self).__init__()
        self.emg = np.load(os.path.join(root_dir, 'emg_train.npy'))
        self.force = np.load(os.path.join(root_dir, 'force_train.npy'))
        self.force_class = np.load(os.path.join(root_dir, 'force_class_train.npy'))

    def __len__(self):
        return len(self.emg)

    def __getitem__(self, idx):
        emg = self.emg[idx]
        force = self.force[idx]
        force_class = self.force_class[idx]
        return torch.from_numpy(emg), torch.from_numpy(force), torch.from_numpy(force_class)



class EMGDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "./dataset", batch_size: int = 64):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def prepare_data(self):
        pass
    
    def prepare_data_per_node(self):
        pass

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = EMGDataset(os.path.join(os.getcwd(), 'Dataset'))
        if stage == "valid":
            self.val_dataset = EMGDataset(os.path.join(os.getcwd(), 'Dataset'))


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.batch_size, num_workers=8, persistent_workers=True, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.batch_size, num_workers=8, persistent_workers=True, shuffle=True, pin_memory=True)

