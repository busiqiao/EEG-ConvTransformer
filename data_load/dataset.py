import pickle

import numpy as np
import torch
from torch.utils.data.dataset import Dataset


class EEGImagesDataset(Dataset):
    """EEGLearn Images Dataset from EEG."""

    def __init__(self, file_path, num_class=6, s=1):
        self.file_path = file_path
        self.num_class = num_class

        with open(self.file_path + f'S{s+1}.pkl', 'rb') as file:
            data, y = pickle.load(file)
            data = np.array(data)  # Convert list of numpy arrays to a single numpy array
            data = torch.tensor(data, dtype=torch.float32)
            self.data = data.unsqueeze(1)
            if self.num_class == 6:
                labels = [item[0] for item in y]
                self.labels = torch.tensor(labels).long()
            elif self.num_class == 72:
                labels = [item[1] for item in y]
                self.labels = torch.tensor(labels).long()
            else:
                raise ValueError('num_class must be 6 or 72')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
