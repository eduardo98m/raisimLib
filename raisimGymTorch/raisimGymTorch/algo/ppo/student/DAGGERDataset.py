import os
import numpy as np

from typing import Dict, Any
from torch.utils.data import Dataset, DataLoader
import torch

class DAGGERDataset(Dataset):
    """
        Dataset para cargar datos de entrenamiento.
    """
    def __init__(self, data_dir: str, begin_H: int, end_H: int, history_len: int=60):
        # Verificamos que el directorio existe y obtenemos los archivos que 
        # contiene.
        assert os.path.exists(data_dir)
        self.data_dir = data_dir
        self.begin_H = begin_H 
        self.end_H = end_H
        self.files = [f for f in os.listdir(data_dir) if f[-4:] == '.npz']

        # Cargamos un archivo para saber el numero de pasos
        self.n_steps = np.load(f'{self.data_dir}/{self.files[0]}')['obs'].shape[0]
        self.history_len = history_len

    def __len__(self) -> int: 
        return len(self.files) * self.n_steps
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        index_file = index // self.n_steps
        index_step = index % self.n_steps

        item = {'file': self.files[index_file]}
        data = np.load(f'{self.data_dir}/{self.files[index_file]}')
        item['H'] = np.zeros((self.history_len, self.end_H - self.begin_H ))
        begin_obs = self.history_len - 1 - min(self.history_len-1, index_step)
        begin_step = max(0, index_step - self.history_len + 1)
        item['obs'] = data['obs'][index_step]
        item['H'][begin_obs:] = data['obs'][begin_step: index_step+1, self.begin_H: self.end_H]
        item['label'] = data['labels'][index_step]
        item['action'] = data['actions'][index_step]

        return item

    def dataloader(self, **args) -> DataLoader:
        """
            Retorna un dataloader del dataset actual. Para mas informacion de
            los argumentos revise la documentacion de torch.utils.data.DataLoader

            Return:
            -------
                 * torch.utils.data.DataLoader: Dataloader que corresponde a este
                    dataset.
        """
        return DataLoader(self, **args)

