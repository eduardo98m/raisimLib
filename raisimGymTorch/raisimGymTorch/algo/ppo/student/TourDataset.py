import os
import numpy as np

from typing import Dict, Any
from torch.utils.data import Dataset, DataLoader

class TourDataset(Dataset):
    """
        Dataset para cargar datos de entrenamiento.
    """
    def __init__(self, data_dir: str, history_len: int=60):
        # Verificamos que el directorio existe y obtenemos los archivos que 
        # contiene.
        assert os.path.exists(data_dir)
        self.data_dir = data_dir
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

        item['obs'] = np.zeros((self.history_len, data['obs'].shape[1]))
        begin_obs = self.history_len-1 - min(self.history_len-1, index_step)
        begin_step = max(0, index_step - self.history_len + 1)
        item['obs'][begin_obs:] = data['obs'][begin_step: index_step+1]
        item['label'] = data['labels'][index_step]

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

