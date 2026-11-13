import os
import numpy as np
import torch

from torch.utils.data import Dataset
from tqdm import tqdm


class SeismicDataset3D(Dataset):

    def __init__(
        self,
        seis_dir,
        fault_dir,
        shape=(128, 128, 128),
        seis_dtype=np.float32,
        fault_dtype=np.float32,
        augment=False
    ):

        self.seis_dir = seis_dir
        self.fault_dir = fault_dir
        self.shape = tuple(shape)

        self.seis_dtype = seis_dtype
        self.fault_dtype = fault_dtype

        self.augment = augment

        if not os.path.isdir(seis_dir):
            raise FileNotFoundError(seis_dir)

        if not os.path.isdir(fault_dir):
            raise FileNotFoundError(fault_dir)

        self.file_list = sorted([
            f for f in os.listdir(seis_dir)
            if f.endswith('.dat')
            and os.path.isfile(
                os.path.join(fault_dir, f)
            )
        ])

        if len(self.file_list) == 0:
            raise RuntimeError(
                "No paired .dat files found."
            )

        self.data_cache = []

        expected_size = int(np.prod(self.shape))

        print(
            f"Pre-loading {len(self.file_list)} samples..."
        )

        for filename in tqdm(self.file_list):

            seis_path = os.path.join(
                self.seis_dir,
                filename
            )

            fault_path = os.path.join(
                self.fault_dir,
                filename
            )

            s_np = np.fromfile(
                seis_path,
                dtype=self.seis_dtype
            )

            f_np = np.fromfile(
                fault_path,
                dtype=self.fault_dtype
            )

            if (
                s_np.size != expected_size
                or f_np.size != expected_size
            ):
                raise ValueError(
                    f"Invalid volume size: {filename}"
                )

            s_np = s_np.reshape(self.shape)

            f_np = f_np.reshape(self.shape)

            # Z-score normalization
            mean = s_np.mean()

            std = s_np.std() + 1e-6

            s_np = (s_np - mean) / std

            # Binary fault labels
            f_np = (f_np > 0.5).astype(
                np.float32
            )

            s_tensor = torch.from_numpy(
                s_np.copy()
            ).unsqueeze(0).float()

            f_tensor = torch.from_numpy(
                f_np.copy()
            ).unsqueeze(0).float()

            self.data_cache.append(
                (s_tensor, f_tensor)
            )

        print(
            f"Loaded {len(self.data_cache)} samples."
        )

    def __len__(self):

        return len(self.data_cache)

    def __getitem__(self, idx):

        s_tensor = self.data_cache[idx][0].clone()

        f_tensor = self.data_cache[idx][1].clone()

        if self.augment:

            # D direction
            if torch.rand(1).item() > 0.5:

                s_tensor = torch.flip(
                    s_tensor,
                    dims=[1]
                )

                f_tensor = torch.flip(
                    f_tensor,
                    dims=[1]
                )

            # H direction
            if torch.rand(1).item() > 0.5:

                s_tensor = torch.flip(
                    s_tensor,
                    dims=[2]
                )

                f_tensor = torch.flip(
                    f_tensor,
                    dims=[2]
                )

            # W direction
            if torch.rand(1).item() > 0.5:

                s_tensor = torch.flip(
                    s_tensor,
                    dims=[3]
                )

                f_tensor = torch.flip(
                    f_tensor,
                    dims=[3]
                )

        return s_tensor, f_tensor
