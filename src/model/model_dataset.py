from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import jpegio as jio
from PIL import Image


class YCbCrImageDataset(Dataset):
    """
    Dataset für Bildmodelle (z. B. EfficientNet, TinyCNN).
    Lädt YCbCr-Bilddaten aus JPEG-Dateien und gibt (Tensor, Label) zurück.

    Args:
        dataframe (pd.DataFrame): Muss eine 'path'-Spalte und eine Labelspalte enthalten.
        transform (callable, optional): Bildtransform (z. B. ToTensor + Normalize).
        target_column (str): Spaltenname für Zielvariable (z. B. "label").
    """

    def __init__(self, dataframe, transform=None, target_column="label"):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.target_column = target_column

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        image = Image.open(row["path"]).convert("YCbCr")
        if self.transform is not None:
            image = self.transform(image)
        label = int(row[self.target_column])
        return image, label

class DCTCoefficientDataset(Dataset):
    """
    Dataset für Modelle im DCT-Raum (z. B. SRNet).
    Lädt DCT-Koeffizienten (ein Kanal) als Float32-Tensor [1, H, W].

    Args:
        dataframe (pd.DataFrame): Muss eine 'path'-Spalte und eine Labelspalte enthalten.
        channel (int): Farbkanal (0 = Y, 1 = Cb, 2 = Cr).
        transform (callable, optional): Optionaler Transform auf den DCT-Tensor.
        target_column (str): Spaltenname der Zielvariable (z. B. "label").
    """

    def __init__(self, dataframe, channel=0, transform=None, target_column="label"):
        self.df = dataframe.reset_index(drop=True)
        self.channel = channel
        self.transform = transform
        self.target_column = target_column

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        coef_array = jio.read(str(row["path"])).coef_arrays[self.channel].astype(np.float32)
        coef_tensor = torch.from_numpy(coef_array).unsqueeze(0)  # [1, H, W]
        if self.transform is not None:
            coef_tensor = self.transform(coef_tensor)
        label = int(row[self.target_column])
        return coef_tensor, label

    
class FusionDataset2(Dataset):
    """
    Dataset für Modelle mit kombiniertem Input:
    - Bild: Y-Kanal aus YCbCr-Farbraum (als Bildtransform)
    - DCT: DCT-Koeffizienten des Y-Kanals als Float32-Tensor [1, H, W]

    Args:
        dataframe (pd.DataFrame): Muss eine 'path'-Spalte und eine Labelspalte enthalten.
        transform (callable, optional): Optionaler Transform für den Y-Kanal (z. B. ToTensor).
        target_column (str): Spaltenname der Zielvariable (z. B. "label").
    """

    def __init__(self, dataframe, transform=None, target_column="label"):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.channel = 0
        self.target_column = target_column

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[tuple[torch.Tensor, torch.Tensor], int]:
        row = self.df.iloc[idx]

        # 1) Y-Kanal extrahieren
        y_image = Image.open(row["path"]).convert("YCbCr").split()[0]  # Nur Y-Kanal (PIL Image)
        if self.transform is not None:
            y_image = self.transform(y_image)  # [1, H, W] oder [3, H, W], je nach Transform

        # 2) DCT-Koeffizienten laden
        coef_array = jio.read(str(row["path"])).coef_arrays[self.channel].astype(np.float32)
        dct_tensor = torch.from_numpy(coef_array).unsqueeze(0)  # [1, H, W]

        label = int(row[self.target_column])
        return (y_image, dct_tensor), label
    

class FusionDataset4(Dataset):
    """
    Dataset mit kombiniertem Input für Steganalyse:
    - Bild: YCbCr-Bild als Float32-Tensor [3, H, W]
    - DCT: DCT-Y-Kanal als Float32-Tensor [1, H, W] (nur AC-Koeffizienten, DC = 0)

    Gibt ein Tupel ((Bild, DCT), Label) zurück.

    Args:
        dataframe (pd.DataFrame): Muss eine 'path'-Spalte und eine Labelspalte enthalten.
        transform (callable, optional): Optionaler Transform auf das YCbCr-Bild (z. B. ToTensor + Normalize).
        target_column (str): Spaltenname der Zielvariable (z. B. "label").
    """

    def __init__(self, dataframe, transform=None, target_column="label"):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.target_column = target_column

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[tuple[torch.Tensor, torch.Tensor], int]:
        row = self.df.iloc[idx]
        path = row["path"]

        # Bildraum
        image = Image.open(path).convert("YCbCr")
        if self.transform is not None:
            image = self.transform(image)  # [3, H, W]

        # DCT-Y (AC only)
        dct_y = jio.read(str(path)).coef_arrays[0].astype(np.float32)
        dct_y[::8, ::8] = 0
        dct_tensor = torch.from_numpy(dct_y).unsqueeze(0)  # [1, H, W]

        label = int(row[self.target_column])
        return (image, dct_tensor), label


class FusionDataset6(Dataset):
    """
    Dataset mit getrenntem Input für Steganalyse:
    - Bild: YCbCr-Bild als Float32-Tensor [3, H, W]
    - DCT: Drei z-standardisierte DCT-Kanäle (Y, Cb, Cr), jeweils AC-only [3, H, W]

    Gibt ein Tupel ((Bild, DCT), Label) zurück.

    Args:
        dataframe (pd.DataFrame): Muss eine 'path'-Spalte und eine Labelspalte enthalten.
        transform (callable, optional): Optionaler Transform auf das YCbCr-Bild (z. B. ToTensor + Normalize).
        target_column (str): Spaltenname der Zielvariable (z. B. "label").
    """

    def __init__(self, dataframe, transform=None, target_column="label"):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.target_column = target_column

    def __len__(self) -> int:
        return len(self.df)

    @staticmethod
    def _load_ac_tensor(jpeg_path: str, channel: int) -> torch.Tensor:
        coef = jio.read(jpeg_path).coef_arrays[channel].astype(np.float32)
        coef[::8, ::8] = 0  # DC-Koeffizienten nullen
        mu, std = coef.mean(), coef.std() + 1e-8
        coef = (coef - mu) / std  # z-Score-Normierung
        return torch.from_numpy(coef).unsqueeze(0)  # [1, H, W]

    def __getitem__(self, idx: int) -> tuple[tuple[torch.Tensor, torch.Tensor], int]:
        row = self.df.iloc[idx]
        path = row["path"]

        # 1) YCbCr-Bild
        image = Image.open(path).convert("YCbCr")
        if self.transform is not None:
            image = self.transform(image)  # [3, H, W]

        # 2) DCT-Kanäle (Y, Cb, Cr)
        dct_y  = self._load_ac_tensor(path, channel=0)
        dct_cb = self._load_ac_tensor(path, channel=1)
        dct_cr = self._load_ac_tensor(path, channel=2)
        dct_tensor = torch.cat([dct_y, dct_cb, dct_cr], dim=0)  # [3, H, W]

        label = int(row[self.target_column])
        return (image, dct_tensor), label
