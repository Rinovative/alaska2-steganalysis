import random

import jpegio as jio
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class AlignedDeterministicCrop:
    """
    Führt einen deterministischen, block-alignierten Crop aus,
    dessen Position aus einem Hash des Bildinhalts abgeleitet wird.

    Args:
        size (int): Zielgröße (z. B. 256)
        block_size (int): Blockgröße (z. B. 8)
    """

    def __init__(self, size: int, block_size: int = 8):
        self.size = size
        self.block_size = block_size

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        x_max = (w - self.size) // self.block_size
        y_max = (h - self.size) // self.block_size

        img.load()
        img_bytes = img.tobytes()
        seed = int.from_bytes(img_bytes[:64], "little", signed=False) % (2**32)
        rnd = random.Random(seed)

        gx = rnd.randint(0, x_max) * self.block_size
        gy = rnd.randint(0, y_max) * self.block_size

        return img.crop((gx, gy, gx + self.size, gy + self.size))


class AlignedRandomCrop:
    """
    Führt einen deterministischen oder zufälligen, block-ausgerichteten Crop aus.

    Wenn `seed` gesetzt wird, ist der Crop reproduzierbar – z. B. nützlich in Validierung/Test,
    um unterschiedliche Bildbereiche deterministisch zu betrachten (z. B. am Rand bei UERD).

    Args:
        size (int): Zielgröße (z. B. 256)
        block_size (int): DCT-Blockgröße (z. B. 8)
        seed (int, optional): Optionaler Seed zur Reproduzierbarkeit
    """

    def __init__(self, size: int, block_size: int = 8, seed: int = None):
        self.size = size
        self.block_size = block_size
        self.seed = seed

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        assert w >= self.size and h >= self.size

        x_max = (w - self.size) // self.block_size
        y_max = (h - self.size) // self.block_size

        if self.seed is not None:
            rnd = random.Random(self.seed)
            gx = rnd.randint(0, x_max) * self.block_size
            gy = rnd.randint(0, y_max) * self.block_size
        else:
            gx = random.randint(0, x_max) * self.block_size
            gy = random.randint(0, y_max) * self.block_size

        return img.crop((gx, gy, gx + self.size, gy + self.size))


class RandomGridShuffle:
    """
    Zerteilt ein quadratisches Bild in ein Raster aus gleich großen Blöcken (grid_size × grid_size)
    und mischt diese zufällig neu an. Die Blockinhalte bleiben erhalten, nur ihre Position ändert sich.

    Dies zerstört den semantischen Bildinhalt, erhält aber die lokalen DCT-Strukturen – ideal zur
    Trennung von Bildinhalt und Stego-Signatur in JPEG-Steganalyse.

    Achtung: Bildbreite und -höhe müssen exakt durch grid_size teilbar sein.

    Args:
        grid_size (int): Anzahl der Blöcke pro Bildseite (z. B. 8 → 8×8 Raster).
    """

    def __init__(self, grid_size=8):
        self.grid_size = grid_size

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w % self.grid_size != 0 or h % self.grid_size != 0:
            raise ValueError(f"Bildgrösse muss durch {self.grid_size} teilbar sein")

        bw, bh = w // self.grid_size, h // self.grid_size
        blocks = []
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                box = (x * bw, y * bh, (x + 1) * bw, (y + 1) * bh)
                blocks.append(img.crop(box))

        random.shuffle(blocks)

        new_img = Image.new(img.mode, (w, h))
        for i, block in enumerate(blocks):
            x = (i % self.grid_size) * bw
            y = (i // self.grid_size) * bh
            new_img.paste(block, (x, y))

        return new_img


class RGBImageDataset(Dataset):
    """
    Dataset für Bildmodelle (z. B. EfficientNet, TinyCNN).
    Lädt **RGB**-Bilddaten aus JPEG-Dateien und gibt (Tensor, Label) zurück.

    Args:
        dataframe (pd.DataFrame):
            Muss eine 'path'-Spalte und eine Label-Spalte enthalten.
        transform (callable, optional):
            Bild-Transform (z. B. `ToTensor` + `Normalize`).
        target_column (str):
            Spaltenname der Zielvariable (z. B. "label").
    """

    def __init__(self, dataframe, transform=None, target_column: str = "label"):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.target_column = target_column

        # Label-Dtype automatisch wählen
        dtype_kind = self.df[self.target_column].dtype.kind
        if dtype_kind == "f":  # float → binär (BCEWithLogitsLoss)
            self.label_dtype = torch.float32
        elif dtype_kind in {"i", "u"}:  # int/uint → (multi-)klassig (CrossEntropy)
            self.label_dtype = torch.long
        else:
            raise ValueError(f"Unbekannter Datentyp für Labels: {self.df[self.target_column].dtype}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]

        # 1) Bild in **RGB** lesen
        image = Image.open(row["path"]).convert("RGB")

        # 2) Optionalen Transform anwenden
        if self.transform:
            image = self.transform(image)

        # 3) Label in passenden Tensor casten
        label = torch.tensor(row[self.target_column], dtype=self.label_dtype)

        return image, label


class YChannelDataset(Dataset):
    """
    Dataset für Steganalyse-Modelle, die nur den Y-Kanal (Luminanz) aus
    JPEG-Bildern nutzen.

    Gibt (Tensor, Label) zurück, wobei der Tensor die Form [1, H, W] hat.

    Args:
        dataframe (pd.DataFrame): Muss eine 'path'-Spalte und eine Labelspalte enthalten.
        transform (callable, optional): Optionaler Transform auf den Y-Kanal
        target_column (str): Spaltenname der Zielvariable (z. B. "label").
    """

    def __init__(self, dataframe, transform=None, target_column: str = "label"):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.target_column = target_column

        dtype_kind = self.df[self.target_column].dtype.kind
        if dtype_kind == "f":  # float → binär
            self.label_dtype = torch.float32
        elif dtype_kind in {"i", "u"}:  # int oder uint → multiclass
            self.label_dtype = torch.long
        else:
            raise ValueError(f"Unbekannter Datentyp für Labels: {self.df[self.target_column].dtype}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        y_image = Image.open(row["path"]).convert("YCbCr").split()[0]
        y_tensor = self.transform(y_image)
        label = torch.tensor(row[self.target_column], dtype=self.label_dtype)
        return y_tensor, label


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
        dtype_kind = self.df[self.target_column].dtype.kind
        self.label_dtype = torch.float32 if dtype_kind == "f" else torch.long

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["path"]).convert("YCbCr")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(row[self.target_column], dtype=self.label_dtype)
        return image, label


class DCTCoefficientDataset(Dataset):
    """
    Dataset für Modelle im DCT-Raum.
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
        dtype_kind = self.df[self.target_column].dtype.kind
        self.label_dtype = torch.float32 if dtype_kind == "f" else torch.long

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        coef_array = jio.read(str(row["path"])).coef_arrays[self.channel].astype(np.float32)
        coef_tensor = torch.from_numpy(coef_array).unsqueeze(0)
        if self.transform:
            coef_tensor = self.transform(coef_tensor)
        label = torch.tensor(row[self.target_column], dtype=self.label_dtype)
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
        dtype_kind = self.df[self.target_column].dtype.kind
        self.label_dtype = torch.float32 if dtype_kind == "f" else torch.long

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        y_image = Image.open(row["path"]).convert("YCbCr").split()[0]
        if self.transform:
            y_image = self.transform(y_image)
        coef_array = jio.read(str(row["path"])).coef_arrays[self.channel].astype(np.float32)
        dct_tensor = torch.from_numpy(coef_array).unsqueeze(0)
        label = torch.tensor(row[self.target_column], dtype=self.label_dtype)
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
        dtype_kind = self.df[self.target_column].dtype.kind
        self.label_dtype = torch.float32 if dtype_kind == "f" else torch.long

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["path"]
        image = Image.open(path).convert("YCbCr")
        if self.transform:
            image = self.transform(image)
        dct_y = jio.read(str(path)).coef_arrays[0].astype(np.float32)
        dct_y[::8, ::8] = 0
        dct_tensor = torch.from_numpy(dct_y).unsqueeze(0)
        label = torch.tensor(row[self.target_column], dtype=self.label_dtype)
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
        dtype_kind = self.df[self.target_column].dtype.kind
        self.label_dtype = torch.float32 if dtype_kind == "f" else torch.long

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _load_ac_tensor(jpeg_path: str, channel: int) -> torch.Tensor:
        coef = jio.read(jpeg_path).coef_arrays[channel].astype(np.float32)
        coef[::8, ::8] = 0
        mu, std = coef.mean(), coef.std() + 1e-8
        coef = (coef - mu) / std
        return torch.from_numpy(coef).unsqueeze(0)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["path"]
        image = Image.open(path).convert("YCbCr")
        if self.transform:
            image = self.transform(image)
        dct_y = self._load_ac_tensor(path, 0)
        dct_cb = self._load_ac_tensor(path, 1)
        dct_cr = self._load_ac_tensor(path, 2)
        dct_tensor = torch.cat([dct_y, dct_cb, dct_cr], dim=0)
        label = torch.tensor(row[self.target_column], dtype=self.label_dtype)
        return (image, dct_tensor), label
