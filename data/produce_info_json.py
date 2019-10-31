import torch
from torchvision import datasets, transforms
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json


class MyDataset(Dataset):
    def __init__(self):
        self.list_ofimages_path=pd.read_csv('train_images.txt', header=None).values
    def __getitem__(self, index):
        filepath = self.list_ofimages_path[index][0]
        im = np.asarray(Image.open(filepath)).astype(np.float32)

        return im

    def __len__(self):
        return len(self.list_ofimages_path)
    

dataset = MyDataset()
print(len(dataset))

loader = DataLoader(
    dataset,
    batch_size=8,
    num_workers=4,
    shuffle=False
)


mean = 0.
std = 0.
nb_samples = 0.
for data in loader:
    batch_samples = data.size(0)
    mean += data.mean((0,1,2))
    std += data.std((0,1,2))
    nb_samples += batch_samples
    progress=100*nb_samples//len(dataset)
    if progress%5==0 and progress>0:
      print(progress)

mean /= nb_samples
std /= nb_samples

info ={"std": list(std.numpy()), "mean": list(mean.numpy())}
with open('info.json', 'w') as f:
    json.dump(str(info), f)
