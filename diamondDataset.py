import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io

class DiamondDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        id = str(int(self.annotations.iloc[index, 0]))
        img_path = os.path.join(self.root_dir, id, f'{id}-photo0.jpg')
        image = io.imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 11]))

        if self.transform:
            image = self.transform(image)

        return (image, y_label)


if __name__ == "__main__":
    diamondDataset = DiamondDataset("Diamond_Pricing_Dataset.csv", "images")
    diamondDataset[0]