
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

class DiamondDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.annotations = pd.read_csv(csv_file)
        print(len(self.annotations))
        image_exists = [os.path.exists(
            os.path.join(self.root_dir, str(int(id)), f'{int(id)}-photo0.jpg')
        ) for id in self.annotations['ID']]
        self.annotations = self.annotations[image_exists]
       
       
        # dropped = []
        # for i in self.annotations['ID']:
        #     i = int(i)
        #     id = str(int(i))
        #     img_path = os.path.join(self.root_dir, id, f'{id}-photo0.jpg')
        #     if not os.path.exists(img_path):
        #         dropped.append(i)
        # self.annotations.drop(dropped, axis=0, inplace=True)


        

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        id = str(int(self.annotations.iloc[index, 0]))
        img_path = os.path.join(self.root_dir, id, f'{id}-photo0.jpg')
        image = Image.open(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 11])).float() # without the float, this label becomes a LongTensor and screws things up when we do .backward()

        if self.transform:
            image = self.transform(image)

        return (image, y_label)


if __name__ == "__main__":
    diamondDataset = DiamondDataset("Diamond_Pricing_Dataset.csv", "images")
    diamondDataset[0]

