import torch
import tempfile
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from diamondDataset import DiamondDataset
    # Training dataset as torch.utils.data.Dataset instance
train_dataset = DiamondDataset(
    'Diamond_Pricing_Dataset.csv', 'images',  # where data is stored
    torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize(256,256)
    ])  # how each sample will be transformed
)
    
print(len(train_dataset))

train_dataset, validation_dataset, test_dataset= torch.utils.data.random_split(
    train_dataset, [1557 , 444, 222]
)

BATCH_SIZE = 64

image, price = train_dataset[0]
print(image.shape)
image = image.transpose(0,2)
image = image.transpose(0,1)
print(image.shape)
# plt.imshow(image)
# plt.show()

dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
for batch in dataloader:
    print(batch)