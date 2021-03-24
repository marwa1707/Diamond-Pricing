import torch
import tempfile
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from diamondDataset import DiamondDataset
from model import Model
import torch.nn.functional as F
    # Training dataset as torch.utils.data.Dataset instance
train_dataset = DiamondDataset(
    'Diamond_Pricing_Dataset.csv', 'images',  # where data is stored
    torchvision.transforms.Compose([
        torchvision.transforms.Resize((256,256)),
        torchvision.transforms.ToTensor(),
        
    ])  # how each sample will be transformed
)
    
print(len(train_dataset))

train_dataset, validation_dataset, test_dataset= torch.utils.data.random_split(
    train_dataset, [1380 , 395, 195]
)


image, price = train_dataset[0]
# print(image.shape)
image = image.transpose(0,2)
image = image.transpose(0,1)
# print(image.shape)
# plt.imshow(image)
# plt.show()

dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
model = Model()
for batch in dataloader:
    features, labels = batch
    labels = labels.reshape(labels.shape[0], 1)
    predictions = model(features)
    loss = F.mse_loss(predictions, labels)
    loss.backward()
    break
    