import torch
import tempfile
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from diamondDataset import DiamondDataset
from model import Model
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

    # Training dataset as torch.utils.data.Dataset instance
train_dataset = DiamondDataset(
    'Diamond_Pricing_Dataset.csv', 'images',  # where data is stored
    torchvision.transforms.Compose([
        torchvision.transforms.Resize((64,64)),
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

dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
model = Model()
adam = torch.optim.Adam(model.parameters(), lr=0.1)
sgd = torch.optim.SGD(model.parameters(), lr=1e-07, momentum=0.9)
writer = SummaryWriter()
lrs = [1e-5, 1e-6, 1e-7, 1e-8]
SGD_Loss = []
step = 0
num_epochs=10
for lr in lrs:
    model = Model()
    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9)
    for epoch in range(num_epochs):
        for batch in dataloader:
            features, labels = batch
            labels = labels.reshape(labels.shape[0], 1)
            predictions = model(features)
            loss = F.mse_loss(predictions, labels)
            SGD_Loss.append(loss)
            print(f'epoch = {epoch}, loss = {loss}')
            writer.add_scalar('loss', loss.item(), global_step=step)
            step += 1
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        
    