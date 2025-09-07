# The MNIST datasets are hosted on yann.lecun.com that has moved under CloudFlare protection
# Run this script to enable the datasets download
# Reference: https://github.com/pytorch/vision/issues/1938

import urllib.request
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

import torch
from torch import nn, optim   # <-- moved optim import up
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                              ])
# Download and load the training data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.003)

epochs = 5
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)
    
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

# (Optional) You were redefining optimizer here unnecessarily, removed to avoid confusion
# optimizer = optim.SGD(model.parameters(), lr=0.01)

# replacement for helper.view_classify
def view_classify(img, ps):
    img = img.view(28, 28)
    plt.figure(figsize=(6,3))
    ax1 = plt.subplot(1,2,1)
    ax1.imshow(img, cmap='viridis')
    ax1.axis('off')
    ax2 = plt.subplot(1,2,2)
    ax2.barh(range(10), ps.numpy().squeeze())
    ax2.set_aspect(0.1)
    ax2.set_yticks(range(10))
    ax2.set_yticklabels(range(10))
    ax2.set_xlim(0, 1.1)
    ax2.set_title('Class Probability')
    plt.show()

images, labels = next(iter(trainloader))
img = images[0].view(1, 784)

# Turn off gradients to speed up this part
with torch.no_grad():
    logps = model(img)

# Output of the network are log-probabilities, need to take exponential for probabilities
ps = torch.exp(logps)
view_classify(img.view(1, 28, 28), ps)
