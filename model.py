from collections import OrderedDict

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision.models as models


# Path to data files - in this tutorial I use PyTorch's generic loaders
# Data is arranged like this root/train_or_test_folder/class_folder
# If you'd liek to structured your daat difrrently, rewrite this part

data_dir = "../brazilian_vs_domestic/data/"
train_dir = data_dir + "train/"
test_dir = data_dir + "test/"


# ImageNet means and standard deviations for each color channel.
# We need this to normalize input images for our pretrained network.

channel_means = [0.485, 0.456, 0.406]
channel_sds = [0.229, 0.224, 0.225]

# Load, transform and normalize the data
train_transforms = transforms.Compose(
    [
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(channel_means, channel_sds),
    ]
)

test_transforms = transforms.Compose(
    [
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(channel_means, channel_sds),
    ]
)

train_data = datasets.ImageFolder(train_dir, train_transforms)
test_data = datasets.ImageFolder(test_dir, test_transforms)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32)


# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Download pretrained model - I use VGG19 architecture, but you can try other ones
vgg19 = models.vgg19(pretrained=True)


def build_classifier(model):

    for param in model.parameters():
        param.requires_grad = False

    # We only add this on top of pre-trained network
    # First dimension in fc1 comes pretrained network of our choice
    classifier = nn.Sequential(
        OrderedDict(
            [
                ("fc1", nn.Linear(25088, 500)),
                ("relu", nn.ReLU()),
                ("fc2", nn.Linear(500, 2)),
                ("output", nn.LogSoftmax(dim=1)),
            ]
        )
    )

    model.classifier = classifier

    return model


# Build it...
model = build_classifier(vgg19)

model.to(device)

criterion = nn.NLLLoss()

optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)


def train_model(epochs):
    steps = 0
    running_loss = 0
    print_every_n_updates = 5

    for epoch in range(epochs):
        for inputs, labels in train_loader:
            steps += 1
            # Send inputs and labels to selected device
            inputs, labels = (inputs.to(device), labels.to(device))

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every_n_updates == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in test_loader:
                        inputs, labels = (inputs.to(device), labels.to(device))
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        # Compute metric - accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(
                    f"Epoch {epoch+1}/{epochs}.. "
                    f"Train loss: {running_loss/print_every_n_updates:.3f}.. "
                    f"Test loss: {test_loss/len(test_loader):.3f}.. "
                    f"Test accuracy: {accuracy/len(test_loader):.3f}"
                )
                running_loss = 0
                model.train()


# ...and train it
train_model(epochs=5)
