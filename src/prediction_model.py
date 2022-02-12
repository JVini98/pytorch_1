from pyexpat import model
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
from neuralNetwork import NeuralNetwork
from random import randrange

model = NeuralNetwork()
model.load_state_dict(torch.load("data/model.pth"))

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()

for i in range(10):

    index = randrange(10000)

    x, y = test_data[index][0], test_data[index][1]

    with torch.no_grad():
        pred = model(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')

    img, label = test_data[index]

    plt.title(f'Predicted: "{predicted}", Actual: "{actual}"')
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
    plt.show()
