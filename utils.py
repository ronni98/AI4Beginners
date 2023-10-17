import torch
from torchvision import datasets, transforms


def get_mnist_data(datafolder="./data"):
	"""Retrieve the MNIST dataset. If the data do not exist, download them from the Internet.
	"""

	return  datasets.MNIST(
		root = datafolder,
		train = True,
		download = True,
		transform = transforms.ToTensor() # turn images to PyTorch tensors
	)
