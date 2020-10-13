import torch
from torch.utils.data import DataLoader, Dataset
import torchvision


def sample_data_loader():
	dataset = Dataset()
	return DataLoader(dataset)


class GAN_Trainer:
	def __init__(self):
		self.generator = None
		self.discriminator = None
		
		self.epochs = 1
		self.batch_size = 32
		self.image_size = (480, 480)
		
		self.data_loader = sample_data_loader()
