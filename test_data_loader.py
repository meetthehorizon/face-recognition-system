import torch
import random
import matplotlib.pyplot as plt	
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader
from src.data.data_loader import DigiFace

def test_dataloader(path='data/raw', transform=None, shuffle=True):
	"""Test dataloader.
	
	Parameters
	----------
	path : str
		Path to images directory
	transform : callable
		Transformation to apply to image and mask
	batch_size : int
		Number of samples per batch
	shuffle : bool
		Whether to shuffle samples"""
	
	dataset = DigiFace(path=path, transform=transform)
	dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
	batch = dataloader.__iter__().__next__()
	
	fig, axes = plt.subplots(4, 8, figsize=(12, 6))
	axes = axes.flatten()
	for i, (image, identity) in enumerate(zip(*batch)):
		image = image.permute(1, 2, 0)
		axes[i].imshow(image)
		axes[i].set_title(identity.item())
		axes[i].axis('off')

	print('DigiFace dataset')
	print('Number of samples:', len(dataset))
	print('Image Shape: ', dataset[0][0].shape)
	print('Number of identities:', len(dataset.identities))
	print('Number of batches:', len(dataloader))
	print('Batch size:', dataloader.batch_size)
	
	plt.tight_layout()
	plt.show()
		
if __name__ == '__main__':
	print('Testing DigiFace')
	test_dataloader(path='data/raw', transform=transforms.PILToTensor(), shuffle=True)
	print('Passed')