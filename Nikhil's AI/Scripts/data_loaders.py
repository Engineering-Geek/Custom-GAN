from torch.utils.data import DataLoader, random_split
from torchvision import datasets as dset
from torchvision import transforms


def custom_dataloader():
	dataset = dset.ImageFolder(
		root=root,
		transform=transforms.Compose([
			transforms.Resize(img_size),
			transforms.CenterCrop(img_size),
			transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		])
	)
	total_length = len(dataset)
	training_length = int(total_length * train_ratio)
	validation_length = int((total_length - training_length) * val_test_ratio)
	test_length = int((total_length - training_length) * (1 - val_test_ratio))
	training_length += total_length - training_length - validation_length - test_length
	print("Creating datasets")
	training_dataset, validation_dataset, testing_dataset = torch.utils.data.random_split(
		dataset, [training_length, validation_length, test_length])
	print("Datasets created")
	training_loader = DataLoader(
		training_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)
	validation_loader = DataLoader(
		validation_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)
	testing_loader = DataLoader(
		testing_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)

