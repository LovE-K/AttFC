from typing import Iterable

# import mxnet as mx
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset.img_dataset import ImageDataset
from dataset.sampler import Sampler


def get_dataloader(
        batch_size,
        roots,
        anno_files,
        num_workers,
        sample_num,
        num_image
) -> Iterable:
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        )
    ])

    datasets = [ImageDataset(root=roots, anno_file=anno_files, img_count=num_image, transform=transform)]
    train_dataset = Sampler(datasets, p_datasets=[1.0],
                            k=sample_num,
                            sampling_base='image')

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    return train_loader
