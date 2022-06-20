import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.models import resnet18

from transforms import GaussianBlur
from datasets import GOES17Dataset
from models import SimSiam
from layers import SymmetrizedNegativeCosineSimilarity

from typing import List, Tuple
from tqdm import tqdm


# Set up data augmentation
def _get_normalize(normalized: bool = True) -> Tuple[List[float], List[float]]:
    if normalized:
        return [0.4888, 0.5074, 0.5162], [0.4247, 0.4173, 0.4087]
    return [0., 0., 0.], [1., 1., 1.]


# Transformation based on SimCLR
def get_train_transform(normalized: bool = True) -> T.Compose:
    return T.Compose([
        T.RandomResizedCrop(224, scale=(0.2, 1.)),
        T.RandomApply([
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4)
        ], p=0.1),
        T.RandomGrayscale(p=0.2),
        T.RandomApply([
            GaussianBlur([0.1, 2.0])
        ], p=0.5),
        # comment out since it's rectangular image
        # T.RandomVerticalFlip(0.5),
        T.RandomHorizontalFlip(0.5),
        T.ToTensor(),
        T.Normalize(*_get_normalize(normalized))
    ])


def get_test_transform(normalized: bool = True) -> T.Compose:
    return T.Compose([
        T.ToTensor(),
        T.Normalize(*_get_normalize(normalized))
    ])


# Set up DataLoader
def goes17_dataloader(img_path: List,
                      transform: T.Compose,
                      batch_size: int = 16,
                      num_workers: int = 8) -> DataLoader:
    dataset = GOES17Dataset(img_path=img_path, transform=transform)

    return DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=True,
                      num_workers=num_workers)


# Set up training

# Normalized std
def _get_normalized_std(z: torch.Tensor) -> float:
    with torch.no_grad():
        z_normalized = F.normalize(z, dim=1)
        z_std = torch.std(z_normalized, dim=1)
        mean_std = z_std.mean()

    return mean_std


# Training epoch iteration
def _train_one_epoch(dataloader: DataLoader,
                     model: nn.Module,
                     optimizer: optim.Optimizer,
                     loss_fn: nn.Module = SymmetrizedNegativeCosineSimilarity,
                     device: str = "mps") -> Tuple[float, float, float]:
    model = model.train()

    total_loss = 0.0
    total_std1 = 0.0
    total_std2 = 0.0

    data_point = len(dataloader.dataset)

    for x1, x2 in tqdm(dataloader, total=data_point):
        optimizer.zero_grad()

        x1 = x1.to(device)
        x2 = x2.to(device)

        z1, p1 = model(x1)
        z2, p2 = model(x2)

        loss = loss_fn(p1, z1, p2, z2)
        loss.backward()

        optimizer.step()

        total_loss += len(x1) * loss.item()
        total_std1 += len(x1) * _get_normalized_std(z1)
        total_std1 += len(x1) * _get_normalized_std(z2)

        total_loss /= data_point
        total_std1 /= data_point
        total_std2 /= data_point

        return total_loss, total_std1, total_std2


def _val_one_epoch(dataloader: DataLoader,
                   model: nn.Module,
                   loss_fn: nn.Module = SymmetrizedNegativeCosineSimilarity,
                   device: str = "mps") -> Tuple[float, float, float]:
    model = model.train()

    total_loss = 0.0
    total_std1 = 0.0
    total_std2 = 0.0

    data_point = len(dataloader.dataset)

    with torch.no_grad():
        for x1, x2 in tqdm(dataloader, total=data_point):
            x1 = x1.to(device)
            x2 = x2.to(device)

            z1, p1 = model(x1)
            z2, p2 = model(x2)

            loss = loss_fn(p1, z1, p2, z2)

            total_loss += len(x1) * loss.item()
            total_std1 += len(x1) * _get_normalized_std(z1)
            total_std1 += len(x1) * _get_normalized_std(z2)

            total_loss /= data_point
            total_std1 /= data_point
            total_std2 /= data_point

        return total_loss, total_std1, total_std2


def _train(train_dataloader: DataLoader,
           val_dataloader: DataLoader,
           model: nn.Module,
           optimizer: optim.Optimizer,
           epochs: int = 20,
           device: str = "mps"):
    model = model.to(device)
    loss_fn = SymmetrizedNegativeCosineSimilarity()

    for epoch in tqdm(range(epochs)):
        print(f'Epoch {epoch}')

        total_loss, total_std1, total_std2 = _train_one_epoch(dataloader=train_dataloader,
                                                              model=model,
                                                              loss_fn=loss_fn,
                                                              optimizer=optimizer)

        print(f'Training loss: {total_loss}')
        print(f'Training STD1: {total_std1}')
        print(f'Training STD2: {total_std2}')

        total_loss, total_std1, total_std2 = _val_one_epoch(dataloader=val_dataloader,
                                                            model=model,
                                                            loss_fn=loss_fn)

        print(f'Validation loss: {total_loss}')
        print(f'Validation STD1: {total_std1}')
        print(f'Validation STD2: {total_std2}')


if __name__ == "__main__":
    init_lr = 0.05
    batch_size = 128
    num_workers = 4
    epochs = 50

    lr = init_lr * batch_size / 256
    momentum = 0.9
    weight_decay = 1e-4

    img_path = sorted(os.listdir("./images"))

    device = "mps" if torch.has_mps else "cpu"
    print(f"Device: {device}")

    encoder = resnet18()
    encoder.fc = nn.Identity()

    input_dim = 512
    proj_hidden_dim = 512
    pred_hidden_dim = 128
    output_dim = 512

    model = SimSiam(encoder=encoder,
                    input_dim=input_dim,
                    proj_hidden_dim=proj_hidden_dim,
                    pred_hidden_dim=pred_hidden_dim,
                    output_dim=output_dim)

    optimizer = optim.SGD(params=model.parameters(),
                          lr=lr,
                          momentum=momentum,
                          weight_decay=weight_decay)

    train_dataloader = goes17_dataloader(img_path=img_path,
                                         transform=get_train_transform(),
                                         batch_size=batch_size,
                                         num_workers=num_workers)

    val_dataloader = goes17_dataloader(img_path=img_path,
                                       transform=get_train_transform(),
                                       batch_size=batch_size,
                                       num_workers=num_workers)

    _train(train_dataloader=train_dataloader,
           val_dataloader=val_dataloader,
           model=model,
           optimizer=optimizer,
           epochs=epochs)
