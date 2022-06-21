import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.models import resnet18
import torchvision.models as backbones

from transforms import GaussianBlur
from datasets import GOES17Dataset
from models import SimSiam
from layers import SymmetrizedNegativeCosineSimilarity

from typing import List, Tuple
from tqdm import tqdm

# Set up encoder
backbone_name = sorted(name for name in backbones.__dict__
                       if name.islower() and not name.startswith("__")
                       and callable(backbones.__dict__[name]))

# Set up os for running
os_name = ['linux', 'windows', 'mac_intel', 'mac_arm']

# Set up argparse
# For training
parser = argparse.ArgumentParser(description="SimSiam on GOES-17 Dataset")
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet34',
                    choices=backbone_name,
                    help='model architecture: ' +
                         ' | '.join(backbone_name) +
                         ' (default: resnet34)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial (base) learning rate (default: 0.05)', dest='lr')
parser.add_argument('--os', default='linux', type=str,
                    choices=os_name,
                    metavar='OS', help='determine OS for environment (default: linux)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver (default: 0.9)')
parser.add_argument('--seed', default=42, type=int,
                    help='seed for initializing training (default: 42)')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='WD', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-w', '--workers', default=8, type=int, metavar='W',
                    help='number of data loading workers (default: 8)')

# For SimSiam config
parser.add_argument('--input-dim', default=512, type=int, dest='input_dim',
                    help='input dimension for SimSiam (default: 512)')
parser.add_argument('--proj-hidden-dim', default=512, type=int, dest='proj_hidden_dim',
                    help='hidden dimension for projection head (default: 512)')
parser.add_argument('--pred-hidden-dim', default=128, type=int, dest='pred_hidden_dim',
                    help='hidden dimension for prediction head (default: 128)')
parser.add_argument('--output-dim', default=512, type=int, dest='output_dim',
                    help='output dimension for SimSiam (default: 512)')


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
    img_path = sorted(os.listdir("./images"))
    args = parser.parse_args()

    # Random seed
    if args.seed:
        torch.manual_seed(args.seed)

    # Determine OS for device
    if args.os and args.os == 'mac_arm' and "1.13.0.dev" in torch.__version__:
        device = "mps" if torch.has_mps else "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    batch_size = args.batch_size
    num_workers = args.workers
    epochs = args.epochs

    lr = args.lr * batch_size / 256

    # Create model
    print(f"Creating model {args.arch}")

    encoder = backbones.__dict__[args.arch]
    # encoder.fc = nn.Identity()

    model = SimSiam(encoder=encoder,
                    input_dim=args.input_dim,
                    proj_hidden_dim=args.proj_hidden_dim,
                    pred_hidden_dim=args.pred_hidden_dim,
                    output_dim=args.output_dim)

    optimizer = optim.SGD(params=model.parameters(),
                          lr=lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

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
