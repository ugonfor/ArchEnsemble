from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from mimo.config import Config
from mimo.model import MIMOModel
from mimo.trainer import MIMOTrainer

import torchvision
import wandb

from mimo.cnn import CNNLayer
from mimo._resnet import ResNet, BasicBlock


parser = ArgumentParser("MIMO Training")
parser.add_argument("--ensemble-num", type=int, default=3)
parser.add_argument("--model", default='simplecnn')
parser.add_argument("--dataset", default='cifar10')
parser.add_argument("--epoch", type=int, default=10)



def main(args):
    wandb.init()
    config = Config(ensemble_num=args.ensemble_num, num_epochs=args.epoch)
    wandb.config.update(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)

        test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)

    elif args.dataset == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_dataset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=transform_train)

        test_dataset = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, transform=transform_test)


    train_dataloaders = [
        DataLoader(
            train_dataset, batch_size=config.batch_size, num_workers=config.num_workers, pin_memory=True, shuffle=True
        )
        for _ in range(config.ensemble_num)
    ]
    test_dataloader = DataLoader(
        test_dataset, batch_size=config.batch_size, num_workers=config.num_workers, pin_memory=True
    )

    if args.model == 'simplecnn':
        backbone = CNNLayer()
    elif args.model == 'resnet20':
        backbone = ResNet(BasicBlock, [3, 3, 3], num_classes=128)
    elif args.model == 'resnet32':
        backbone = ResNet(BasicBlock, [5, 5, 5], num_classes=128)



    model = MIMOModel(backbone, config.ensemble_num).to(device)
    wandb.watch(model)
    trainer = MIMOTrainer(config, model, train_dataloaders, test_dataloader, device)
    trainer.train()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
