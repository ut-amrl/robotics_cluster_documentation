#!/usr/bin/env python3
"""CIFAR-10 training example for Stampede3 amd-rtx nodes.

Trains a ResNet-18 on CIFAR-10 using a single GPU. Demonstrates:
  - DataLoader with pinned memory and multiple workers
  - Mixed-precision training (torch.amp)
  - Learning rate scheduling
  - Model checkpointing

Usage:
    python train_cifar10.py                      # defaults: 5 epochs, GPU 0
    python train_cifar10.py --epochs 20          # 20 epochs
    python train_cifar10.py --gpu 3              # use GPU 3
    python train_cifar10.py --output $SCRATCH/cifar10_checkpoints
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torchvision import datasets, transforms
from torchvision.models import resnet18


def get_args():
    parser = argparse.ArgumentParser(description="CIFAR-10 ResNet-18 training")
    parser.add_argument("--epochs", type=int, default=5, help="number of epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="batch size")
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index")
    parser.add_argument("--workers", type=int, default=4, help="data loader workers")
    parser.add_argument("--output", type=str, default="./checkpoints",
                        help="checkpoint output directory")
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="dataset download directory")
    return parser.parse_args()


def main():
    args = get_args()

    # Device setup
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device} ({torch.cuda.get_device_name(args.gpu)})")
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
    print(f"Config: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")

    # Data augmentation and normalization
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    train_dataset = datasets.CIFAR10(root=args.data_dir, train=True,
                                     download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root=args.data_dir, train=False,
                                    download=True, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # Model: ResNet-18 adapted for CIFAR-10 (32x32 images, 10 classes)
    model = resnet18(num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # Remove maxpool for small images
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler("cuda")

    os.makedirs(args.output, exist_ok=True)
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        # --- Training ---
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        t0 = time.time()

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad()
            with autocast("cuda"):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_time = time.time() - t0
        train_acc = 100.0 * correct / total
        train_loss /= total

        # --- Evaluation ---
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                with autocast("cuda"):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                test_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        test_acc = 100.0 * correct / total
        test_loss /= total
        scheduler.step()

        print(f"Epoch {epoch}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.2f}% | "
              f"Time: {train_time:.1f}s | "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "test_acc": test_acc,
            }, os.path.join(args.output, "best_model.pt"))

    print(f"\nTraining complete. Best test accuracy: {best_acc:.2f}%")
    print(f"Checkpoint saved to: {os.path.join(args.output, 'best_model.pt')}")


if __name__ == "__main__":
    main()
