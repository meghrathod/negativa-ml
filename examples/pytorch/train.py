#!/usr/bin/env python3
"""
Simple MobileNetV2 training script for CIFAR-10.

This script is intended to create a realistic workload for Negativa‑ML tracing.
It trains for a small number of epochs to exercise CUDA kernels and shared
libraries.  Feel free to reduce the number of epochs or batch size for quicker runs.
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Transformations for CIFAR-10: resize to 224×224 because MobileNetV2 expects 224 input
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])

    # Download CIFAR-10 dataset
    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True
    )

    # Load pre-defined MobileNetV2 from torchvision.  num_classes=10 for CIFAR-10.
    model = torchvision.models.mobilenet_v2(num_classes=10)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    num_epochs = 2  # Increase to 10–20 for a more realistic workload

    start = time.time()
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (batch_idx + 1) % 50 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Step {batch_idx+1}, Loss: {running_loss / 50:.4f}")
                running_loss = 0.0

    end = time.time()
    print(f"Training completed in {(end - start):.2f} seconds")

if __name__ == "__main__":
    main()
