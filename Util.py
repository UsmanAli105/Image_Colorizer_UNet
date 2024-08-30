import numpy as np
import torch
import os

def normalize_image(x):
    return x/255.0

def split_image_files(image_files):
    n = len(image_files)
    train_size = 0.7
    split_index = int(n * train_size)
    
    train_image_files = image_files[:split_index]
    test_image_files = image_files[split_index:]
    
    return train_image_files, test_image_files


def train(net, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    train_losses = []
    val_losses = []

    net.to(device=device)

    for epoch in range(num_epochs):
        temp_loss = []
        temp_score = []
        net.train()
        for gray_images, color_images in train_loader:
            gray_images, color_images = gray_images.to(device=device), color_images.to(device=device)

            optimizer.zero_grad()

            outputs = net(gray_images)

            loss = criterion(outputs, color_images)
            temp_loss.append(loss.item())
            loss.backward()
            optimizer.step()

        train_loss = np.mean(temp_loss)
        train_score = np.mean(temp_score)
        train_losses.append(train_loss)

        val_loss, val_score, _, _ = eval_model(net, val_loader, criterion, device)
        val_losses.append(val_loss.item())

        print(
            f'Epoch: [{epoch + 1}/{num_epochs}]\tTrain Score: {train_score:.4f}\tTrain Loss: {train_loss:.4f}\tVal Score: {val_score:.4f}\tVal Loss: {val_loss:.4f}')
    return net, train_losses, val_losses


def eval_model(_net, _val_loader, _criterion, device):
    _net.eval()
    val_loss = 0.0
    with torch.no_grad():
        for gray_images, color_images in _val_loader:
            gray_images, color_images = gray_images.to(device=device), color_images.to(device=device)
            outputs = _net(gray_images)
            val_loss += _criterion(outputs, color_images)
        val_loss /= len(_val_loader)
    return val_loss

def save_net(_net, path, file_name):
    if not os.path.isdir(path):
        os.makedirs(path)
    torch.save(_net.state_dict(), str(path + '/' + file_name + '.pth'))
