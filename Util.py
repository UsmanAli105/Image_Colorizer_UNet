import numpy as np
import torch
import os

def normalize_image(x):
    """
    Normalize an image by scaling pixel values to the range [0, 1].

    This function takes an image represented as a numerical array and normalizes its pixel values by dividing by 255.0, which scales the values to the range [0, 1].

    Args:
        x (numpy.ndarray or torch.Tensor): The image data to be normalized, with pixel values typically in the range [0, 255].

    Returns:
        numpy.ndarray or torch.Tensor: The normalized image data with pixel values scaled to the range [0, 1].
    """
    return x/255.0

def split_image_files(image_files):
    """
    Split a list of image file paths into training and testing sets.

    This function divides the list of image file paths into two subsets: one for training and one for testing. The split is determined by a specified ratio, with the default ratio being 70% for training and 30% for testing.

    Args:
        image_files (list of str): A list of file paths to the image files to be split.

    Returns:
        tuple: A tuple containing:
            - A list of image file paths for the training set.
            - A list of image file paths for the testing set.
    """
    n = len(image_files)
    train_size = 0.7
    split_index = int(n * train_size)
    
    train_image_files = image_files[:split_index]
    test_image_files = image_files[split_index:]
    
    return train_image_files, test_image_files


def train(net, train_loader, val_loader, criterion, optimizer, num_epochs, device, output_activation):
    """
    Train a PyTorch model and evaluate its performance on a validation dataset.

    This function trains the model for a specified number of epochs using the provided training data loader and evaluates its performance using the validation data loader. The average training and validation losses are computed and returned.

    Args:
        net (torch.nn.Module): The PyTorch model to be trained.
        train_loader (torch.utils.data.DataLoader): The data loader providing the training dataset.
        val_loader (torch.utils.data.DataLoader): The data loader providing the validation dataset.
        criterion (torch.nn.Module): The loss function used to compute the training and validation losses.
        optimizer (torch.optim.Optimizer): The optimizer used to update the model's parameters.
        num_epochs (int): The number of epochs to train the model.
        device (torch.device): The device (CPU or GPU) on which the model and data are located.

    Returns:
        tuple: A tuple containing:
            - The trained PyTorch model.
            - A list of average training losses for each epoch.
            - A list of average validation losses for each epoch.
    """
    train_losses = []
    val_losses = []

    net.to(device=device)

    for epoch in range(num_epochs):
        temp_loss = []
        net.train()
        for gray_images, color_images in train_loader:
            gray_images, color_images = gray_images.to(device=device), color_images.to(device=device)

            optimizer.zero_grad()

            outputs = net(gray_images)
            outputs = output_activation(outputs)

            loss = criterion(outputs, color_images)
            temp_loss.append(loss.item())
            loss.backward()
            optimizer.step()

        train_loss = np.mean(temp_loss)
        train_losses.append(train_loss)

        val_loss = eval_model(net, val_loader, criterion, device)
        val_losses.append(val_loss)

        print(
            f'Epoch: [{epoch + 1}/{num_epochs}]\tTrain Loss: {train_loss:.4f}\tVal Loss: {val_loss:.4f}')
    return net, train_losses, val_losses


def eval_model(net, val_loader, criterion, device):
    """
    Evaluate the performance of a PyTorch model on a validation dataset.

    This function sets the model to evaluation mode and computes the average validation loss over all batches in the validation data loader.

    Args:
        net (torch.nn.Module): The PyTorch model to be evaluated.
        val_loader (torch.utils.data.DataLoader): The data loader providing the validation dataset.
        criterion (torch.nn.Module): The loss function used to compute the validation loss.
        device (torch.device): The device (CPU or GPU) on which the model and data are located.

    Returns:
        float: The average validation loss across all batches.
    
    Raises:
        ValueError: If the validation data loader is empty.
    """
    net.eval()
    val_loss = 0.0
    num_batches = len(val_loader)

    if num_batches == 0:
        raise ValueError("Validation loader is empty. Cannot perform evaluation.")

    with torch.no_grad():
        for gray_images, color_images in val_loader:
            gray_images, color_images = gray_images.to(device), color_images.to(device)
            outputs = net(gray_images)
            loss = criterion(outputs, color_images)
            val_loss += loss.item()
    
    val_loss /= num_batches
    return val_loss


def save_net(net, path, file_name):
    """
    Save the state dictionary of a PyTorch model to a file.

    This function saves the parameters of the given PyTorch model to a specified file path.
    If the directory for the path does not exist, it will be created.

    Args:
        net (torch.nn.Module): The PyTorch model instance whose state dictionary is to be saved.
        path (str): The directory path where the model file will be saved.
        file_name (str): The name of the file to save the model's state dictionary (without the '.pth' extension).

    Returns:
        None
    """
    if not os.path.isdir(path):
        os.makedirs(path)
    torch.save(net.state_dict(), os.path.join(path, f"{file_name}.pth"))

def load_model(saved_models_folder_path, saved_model_name):
    """
    Load the state dictionary of a PyTorch model from a checkpoint file.

    This function loads the state dictionary from a specified file path and returns it. The state dictionary contains the model's parameters.

    Args:
        saved_models_folder_path (str): The directory path where the model checkpoint file is located.
        saved_model_name (str): The name of the checkpoint file (without the '.pth' extension).

    Returns:
        dict: The state dictionary of the PyTorch model.
    """
    checkpoint_path = os.path.join(saved_models_folder_path, f"{saved_model_name}.pth")
    state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=True)
    return state_dict