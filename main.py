from Constants import *
from Dataloader import *
from Util import *
from Model import *
import sys
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

if __name__ == '__main__':
    
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda:0") 
        print(f'Using Device: {torch.cuda.current_device()}')
    else:
        DEVICE = torch.device("cpu")
        print(f'Using Device: cpu')
    try:
        args = sys.argv
        n = len(args)
        if n < 3:
            raise Exception(NO_ARGUMENT_ERROR_MSG)
        
        DATASET_PATH = args[1]
        MODE = args[2]

        if not os.path.exists(DATASET_PATH):
            raise Exception(DATASET_PATH_NOT_EXISTS_ERROR_MSG)

        image_transforms = transforms.Compose([
            normalize_image,
            transforms.Resize((256, 256), antialias=True),
        ])

        image_files = os.listdir(DATASET_PATH)

        train_image_files, test_image_files = split_image_files(image_files)

        train_dataset = CustomDataset(train_image_files, DATASET_PATH, image_transforms)
        test_dataset = CustomDataset(test_image_files, DATASET_PATH, image_transforms)

        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

        grayscale_image, color_image = train_dataset.__getitem__(0)

        grayscale_image = grayscale_image.permute(1, 2, 0)
        color_image = color_image.permute(1, 2, 0)

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))        
        axs[0].imshow(grayscale_image, cmap='gray')
        axs[0].set_title('Grayscale Image')
        axs[0].axis('off')
        axs[1].imshow(color_image)
        axs[1].set_title('Color Image')
        axs[1].axis('off')        
        plt.show()

        input_size = 1
        output_size = 3
        model = UNet(input_size, output_size)

        output_activation = nn.Sigmoid()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        if MODE == TRAIN:
            net, train_losses, val_losses = train(model, train_dataloader, test_dataloader, criterion, optimizer, 10, DEVICE)
            save_net(model, SAVED_MODELS_FOLDER_PATH, SAVED_MODEL_NAME)
        elif MODE == TEST:
            pass





    except Exception as e:
        print(e)
        sys.exit(0)