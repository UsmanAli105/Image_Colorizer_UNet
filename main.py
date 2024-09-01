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
        print(f'Using Device: {torch.cuda.get_device_name(0)}')
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

        BATCH_SIZE = 8
        NUM_EPOCH = 25
        LEARNING_RATE = 10e-2
        MOMENTUM = 0.9

        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

        # grayscale_image, color_image = train_dataset.__getitem__(0)
        #
        # grayscale_image = grayscale_image.permute(1, 2, 0)
        # color_image = color_image.permute(1, 2, 0)
        #
        # fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        # axs[0].imshow(grayscale_image, cmap='gray')
        # axs[0].set_title('Grayscale Image')
        # axs[0].axis('off')
        # axs[1].imshow(color_image)
        # axs[1].set_title('Color Image')
        # axs[1].axis('off')
        # plt.show()

        input_size = 1
        output_size = 3
        model = UNet(input_size, output_size)

        output_activation = nn.Sigmoid()
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

        if MODE == TRAIN:
            net, train_losses, val_losses = train(model,
                                                  train_dataloader,
                                                  test_dataloader,
                                                  criterion,
                                                  optimizer,
                                                  NUM_EPOCH,
                                                  DEVICE,
                                                  output_activation)
            save_net(model, SAVED_MODELS_FOLDER_PATH, SAVED_MODEL_NAME)
        elif MODE == TEST:
            # Load the saved model
            saved_weights = load_model(SAVED_MODELS_FOLDER_PATH, SAVED_MODEL_NAME)
            model.load_state_dict(saved_weights)
            model.to(DEVICE)
            model.eval()

            # Load a test image
            test_grayscale_image, test_colored_image = train_dataset.__getitem__(56)
            test_grayscale_image = test_grayscale_image.unsqueeze(0).to(DEVICE)

            # Make prediction
            with torch.no_grad():
                model_output = model(test_grayscale_image)
                model_output = model_output.squeeze(0).cpu()  # Remove batch dimension and move to CPU
            model_output = model_output if not output_activation else output_activation(model_output)
            # Convert the model output to an image format
            model_output_image = model_output.permute(1, 2, 0).clamp(0, 1)  # Ensure values are between 0 and 1

            # Convert the test grayscale image to RGB for visualization
            test_grayscale_image = test_grayscale_image.squeeze(0).cpu().permute(1, 2, 0)
            test_grayscale_image = test_grayscale_image.repeat(1, 1, 3)  # Repeat channels for RGB
            test_colored_image = test_colored_image.cpu().permute(1, 2, 0)

            # Display the images
            fig, axs = plt.subplots(1, 3)
            axs[1].imshow(test_colored_image)
            axs[1].set_title('Actual Colors')
            axs[1].axis('off')
            axs[0].imshow(test_grayscale_image)
            axs[0].set_title('Black and White Image')
            axs[0].axis('off')
            axs[2].imshow(model_output_image)
            axs[2].set_title('Predicted Colors')
            axs[2].axis('off')
            plt.show()

        elif MODE == CONTINUE_TRAINING:
            # Load the saved model
            saved_weights = load_model(SAVED_MODELS_FOLDER_PATH, SAVED_MODEL_NAME)
            model.load_state_dict(saved_weights)
            model.to(DEVICE)

            # Resume training
            net, train_losses, val_losses = train(model,
                                                  train_dataloader,
                                                  test_dataloader,
                                                  criterion,
                                                  optimizer,
                                                  NUM_EPOCH,
                                                  DEVICE,
                                                  output_activation)
            save_net(model, SAVED_MODELS_FOLDER_PATH, SAVED_MODEL_NAME)

        elif MODE == INFERENCE:
            saved_weights = load_model(SAVED_MODELS_FOLDER_PATH, SAVED_MODEL_NAME)
            model.load_state_dict(saved_weights)
            model.to(DEVICE)

            image_path = args[3]
            if not os.path.exists(image_path):
                raise DATASET_PATH_NOT_EXISTS_ERROR_MSG
            image_data = read_image(image_path).float()
            image_data_transformed = (image_transforms(convert_to_grayscale(image_data)).unsqueeze(dim=0).to(DEVICE))
            with torch.no_grad():
                output = model(image_data_transformed).detach().cpu()

            prediction = output_activation(output)
            prediction = np.squeeze(prediction)
            predicted_image = prediction.permute(1, 2, 0)

            fig, axs = plt.subplots(1, 3)
            axs[1].set_title('Actual Colors')
            axs[1].imshow(image_transforms(image_data).permute(1, 2, 0))
            axs[1].axis('off')
            axs[0].imshow(convert_to_grayscale(image_transforms(image_data)).permute(1, 2, 0), cmap='gray')
            axs[0].set_title('Black and White Image')
            axs[0].axis('off')
            axs[2].imshow(predicted_image)
            axs[2].set_title('Predicted Colors')
            axs[2].axis('off')
            plt.show()

        print('Program Finished')


    except Exception as e:
        print(e)
        sys.exit(0)
