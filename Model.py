import torch.nn as nn
import torch


class DoubleConvLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(DoubleConvLayer, self).__init__()
        self.conv1 = nn.Conv2d(input_size, output_size, kernel_size=3, padding=1)
        self.LeakyReLU = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(output_size, output_size, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.LeakyReLU(x)
        x = self.conv2(x)
        x = self.LeakyReLU(x)
        return x


class DownSampleBlock(nn.Module):
    def __init__(self, input_size, output_size):
        super(DownSampleBlock, self).__init__()
        self.double_conv = DoubleConvLayer(input_size, output_size)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x_cov = self.double_conv(x)
        x_pooled = self.max_pool(x_cov)
        return x_pooled, x_cov


class UpSampleBlock(nn.Module):
    def __init__(self, input_size, output_size) -> None:
        super(UpSampleBlock, self).__init__()
        self.up_sample = nn.ConvTranspose2d(input_size, input_size // 2, kernel_size=2, stride=2)
        self.double_conv = DoubleConvLayer(input_size, output_size)

    def forward(self, input_x, skip_layer_input):
        x_up_sampled = self.up_sample(input_x)
        x_concatenated = torch.cat((x_up_sampled, skip_layer_input), dim=1)
        return self.double_conv(x_concatenated)


class Encoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(Encoder, self).__init__()
        self.down_sample_block_1 = DownSampleBlock(input_size, 64)
        self.down_sample_block_2 = DownSampleBlock(64, 128)
        self.down_sample_block_3 = DownSampleBlock(128, 256)
        self.down_sample_block_4 = DownSampleBlock(256, output_size)
        self.skip_layer = True

    def forward(self, x):
        x, x_conv_1 = self.down_sample_block_1(x)
        x, x_conv_2 = self.down_sample_block_2(x)
        x, x_conv_3 = self.down_sample_block_3(x)
        x, x_conv_4 = self.down_sample_block_4(x)

        zeros_like_x_conv1 = torch.zeros_like(x_conv_1)
        zeros_like_x_conv2 = torch.zeros_like(x_conv_2)
        zeros_like_x_conv3 = torch.zeros_like(x_conv_3)
        zeros_like_x_conv4 = torch.zeros_like(x_conv_4)

        if self.skip_layer:
            return x, x_conv_1, x_conv_2, x_conv_3, x_conv_4
        else:
            return x, zeros_like_x_conv1, zeros_like_x_conv2, zeros_like_x_conv3, zeros_like_x_conv4


class Decoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(Decoder, self).__init__()
        self.up_sample_block_1 = UpSampleBlock(input_size, 512)
        self.up_sample_block_2 = UpSampleBlock(512, 256)
        self.up_sample_block_3 = UpSampleBlock(256, 128)
        self.up_sample_block_4 = UpSampleBlock(128, output_size)

    def forward(self, x, x_conv_1, x_conv_2, x_conv_3, x_conv_4):
        x = self.up_sample_block_1(x, x_conv_4)
        x = self.up_sample_block_2(x, x_conv_3)
        x = self.up_sample_block_3(x, x_conv_2)
        x = self.up_sample_block_4(x, x_conv_1)
        return x


class UNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(UNet, self).__init__()
        self.encoder = Encoder(input_size, 512)
        self.bottle_neck_layer = DoubleConvLayer(512, 1024)
        self.decoder = Decoder(1024, 64)
        self.conv_1x1 = nn.Conv2d(64, output_size, kernel_size=1)

    def forward(self, x):
        x, x_conv_1, x_conv_2, x_conv_3, x_conv_4 = self.encoder(x)
        x = self.bottle_neck_layer(x)
        x = self.decoder(x, x_conv_1, x_conv_2, x_conv_3, x_conv_4)
        x = self.conv_1x1(x)
        return x
