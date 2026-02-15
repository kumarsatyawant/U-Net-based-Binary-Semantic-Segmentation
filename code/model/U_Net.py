import torch
import torch.nn as nn

def double_conv(input_channel, output_channel):
    conv = nn.Sequential(
        nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(output_channel),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(output_channel, output_channel, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(output_channel),
        nn.LeakyReLU(inplace=True)
    )
    return conv

class U_Net(nn.Module):
    def __init__(self, num_class=1):
        super(U_Net, self).__init__()
        # Max Pooling
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        #Down Convolution
        self.conv_down_1 = double_conv(3, 64)
        self.conv_down_2 = double_conv(64, 128)
        self.conv_down_3 = double_conv(128, 256)
        self.conv_down_4 = double_conv(256, 512)
        self.conv_down_5 = double_conv(512, 1024)

        #Up Sampling
        self.up_sample_1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.up_sample_2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.up_sample_3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.up_sample_4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)

        #Up Convolution
        self.conv_up_1 = double_conv(1024, 512)
        self.conv_up_2 = double_conv(512, 256)
        self.conv_up_3 = double_conv(256, 128)
        self.conv_up_4 = double_conv(128, 64)

        #Output layer
        self.out = nn.Conv2d(in_channels=64, out_channels=num_class, kernel_size=1, padding=0)

    def forward(self, image_dim):
        #Encoder Network
        out_1 = self.conv_down_1(image_dim) #Skip Connection
        out_2 = self.max_pool(out_1)
        out_3 = self.conv_down_2(out_2) #Skip Connection
        out_4 = self.max_pool(out_3)
        out_5 = self.conv_down_3(out_4) #Skip Connection
        out_6 = self.max_pool(out_5)
        out_7 = self.conv_down_4(out_6) #Skip Connection
        out_8 = self.max_pool(out_7)
        out_9 = self.conv_down_5(out_8)

        #Decoder Network
        up_sample_1 = self.up_sample_1(out_9)
        conv_up_1 = self.conv_up_1(torch.cat([up_sample_1, out_7], 1))

        up_sample_2 = self.up_sample_2(conv_up_1)
        conv_up_2 = self.conv_up_2(torch.cat([up_sample_2, out_5], 1))

        up_sample_3 = self.up_sample_3(conv_up_2)
        conv_up_3 = self.conv_up_3(torch.cat([up_sample_3, out_3], 1))

        up_sample_4 = self.up_sample_4(conv_up_3)
        conv_up_4 = self.conv_up_4(torch.cat([up_sample_4, out_1], 1))
        output = self.out(conv_up_4)
        
        return output