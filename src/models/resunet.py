import torch 
import torch.nn as nn

class ResidualConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv_block(x) + self.shortcut(x))
    
class ResUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.down1 = ResidualConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.down2 = ResidualConv(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.down3 = ResidualConv(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.bottleneck = ResidualConv(256, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv3 = ResidualConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = ResidualConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv1 = ResidualConv(128, 64)

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))

        b = self.bottleneck(self.pool3(d3))

        u3 = self.up3(b)
        u3 = torch.cat((u3, d3), dim=1)
        u3 = self.conv3(u3)

        u2 = self.up2(u3)
        u2 = torch.cat((u2, d2), dim=1)
        u2 = self.conv2(u2)

        u1 = self.up1(u2)
        u1 = torch.cat((u1, d1), dim=1)
        u1 = self.conv1(u1)

        return self.final(u1)