import torch 
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.block(x)

class AttentionGate(nn.Module):
    def __init__(self, gate_channels, skip_channels, inter_channels):
        super().__init__()

        self.gate_conv = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels),
        )

        self.skip_conv = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels)
        )

        self.attention = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
    
    def forward(self, gate, skip):
        gate_features = self.gate_conv(gate)
        skip_features = self.skip_conv(skip)
        attention_map = self.attention(gate_features + skip_features)
        return skip * attention_map

class AttentionUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        self.down1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.bottleneck = DoubleConv(256, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att3 = AttentionGate(
            gate_channels=256,
            skip_channels=256,
            inter_channels=128,
        )
        self.conv3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att2 = AttentionGate(
            gate_channels=128,
            skip_channels=128,
            inter_channels=64,
        )
        self.conv2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att1 = AttentionGate(
            gate_channels=64,
            skip_channels=64,
            inter_channels=32,
        )
        self.conv1 = DoubleConv(128, 64)

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))

        b = self.bottleneck(self.pool3(d3))

        u3 = self.up3(b)
        d3 = self.att3(gate=u3, skip=d3)
        u3 = torch.cat((u3, d3), dim=1)
        u3 = self.conv3(u3)

        u2 = self.up2(u3)
        d2 = self.att2(gate=u2, skip=d2)
        u2 = torch.cat((u2, d2), dim=1)
        u2 = self.conv2(u2)

        u1 = self.up1(u2)
        d1 = self.att1(gate=u1, skip=d1)
        u1 = torch.cat((u1, d1), dim=1)
        u1 = self.conv1(u1)

        return self.final(u1)