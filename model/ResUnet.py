class ResBlock(nn.Module):
    """Residual block with skip connections"""

    def __init__(self, in_ch, out_ch):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.Sequential()
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1),
                nn.BatchNorm2d(out_ch)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(residual)
        out = self.relu(out)

        return out


class ResUNet(nn.Module):
    """
    ResUNet with residual connections.
    Better gradient flow, faster convergence.
    """

    def __init__(self, in_ch=1, out_ch=1):
        super(ResUNet, self).__init__()

        # Encoder
        self.inc = ResBlock(in_ch, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ResBlock(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ResBlock(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), ResBlock(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), ResBlock(512, 1024))

        # Decoder
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv_up1 = ResBlock(1024, 512)

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up2 = ResBlock(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up3 = ResBlock(256, 128)

        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up4 = ResBlock(128, 64)

        self.outc = nn.Conv2d(64, out_ch, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.conv_up1(x)

        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.conv_up2(x)

        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv_up3(x)

        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_up4(x)

        out = self.outc(x)
        return torch.sigmoid(out)
