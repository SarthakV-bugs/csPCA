class AttentionBlock(nn.Module):
    """Attention gate for focusing on relevant features"""

    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class AttentionUNet2D(nn.Module):
    """
    U-Net with attention gates.
    DSC improvement: +3-5% over standard U-Net.
    """

    def __init__(self, in_ch=1, out_ch=1):
        super(AttentionUNet2D, self).__init__()

        # Encoder
        self.inc = DoubleConv2D(in_ch, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv2D(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv2D(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv2D(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv2D(512, 1024))

        # Decoder with attention
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.att1 = AttentionBlock(F_g=512, F_l=512, F_int=256)
        self.conv_up1 = DoubleConv2D(1024, 512)

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att2 = AttentionBlock(F_g=256, F_l=256, F_int=128)
        self.conv_up2 = DoubleConv2D(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att3 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.conv_up3 = DoubleConv2D(256, 128)

        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att4 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        self.conv_up4 = DoubleConv2D(128, 64)

        self.outc = nn.Conv2d(64, out_ch, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder with attention gates
        d4 = self.up1(x5)
        x4 = self.att1(g=d4, x=x4)
        d4 = torch.cat([x4, d4], dim=1)
        d4 = self.conv_up1(d4)

        d3 = self.up2(d4)
        x3 = self.att2(g=d3, x=x3)
        d3 = torch.cat([x3, d3], dim=1)
        d3 = self.conv_up2(d3)

        d2 = self.up3(d3)
        x2 = self.att3(g=d2, x=x2)
        d2 = torch.cat([x2, d2], dim=1)
        d2 = self.conv_up3(d2)

        d1 = self.up4(d2)
        x1 = self.att4(g=d1, x=x1)
        d1 = torch.cat([x1, d1], dim=1)
        d1 = self.conv_up4(d1)

        out = self.outc(d1)
        return torch.sigmoid(out)
