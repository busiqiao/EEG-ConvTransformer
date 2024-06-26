import torch
import torch.nn as nn
from model.lfe import LocFeaExtractor
from model.msa_cfe import CTBlock


class ConvTransformer(nn.Module):
    def __init__(self, num_classes, channels=8, num_heads=2, E=16, F=256, T=32, depth=2):
        super().__init__()
        self.lfe = LocFeaExtractor(channels=channels)
        self.blocks = nn.ModuleList([
            CTBlock(channels=channels, num_heads=num_heads, E=E)
            for _ in range(depth)])

        p = ((T - 8 + 2 * 0) // 4 + 1) ** 2
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=F // 2,
                               kernel_size=(p, 3), stride=(1, 1), padding=(0, 1))
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=F // 2,
                               kernel_size=(p, 5), stride=(1, 1), padding=(0, 2))
        self.bn = nn.BatchNorm2d(num_features=F)
        self.elu = nn.ELU()
        self.fla = nn.Flatten(start_dim=1, end_dim=-1)
        self.classifier = nn.Sequential(
            nn.Linear(in_features=F * T, out_features=500),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(in_features=500, out_features=100),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=num_classes),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        # [b, 1,  M, M, T]
        x = self.lfe(x)  # [b, c, p=m*m, T]
        for blk in self.blocks:
            x = blk(x)  # [b, c, p, T]
        x1 = self.conv1(x)  # [b, F/2, 1, T]
        x2 = self.conv2(x)  # [b, F/2, 1, T]
        x = torch.cat((x1, x2), dim=1)  # [b, F, 1, T]
        x = self.bn(x)
        x = self.elu(x)
        x = self.fla(x)  # [b, F*T]
        x = self.classifier(x)  # [b, classes]
        return x
