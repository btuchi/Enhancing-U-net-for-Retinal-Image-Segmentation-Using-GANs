from torch import nn
from torchvision import models
import torch.nn.functional as F
import torch


class block(nn.Module):
    def __init__(self,in_filters,n_filters):
        super(block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_filters, n_filters, 3, stride=1, padding=1),
            nn.BatchNorm2d(n_filters),
            nn.ReLU()
        )
    def forward(self, x):
        x=self.conv(x)
        return x

# 就是一个U-Net
class generator(nn.Module):
    # initializers
    def __init__(self, n_filters=32):
        super(generator, self).__init__()
        self.down1=nn.Sequential(
            block(3,n_filters),
            block(n_filters,n_filters),
            nn.MaxPool2d((2, 2)))
        self.down2 = nn.Sequential(
            block(n_filters, 2*n_filters),
            block(2*n_filters, 2*n_filters),
            nn.MaxPool2d((2, 2)))
        self.down3 = nn.Sequential(
            block(2*n_filters, 4*n_filters),
            block(4*n_filters, 4*n_filters),
            nn.MaxPool2d((2, 2)))
        self.down4 = nn.Sequential(
            block(4*n_filters, 8 * n_filters),
            block(8 * n_filters, 8 * n_filters),
            nn.MaxPool2d((2, 2)))
        self.down5 = nn.Sequential(
            block(8 * n_filters, 16 * n_filters),
            block(16 * n_filters, 16 * n_filters))

        self.up1=nn.Sequential(
            block(16 * n_filters+8*n_filters, 8 * n_filters),
            block(8 * n_filters, 8 * n_filters))
        self.up2 = nn.Sequential(
            block(8 * n_filters+4*n_filters, 4 * n_filters),
            block(4 * n_filters, 4 * n_filters))
        self.up3 = nn.Sequential(
            block(4 * n_filters+2*n_filters,2 * n_filters),
            block(2 * n_filters, 2 * n_filters))
        self.up4 = nn.Sequential(
            block(2 * n_filters+n_filters,  n_filters),
            block( n_filters,  n_filters))

        self.out = nn.Conv2d(n_filters, 1, kernel_size=1)

    # forward method
    def forward(self, x): # torch.Size([2, 3, 640, 640])

        x1 = self.down1(x) # torch.Size([2, 3, 640, 640]) -> torch.Size([2, 32, 320, 320])
        x2 = self.down2(x1) # torch.Size([2, 32, 320, 320]) -> torch.Size([2, 64, 160, 160])
        x3 = self.down3(x2) # torch.Size([2, 64, 160, 160]) -> torch.Size([2, 128, 80, 80])
        x4 = self.down4(x3) # torch.Size([2, 128, 80, 80]) -> torch.Size([2, 256, 40, 40])
        x5 = self.down5(x4) # torch.Size([2, 256, 40, 40]) -> torch.Size([2, 512, 40, 40])

        x = self.up1(F.upsample(torch.cat((x4, x5), dim=1),scale_factor=2)) # torch.Size([2, 256, 80, 80])
        x = self.up2(F.upsample(torch.cat((x, x3), dim=1), scale_factor=2)) # torch.Size([2, 128, 160, 160])
        x = self.up3(F.upsample(torch.cat((x, x2), dim=1), scale_factor=2)) # torch.Size([2, 64, 320, 320])
        x = self.up4(F.upsample(torch.cat((x, x1), dim=1), scale_factor=2)) # torch.Size([2, 32, 640, 640])
        x = F.sigmoid(self.out(x)) # torch.Size([2, 32, 640, 640]) -> torch.Size([2, 1, 640, 640])

        return x

class discriminator(nn.Module):
    def __init__(self,n_filters):
        super(discriminator,self).__init__()
        self.down1 = nn.Sequential(
            block(4, n_filters),
            block(n_filters, n_filters),
            nn.MaxPool2d((2, 2)))
        self.down2 = nn.Sequential(
            block(n_filters, 2 * n_filters),
            block(2 * n_filters, 2 * n_filters),
            nn.MaxPool2d((2, 2)))
        self.down3 = nn.Sequential(
            block(2 * n_filters, 4 * n_filters),
            block(4 * n_filters, 4 * n_filters),
            nn.MaxPool2d((2, 2)))
        self.down4 = nn.Sequential(
            block(4 * n_filters, 8 * n_filters),
            block(8 * n_filters, 8 * n_filters),
            nn.MaxPool2d((2, 2)))
        self.down5 = nn.Sequential(
            block(8 * n_filters, 16 * n_filters),
            block(16 * n_filters, 16 * n_filters))
        self.out = nn.Linear(16*n_filters,1)

    def forward(self, x): # torch.Size([2, 4, 640, 640])
        x = self.down1(x) # torch.Size([2, 4, 640, 640]) -> torch.Size([2, 32, 320, 320])
        x = self.down2(x) # torch.Size([2, 32, 320, 320]) -> torch.Size([2, 64, 160, 160])
        x = self.down3(x) # torch.Size([2, 64, 160, 160]) -> torch.Size([2, 128, 80, 80])
        x = self.down4(x) # torch.Size([2, 128, 80, 80]) -> torch.Size([2, 256, 40, 40])
        x = self.down5(x) # torch.Size([2, 256, 40, 40]) -> torch.Size([2, 512, 40, 40])

        x = F.avg_pool2d(x, kernel_size=x.size()[2:]) # torch.Size([2, 512, 40, 40]) -> torch.Size([2, 512, 1, 1])
        x = x.view(x.size()[0], -1) # torch.Size([2, 512, 1, 1]) -> torch.Size([2, 512])

        x=self.out(x) # torch.Size([2, 512]) -> torch.Size([2, 1])
        x = F.sigmoid(x)
        return x


if __name__=='__main__':
    D=discriminator(32).cuda()
    t=torch.ones((2,4,512,512)).cuda()
    res = D(t)
    print(res)