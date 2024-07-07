import torch
from misc.utils import make_trainable
from model import discriminator, generator
from dataset import MyDataset
from transform import ReLabel, ToLabel
from torchvision.transforms import Compose, Normalize, ToTensor
import tqdm

from misc.data_aug import ColorAug,Random_horizontal_flip,Random_vertical_flip,Compose_imglabel,Random_crop

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--add_gan_loss', type=bool, default=True)
args = parser.parse_args()


"""
    数据准备部分
    准备部分的结果是得到一个torch的dataloader
"""

trainloader = torch.utils.data.DataLoader(
    MyDataset("data/"),
    batch_size=2,
    shuffle=True,
    num_workers=4
)

total_epoches = 1000

"""
    模型准备部分
    准备部分的结果是得到一个torch的dataloader
"""



G = generator(n_filters=32).cuda()
optimizer_G = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.9))

bce_loss = torch.nn.BCELoss()

if args.add_gan_loss:
    D = discriminator(n_filters=32).cuda()
    optimizer_D = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.9))
    gan_loss_percent = 0.03

    batch_size = trainloader.batch_size
    real_pair_gt = torch.ones((batch_size, 1)).cuda()
    fake_pair_gt = torch.zeros((batch_size, 1)).cuda()

    for epoch in range(total_epoches):
        D.train()
        G.train()
        #train D
        make_trainable(D, True)
        make_trainable(G, False)
        for idx, (real_imgs, real_labels) in tqdm.tqdm(enumerate(trainloader)):
            bs = real_imgs.shape[0] # 2
            real_imgs = real_imgs.cuda() # torch.Size([2, 3, 640, 640])
            real_labels = real_labels.unsqueeze(1).cuda() # torch.Size([2, 1, 640, 640])
            optimizer_D.zero_grad()

            real_pair = torch.cat((real_imgs, real_labels), dim=1) # torch.Size([2, 3+1, 640, 640])
            real_pair_pred = D(real_pair) # torch.Size([2, 1])
            
            fake_pair = torch.cat((real_imgs, G(real_imgs)), dim=1) # torch.Size([2, 4, 640, 640])
            fake_pair_pred = D(fake_pair) # torch.Size([2, 1])

            d_real_pair_loss = bce_loss(real_pair_pred, real_pair_gt)
            d_fake_pair_loss = bce_loss(fake_pair_pred, fake_pair_gt)

            d_loss = d_real_pair_loss + d_fake_pair_loss
            d_loss.backward()

            optimizer_D.step()

        #train G
        make_trainable(D,False)
        make_trainable(G,True)
        for idx,(real_imgs,real_labels) in tqdm.tqdm(enumerate(trainloader)):
            optimizer_G.zero_grad()

            real_imgs = real_imgs.cuda() # torch.Size([2, 3, 640, 640])
            real_labels = real_labels.cuda() # torch.Size([2, 640, 640])
            pred_labels=G(real_imgs) # torch.Size([2, 640, 640]) -> torch.Size([2, 1, 640, 640])

            Seg_Loss = bce_loss(pred_labels, real_labels.unsqueeze(1)) # 2, 1, 640, 640 -> scaler

            fake_pair = torch.cat((real_imgs, pred_labels), dim=1) # torch.Size([2, 3 + 1, 640, 640])
            fake_pair_pred = D(fake_pair) # torch.Size([2, 1])
            d_fake_pair_loss = bce_loss(fake_pair_pred, real_pair_gt)

            g_loss= d_fake_pair_loss * gan_loss_percent + Seg_Loss
            g_loss.backward()

            optimizer_G.step()
        
        print(f"epoch[{epoch}/{total_epoches}] segloss: {Seg_Loss}")

        if (epoch + 1) % 50 == 0:
            torch.save(G.state_dict(), f"output/enhanced_unet/epoch{epoch+1}_G.pth")
            # torch.save(D.state_dict(), f"output/epoch{epoch+1}_D.pth")


else:
    for epoch in range(total_epoches):
        for idx,(real_imgs,real_labels) in tqdm.tqdm(enumerate(trainloader)):
            optimizer_G.zero_grad()

            real_imgs = real_imgs.cuda() # torch.Size([2, 3, 640, 640])
            real_labels = real_labels.cuda() # torch.Size([2, 640, 640])
            pred_labels=G(real_imgs) # torch.Size([2, 640, 640]) -> torch.Size([2, 1, 640, 640])

            Seg_Loss = bce_loss(pred_labels, real_labels.unsqueeze(1)) # 2, 1, 640, 640 -> scaler

            g_loss= Seg_Loss
            g_loss.backward()
            optimizer_G.step()
        
        print(f"epoch[{epoch}/{total_epoches}] segloss: {Seg_Loss}")

        if (epoch + 1) % 50 == 0:
            torch.save(G.state_dict(), f"output/unet/epoch{epoch+1}_G.pth")
            # torch.save(D.state_dict(), f"output/epoch{epoch+1}_D.pth")