import torch
import os
import os.path as osp
import numpy as np
from PIL import Image
from misc.data_aug import ColorAug,Add_Gaussion_noise,Random_horizontal_flip,Random_vertical_flip,Compose_imglabel,Random_crop
import collections
from transform import ReLabel, ToLabel, Scale
from transform import HorizontalFlip, VerticalFlip
from torchvision.transforms import Compose
from torchvision.transforms import Compose, CenterCrop, Normalize, ToTensor

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, root, split="training"):
        self.root = root
        self.split = split

        self.files = collections.defaultdict(list)
        if self.split == "training":
            self.img_transform = Compose([
                ColorAug(),
                ToTensor(),
                Normalize([.585, .256, .136], [.229, .124, .095]),
            ])
        else:
            self.img_transform = Compose([
                ToTensor(),
                Normalize([.585, .256, .136], [.229, .124, .095]),
            ])
        self.label_transform = Compose([
            ToLabel(),
            ReLabel(255, 1),
        ])
        self.image_label_transform=Compose_imglabel([
            Random_horizontal_flip(0.5), # transform
            Random_vertical_flip(0.5),
        ])

        data_dir = osp.join(root, "DRIVE", split)

        img_dir = osp.join(data_dir, "images")
        vessel_dir = os.path.join(data_dir, "1st_manual")
        mask_dir = os.path.join(data_dir, "mask")

        for img_file_name, label_file_name, mask_file_name in zip(os.listdir(img_dir), os.listdir(vessel_dir), os.listdir(mask_dir)):
            self.files[split].append({
                "img": os.path.join(img_dir, img_file_name),
                "label": os.path.join(vessel_dir, label_file_name),
                "mask": os.path.join(mask_dir, mask_file_name)
            })

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        datafiles = self.files[self.split][index]

        img_file = datafiles["img"]
        img = Image.open(img_file).convert('RGB')

        label_file = datafiles["label"]
        label = Image.open(label_file).convert("P")

        mask_file = datafiles["mask"]
        mask = Image.open(mask_file).convert("P")

        if self.split == "training":
            img, label = self.image_label_transform(img, label)
            img = self.img_transform(img)
            label = self.label_transform(label)
        else:
            img = self.img_transform(img)
            label = self.label_transform(label)
            mask = self.label_transform(mask)
            mask = torch.nn.functional.pad(mask, (0, 640-565, 0, 640-584))

        img = torch.nn.functional.pad(img, (0, 640-565, 0, 640-584))
        label = torch.nn.functional.pad(label, (0, 640-565, 0, 640-584))

        if self.split == "training":
            return img, label
        else:
            return img, label, mask, [img_file, label_file]
    




if __name__ == '__main__':

    input_transform = Compose([
        ColorAug(),
        Add_Gaussion_noise(prob=0.5),
        #Scale((512, 512), Image.BILINEAR),
        ToTensor(),
        Normalize([.485, .456, .406], [.229, .224, .225]),

    ])
    target_transform = Compose([
        #Scale((512, 512), Image.NEAREST),
        #ToSP(512),
        ToLabel(),
        ReLabel(255, 1),
    ])

    img_label_transform = Compose_imglabel([
        Random_crop(512,512),
        Random_horizontal_flip(0.5),
        Random_vertical_flip(0.5),
    ])
    dst = MyDataset("./", img_transform=input_transform,label_transform=target_transform,image_label_transform=img_label_transform)
    trainloader = torch.utils.data.DataLoader(dst, batch_size=1)

    for i, data in enumerate(trainloader):
        imgs, labels = data



