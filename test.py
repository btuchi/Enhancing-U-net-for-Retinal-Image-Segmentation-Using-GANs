import torch
from dataset import MyDataset
from torchvision.transforms import Compose, Normalize, ToTensor
from transform import ReLabel, ToLabel
from model import discriminator, generator
import numpy as np
from PIL import Image
import shutil
import os
import sklearn.metrics

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="unet")
args = parser.parse_args()

# args.model = "enhanced_unet"

valloader = torch.utils.data.DataLoader(
    MyDataset("data/", split='test'),
    batch_size=1,
    shuffle=False
)

G = generator(n_filters=32)
G.eval()

epoch = 1000


G_state_dict = torch.load(f"output/{args.model}/epoch{epoch}_G.pth", map_location="cpu")
G.load_state_dict(G_state_dict)


outputs_list, real_labels_list = [], []


for i, (real_imgs, real_labels, real_masks, source_files) in enumerate(valloader):
    with torch.no_grad():
        outputs = G(real_imgs)
    outputs = outputs.squeeze()[:584, :565]
    outputs = outputs.cpu().numpy()
    

    real_labels = real_labels.squeeze()[:584, :565]
    real_labels = real_labels.cpu().numpy()

    real_masks = real_masks.squeeze()[:584, :565]
    real_masks = real_masks.bool().cpu().numpy()

    masked_outputs = outputs[real_masks]         # (584, 565) -> (sum(real_masks), )
    masked_real_labels = real_labels[real_masks] # (584, 565) -> (sum(real_masks), )

    outputs_list.append(masked_outputs)
    real_labels_list.append(masked_real_labels)
    
    os.makedirs(f"output/{args.model}/case_{i}", exist_ok=True)
    Image.fromarray((outputs * 255).astype(np.uint8)).save(f"output/{args.model}/case_{i}/generate_mask.png")
    shutil.copyfile(source_files[0][0], f"output/{args.model}/case_{i}/source_image.tif")
    shutil.copyfile(source_files[1][0], f"output/{args.model}/case_{i}/real_mask.gif")


all_real_lables = np.concatenate(real_labels_list)
all_outputs = np.concatenate(outputs_list)

# import pdb; pdb.set_trace()
fpr, tpr, _ = sklearn.metrics.roc_curve(all_real_lables, all_outputs)
precision, recall, _ = sklearn.metrics.precision_recall_curve(all_real_lables, all_outputs)

AUC_ROC_SCORE = sklearn.metrics.roc_auc_score(all_real_lables, all_outputs)
AUC_PR_SCORE = sklearn.metrics.auc(recall, precision)
print(f"epoch: {epoch}")
print(f"AUC_ROC_SCORE: {AUC_ROC_SCORE}")
print(f"AUC_PR_SCORE: {AUC_PR_SCORE}")

def dice_coefficient(
    logits,
    targets,
    scale=1000,
    eps=1e-6,
):
    probs = logits
    numerator = 2 * (probs / scale * targets).sum()
    denominator = (probs / scale).sum() + (targets / scale).sum()
    res =  (numerator + eps) / (denominator + eps)
    return res

dice = dice_coefficient(all_outputs, all_real_lables)
print(f"DICE: {dice}\n")

d = {
    "fpr":fpr,
    "tpr":tpr,
    "precision":precision,
    "recall":recall
}
torch.save(d, open(f"output/{args.model}/plot.pth", "wb"))