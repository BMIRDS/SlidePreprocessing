"""
This script extracts features from slide patches using a pre-trained ResNet model.

Usage: The script is executed from the command line and takes arguments for 
the study name, patch size, magnification, number of layers, and batch size.

Command-line arguments:

'-c', '--study': The name of the study (default: 'TCGA_BLCA')
'-j', '--num-workers': The number of workers (default: 10)
'-m', '--magnification': The magnification of the slide patches (default: 10)
'-s', '--patch-size': The size of the slide patches (default: 224)
'-b', '--batch-size': The batch size for processing patches (default: 256)
'-l', '--num-layers': The number of layers for the ResNet model (default: 18)

"""

from PIL import Image
import argparse
import os
import shutil

from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import pandas as pd
import torch
import torchvision
import tqdm


parser = argparse.ArgumentParser(description='Feature extraction')
parser.add_argument('-c', '--study', 
                    type=str, 
                    default='TCGA_BLCA',
                    help="Name of the study to be used. Default is TCGA_BLCA.")
parser.add_argument('-j', '--num-workers', 
                    type=int, 
                    default=4,
                    help="Number of workers for data loading. Default is 4.")
parser.add_argument('-m', '--magnification', 
                    type=int, 
                    default=10,
                    help="Magnification level of slide patches. Default is 10.")
parser.add_argument('-s', '--patch-size', 
                    type=int, 
                    default=224,
                    help="Size of the slide patches. Default is 224.")
parser.add_argument('-b', '--batch-size', 
                    type=int, 
                    default=256,
                    help="Batch size for processing slide patches. Default is 256.")
parser.add_argument('-l', '--num-layers', 
                    type=int, 
                    default=18,
                    help="Number of layers in the ResNet model. Default is 18.")
args = parser.parse_args()

assert args.num_layers in [18,34,50]

# load slide patches
class SlidesDataset(Dataset):

    def __init__(
        self,
        data_file,
        transform=None,
    ):

        self.df = data_file
        self.transform = transform
        self.files = self.df.file.to_list()

    def __len__(self):
        return self.df.shape[0]

    def sample_patch(self, idx):
        idx = idx % self.df.shape[0]
        fname = self.files[idx]
        img = np.array(Image.open(fname))
        if self.transform:
            img = self.transform(img)
        return img

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.sample_patch(idx)


def create_model(num_layers, pretrain, num_classes):
    assert num_layers in [18, 34, 50, 101, 152]
    architecture = "resnet{}".format(num_layers)
    model_constructor = getattr(torchvision.models, architecture)
    model = model_constructor(num_classes=num_classes)

    if pretrain is True:
        print("Loading pretrained model!")
        try:
            pretrained = model_constructor(weights='IMAGENET1K_V1').state_dict()
        except:
            pretrained = model_constructor(pretrained=True).state_dict()
        if num_classes != pretrained['fc.weight'].size(0):
            del pretrained['fc.weight'], pretrained['fc.bias']
        model.load_state_dict(pretrained, strict=False)
    return model


df = pd.read_pickle(
    f'meta/{args.study}/patches_meta-mag_{args.magnification}-size_{args.patch_size}.pickle'
)
df.head()


PATH_MEAN = [0.7968, 0.6492, 0.7542]
PATH_STD = [0.1734, 0.2409, 0.1845]

model = create_model(args.num_layers, True, 1)
model.fc = nn.Identity()
model.cuda()
model.eval()

trf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(args.patch_size),
    transforms.Normalize(PATH_MEAN, PATH_STD)
])

ds = SlidesDataset(df, transform=trf)
dl = torch.utils.data.DataLoader(ds,
                                 shuffle=False,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers,
                                 pin_memory=False,
                                 drop_last=False)


save_dir = f"features/{args.study}/mag_{args.magnification}-size_{args.patch_size}/resnet_{args.num_layers}/"
if os.path.isdir(save_dir):
    shutil.rmtree(save_dir)
os.makedirs(
    f"features/{args.study}/mag_{args.magnification}-size_{args.patch_size}/resnet_{args.num_layers}/",
    exist_ok=True)

for i, imgs in tqdm.tqdm(enumerate(dl), total=len(dl)):
    ft_i = model(imgs.cuda())
    torch.save(
        ft_i,
        f"features/{args.study}/mag_{args.magnification}-size_{args.patch_size}/resnet_{args.num_layers}/{i:06d}.pt"
    )
