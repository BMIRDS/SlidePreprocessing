import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torchvision
from torch import nn
from PIL import Image
from torchvision import transforms
import argparse
import tqdm
import os
import shutil

parser = argparse.ArgumentParser(description='Feature extraction')
parser.add_argument('-c', '--study', type=str, default='TCGA_BLCA')
parser.add_argument('-j', '--num-workers', type=int, default=10)
parser.add_argument('-m', '--magnification', type=int, default=10)
parser.add_argument('-s', '--patch-size', type=int, default=224)
parser.add_argument('-b', '--batch-size', type=int, default=256)
parser.add_argument('-l', '--num-layers', type=int, default=18)
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
