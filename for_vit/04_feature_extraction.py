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
import shutil

from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import pandas as pd
import torch
import torchvision
import tqdm

from utils.config import Config, default_options
from utils.print_utils import print_intro, print_outro
from utils.io_utils import create_patches_meta_path, create_features_dir

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


def main():
    args = default_options()
    config = Config(
        args.default_config_file,
        args.user_config_file)

    study_name = config.study.study_name
    magnification = config.patch.magnification
    patch_size = config.patch.patch_size
    patches_meta_path = create_patches_meta_path(study_name, magnification, patch_size)
    df = pd.read_pickle(patches_meta_path)
    df.head()

    PATH_MEAN = config.feature.path_mean
    PATH_STD = config.feature.path_std

    num_layers = int(config.patch.backbone.replace('resnet_', ''))
    assert num_layers in [18,34,50]

    model = create_model(num_layers, True, 1)
    model.fc = nn.Identity()
    model.cuda()
    model.eval()

    trf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(patch_size),
        transforms.Normalize(PATH_MEAN, PATH_STD)
    ])

    ds = SlidesDataset(df, transform=trf)
    dl = torch.utils.data.DataLoader(ds,
                                    shuffle=False,
                                    batch_size=config.patch.batch_size,
                                    num_workers=config.patch.num_workers,
                                    pin_memory=False,
                                    drop_last=False)


    features_save_dir = create_features_dir(study_name, magnification, patch_size, num_layers)
    if features_save_dir.is_dir():
        shutil.rmtree(features_save_dir)
    features_save_dir.mkdir(parents=True, exist_ok=True)

    for i, imgs in tqdm.tqdm(enumerate(dl), total=len(dl)):
        ft_i = model(imgs.cuda())
        torch.save(ft_i, features_save_dir / f"{i:06d}.pt")

if __name__ == '__main__':
    print_intro(__file__)
    main()
    print_outro(__file__)
