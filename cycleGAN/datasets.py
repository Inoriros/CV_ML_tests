import os
import glob
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image

import torchvision.transforms as trsForms

class ImageDataset(Dataset):
    def __init__(self, rootPth="", transform=None, model="train"):
        self.transform = trsForms.Compose(transform)
        # set path for A and B
        self.pathA = os.path.join(rootPth, model, "A/*")
        self.pathB = os.path.join(rootPth, model, "B/*")
        # put into list
        self.listA = glob.glob(self.pathA)
        self.listB = glob.glob(self.pathB)

    def __getitem__(self, index):
        im_pathA = self.listA[index % len(self.listA)]
        im_pathB = random.choice(self.listB)
        # get original images
        image_A = Image.open(im_pathA)
        image_B = Image.open(im_pathB)
        # image process to suitable data
        item_A = self.transform(image_A)
        item_B = self.transform(image_B)
        return {"A":item_A, "B":item_B}

    def __len__(self):
        return max(len(self.listA), len(self.listB))


if __name__=='__main__':
    # Change the root path to your setting
    rootPth = "~/Datasets/apple2orange"
    transform_ = [trsForms.Resize((256 ,256), trsForms.InterpolationMode.BILINEAR), trsForms.ToTensor()]
    dataloader = DataLoader(ImageDataset(rootPth, transform_, "train"), batch_size=1, shuffle=True, num_workers=1)

    for i, batch in enumerate(dataloader):
        print(i)
        print(batch)

