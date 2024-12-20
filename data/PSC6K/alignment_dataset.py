import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from skimage import io


class AlignmentDataset(Dataset):
    """
    Edited from: Proposal Flow image pair dataset
    https://github.com/ignacio-rocco/weakalign/blob/master/data/pf_dataset.py
    Comment: the GeometricTnf is replaced with F.interpolate for simplicity.

    Args:
        csv_file (string): Path to the csv file with image names.
                           Should use the csv with suffix "_st" (source/target).
        dataset_path (string): Directory with the images.
        output_size (2-tuple): Desired output size
        transform (callable): Transformation for post-processing the training pair (eg. image normalization)

    """

    def __init__(self, csv_file, dataset_path, output_size=(224, 224), transform=None, category=None,
                 pck_procedure='scnet'):

        self.category_names = ['airplane', 'alarm', 'ant', 'ape', 'apple', 'armor', 'axe', 'banana', 'bat', 'bear',
                               'bee', 'beetle', 'bell', 'bench', 'bicycle', 'blimp', 'bread', 'butterfly', 'cabin',
                               'camel', 'candle', 'cannon', 'car', 'castle', 'cat', 'chair', 'chicken', 'church',
                               'couch', 'cow', 'crab', 'crocodilian', 'cup', 'deer', 'dog', 'dolphin', 'door', 'duck',
                               'elephant', 'eyeglasses', 'fan', 'fish', 'flower', 'frog', 'geyser', 'giraffe', 'guitar',
                               'hamburger', 'hammer', 'harp', 'hat', 'hedgehog', 'helicopter', 'hermit', 'horse',
                               'hot-air', 'hotdog', 'hourglass', 'jack-o-lantern', 'jellyfish', 'kangaroo', 'knife',
                               'lion', 'lizard', 'lobster', 'motorcycle', 'mouse', 'mushroom', 'owl', 'parrot', 'pear',
                               'penguin', 'piano', 'pickup', 'pig', 'pineapple', 'pistol', 'pizza', 'pretzel', 'rabbit',
                               'raccoon', 'racket', 'ray', 'rhinoceros', 'rifle', 'rocket', 'sailboat', 'saw',
                               'saxophone', 'scissors', 'scorpion', 'sea', 'seagull', 'seal', 'shark', 'sheep', 'shoe',
                               'skyscraper', 'snail', 'snake', 'songbird', 'spider', 'spoon', 'squirrel', 'starfish',
                               'strawberry', 'swan', 'sword', 'table', 'tank', 'teapot', 'teddy', 'tiger', 'tree',
                               'trumpet', 'turtle', 'umbrella', 'violin', 'volcano', 'wading', 'wheelchair', 'windmill',
                               'window', 'wine', 'zebra']
        self.out_h, self.out_w = output_size
        self.pairs = pd.read_csv(csv_file)
        self.category = self.pairs.iloc[:, 2].to_numpy().astype('float')
        if category is not None:
            cat_idx = np.nonzero(self.category == category)[0]
            self.category = self.category[cat_idx]
            self.pairs = self.pairs.iloc[cat_idx, :]
        self.img_A_names = self.pairs.iloc[:, 0]
        self.img_B_names = self.pairs.iloc[:, 1]
        self.point_A_coords = self.pairs.iloc[:, 3:5]
        self.point_B_coords = self.pairs.iloc[:, 5:]
        self.dataset_path = dataset_path
        self.transform = transform

        self.pck_procedure = pck_procedure

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        # get pre-processed images
        image_A, im_size_A = self.get_image(self.img_A_names, idx)
        image_B, im_size_B = self.get_image(self.img_B_names, idx)

        # get pre-processed point coords
        point_A_coords = self.get_points(self.point_A_coords, idx)
        point_B_coords = self.get_points(self.point_B_coords, idx)

        # compute PCK reference length L_pck (equal to max bounding box side in image_A)
        N_pts = torch.sum(torch.ne(point_A_coords[0, :], -1))

        if self.pck_procedure == 'pf':
            L_pck = torch.FloatTensor(
                [torch.max(point_A_coords[:, :N_pts].max(1)[0] - point_A_coords[:, :N_pts].min(1)[0])])
        elif self.pck_procedure == 'scnet':
            # modification to follow the evaluation procedure of SCNet
            point_A_coords[0, 0:N_pts] = point_A_coords[0, 0:N_pts] * 224 / im_size_A[1]
            point_A_coords[1, 0:N_pts] = point_A_coords[1, 0:N_pts] * 224 / im_size_A[0]

            point_B_coords[0, 0:N_pts] = point_B_coords[0, 0:N_pts] * 224 / im_size_B[1]
            point_B_coords[1, 0:N_pts] = point_B_coords[1, 0:N_pts] * 224 / im_size_B[0]

            im_size_A[0:2] = torch.FloatTensor([224, 224])
            im_size_B[0:2] = torch.FloatTensor([224, 224])

            L_pck = torch.FloatTensor([224.0])

        sample = {'source_image': image_A, 'target_image': image_B, 'source_im_size': im_size_A,
                  'target_im_size': im_size_B, 'source_points': point_A_coords, 'target_points': point_B_coords,
                  'L_pck': L_pck}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_image(self, img_name_list, idx):
        img_name = os.path.join(self.dataset_path, img_name_list.iloc[idx])
        image = io.imread(img_name)

        # get image size
        im_size = np.asarray(image.shape)

        # convert to torch Variable
        image = np.expand_dims(image.transpose((2, 0, 1)), 0)
        image = torch.Tensor(image.astype(np.float32))
        image = F.interpolate(image, (self.out_h, self.out_w)).squeeze(0)

        im_size = torch.Tensor(im_size.astype(np.float32))

        return (image, im_size)

    def get_points(self, point_coords_list, idx):
        X = np.fromstring(point_coords_list.iloc[idx, 0], sep=';')
        Y = np.fromstring(point_coords_list.iloc[idx, 1], sep=';')
        Xpad = -np.ones(20)
        Xpad[:len(X)] = X
        Ypad = -np.ones(20)
        Ypad[:len(X)] = Y
        point_coords = np.concatenate((Xpad.reshape(1, 20), Ypad.reshape(1, 20)), axis=0)

        # make arrays float tensor for subsequent processing
        point_coords = torch.Tensor(point_coords.astype(np.float32))
        return point_coords
