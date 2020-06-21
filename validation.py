import argparse
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

import dataset
import utils

if __name__ == "__main__":

    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # Pre-train, saving, and loading parameters
    parser.add_argument('--pre_train', type = bool, default = False, help = 'pre-train ot not')
    parser.add_argument('--load_name', type = str, \
        default = './COCO2014_sigma10_models/DnCNN_epoch50_bs16_mu0_sigma10.pth', \
            help = 'test model name')
    # Initialization parameters
    parser.add_argument('--pad', type = str, default = 'reflect', help = 'pad type of networks')
    parser.add_argument('--activ', type = str, default = 'relu', help = 'activation type of networks')
    parser.add_argument('--norm', type = str, default = 'bn', help = 'normalization type of networks')
    parser.add_argument('--in_channels', type = int, default = 3, help = 'input channels for generator')
    parser.add_argument('--out_channels', type = int, default = 3, help = 'output channels for generator')
    parser.add_argument('--start_channels', type = int, default = 64, help = 'start channels for generator')
    parser.add_argument('--m_block', type = int, default = 17, help = 'number of convolutional blocks in DnCNN, 17 or 20')
    # Dataset parameters
    parser.add_argument('--baseroot', type = str, \
        default = "C:\\Users\\yzzha\\Desktop\\dataset\\ILSVRC2012_val_256_maksed", \
            help = 'the testing folder')
    parser.add_argument('--batch_size', type = int, default = 1, help = 'test batch size, always 1')
    parser.add_argument('--crop_size', type = int, default = 256, help = 'single patch size')
    '''
    parser.add_argument('--geometry_aug', type = bool, default = False, help = 'geometry augmentation (scaling)')
    parser.add_argument('--angle_aug', type = bool, default = False, help = 'geometry augmentation (rotation, flipping)')
    parser.add_argument('--scale_min', type = float, default = 1, help = 'min scaling factor')
    parser.add_argument('--scale_max', type = float, default = 1, help = 'max scaling factor')
    '''
    parser.add_argument('--max_white_value', type = int, default = 255, help = 'dynamic range of images')
    parser.add_argument('--mu', type = float, default = 0, help = 'min scaling factor')
    parser.add_argument('--sigma', type = float, default = 0.05, help = 'max scaling factor')
    opt = parser.parse_args()
    print(opt)

    # ----------------------------------------
    #       Initialize testing dataset
    # ----------------------------------------

    # Define the dataset
    testset = dataset.FullResDenoisingDataset(opt, opt.baseroot)
    print('The overall number of images equals to %d' % len(testset))

    # Define the dataloader
    dataloader = DataLoader(testset, batch_size = opt.batch_size, pin_memory = True)

    # ----------------------------------------
    #                 Testing
    # ----------------------------------------

    generator = utils.create_generator(opt)
    generator = generator.cuda()

    for batch_idx, (noisy_img, img, imgpath) in enumerate(dataloader):

        # To Tensor
        noisy_img = noisy_img.cuda()
        img = img.cuda()
        imgpath = imgpath[0]

        # Generator output
        res_img = generator(noisy_img)
        recon_img = noisy_img - res_img
        print(recon_img.shape)

        # convert to visible image format
        img = img.cpu().numpy().reshape(3, opt.crop_size, opt.crop_size).transpose(1, 2, 0)
        img = img * 255
        img = img.astype(np.uint8)
        recon_img = recon_img.detach().cpu().numpy().reshape(3, opt.crop_size, opt.crop_size).transpose(1, 2, 0)
        recon_img = recon_img * 255
        recon_img = recon_img.astype(np.uint8)

        # show
        show_img = np.concatenate((img, recon_img), axis = 1)
        r, g, b = cv2.split(show_img)
        show_img = cv2.merge([b, g, r])
        cv2.imshow('comparison.jpg', show_img)
        cv2.waitKey(1)
        cv2.imwrite('result_%d.jpg' % batch_idx, recon_img)
