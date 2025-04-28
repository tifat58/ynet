#!/usr/bin/python
#
# Copyright 2022 Azade Farshad
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

import argparse
from os import path, makedirs

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from torchinfo import summary

from data import get_data
from networks import UNet, get_model
from utils import plot_mult, plot_single, per_class_dice, mIOU

import torchvision.transforms as tf
import cv2


SEG_MARKERS = {
    0 : "Region above the retina (RaR)",
    1 : "ILM: Inner limiting membrane",
    2 : "NFL-IPL: Nerve fiber ending to Inner plexiform layer",
    3 : "INL: Inner Nuclear layer",
    4 : "OPL: Outer plexiform layer",
    5 : "ONL-ISM: Outer Nuclear layer to Inner segment myeloid",
    6 : "ISE: Inner segment ellipsoid",
    7 : "OS-RPE: Outer segment to Retinal pigment epithelium",
    8 : "Region below RPE (RbR)",
    -1 : "void"
}

class TransformOCTBilinear(object):
    def __new__(cls, img_size=(128,128),*args, **kwargs):
        return tf.Compose([
            tf.Resize(img_size)
            ])

class TransformStandardization(object):
    """
    Standardizaton / z-score: (x-mean)/std
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        return (image - self.mean) / self.std

    def __repr__(self):
        return self.__class__.__name__ + f": mean {self.mean}, std {self.std}"

def argument_parser():
    parser = argparse.ArgumentParser()

    # Optimization hyperparameters
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--n_classes', default=9, type=int)

    # Dataset options
    parser.add_argument('--dataset', default='Duke', choices=["Duke", "UMN"])
    parser.add_argument('--image_size', default='224', type=int)

    parser.add_argument('--image_dir', default="./DukeData/")

    # Network options
    parser.add_argument('--g_ratio', default=0.5, type=float)

    # Other options
    parser.add_argument('--device', default="cuda", choices=["cuda", "cpu"])
    parser.add_argument('--seed', default=7, type=int)
    parser.add_argument('--print_params', default=False)
    parser.add_argument('--pretrained_path', default="./pretrained_models/")

    parser.add_argument('--input', default='input.png')
    parser.add_argument('--output', default='outputs/')

    return parser


def plot_samples_2mod(model1, model2, testset, idx_=None):
    model1.eval()
    model2.eval()
    plt.axis('off')
    plt.rcParams["text.usetex"] = True

    if idx_ is None:
        idx_ = np.random.randint(0, len(testset))

    img, label = testset.__getitem__(idx_)

    img = img.unsqueeze(0).to(device='cuda')
    label_e1 = label.unsqueeze(0).to(device='cuda')

    pred1 = model1(img)
    pred2 = model2(img)
    _, idx1 = torch.max(pred1, 1)
    _, idx2 = torch.max(pred2, 1)

    im_out = img[0][0].cpu().numpy()
    lb_np_e1 = label_e1[0][0].cpu().numpy()
    pred1_np = idx1[0].detach().cpu().numpy()
    pred2_np = idx2[0].detach().cpu().numpy()

    labels = [im_out, lb_np_e1, pred1_np, pred2_np]
    names = ["Input Image", "Expert 1", "U-Net", r'$\Upsilon$'"-Net (Ours)"]

    plot_mult(labels, names, True, idx_)


def qual_eval(testset, model, model_2):
    for i in range(len(testset)):
        plot_samples_2mod(model, model_2, testset, idx_=i)


def quant_eval(model, test_loader, n_classes, device="cuda"):
    dice = 0
    dice_all = np.zeros(n_classes)
    iou_all = 0
    counter = 0

    for img, label in tqdm.tqdm(test_loader):
        img = img.to(device=device)
        label = label.to(device=device)
        label_oh = torch.nn.functional.one_hot(label, num_classes=n_classes)

        pred = model(img)
        max_val, idx = torch.max(pred, 1)
        pred_oh = torch.nn.functional.one_hot(idx, num_classes=n_classes)

        d1, d2 = per_class_dice(pred_oh, label_oh, n_classes)
        iou = mIOU(label, pred, n_classes)
        iou_all += iou
        dice += d1
        dice_all += d2

        counter += 1

    dice_all = dice_all / counter
    iou_all = iou_all / counter
    dice_all = [round(x, 2) for x in dice_all]
    dice = np.mean(dice_all[1:])
    print(" Mean Dice: ", dice, "Dice All:", dice_all, "mIoU All: ", iou_all)


def eval_unet_vs_ynet(testloader, testset, args):
    n_classes = args.n_classes

    unet_model = UNet(1, n_classes).to(args.device)
    unet_path = path.join(args.pretrained_path, "unet.pt")
    unet_model.load_state_dict(torch.load(unet_path))

    ynet_model = get_model("y_net_gen_ffc", ratio=args.g_ratio, num_classes=n_classes).to(args.device)
    ynet_path = path.join(args.pretrained_path, "y_net_gen_ffc.pt")
    ynet_model.load_state_dict(torch.load(ynet_path))

    unet_model.eval()
    ynet_model.eval()

    print("UNet Dice Score:")
    quant_eval(unet_model, testloader, n_classes=n_classes)
    print("YNet Dice Score:")
    quant_eval(ynet_model, testloader, n_classes=n_classes)
    print("Generating Qualitative Results")
    if not path.exists("./figs"):
        makedirs("./figs")
    qual_eval(testset, unet_model, ynet_model)


def process_image(img, img_size=224):
    # convert to tensor
    img = img.squeeze()
    img = torch.Tensor(img).reshape(1, 1, *img.shape)
    # apply transforms
    normalize = TransformStandardization((46.3758),
                                         (53.9434))
    size_transform = TransformOCTBilinear(img_size=(img_size, img_size))
    img = size_transform(img)
    img = normalize(img)
    return img


def print_params(n_classes):
    input_shape = (1, 1, 224, 224)

    unet_model = UNet(1, n_classes).cuda()
    ynet_model = get_model("y_net_gen", ratio=0.5).cuda()

    print("UNet")
    summary(unet_model, input_shape)

    print("YNet")
    summary(ynet_model, input_shape)


def eval_single_ynet(image_path: str, output_path : str, cmap='viridis'):

    args = argument_parser().parse_args()
    device = args.device
    img_size = args.image_size
    if args.print_params:
        print_params(args.n_classes)
    n_classes = args.n_classes

    # load image for inference
    try:
        img = cv2.imread(image_path, 0)
        print("Reading input image from: ", image_path)
    except Exception as e:
        img = None
        print(e)

    initial_img_size = img.shape

    p_img = process_image(img, img_size=224)
    # get model
    ynet_model = get_model("y_net_gen_ffc", ratio=args.g_ratio, num_classes=n_classes).to(args.device)
    ynet_path = path.join(args.pretrained_path, "y_net_gen_ffc.pt")
    ynet_model.load_state_dict(torch.load(ynet_path))
    ynet_model.eval()


    p_img = p_img.to(device=device)
    pred = ynet_model(p_img)
    _, idx1 = torch.max(pred, 1)
    pred1_np = idx1[0].detach().cpu().numpy()
    # pred1_res= cv2.resize(pred1_np, dsize=(420,420), interpolation=cv2.INTER_CUBIC)
    # im_out = img[0][0].cpu().numpy()

    # saving image
    file_path = output_path
    plt.axis('off')
    plt.imshow(pred1_np, cmap=cmap)
    plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
    print('saving pred image to: ', file_path)
    print('\n')

    # return prediction image and  its file_path
    return pred1_np, file_path


def compute_regions(img1, img2):

    total_area = img1.shape[0] * img1.shape[1]
    # print("Total Area: ", total_area)

    colors1, counts1 = np.unique(img1.flatten(),
                               return_counts=True,
                               axis=0)

    colors2, counts2 = np.unique(img2.flatten(),
                               return_counts=True,
                               axis=0)

    comment_dict = {}
    for i in range(len(colors1)):

        if colors1[i] in colors2:
            idx = np.where(colors2 == colors1[i])[0][0]

            diff = counts1[i] - counts2[idx]
            # print("Diff: ", diff)
            if diff > 0:
                diff_percnt = (diff / counts1[i]) * 100
                comment_dict[SEG_MARKERS[colors1[i]]] = 'decreased by ' + str(round(diff_percnt,2)) + '%'
                # pred_str = SEG_MARKERS[colors1[i]] + " has decreased by " + str(round(diff_percnt,2)) + "%"
            elif diff < 0:
                diff_percnt = (abs(diff) / counts1[i]) * 100
                comment_dict[SEG_MARKERS[colors1[i]]] = 'increased by ' + str(round(diff_percnt,2)) + '%'
                # pred_str = SEG_MARKERS[colors1[i]] + " has increased by " + str(round(diff_percnt,2)) + "%"
            else:
                comment_dict[SEG_MARKERS[colors1[i]]] = 'unchanged'
                # pred_str = SEG_MARKERS[colors1[i]] + " has remained unchanged"

    return comment_dict


if __name__ == "__main__":

    img1_path = 'oct_image/Zert-AMD-AnnotatorID_001-15.png'
    img1_out_path = 'output/Zert-AMD-AnnotatorID_001-15_out.png'

    img2_path = 'oct_image/Zert-AMD-AnnotatorID_002-35.png'
    img2_out_path = 'output/Zert-AMD-AnnotatorID_002-35.png'

    # cmap = 'gray' / 'viridis' for grayscale or color images respectively
    pred1, pred1_path = eval_single_ynet(img1_path, img1_out_path, cmap='gray')
    pred2, pred2_path = eval_single_ynet(img2_path, img2_out_path, cmap='gray')

    comment_dict = compute_regions(pred1, pred2)
    for k,v in comment_dict.items():
        print(k + " has " + v)