import torch
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import yaml
from torchsummary import summary

from model import build_model
from datasets.mnist import MNIST


def yaml_parser(yaml_path):
    with open(yaml_path, 'r') as file:
        opt = argparse.Namespace(**yaml.load(file.read(), Loader=yaml.FullLoader))
    opt.GLOBAL = argparse.Namespace(**opt.GLOBAL)
    opt.MODEL = argparse.Namespace(**opt.MODEL)

    return opt


def get_args():
    parser = argparse.ArgumentParser(description='Visualization')
    parser.add_argument('--yaml', default='config/visualize.yaml', type=str, help='output model name')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter')

    return parser.parse_args()


@torch.no_grad()
def reconstruction_visualize(cfg):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = "cuda:0"

    dataset = MNIST(cfg, 'train')
    print(f"Processing dataset with {len(dataset)} images.")

    model = build_model.get_model(cfg, 0)

    model.load_state_dict(torch.load(cfg.GLOBAL.RESUME_PATH)['state_dict_backbone'])
    model.cuda()
    model.eval()

    summary(model, (1, 28, 28))

    for _, data in enumerate(dataset):
        image, label = data
        C, H, W = image.shape
        image = image[None]
        image_generate = model(image.to(device))

        image_np = torch.squeeze(image).cpu().numpy()
        image_generate_np = torch.squeeze(image_generate).cpu().numpy()

        print(((image_np - image_generate_np)**2).mean())

        # image_generate_np = (image_generate_np -

        plt.subplot(121)
        plt.imshow(image_np)
        plt.subplot(122)
        plt.imshow(image_generate_np)
        plt.show()


if __name__ == '__main__':
    args = get_args()
    cfg = yaml_parser(args.yaml)
    reconstruction_visualize(cfg)
