from .AE import LinearAutoEncoder, ConvAutoencoder
from .VAE import VAE

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


def get_model(cfg, num_classes):
    if cfg.MODEL.NAME == 'LAE':
        model = LinearAutoEncoder(cfg.MODEL.IN_CHANNEL, cfg.MODEL.LATENT_DIM, cfg.MODEL.LAE_ENCODER_HIDDEN_DIM,
                                  cfg.MODEL.LAE_DECODER_HIDDEN_DIM)
        if cfg.GLOBAL.PRETRAINED_MODEL:
            pretrained_weights = torch.load(cfg.GLOBAL.PRETRAINED_MODEL)  # 加载预训练模型
            load_pretrained_dict = {k: v for k, v in pretrained_weights.items()
                                    if model.state_dict()[k].numel() == v.numel()}  # 加载结构一致的权重
            model.load_state_dict(load_pretrained_dict, strict=False)

    elif cfg.MODEL.NAME == 'CAE':
        model = ConvAutoencoder(cfg.MODEL.IN_CHANNEL, cfg.MODEL.LATENT_DIM)
        if cfg.GLOBAL.PRETRAINED_MODEL:
            pretrained_weights = torch.load(cfg.GLOBAL.PRETRAINED_MODEL)  # 加载预训练模型
            load_pretrained_dict = {k: v for k, v in pretrained_weights.items()
                                    if model.state_dict()[k].numel() == v.numel()}  # 加载结构一致的权重
            model.load_state_dict(load_pretrained_dict, strict=False)

    elif cfg.MODEL.NAME == 'VAE':
        model = VAE(cfg.MODEL.IN_CHANNEL, cfg.MODEL.LATENT_DIM, cfg.MODEL.VAE_HIDDEN_DIM)
        if cfg.GLOBAL.PRETRAINED_MODEL:
            pretrained_weights = torch.load(cfg.GLOBAL.PRETRAINED_MODEL)  # 加载预训练模型
            load_pretrained_dict = {k: v for k, v in pretrained_weights.items()
                                    if model.state_dict()[k].numel() == v.numel()}  # 加载结构一致的权重
            model.load_state_dict(load_pretrained_dict, strict=False)

    return model


def is_parallel(model):
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def de_parallel(model):
    return model.module if is_parallel(model) else model


def parallel_model(model, device, rank, local_rank):
    # DDP mode
    ddp_mode = device.type != 'cpu' and rank != -1
    if ddp_mode:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    return model
