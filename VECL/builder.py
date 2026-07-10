import torch
import torchvision.transforms as transforms
from . import models
from . import lightning
from . import datasets


def build_data_module(cfg):
    if cfg.data.dataset.lower() == 'pretrain':
        data_module = datasets.DATA_MODULES['pretrain']
    else:
        data_module = datasets.DATA_MODULES['finetune']
    return data_module(cfg)

def build_lightning_model(cfg, dm):
    module = lightning.LIGHTNING_MODULES[cfg.phase.lower()]
    module = module(cfg)
    module.dm = dm
    return module

def build_VECL_model(cfg):
    VECL_model = models.VECL_Model.VECL_model(cfg)
    return VECL_model

def build_classification_model(cfg_ckpt, cfg_classify, cfg):
    classification_model = models.pretrained_VECL_model.PretrainedImageClassifier(cfg_ckpt, cfg_classify, cfg)
    return classification_model

def build_fusion_module(cfg):
    fusion_module = models.fusion_module.Fusion_Model(cfg)
    return fusion_module

def build_img_model(cfg):
    image_model = models.vision_model.ImageEncoder(cfg)
    return image_model

def build_text_model(cfg):
    text_model = models.text_model.BertEncoder(cfg)
    return text_model

def build_optimizer(cfg, lr, model):

    # get params for optimization
    params = []
    for p in model.parameters():
        if p.requires_grad:
            params.append(p)

    # define optimizers
    if cfg.train.optimizer.name == "SGD":
        return torch.optim.SGD(
            params, lr=lr, momentum=cfg.train.optimizer.momentum, weight_decay=cfg.train.optimizer.weight_decay
        )
    elif cfg.train.optimizer.name == "Adam":
        return torch.optim.Adam(
            params,
            lr=lr,
            weight_decay=cfg.train.optimizer.weight_decay,
            betas=(0.5, 0.999),
        )
    elif cfg.train.optimizer.name == "AdamW":
        return torch.optim.AdamW(
            params, lr=lr, weight_decay=cfg.train.optimizer.weight_decay
        )

def build_scheduler(cfg, optimizer, dm=None):

    if cfg.train.scheduler.name == "warmup":

        def lambda_lr(epoch):
            if epoch <= 3:
                return 0.001 + epoch * 0.003
            if epoch >= 22:
                return 0.01 * (1 - epoch / 200.0) ** 0.9
            return 0.01

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_lr)
    elif cfg.train.scheduler.name == "cos":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    elif cfg.train.scheduler.name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=5
        )
    elif cfg.train.scheduler.name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
    else:
        scheduler = None

    if cfg.lightning.trainer.val_check_interval is not None:
        cfg.train.scheduler.interval = "step"
        num_iter = len(dm.train_dataloader().dataset)
        if type(cfg.lightning.trainer.val_check_interval) == float:
            frequency = int(num_iter * cfg.lightning.trainer.val_check_interval)
            cfg.train.scheduler.frequency = frequency
        else:
            cfg.train.scheduler.frequency = cfg.lightning.trainer.val_check_interval

    scheduler = {
        "scheduler": scheduler,
        "monitor": cfg.train.scheduler.monitor,
        "interval": cfg.train.scheduler.interval,
        "frequency": cfg.train.scheduler.frequency,
    }
    return scheduler

def build_transformation(cfg, split):

    t = []
    if split == "train":

        if cfg.transforms.random_crop is not None:
            t.append(transforms.RandomCrop(cfg.transforms.random_crop.crop_size))

        if cfg.transforms.random_horizontal_flip is not None:
            t.append(
                transforms.RandomHorizontalFlip(p=cfg.transforms.random_horizontal_flip)
            )

        if cfg.transforms.random_affine is not None:
            t.append(
                transforms.RandomAffine(
                    cfg.transforms.random_affine.degrees,
                    translate=[*cfg.transforms.random_affine.translate],
                    scale=[*cfg.transforms.random_affine.scale],
                )
            )

        if cfg.transforms.color_jitter is not None:
            t.append(
                transforms.ColorJitter(
                    brightness=[*cfg.transforms.color_jitter.bightness],
                    contrast=[*cfg.transforms.color_jitter.contrast],
                )
            )
    else:
        if cfg.transforms.random_crop is not None:
            t.append(transforms.CenterCrop(cfg.transforms.random_crop.crop_size))

    t.append(transforms.ToTensor())
    if cfg.transforms.norm is not None:
        if cfg.transforms.norm == "imagenet":
            t.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
        elif cfg.transforms.norm == "half":
            t.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        elif cfg.transforms.norm == "CXR_MAE":
            t.append(transforms.Normalize(mean=[0.4978], std=[0.2449]))
        elif cfg.transforms.norm == "CLIP":
            t.append(transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)))
        else:
            raise NotImplementedError("Normaliation method not implemented")

    return transforms.Compose(t)
