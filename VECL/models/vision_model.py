import torch.nn as nn
from . import transformer_backbones
from VECL.paths import M3AE_VITB16_PATH


class ImageEncoder(nn.Module):
    def __init__(self, cfg):
        super(ImageEncoder, self).__init__()
        self.cfg = cfg
        model_function = getattr(transformer_backbones, cfg.model.vision.model_name)
        if cfg.model.vision.pretrained == True:
            if not cfg.model.vision.pretrained_path:
                cfg.model.vision.pretrained_path = str(M3AE_VITB16_PATH)
            self.model, self.feature_dim = model_function(cfg.model.vision.pretrained_path)
        else:
            raise ValueError("Unpretrained Image Encoder is not supported.")
        self.global_embedder = nn.Identity()
        self.local_embedder = nn.Identity()
        if cfg.model.vision.freeze_encoder:
            print("Freezing VIT model")
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x, get_local=False):
        global_ft, local_ft = self.vit_forward(x)
        if get_local:
            return global_ft, local_ft
        else:
            return global_ft

    def generate_embeddings(self, global_features, local_features):
        global_emb = self.global_embedder(global_features)
        local_emb = self.local_embedder(local_features)
        return global_emb, local_emb

    def vit_forward(self, x):
        x, local_features = self.model(x)
        return x, local_features