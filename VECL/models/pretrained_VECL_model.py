import torch.nn as nn
import torch
from .. import builder
from .. import loss


class PretrainedImageClassifier(nn.Module):
    def __init__(self, cfg_ckpt, cfg_classify, cfg):
        super(PretrainedImageClassifier, self).__init__()
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.classify_loss = loss.classification_loss.CLASSIFYLoss()
        self.classify_plus_loss = loss.classification_loss.CLASSIFY_PLUS_Loss()
        self.experiment_name = cfg.experiment_name
        self.cfg_ckpt = cfg_ckpt
        self.cfg_classify = cfg_classify
        self.text_encoder = builder.build_text_model(self.cfg_ckpt)
        self.img_encoder = builder.build_img_model(self.cfg_ckpt)
        self.fusion_module = builder.build_fusion_module(self.cfg_ckpt)
        ckpt = torch.load(self.cfg_classify.model.ckpt_path, map_location=self.device)
        ckpt_dict = ckpt["state_dict"]
        fixed_img_encoder_ckpt_dict = {}
        fixed_text_encoder_ckpt_dict = {}
        fixed_fusion_module_ckpt_dict = {}
        img_encoder_weights = self.img_encoder.state_dict()
        text_encoder_weights = self.text_encoder.state_dict()
        fusion_module_weights = self.fusion_module.state_dict()
        for k, v in ckpt_dict.items():
            new_key = k.split("VECL_model.")[-1]
            if "img_encoder" in new_key:
                new_key = k.split("img_encoder.")[-1]
            if "text_encoder" in new_key:
                new_key = k.split("text_encoder.")[-1]
            if "fusion_module" in new_key:
                new_key = k.split("fusion_module.")[-1]
            if new_key in img_encoder_weights:
                fixed_img_encoder_ckpt_dict[new_key] = v
            if new_key in text_encoder_weights:
                fixed_text_encoder_ckpt_dict[new_key] = v
            if new_key in fusion_module_weights:
                fixed_fusion_module_ckpt_dict[new_key] = v
        self.img_encoder.load_state_dict(fixed_img_encoder_ckpt_dict, strict=True)
        self.text_encoder.load_state_dict(fixed_text_encoder_ckpt_dict, strict=True)
        self.fusion_module.load_state_dict(fixed_fusion_module_ckpt_dict, strict=True)
        for param in self.img_encoder.parameters():
            param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for param in self.fusion_module.parameters():
            param.requires_grad = False
        for name, param in self.fusion_module.named_parameters():
            if 'mlp_head' in name:
                param.requires_grad = True

    def image_encoder_forward(self, imgs):
        img_feat_g, img_emb_l = self.img_encoder(imgs, get_local=True)
        img_emb_g, img_emb_l = self.img_encoder.generate_embeddings(img_feat_g, img_emb_l)
        return img_emb_l, img_emb_g

    def text_encoder_forward(self, caption_ids, attention_mask, token_type_ids):
        text_emb_l, text_emb_g, sents = self.text_encoder(caption_ids, attention_mask, token_type_ids)
        return text_emb_l, text_emb_g, sents

    def calc_loss(self, i2t_cls, t2i_cls, label):

        loss = 0

        ce_loss0 = self.classify_plus_loss(i2t_cls, label)
        ce_loss1 = self.classify_plus_loss(t2i_cls, label)
        loss += ce_loss0
        loss += ce_loss1

        return loss

    def forward(self, image, text, caption_ids, attention_mask, token_type_ids):
        img_emb_l, img_emb_g = self.image_encoder_forward(image)
        img_emb_l_ = img_emb_l.view(img_emb_l.size(0), img_emb_l.size(1), -1)  # [512, 768, 14, 14] -> [512, 768, 196]
        img_emb_l_ = img_emb_l_.permute(0, 2, 1)  # patch_num b dim # [196, 512, 768]
        text_emb_l, text_emb_g, sents = self.text_encoder_forward(caption_ids, attention_mask, token_type_ids)
        text_emb_l_ = text_emb_l.view(text_emb_l.size(0), text_emb_l.size(1), -1)
        text_emb_l_ = text_emb_l_.permute(0, 2, 1)  # patch_num b dim # [97, 512, 768]
        i2t_cls = self.fusion_module(torch.cat([img_emb_g.unsqueeze(1), img_emb_l_], dim=1), text_emb_g).squeeze(-1)
        t2i_cls = self.fusion_module(torch.cat([text_emb_g.unsqueeze(1), text_emb_l_], dim=1), img_emb_g).squeeze(-1)
        t2i_cls = t2i_cls.transpose(1, 0)
        return i2t_cls, t2i_cls
