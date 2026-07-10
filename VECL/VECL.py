import os
import torch
import pandas as pd
from . import builder
from .paths import remap_image_path
from typing import Union
import omegaconf


def load_VECL(
    ckpt_path = None,
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
    cfg_path = None
):
    if not os.path.exists(ckpt_path):
        raise RuntimeError(f"Model {ckpt_path} not found.")

    ckpt = torch.load(ckpt_path, map_location=device)
    try:
        cfg = ckpt["hyper_parameters"]
    except:
        cfg = omegaconf.OmegaConf.load(cfg_path)

    ckpt_dict = ckpt["state_dict"]

    VECL_model = builder.build_VECL_model(cfg).to(device)

    model_weights = VECL_model.state_dict()

    if 'pretrain' in cfg.experiment_name:
        prefix = 'VECL_model.'
    elif 'finetune_classification' in cfg.experiment_name:
        prefix = 'classification_model.'
    else:
        raise ValueError('prefix error')

    fixed_ckpt_dict = {}
    for k, v in ckpt_dict.items():
        new_key = k.split(prefix)[-1]
        if new_key in model_weights:
            fixed_ckpt_dict[new_key] = v
    ckpt_dict = fixed_ckpt_dict
    VECL_model.load_state_dict(ckpt_dict, strict=True)
    return VECL_model


def classification_inference(VECL_model, imgs, cls_txt_mapping):

    caption_ids = []
    attention_mask = []
    token_type_ids = []
    for cls_name, txts in cls_txt_mapping.items():
        caption_ids.append(txts["caption_ids"])
        attention_mask.append(txts["attention_mask"])
        token_type_ids.append(txts["token_type_ids"])

    caption_ids = torch.cat(caption_ids, dim=0)
    attention_mask = torch.cat(attention_mask, dim=0)
    token_type_ids = torch.cat(token_type_ids, dim=0)
    text_batch = {"caption_ids": caption_ids, "attention_mask": attention_mask, "token_type_ids":token_type_ids}

    cls_similarity = get_similarities(VECL_model, imgs, text_batch)
    class_similarities_p = pd.DataFrame(cls_similarity[:,:,0], columns=cls_txt_mapping.keys())
    class_similarities_0 = pd.DataFrame(cls_similarity[:,:,1], columns=cls_txt_mapping.keys())
    class_similarities_n = pd.DataFrame(cls_similarity[:,:,2], columns=cls_txt_mapping.keys())

    return [class_similarities_p, class_similarities_0, class_similarities_n]

def get_similarities(VECL_model, imgs, txts):
    with torch.no_grad():
        VECL_model.eval()
        label_img_emb_l, label_img_emb_g = VECL_model.image_encoder_forward(imgs)
        query_emb_l, query_emb_g, _ = VECL_model.text_encoder_forward(txts["caption_ids"], txts["attention_mask"], txts["token_type_ids"])
        bs = label_img_emb_g.size(0)
        cls_bs = []
        for i in range(bs):
            label_img_emb_l_ = label_img_emb_l[i:i+1].view(label_img_emb_l[i:i+1].size(0), label_img_emb_l[i:i+1].size(1), -1)
            label_img_emb_g_ = label_img_emb_g[i:i+1]
            label_img_emb_l_ = label_img_emb_l_.permute(0, 2, 1) #patch_num b dim
            query_emb_l_ = query_emb_l.view(query_emb_l.size(0), query_emb_l.size(1), -1)
            query_emb_l_ = query_emb_l_.permute(0, 2, 1) #patch_num b dim # [97, b, 768]
            i2t_cls = VECL_model.fusion_module(torch.cat([label_img_emb_g_.unsqueeze(1) , label_img_emb_l_], dim=1), query_emb_g)
            t2i_cls = VECL_model.fusion_module(torch.cat([query_emb_g.unsqueeze(1) , query_emb_l_], dim=1), label_img_emb_g_)
            i2t_cls = i2t_cls.squeeze(-1)
            t2i_cls = t2i_cls.squeeze(-1).transpose(1,0)
            cls = (i2t_cls + t2i_cls) / 2
            cls_bs.append(cls)
        cls = torch.cat(cls_bs, dim=0)
        return cls.detach().cpu().numpy()