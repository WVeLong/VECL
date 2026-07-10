import torch
import torch.nn as nn
from .. import builder
from .. import loss
import re
from nltk.tokenize import RegexpTokenizer
from transformers import AutoTokenizer
import cv2
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import numpy as np


class VECL_model(nn.Module):
    def __init__(self, cfg):
        super(VECL_model, self).__init__()

        self.cfg = cfg
        self.text_encoder = builder.build_text_model(cfg)
        self.img_encoder = builder.build_img_model(cfg)
        self.fusion_module = builder.build_fusion_module(cfg)
        self.batch_size = self.cfg.train.batch_size
        self.pretrain_loss = loss.pretrain_loss.ThreeD_InfoNCE_Loss()
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.text.bert_type)
        self.ixtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}

    def text_encoder_forward(self, caption_ids, attention_mask, token_type_ids):
        text_emb_l, text_emb_g, sents = self.text_encoder(caption_ids, attention_mask, token_type_ids)
        return text_emb_l, text_emb_g, sents

    def image_encoder_forward(self, imgs):
        img_feat_g, img_emb_l = self.img_encoder(imgs, get_local=True)
        img_emb_g, img_emb_l = self.img_encoder.generate_embeddings(img_feat_g, img_emb_l)
        return img_emb_l, img_emb_g

    def calc_loss(self, i2t_cls, t2i_cls, label_sample, label_list):
        loss = 0
        ce_loss0 = self.pretrain_loss(i2t_cls, label_sample, label_list)
        ce_loss1 = self.pretrain_loss(t2i_cls, label_sample, label_list)
        loss += ce_loss0
        loss += ce_loss1
        return loss

    def forward(self, x):

        # img encoder branch
        img_emb_l, img_emb_g = self.image_encoder_forward(x["imgs"])
        img_emb_l_ = img_emb_l.view(img_emb_l.size(0), img_emb_l.size(1), -1)  # [B, 768, 14, 14] -> [B, 768, 196]
        img_emb_l_ = img_emb_l_.permute(0, 2, 1)  # patch_num b dim # [196, B, 768]

        # text encorder branch
        text_emb_l, text_emb_g, sents = self.text_encoder_forward(x["caption_ids"], x["attention_mask"], x["token_type_ids"])
        text_emb_l_ = text_emb_l.view(text_emb_l.size(0), text_emb_l.size(1), -1)
        text_emb_l_ = text_emb_l_.permute(0, 2, 1)  # patch_num b dim # [97, B, 768]

        # fusion
        i2t_cls = self.fusion_module(torch.cat([img_emb_g.unsqueeze(1), img_emb_l_], dim=1), text_emb_g).squeeze(-1)
        t2i_cls = self.fusion_module(torch.cat([text_emb_g.unsqueeze(1), text_emb_l_], dim=1), img_emb_g).squeeze(-1)
        t2i_cls = t2i_cls.transpose(1, 0)

        return i2t_cls, t2i_cls

    def process_text(self, text, device):

        if type(text) == str:
            text = [text]

        processed_text_tensors = []
        for t in text:
            # use space instead of newline
            t = t.replace("\n", " ")

            # split sentences
            splitter = re.compile("[0-9]+\.")
            captions = splitter.split(t)
            captions = [point.split(".") for point in captions]
            captions = [sent for point in captions for sent in point]

            all_sents = []

            for t in captions:
                t = t.replace("\ufffd\ufffd", " ")
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(t.lower())

                if len(tokens) <= 1:
                    continue

                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                all_sents.append(" ".join(included_tokens))

            t = " ".join(all_sents)

            text_tensors = self.tokenizer(
                t,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=self.cfg.data.text.word_num,
            )
            text_tensors["sent"] = [
                self.ixtoword[ix] for ix in text_tensors["input_ids"][0].tolist()
            ]
            processed_text_tensors.append(text_tensors)

        caption_ids = torch.stack([x["input_ids"] for x in processed_text_tensors])
        attention_mask = torch.stack(
            [x["attention_mask"] for x in processed_text_tensors]
        )
        token_type_ids = torch.stack(
            [x["token_type_ids"] for x in processed_text_tensors]
        )

        if len(text) == 1:
            caption_ids = caption_ids.squeeze(0).to(device)
            attention_mask = attention_mask.squeeze(0).to(device)
            token_type_ids = token_type_ids.squeeze(0).to(device)
        else:
            caption_ids = caption_ids.squeeze().to(device)
            attention_mask = attention_mask.squeeze().to(device)
            token_type_ids = token_type_ids.squeeze().to(device)

        cap_lens = []
        for txt in text:
            cap_lens.append(len([w for w in txt if not w.startswith("[")]))

        return {
            "caption_ids": caption_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "cap_lens": cap_lens,
        }

    def process_class_prompts(self, class_prompts, device):
        cls_2_processed_txt = {}
        for k, v in class_prompts.items():
            cls_2_processed_txt[k] = self.process_text(v, device)
        return cls_2_processed_txt

    def process_img(self, paths, device, delete_index, i, bs):

        transform = builder.build_transformation(self.cfg, split="test")

        if type(paths) == str:
            paths = [paths]

        # 初始化一个固定大小的列表
        all_imgs = [None] * len(paths)

        # 定义处理图像的函数
        def process_image(p, index, all_imgs, delete_index):
            if p is None:
                delete_index.append(i * bs + index)
                return

            # 检查文件路径是否存在
            if not os.path.exists(p):
                return

                # 读取图像
            x = cv2.imread(str(Path(p)), 0)
            if x is None:
                delete_index.append(i * bs + index)
                return

            # 对图像进行变换处理
            x = self._resize_img(x, self.cfg.data.image.imsize)
            img = Image.fromarray(x).convert("RGB")
            img = transform(img)

            # 将处理后的图像放入对应的索引位置
            all_imgs[index] = torch.tensor(img)

        # 控制并发线程数
        with ThreadPoolExecutor(max_workers=16) as executor:
            for index, p in enumerate(paths):
                executor.submit(process_image, p, index, all_imgs, delete_index)

        all_imgs = [img for img in all_imgs if img is not None]
        if len(all_imgs) == 0:
            raise RuntimeError("No readable images found in this batch. Check VECL_*_ROOT dataset path environment variables.")
        all_imgs = torch.stack(all_imgs).to(device)

        return all_imgs

    def _resize_img(self, img, scale):
        """
        Args:
            img - image as numpy array (cv2)
            scale - desired output image-size as scale x scale
        Return:
            image resized to scale x scale with shortest dimension 0-padded
        """
        size = img.shape
        max_dim = max(size)
        max_ind = size.index(max_dim)

        # Resizing
        if max_ind == 0:
            # image is heigher
            wpercent = scale / float(size[0])
            hsize = int((float(size[1]) * float(wpercent)))
            desireable_size = (scale, hsize)
        else:
            # image is wider
            hpercent = scale / float(size[1])
            wsize = int((float(size[0]) * float(hpercent)))
            desireable_size = (wsize, scale)
        resized_img = cv2.resize(
            img, desireable_size[::-1], interpolation=cv2.INTER_AREA
        )  # this flips the desireable_size vector

        # Padding
        if max_ind == 0:
            # height fixed at scale, pad the width
            pad_size = scale - resized_img.shape[1]
            left = int(np.floor(pad_size / 2))
            right = int(np.ceil(pad_size / 2))
            top = int(0)
            bottom = int(0)
        else:
            # width fixed at scale, pad the height
            pad_size = scale - resized_img.shape[0]
            top = int(np.floor(pad_size / 2))
            bottom = int(np.ceil(pad_size / 2))
            left = int(0)
            right = int(0)
        resized_img = np.pad(
            resized_img, [(top, bottom), (left, right)], "constant", constant_values=0
        )

        return resized_img