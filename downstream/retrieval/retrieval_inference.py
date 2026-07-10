from tqdm import tqdm
import argparse
import numpy as np
import torch
import pandas as pd
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from VECL import VECL
from VECL.paths import output_path, remap_image_path
from transformers import AutoTokenizer
import cv2
from PIL import Image
import torchvision.transforms as transforms
import pickle
import torch.nn.functional as F
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer_path', type=str, default=None)
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--test_data_path', type=str, default=None)
    parser.add_argument('--MIMIC_data_path', type=str, default=None)
    parser.add_argument('--report_corpus_path', type=str, default=None)
    parser.add_argument('--split_id', type=int, default=None)
    parser.add_argument('--num_chunks', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    args = parser.parse_args()
    return args

def split_list_into_chunks(lst, num_chunks):
    chunk_size = len(lst) // num_chunks
    chunks = [lst[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)]
    remainder = len(lst) % num_chunks
    for i in range(remainder):
        chunks[i].append(lst[num_chunks * chunk_size + i])
    return chunks

def get_imgs(img_path, transform=None):
    x = cv2.imread(str(img_path), 0)
    x = resize_img(x, 256)
    img = Image.fromarray(x).convert("RGB")
    if transform is not None:
        img = transform(img)
    return img

def resize_img(img, scale):
    size = img.shape
    max_dim = max(size)
    max_ind = size.index(max_dim)
    if max_ind == 0:
        wpercent = scale / float(size[0])
        hsize = int((float(size[1]) * float(wpercent)))
        desireable_size = (scale, hsize)
    else:
        hpercent = scale / float(size[1])
        wsize = int((float(size[0]) * float(hpercent)))
        desireable_size = (wsize, scale)
    resized_img = cv2.resize(img, desireable_size[::-1], interpolation=cv2.INTER_AREA)
    if max_ind == 0:
        pad_size = scale - resized_img.shape[1]
        left = int(np.floor(pad_size / 2))
        right = int(np.ceil(pad_size / 2))
        top = int(0)
        bottom = int(0)
    else:
        pad_size = scale - resized_img.shape[0]
        top = int(np.floor(pad_size / 2))
        bottom = int(np.ceil(pad_size / 2))
        left = int(0)
        right = int(0)
    resized_img = np.pad(resized_img, [(top, bottom), (left, right)], "constant", constant_values=0)
    return resized_img

def build_transformation():
    t = []
    t.append(transforms.RandomCrop(224))
    t.append(transforms.RandomHorizontalFlip(0.3))
    t.append(transforms.RandomAffine(30, translate=[0.1, 0.1], scale=[0.9, 1.1]))
    t.append(transforms.ColorJitter(brightness=[0.8, 1.2], contrast=[0.8, 1.2]))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean=[0.4978], std=[0.2449]))
    return transforms.Compose(t)

def batch_generator(data_list, batch_size):
    batch_size = min(batch_size, len(data_list))
    num_batches = (len(data_list) + batch_size - 1) // batch_size
    for i in range(num_batches):
        start_index = i * batch_size
        end_index = start_index + batch_size
        yield data_list[start_index:end_index]

def compute_logits(img_emb, text_emb):
    logit_scale = nn.Parameter(torch.log(torch.tensor(1 / 0.07)))
    logit_scale.data = torch.clamp(logit_scale.data, 0, 4.6052)
    logit_scale = logit_scale.exp()
    logits_per_text = torch.matmul(text_emb, img_emb.t()) * logit_scale
    return logits_per_text.t()


if __name__ == '__main__':

    args = parse_args()

    report_corpus = pd.read_csv(args.MIMIC_data_path)['Report Impression'].tolist()

    report_corpus_chunk = split_list_into_chunks(report_corpus, args.num_chunks)
    report_corpus_work = report_corpus_chunk[args.split_id - 1]
    test_data = pd.read_csv(args.test_data_path).values.tolist()
    del report_corpus, report_corpus_chunk

    # Init Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = []

    VECL_model = VECL.load_VECL(ckpt_path=args.ckpt_path, device=device)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    transform = build_transformation()

    # Model Inference
    with torch.no_grad():
        VECL_model.eval()
        idx = 0
        for batch in batch_generator(test_data, args.batch_size):
            imgs = []
            with ThreadPoolExecutor() as executor:
                imgs = torch.stack(list(executor.map(lambda test_data_item: get_imgs(remap_image_path(test_data_item[0]), transform), batch))).to(device)
            SimR_batch = [[] for _ in range(len(batch))]
            for batch_text in tqdm(batch_generator(report_corpus_work, args.batch_size), desc=f'当前子集: {args.split_id} | 当前批次: {idx}/{(len(test_data) + args.batch_size - 1) // args.batch_size}'):
                ids, tokens, attention = [], [], []
                for sent in batch_text:
                    input_ids = tokenizer(sent, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
                    x_len = len([t for t in input_ids["input_ids"][0] if t != 0])
                    ids.append(input_ids["input_ids"])
                    tokens.append(input_ids["token_type_ids"])
                    attention.append(input_ids["attention_mask"])
                caption_ids = torch.stack(ids).squeeze(1).to(device)
                token_type_ids = torch.stack(tokens).squeeze(1).to(device)
                attention_mask = torch.stack(attention).squeeze(1).to(device)
                query_emb_l, query_emb_g, _ = VECL_model.text_encoder_forward(caption_ids, attention_mask, token_type_ids)
                label_img_emb_l, label_img_emb_g = VECL_model.image_encoder_forward(imgs)
                label_img_emb_l = label_img_emb_l.view(label_img_emb_l.size(0), label_img_emb_l.size(1), -1)
                label_img_emb_l = label_img_emb_l.permute(0, 2, 1)
                query_emb_l_ = query_emb_l.view(query_emb_l.size(0), query_emb_l.size(1), -1)
                query_emb_l_ = query_emb_l_.permute(0, 2, 1)
                i2t_cls = VECL_model.fusion_module(torch.cat([label_img_emb_g.unsqueeze(1), label_img_emb_l], dim=1), query_emb_g, use_MLP=True, return_atten=False)
                t2i_cls = VECL_model.fusion_module(torch.cat([query_emb_g.unsqueeze(1), query_emb_l_], dim=1), label_img_emb_g, use_MLP=True, return_atten=False)
                i2t_cls = i2t_cls.squeeze(-1)
                t2i_cls = t2i_cls.squeeze(-1).transpose(1, 0)
                SimR_item = (i2t_cls + t2i_cls) / 2
                SimR_item = F.softmax(SimR_item, dim=2)
                SimR_item_norm = SimR_item[: ,: ,0].tolist()
                for i in range(len(SimR_batch)):
                    SimR_batch[i] = SimR_batch[i] + SimR_item_norm[i]
            for Index in range(len(SimR_batch)):
                row = SimR_batch[Index]
                max_value = max(row)
                max_index = row.index(max_value)
                result.append({max_index: max_value})
            idx += 1

    out_dir = output_path('retrieval_based_report_generation', 'new')
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f'split_{args.split_id}.pkl', 'wb') as f:
        pickle.dump(result, f)
    print(f'子集{args.split_id}计算完成')