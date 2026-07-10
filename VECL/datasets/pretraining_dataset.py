import re
import os
import numpy as np
import pandas as pd
import cv2
import tqdm
import pickle
import numpy.random as random
import torch
import torch.utils.data as data
import json
from PIL import Image
from nltk.tokenize import RegexpTokenizer
from transformers import AutoTokenizer
from VECL.constants import *
from VECL.paths import CHESTXRAY14_ROOT, CHESTXDET10_ROOT, CHEXPERT_ROOT, dataset_path, remap_image_path
import ast
from sklearn.preprocessing import MultiLabelBinarizer
import concurrent.futures


class FinetuneDataset(data.Dataset):
    def __init__(self, cfg, transform=None):

        self.cfg = cfg
        self.cfg_classify = cfg
        self.transform = transform
        self.finetune_dataset = cfg.data.dataset

        if self.finetune_dataset == 'OpenI':
            if OpenI_DATA_DIR == None:
                raise RuntimeError("OpenI data path empty")

            # creat label
            mapping = dict()
            mapping["Pleural_Thickening"] = ["pleural thickening"]
            mapping["Infiltration"] = ["Infiltrate"]
            mapping["Atelectasis"] = ["Atelectases"]
            csv = pd.read_csv(OpenI_LABEL_CSV)
            csv = csv.replace(np.nan, "-1")
            gt = []
            for pathology in OpenI_pathologies:
                mask = csv["labels_automatic"].str.contains(pathology.lower())
                if pathology in mapping:
                    for syn in mapping[pathology]:
                        mask |= csv["labels_automatic"].str.contains(syn.lower())
                gt.append(mask.values)
            gt = np.asarray(gt).T
            gt = gt.astype(np.float32)
            gt[np.where(np.sum(gt, axis=1) == 0), -1] = 1
            label = gt[:, :-1]

            # Rename pathologies
            pathologies = np.char.replace(OpenI_pathologies, "Opacity", "Lung Opacity")
            pathologies = np.char.replace(pathologies, "Lesion", "Lung Lesion")
            pathologies = np.char.replace(pathologies, "Pleural_Thickening", "pleural thickening")
            pathologies = np.char.replace(pathologies, "Infiltration", "Infiltrate")
            pathologies = np.char.replace(pathologies, "Atelectasis", "Atelectases")

            # read in csv file
            self.df = pd.read_csv(OpenI_TRAIN_CSV)

            # get path
            self.df[OpenI_PATH_COL] = self.df[OpenI_PATH_COL].apply(lambda x: remap_image_path(x))

            # clean nan data
            delete_index = [0, 1]
            self.df = self.df.values.tolist()
            self.df = [self.df[i] for i in range(len(self.df)) if i not in delete_index]
            label = [label[i] for i in range(len(label)) if i not in delete_index]
            self.df = pd.DataFrame(self.df, columns=[OpenI_PATH_COL])
            label = pd.DataFrame(label, columns=pathologies.tolist()[:-1])

            # sample data
            self.test_input = self.df.sample(frac=0.4, random_state=42)
            self.test_label = label.sample(frac=0.4, random_state=42)
            self.train_input = self.df.drop(self.test_input.index)
            self.train_label = label.drop(self.test_label.index)
            self.train_input = self.train_input.sample(frac=self.cfg_classify.data.frac, random_state=42)
            self.train_label = self.train_label.sample(frac=self.cfg_classify.data.frac, random_state=42)

            test_input_save_path = OpenI_TEST_INPUT_CSV
            test_label_save_path = OpenI_TEST_LABEL_CSV

            if not (os.path.exists(test_input_save_path) or os.path.exists(test_label_save_path)):
                self.test_input.to_csv(test_input_save_path, index=False, header=self.test_input.columns.tolist())
                self.test_label.to_csv(test_label_save_path, index=False, header=self.test_label.columns.tolist())

            # text data
            with open(self.cfg_classify.data.text.path, 'r') as f:
                cls_prompts = json.load(f)
            bert_type = self.cfg_classify.model.text.bert_type
            self.tokenizer = AutoTokenizer.from_pretrained(bert_type)
            self.idxtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}
            processed_txt = {}
            for k, v in cls_prompts.items():
                processed_txt[k] = self.process_text(v, "cpu")
            self.oral_texts = []
            for k, v in cls_prompts.items():
                self.oral_texts.append(v[0])
            caption_ids, attention_mask, token_type_ids = [], [], []
            for cls_name, txts in processed_txt.items():
                caption_ids.append(txts["caption_ids"])
                attention_mask.append(txts["attention_mask"])
                token_type_ids.append(txts["token_type_ids"])
            caption_ids = torch.cat(caption_ids, dim=0)
            attention_mask = torch.cat(attention_mask, dim=0)
            token_type_ids = torch.cat(token_type_ids, dim=0)
            self.text_batch = {"caption_ids": caption_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids}
        elif self.finetune_dataset == 'ChestXray14':
            if ChestXray14_DATA_DIR == None:
                raise RuntimeError("ChestXray14 data path empty")

            # read test label and data
            data = pd.read_csv(ChestXray14_TRAIN_DATA, sep=' ', names=[ChestXray14_PATH_COL] + ChestXray14_pathologies)
            self.label = data[ChestXray14_pathologies].values
            self.df = data[ChestXray14_PATH_COL].values
            self.label = pd.DataFrame(self.label, columns=ChestXray14_pathologies)
            self.df = pd.DataFrame(self.df, columns=['Path'])

            # get path
            self.df['Path'] = self.df['Path'].apply(lambda x: str(CHESTXRAY14_ROOT / x[11:]))

            # read train label and data
            data_train = pd.read_csv(dataset_path('ChestXray14', 'train.csv'), names=['file', 'labels'])
            train_file = pd.DataFrame(data_train['file'].values, columns=['Path'])
            train_label = data_train['labels'].values.tolist()
            train_label_eval = []
            for item in train_label:
                train_label_eval.append(eval(item))
            train_label = pd.DataFrame(train_label_eval, columns=ChestXray14_pathologies)

            # sample data
            self.test_input = self.df
            self.test_label = self.label
            self.train_input = train_file
            self.train_label = train_label
            self.train_input = self.train_input.sample(frac=self.cfg_classify.data.frac, random_state=42)
            self.train_label = self.train_label.sample(frac=self.cfg_classify.data.frac, random_state=42)

            test_input_save_path = ChestXray14_TEST_INPUT_CSV
            test_label_save_path = ChestXray14_TEST_LABEL_CSV

            if not (os.path.exists(test_input_save_path) or os.path.exists(test_label_save_path)):
                self.test_input.to_csv(test_input_save_path, index=False, header=self.test_input.columns.tolist())
                self.test_label.to_csv(test_label_save_path, index=False, header=self.test_label.columns.tolist())

            # text data
            with open(self.cfg_classify.data.text.path, 'r') as f:
                cls_prompts = json.load(f)
            bert_type = self.cfg_classify.model.text.bert_type
            self.tokenizer = AutoTokenizer.from_pretrained(bert_type)
            self.idxtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}
            processed_txt = {}
            for k, v in cls_prompts.items():
                processed_txt[k] = self.process_text(v, "cpu")
            self.oral_texts = []
            for k, v in cls_prompts.items():
                self.oral_texts.append(v[0])
            caption_ids, attention_mask, token_type_ids = [], [], []
            for cls_name, txts in processed_txt.items():
                caption_ids.append(txts["caption_ids"])
                attention_mask.append(txts["attention_mask"])
                token_type_ids.append(txts["token_type_ids"])
            caption_ids = torch.cat(caption_ids, dim=0)
            attention_mask = torch.cat(attention_mask, dim=0)
            token_type_ids = torch.cat(token_type_ids, dim=0)
            self.text_batch = {"caption_ids": caption_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids}
        elif self.finetune_dataset == 'ChestXDet10':
            if ChestXDet10_DATA_DIR == None:
                raise RuntimeError("ChestXDet10 data path empty")

            # read label and data
            self.df = pd.read_csv(ChestXDet10_TRAIN_INPUT)
            self.df[ChestXDet10_PATH_COL] = self.df[ChestXDet10_PATH_COL].apply(
                lambda x: remap_image_path(x))
            with open(ChestXDet10_TRAIN_LABEL, 'r') as f:
                data = json.load(f)
            all_label = []
            for d in data:
                all_label.append(d['syms'])
            mlb = MultiLabelBinarizer(classes=ChestXDet10_pathologies)
            label = mlb.fit_transform(all_label)
            self.label = pd.DataFrame(np.asarray(label))

            # read train label and data
            train_file = []
            train_label = []
            with open(dataset_path('ChestXDet10', 'train.json'), 'r') as f:
                data_train = json.load(f)
            for d in data_train:
                train_label.append(d['syms'])
                train_file.append(str(CHESTXDET10_ROOT / 'train_data' / d['file_name']))
            mlb = MultiLabelBinarizer(classes=ChestXDet10_pathologies)
            train_label = mlb.fit_transform(train_label)
            train_label = pd.DataFrame(np.asarray(train_label))

            # sample data
            self.test_input = self.df
            self.test_label = self.label
            self.train_input = pd.DataFrame(train_file, columns=['Path'])
            self.train_label = train_label
            self.train_input = self.train_input.sample(frac=self.cfg_classify.data.frac, random_state=42)
            self.train_label = self.train_label.sample(frac=self.cfg_classify.data.frac, random_state=42)

            self.train_input.columns = [ChestXDet10_PATH_COL]
            self.train_label.columns = ChestXDet10_pathologies
            self.test_input.columns = [ChestXDet10_PATH_COL]
            self.test_label.columns = ChestXDet10_pathologies

            test_input_save_path = ChestXDet10_TEST_INPUT_CSV
            test_label_save_path = ChestXDet10_TEST_LABEL_CSV

            if not (os.path.exists(test_input_save_path) or os.path.exists(test_label_save_path)):
                self.test_input.to_csv(test_input_save_path, index=False, header=self.test_input.columns.tolist())
                self.test_label.to_csv(test_label_save_path, index=False, header=self.test_label.columns.tolist())

            # text data
            with open(self.cfg_classify.data.text.path, 'r') as f:
                cls_prompts = json.load(f)
            bert_type = self.cfg_classify.model.text.bert_type
            self.tokenizer = AutoTokenizer.from_pretrained(bert_type)
            self.idxtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}
            processed_txt = {}
            for k, v in cls_prompts.items():
                processed_txt[k] = self.process_text(v, "cpu")
            self.oral_texts = []
            for k, v in cls_prompts.items():
                self.oral_texts.append(v[0])
            caption_ids, attention_mask, token_type_ids = [], [], []
            for cls_name, txts in processed_txt.items():
                caption_ids.append(txts["caption_ids"])
                attention_mask.append(txts["attention_mask"])
                token_type_ids.append(txts["token_type_ids"])
            caption_ids = torch.cat(caption_ids, dim=0)
            attention_mask = torch.cat(attention_mask, dim=0)
            token_type_ids = torch.cat(token_type_ids, dim=0)
            self.text_batch = {"caption_ids": caption_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids}
        elif self.finetune_dataset == 'Chexpert':
            if CHEXPERT_ROOT is None:
                raise RuntimeError("CheXpert data path empty")

            key = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]

            # read in csv file
            self.df = pd.read_csv(dataset_path('Chexpert', 'test_labels.csv'))
            self.df_train = pd.read_csv(dataset_path('Chexpert', 'train_cheXbert.csv'))

            # prepare data
            self.df['Path'] = self.df['Path'].apply(lambda x: remap_image_path('/defaultShare/CheXpert/test/' + x))
            test_label = self.df[key]
            train_label = self.df_train[key].fillna(0)
            train_input = pd.read_csv(dataset_path('Chexpert', 'train.csv'))

            # sample data
            self.test_input = pd.DataFrame(self.df['Path'],columns=['Path'])
            self.test_label = test_label
            self.train_input = train_input
            self.train_label = train_label
            self.train_input = self.train_input.sample(frac=self.cfg_classify.data.frac, random_state=42)
            self.train_label = self.train_label.sample(frac=self.cfg_classify.data.frac, random_state=42)

            test_input_save_path = dataset_path('Chexpert', 'test_input.csv')
            test_label_save_path = dataset_path('Chexpert', 'test_label.csv')

            if not (os.path.exists(test_input_save_path) or os.path.exists(test_label_save_path)):
                self.test_input.to_csv(test_input_save_path, index=False, header=self.test_input.columns.tolist())
                self.test_label.to_csv(test_label_save_path, index=False, header=self.test_label.columns.tolist())

            # text data
            with open(self.cfg_classify.data.text.path, 'r') as f:
                cls_prompts = json.load(f)
            bert_type = self.cfg_classify.model.text.bert_type
            self.tokenizer = AutoTokenizer.from_pretrained(bert_type)
            self.idxtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}
            processed_txt = {}
            for k, v in cls_prompts.items():
                processed_txt[k] = self.process_text(v, "cpu")
            self.oral_texts = []
            for k, v in cls_prompts.items():
                self.oral_texts.append(v[0])
            caption_ids, attention_mask, token_type_ids = [], [], []
            for cls_name, txts in processed_txt.items():
                caption_ids.append(txts["caption_ids"])
                attention_mask.append(txts["attention_mask"])
                token_type_ids.append(txts["token_type_ids"])
            caption_ids = torch.cat(caption_ids, dim=0)
            attention_mask = torch.cat(attention_mask, dim=0)
            token_type_ids = torch.cat(token_type_ids, dim=0)
            self.text_batch = {"caption_ids": caption_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids}
        elif self.finetune_dataset == 'PadChest':

            train_input = dataset_path('PadChest', 'train_input.csv')
            train_output = dataset_path('PadChest', 'train_output.csv')

            self.df = pd.read_csv(train_input)
            self.label = pd.read_csv(train_output, header=None)

            # sample data
            self.test_input = self.df.sample(frac=0.4, random_state=42)
            self.test_label = self.label.sample(frac=0.4, random_state=42)
            self.train_input = self.df.drop(self.test_input.index)
            self.train_label = self.label.drop(self.test_label.index)
            self.train_input = self.train_input.sample(frac=self.cfg_classify.data.frac, random_state=42)
            self.train_label = self.train_label.sample(frac=self.cfg_classify.data.frac, random_state=42)

            self.train_input.columns = ['Path']

            test_input_save_path = dataset_path('PadChest', 'test_input.csv')
            test_label_save_path = dataset_path('PadChest', 'test_label.csv')

            if not (os.path.exists(test_input_save_path) or os.path.exists(test_label_save_path)):
                self.test_input.to_csv(test_input_save_path, index=False, header=self.test_input.columns.tolist())
                self.test_label.to_csv(test_label_save_path, index=False, header=self.test_label.columns.tolist())

            # text data
            with open(self.cfg_classify.data.text.path, 'r') as f:
                cls_prompts = json.load(f)
            bert_type = self.cfg_classify.model.text.bert_type
            self.tokenizer = AutoTokenizer.from_pretrained(bert_type)
            self.idxtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}
            processed_txt = {}
            for k, v in cls_prompts.items():
                processed_txt[k] = self.process_text(v, "cpu")
            self.oral_texts = []
            for k, v in cls_prompts.items():
                self.oral_texts.append(v[0])
            caption_ids, attention_mask, token_type_ids = [], [], []
            for cls_name, txts in processed_txt.items():
                caption_ids.append(txts["caption_ids"])
                attention_mask.append(txts["attention_mask"])
                token_type_ids.append(txts["token_type_ids"])
            caption_ids = torch.cat(caption_ids, dim=0)
            attention_mask = torch.cat(attention_mask, dim=0)
            token_type_ids = torch.cat(token_type_ids, dim=0)
            self.text_batch = {"caption_ids": caption_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids}
        else:
            raise ValueError('The finetune dataset is not supported!')

    def __getitem__(self, index):
        row_input = self.train_input.iloc[index]
        row_label = self.train_label.iloc[index]
        # get image
        img_path = row_input['Path']
        image = self.get_imgs(img_path, self.transform)
        # get labels
        label = row_label.tolist()
        label = torch.tensor(label)
        # get text
        text = self.text_batch
        oral_text = self.oral_texts
        return image, label, text, oral_text

    def __len__(self):
        return len(self.train_input)

    def get_imgs(self, img_path, transform=None):

        x = cv2.imread(img_path, 0)

        # tranform images
        x = self._resize_img(x, self.cfg_classify.data.image.imsize)
        img = Image.fromarray(x).convert("RGB")
        img = transform(img)

        return img

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
        resized_img = cv2.resize(img, desireable_size[::-1], interpolation=cv2.INTER_AREA)  # this flips the desireable_size vector

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
        resized_img = np.pad(resized_img, [(top, bottom), (left, right)], "constant", constant_values=0)

        return resized_img

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
                max_length=self.cfg_classify.data.text.word_num,
            )
            text_tensors["sent"] = [
                self.idxtoword[ix] for ix in text_tensors["input_ids"][0].tolist()
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


def Classify_collate_fn(batch):
    """sort sequence"""

    images, labels, caption_ids, attention_mask, token_type_ids = [], [], [], [], []

    # flattern
    for b in batch:
            image, label, text, oral_text = b
            images.append(image)
            labels.append(label)
    # stack
    try:
        images = torch.stack(images)
    except:
        pass

    labels = torch.stack(labels)

    # add to dictionary
    return_dict = {
        "caption_ids": batch[0][2]["caption_ids"],
        "token_type_ids": batch[0][2]["token_type_ids"],
        "attention_mask": batch[0][2]["attention_mask"],
        "images": images,
        "texts": batch[0][-1],
        "labels": labels
    }

    return return_dict
    

class MultimodalPretrainingDataset(data.Dataset):
    def __init__(self, cfg, split="train", transform=None):

        if MIMIC_DATA_DIR is None:
            raise RuntimeError(
                "MIMIC data path empty\n"
                + "Make sure to download data from:\n"
                + "    https://stanfordmlgroup.github.io/competitions/MIMIC/"
                + f" and update MIMIC_DATA_DIR in ./CARZero/constants.py"
            )

        self.cfg = cfg
        self.transform = transform
        self.max_word_num = self.cfg.data.text.captions_per_image
        self.dropout = 0
        self.label_num = 24

        # read MIMIC image-report pair
        csv_path = os.path.join(MIMIC_DATA_DIR, MIMIC_MASTER_CSV_XH)
        self.df = pd.read_csv(csv_path)
        filtered_df = self.df[self.df[MIMIC_VIEW_COL] == 'Frontal']
        self.filtered_indices = filtered_df.index
        self.df = filtered_df.reset_index(drop=True)

        # load studies and study to text mapping
        filename = 'captions_' + 'drop_' + str(self.dropout) + '_' + 'num_' + str(self.label_num) + '_' + '.pickle'
        # filename = 'chexpert_labeler_final.pickle'
        self.path2sent, self.path2label, self.to_remove = self.load_text_data(filename)

        # filter studies to use for current split
        filenames = self.df[self.df[MIMIC_SPLIT_COL] == split][MIMIC_PATH_COL].tolist()
        filenames = [f for f in filenames if f not in self.to_remove]
        with open(os.path.join(MIMIC_DATA_DIR, 'cxr_report_noise_sample.pickle'), 'rb') as f:
            noise_sample = pickle.load(f)
        tempnoise_sample = [os.path.basename(f) for f in noise_sample]
        self.filenames = [f for f in filenames if os.path.basename(f) not in tempnoise_sample]

        # create BERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.text.bert_type)

    def label_dropout(self, input_list, dropout):
        def replace(item):
            if isinstance(item, str) and ',' in item:
                sub_items = [replace(sub.strip()) for sub in item.split(',')]
                return ', '.join(sub_items)
            return '25' if random.random() < dropout else item
        result = []
        for element in input_list:
            replaced_element = replace(element)
            result.append(replaced_element)
        return result

    def label_mask(self, input_list, mask_list):
        def replace(item):
            if isinstance(item, str) and ',' in item:
                sub_items = [replace(sub.strip()) for sub in item.split(',')]
                return ', '.join(sub_items)
            return '25' if item in mask_list else item
        result = []
        for element in input_list:
            replaced_element = replace(element)
            result.append(replaced_element)
        return result

    def parse_string_to_list(self, string):
        try:
            string = re.sub(r"(?<=\[|,)\s*'(.*?)'\s*(?=,|\])", r'"\1"', string)
            result = ast.literal_eval(string)
            if isinstance(result, list):
                return result
            else:
                raise ValueError("The parsed result is not a list.")
        except (SyntaxError, ValueError) as e:
            print(f"Error parsing string: {e}")
            return None

    def load_text_data(self, filename):

        # get study to captions mapping
        filepath = os.path.join(MIMIC_DATA_DIR, filename)
        if not os.path.isfile(filepath):
            print(f"Caption file {filepath} does not exit. Creating captions...")

            # read report sentences and labels
            sent_path = os.path.join(MIMIC_DATA_DIR, SENT_Path)
            sent_str = pd.read_csv(sent_path, header=None)
            sent_str = sent_str.loc[self.filtered_indices].reset_index(drop=True).values.tolist()
            label_path = os.path.join(MIMIC_DATA_DIR, LABEL_Path)
            label_str = pd.read_csv(label_path, header=None)
            label_str = label_str.loc[self.filtered_indices].reset_index(drop=True).values.tolist()
            sent = []
            for sent_sample in tqdm.tqdm(sent_str, desc='Loading chopped sentences'):
                sent.append(self.parse_string_to_list(sent_sample[0]))
            label = []
            for label_sample in tqdm.tqdm(label_str, desc='Loading sentences labels'):
                temp = [element for element in label_sample[:100] if element != '0' and element != 0]
                label.append(temp)

            # label dropout
            dropout_label = []
            for oral_label in tqdm.tqdm(label, desc=f'Label dropout, dropout = {self.dropout}'):
                label_dropout = self.label_dropout(oral_label, self.dropout)
                dropout_label.append(label_dropout)
            label = [[element for element in row] for row in dropout_label]
            del dropout_label

            # label mask
            mask_label = []
            mask_list = ['2-', '2+', '4-', '4+', '3-', '3+', '17-', '17+', '15-', '15+', '20-', '20+',
                         '6-', '6+', '8-', '8+', '1-', '1+', '5-', '5+', '10-', '10+', '16-', '16+',
                         '12-', '12+', '9-', '9+', '13-', '13+', '24-', '24+', '21-', '21+', '11-', '11+',
                         '18-', '18+', '7-', '7+', '14-', '14+', '22-', '22+', '19-', '19+', '23-', '23+']
            mask_list = mask_list[self.label_num * 2:]
            print('mask label:', mask_list)
            for oral_label in tqdm.tqdm(label, desc=f'Label mask, label num = {self.label_num}'):
                label_mask = self.label_mask(oral_label, mask_list)
                mask_label.append(label_mask)
            label = mask_label
            del mask_label

            # store data
            path2sent, path2label, to_remove = self.create_path_2_sent_mapping(self.df, sent, label)
            with open(filepath, "wb") as f:
                pickle.dump([path2sent, path2label, to_remove], f, protocol=2)
                print("Save to: ", filepath)
        else:
            with open(filepath, "rb") as f:
                print(f"Loading captions from {filepath}")
                path2sent, path2label, to_remove = pickle.load(f)

        return path2sent, path2label, to_remove

    def get_caption(self, path):
        series_sents = self.path2sent[path]
        series_labels = self.path2label[path]
        sent_ix = random.randint(0, len(series_sents))
        sent = series_sents[sent_ix]
        label_sample = series_labels[sent_ix]
        label_list = series_labels
        tokens = self.tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.cfg.data.text.word_num,
        )
        x_len = len([t for t in tokens["input_ids"][0] if t != 0])
        return tokens, x_len, label_sample, label_list

    def get_imgs(self, img_path, transform=None):
        img_abs_path = os.path.join(PWD_Path, img_path.replace("/defaultShare/MIMIC-CXR/", ""))
        x = cv2.imread(str(img_abs_path), 0)
        x = self._resize_img(x, self.cfg.data.image.imsize)
        img = Image.fromarray(x).convert("RGB")
        if transform is not None:
            img = transform(img)
        return img

    def __getitem__(self, index):
        key = self.filenames[index]
        imgs = self.get_imgs(key, self.transform)
        caps, cap_len, label_sample, label_list = self.get_caption(key)
        return imgs, caps, cap_len, key, label_sample, label_list

    def __len__(self):
        return len(self.filenames)

    def create_path_2_sent_mapping(self, df, sent, label):
        sent_lens, num_sents, to_remove = [], [], []
        path2sent = {}
        path2label = {}
        for idx, row in tqdm.tqdm(df.iterrows(), total=df.shape[0], desc='Creating captions'):

            # pick impression, findings, last_paragraph
            captions = ""
            if type(row[MIMIC_REPORT_COL]) == str:
                captions += row[MIMIC_REPORT_COL]
            img_path = row[MIMIC_PATH_COL]
            img_abs_path = os.path.join(PWD_Path, img_path.replace("/defaultShare/MIMIC-CXR/", ""))
            if not os.path.exists(img_abs_path):
                to_remove.append(row[MIMIC_PATH_COL])

            # remove empty reports
            if len(captions) == 0:
                to_remove.append(row[MIMIC_PATH_COL])
            if len(sent[idx]) == 0:
                to_remove.append(row[MIMIC_PATH_COL])
            else:
                study_sent = []
                study_label = []
                for cap, lab in zip(sent[idx], label[idx]):
                    initial_length = len(study_sent)
                    tokenizer = RegexpTokenizer(r"\w+")
                    tokens = tokenizer.tokenize(cap.lower())
                    included_tokens = []
                    for t in tokens:
                        t = t.encode("ascii", "ignore").decode("ascii")
                        if len(t) > 0:
                            included_tokens.append(t)
                    study_sent.append(" ".join(included_tokens))
                    sent_lens.append(len(cap))
                    if type(lab) != str:
                        lab = str(lab)
                    if len(study_sent) > initial_length:
                        study_label.append(lab)
                num_sents.append(len(study_sent))
                path2sent[row[MIMIC_PATH_COL]] = study_sent
                path2label[row[MIMIC_PATH_COL]] = study_label

        # get report word/setence statistics
        sent_lens = np.array(sent_lens)
        num_sents = np.array(num_sents)
        print(f"sent lens: {sent_lens.min()},{sent_lens.mean()},{sent_lens.max()} [{np.percentile(sent_lens, 5)}, {np.percentile(sent_lens, 95)}]")
        print(f"num sents: {num_sents.min()},{num_sents.mean()},{num_sents.max()} [{np.percentile(num_sents, 5)}, {np.percentile(num_sents, 95)}]")

        return path2sent, path2label, to_remove

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
    

def multimodal_collate_fn(batch):

    imgs, cap_len, ids, tokens, attention, path, label_sample, label_list = [], [], [], [], [], [], [], []

    # flattern
    for b in batch:
        img, cap, cap_l, p, l_s, l_l = b
        imgs.append(img)
        cap_len.append(cap_l)
        ids.append(cap["input_ids"])
        tokens.append(cap["token_type_ids"])
        attention.append(cap["attention_mask"])
        path.append(p)
        label_sample.append(l_s)
        label_list.append(l_l)

    # stack
    imgs = torch.stack(imgs)
    ids = torch.stack(ids).squeeze()
    tokens = torch.stack(tokens).squeeze()
    attention = torch.stack(attention).squeeze()

    # sort and add to dictionary
    sorted_cap_lens, sorted_cap_indices = torch.sort(torch.tensor(cap_len), 0, True)
    return_dict = {
        "caption_ids": ids[sorted_cap_indices],
        "token_type_ids": tokens[sorted_cap_indices],
        "attention_mask": attention[sorted_cap_indices],
        "imgs": imgs[sorted_cap_indices],
        "cap_lens": sorted_cap_lens,
        "path": path,
        "label_sample": [label_sample[i] for i in sorted_cap_indices] if label_sample else [],
        "label_list": [label_list[i] for i in sorted_cap_indices] if label_list else [],
    }

    return return_dict