import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from VECL import VECL
from VECL.paths import dataset_path, remap_image_path
import os
import argparse
from tqdm import tqdm
import time
from utils import *


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default=None, help="None")
    parser.add_argument("--ckpt_path", type=str, help="Checkpoint path for the save model", default=None)
    parser.add_argument("--evaluation_method", type=str, default="POS", help="None")
    parser.add_argument("--test_Dateset", type=str, default=None, help="None")
    parser.add_argument("--output_path", type=str, default=None, help="None")
    return parser

def obtain_simr(image_path, text_path, args):
    df = pd.read_csv(image_path)
    with open(text_path, "r") as f:
        cls_prompts = json.load(f)

    # load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    VECL_model = VECL.load_VECL(ckpt_path=args.ckpt_path, device=device)
    processed_txt = VECL_model.process_class_prompts(cls_prompts, device)

    # batch size
    bs = 256
    image_list = split_list(df["Path"].tolist(), bs)
    delete_index = []

    for i, img in tqdm(enumerate(image_list), total=len(image_list)):
        img_processed = []
        for index in range(len(img)):
            img[index] = remap_image_path(img[index])
            if not os.path.exists(img[index]):
                print("样本不存在: ", img[index])
                delete_index.append(i * bs + index)
                img_processed.append(None)
                continue

            img_processed.append(img[index])

        processed_imgs = VECL_model.process_img(img_processed, device, delete_index, i, bs)
        similarities = VECL.classification_inference(VECL_model, processed_imgs, processed_txt)

        if i == 0:
            similar = similarities
        else:
            similars_p = pd.concat([similar[0], similarities[0]], axis=0)
            similars_0 = pd.concat([similar[1], similarities[1]], axis=0)
            similars_n = pd.concat([similar[2], similarities[2]], axis=0)
            similar = [similars_p, similars_0, similars_n]

    return similar, delete_index


if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()

    images = [
        str(dataset_path("OpenI", "openi_multi_label_image.csv")),
        str(dataset_path("Chexpert", "chexpert5_test_image.csv")),
        str(dataset_path("ChestXray14", "chestxray14_test_image.csv")),
        str(dataset_path("ChestXDet10", "chestXDet10_test_image.csv")),
        str(dataset_path("PadChest", "padchest_multi_label_image.csv")),
    ]

    if args.evaluation_method == "POS":
        texts = [
            str(dataset_path("OpenI", "openi_multi_label_text.json")),
            str(dataset_path("Chexpert", "chexpert5_test_text.json")),
            str(dataset_path("ChestXray14", "chestxray14_test_text.json")),
            str(dataset_path("ChestXDet10", "chestXDet10_test_text.json")),
            str(dataset_path("PadChest", "padchest_multi_label_text.json")),
        ]
    elif args.evaluation_method == "PNC":
        texts = [
            str(dataset_path("OpenI", "openi_multi_label_text_plus.json")),
            str(dataset_path("Chexpert", "chexpert5_test_text_plus.json")),
            str(dataset_path("ChestXray14", "chestxray14_test_text_plus.json")),
            str(dataset_path("ChestXDet10", "chestXDet10_test_text_plus.json")),
            str(dataset_path("PadChest", "padchest_multi_label_text_plus.json")),
        ]
    else:
        raise ValueError(f'evaluation_method {args.evaluation_method} is not supported !')

    labels = [
        str(dataset_path("OpenI", "custom.csv")),
        str(dataset_path("Chexpert", "test_labels.csv")),
        str(dataset_path("ChestXray14", "test_list.txt")),
        str(dataset_path("ChestXDet10", "test.json")),
        str(dataset_path("PadChest", "manual_image.json"))
    ]

    test_list = args.test_Dateset.split(',')
    test_list = [int(i) for i in test_list]
    delete_dict = {}
    predict_result = [None] * len(images)
    
    for i, (img, txt) in enumerate(zip(images, texts)):
        if i in test_list:
            start = time.time()
            similarities, delete_index = obtain_simr(img, txt, args)
            delete_dict[img] = delete_index
            predict_result[i] = similarities
            print(time.time() - start)

    output = []
    for i in test_list:
        if predict_result[i] is None:
            continue
        if i == 0:
            print('##### Openi #####')
            auc, f1, mcc, map = tripple_openi_rusult_merge(predict_result[i], labels[i], delete_dict, args)
            output.append('##### Openi #####')
            output.append(f'AUC = {auc}, F1 = {f1}, MCC = {mcc}, mAP = {map}')
            output.append('')
        elif i == 1:
            print("##### Chexpert5 #####")
            auc, f1, mcc, map = triple_Chexpert5_result(predict_result[i], labels[i], args)
            output.append('##### Chexpert5 #####')
            output.append(f'AUC = {auc}, F1 = {f1}, MCC = {mcc}, mAP = {map}')
            output.append('')
        elif i == 2:
            print('##### ChestXray14 #####')
            auc, f1, mcc, map = triple_Chexpert14_result(predict_result[i], labels[i], args)
            output.append('##### ChestXray14 #####')
            output.append(f'AUC = {auc}, F1 = {f1}, MCC = {mcc}, mAP = {map}')
            output.append('')
        elif i == 3:
            print('##### ChestXDet10 #####')
            auc, f1, mcc, map = triple_ChestXDet10_result(predict_result[i], labels[i], args)
            output.append('##### ChestXDet10 #####')
            output.append(f'AUC = {auc}, F1 = {f1}, MCC = {mcc}, mAP = {map}')
            output.append('')
        elif i == 4:
            print("##### Padchest #####")
            auc, f1, mcc, map, auc_20, f1_20, mcc_20, map_20 = tripple_padchest_rusult_merge(predict_result[i], labels[i], delete_dict, args)
            output.append('##### Padchest #####')
            output.append(f'AUC = {auc}, F1 = {f1}, MCC = {mcc}, mAP = {map}')
            output.append('')
            output.append('##### Padchest20 #####')
            output.append(f'AUC = {auc_20}, F1 = {f1_20}, MCC = {mcc_20}, mAP = {map_20}')
        else:
            raise ValueError('Error Dataset')

    with open(args.output_path, "w", encoding="utf-8") as file:
        for line in output:
            file.write(line + "\n")