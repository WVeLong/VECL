import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from utils import *
from VECL.paths import remap_image_path
import os
from tqdm import tqdm
import argparse
from VECL import VECL


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ckpt_dir", type=str, default=None, help="Checkpoint path for the save model"
    )
    parser.add_argument(
        "--evaluation_method", type=str, default='PNC', help="None"
    )
    parser.add_argument(
        "--dataset_name", type=str, default='OpenI', help="None"
    )
    parser.add_argument(
        "--train_pct", type=str, default='0.01', help="None"
    )
    parser.add_argument(
        "--cfg_path", type=str, default=None, help="None"
    )
    parser.add_argument(
        "--experiment_name", type=str, default='VECL', help="None"
    )
    parser.add_argument(
        "--output_path", type=str, default=None, help="None"
    )

    return parser


def obtain_simr_fintune(image_path, text_path, ckpt_path, args):

    df = pd.read_csv(image_path)
    with open(text_path, "r") as f:
        cls_prompts = json.load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bs = 256
    image_list = split_list(df["Path"].tolist(), bs)
    delete_index = []

    VECL_model = VECL.load_VECL(ckpt_path=ckpt_path, device=device, cfg_path=args.cfg_path)
    processed_txt = VECL_model.process_class_prompts(cls_prompts, device)

    for i, img in tqdm(enumerate(image_list),desc='process batch'):
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


if __name__ == '__main__':

    parser = get_parser()
    args = parser.parse_args()

    images = [
        "./Dataset/OpenI/test_input.csv",
        "./Dataset/Chexpert/chexpert5_test_image.csv",
        "./Dataset/ChestXray14/chestxray14_test_image.csv",
        "./Dataset/ChestXDet10/chestXDet10_test_image.csv",
        "./Dataset/PadChest/test_input.csv",
    ]

    if args.evaluation_method == "POS":
        texts = [
            "./Dataset/OpenI/openi_multi_label_text.json",
            "./Dataset/Chexpert/chexpert5_test_text.json",
            "./Dataset/ChestXray14/chestxray14_test_text.json",
            "./Dataset/ChestXDet10/chestXDet10_test_text.json",
            "./Dataset/PadChest/padchest_multi_label_text.json",
        ]
    elif args.evaluation_method == "PNC":
        texts = [
            "./Dataset/OpenI/openi_multi_label_text_plus.json",
            "./Dataset/Chexpert/chexpert5_test_text_plus.json",
            "./Dataset/ChestXray14/chestxray14_test_text_plus.json",
            "./Dataset/ChestXDet10/chestXDet10_test_text_plus.json",
            "./Dataset/PadChest/padchest_multi_label_text_plus.json",
        ]
    else:
        raise ValueError(f'evaluation_method {args.evaluation_method} is not supported !')

    labels = [
        "./Dataset/OpenI/test_label.csv",
        "./Dataset/Chexpert/test_labels.csv",
        "./Dataset/ChestXray14/test_list.txt",
        "./Dataset/ChestXDet10/test.json",
        "./Dataset/PadChest/manual_image.json"
    ]

    datasets_map = {'OpenI': 0, 'Chexpert': 1, 'ChestXray14': 2, 'ChestXDet10': 3, 'PadChest': 4}
    dataset_name = args.dataset_name
    dataset_index = datasets_map[dataset_name]
    delete_dict = {}

    file_list = [f for f in os.listdir(args.ckpt_dir) if os.path.isfile(os.path.join(args.ckpt_dir, f))]
    ckpt_list = [f for f in file_list if f not in ['best_ckpts.yaml', 'last.ckpt']]
    sorted_ckpt_list = sorted(ckpt_list, key=lambda x: int(x.split('-')[0].split('=')[1]))

    AUC_list = []
    F1_list = []
    MCC_list = []
    mAP_list = []
    num_ckpt = []

    for i in tqdm(range(len(sorted_ckpt_list))):

        ckpt_path = os.path.join(args.ckpt_dir, sorted_ckpt_list[i])
        similarities, delete_index = obtain_simr_fintune(images[dataset_index], texts[dataset_index], ckpt_path, args)
        delete_dict[dataset_name] = delete_index

        if dataset_index == 0:
            auc, f1, mcc, map = tripple_openi_rusult_merge(similarities, labels[dataset_index], delete_dict, args)
        elif dataset_index == 1:
            auc, f1, mcc, map = triple_Chexpert5_result(similarities, labels[dataset_index], args)
        elif dataset_index == 2:
            auc, f1, mcc, map = triple_Chexpert14_result(similarities, labels[dataset_index], args)
        elif dataset_index == 3:
            auc, f1, mcc, map = triple_ChestXDet10_result(similarities, labels[dataset_index], args)
        elif dataset_index == 4:
            auc, f1, mcc, map, auc_20, f1_20, mcc_20, map_20 = tripple_padchest_rusult_merge(similarities, labels[dataset_index], delete_dict, args)
        else:
            raise ValueError('Error Dataset')

        AUC_list.append(auc)
        F1_list.append(f1)
        MCC_list.append(mcc)
        mAP_list.append(map)
        num_ckpt.append(i + 1)

        print('max_AUC: ', max(AUC_list))
        print('max_F1: ', max(F1_list))
        print('max_MCC: ', max(MCC_list))
        print('max_mAP: ', max(mAP_list))
        print('best_epoch: ', AUC_list.index(max(AUC_list)) + 1)

    max_AUC = max(AUC_list)
    best_epoch = AUC_list.index(max_AUC)
    print('max_AUC: ', max_AUC, '; best_epoch: ', best_epoch)

    df = pd.DataFrame({'epoch': num_ckpt, 'AUC': AUC_list, 'F1': F1_list, 'MCC': MCC_list, 'mAP': mAP_list})
    df.columns = ['epoch', 'AUC', 'F1', 'MCC', 'mAP']
    output_path = os.path.join(args.output_path, args.dataset_name)
    file_name = args.experiment_name + '_' + args.train_pct + '_' + args.evaluation_method + '.csv'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    output_path = os.path.join(output_path, args.train_pct)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    output_path = os.path.join(output_path, file_name)
    pd.DataFrame(df).to_csv(output_path, index=False)
    print('Inference Over')