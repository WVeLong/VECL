import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import MultiLabelBinarizer
from mertics import calculate_scores, calculate_scores_by_class
import torch
import os
from VECL.paths import dataset_path


def split_list(lst, chunk_size):
    result = []
    for i in range(0, len(lst), chunk_size):
        chunk = lst[i:i+chunk_size]
        result.append(chunk)
    return result

def res_process(predict_result):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    p_value_list = torch.tensor(predict_result[0].values.tolist(), device=device)
    z_value_list = torch.tensor(predict_result[1].values.tolist(), device=device)
    n_value_list = torch.tensor(predict_result[2].values.tolist(), device=device)

    # 向量化处理
    combined = torch.stack((p_value_list, z_value_list, n_value_list), dim=-1)
    exp_combined = torch.exp(combined)
    softmax_values = exp_combined / exp_combined.sum(dim=-1, keepdim=True)

    # 提取softmax结果
    predict_p = softmax_values[:, :, 0].cpu().numpy()  # 转回CPU并返回为numpy数组
    return predict_p

def tripple_openi_rusult_merge(predict_result, label_file_path, delete_dict, args):
    pathologies = [
        "Atelectasis",
        "Cardiomegaly",
        "Effusion",
        "Infiltration",
        "Mass",
        "Nodule",
        "Pneumonia",
        "Pneumothorax",
        "Edema",
        "Emphysema",
        "Fibrosis",
        "Pleural_Thickening",
        "Hernia",
        "Fracture",
        "Opacity",
        "Lesion",
        "Calcified Granuloma",
        "Lung Granuloma",
        "No_Finding",
    ]

    test_output_path = dataset_path('OpenI', 'test_label.csv')

    if 'ckpt_dir' in args:  # finetune inference
        if os.path.exists(test_output_path):
            label = pd.read_csv(test_output_path, header=None).iloc[1:].values.tolist()
            label_new = []
            for item in label:
                temp = []
                for t in item:
                    temp.append(eval(t))
                label_new.append(temp)
            label = np.array(label_new)
        else:
            raise ValueError('finetune inference data is not found')
    else:
        mapping = dict()
        mapping["Pleural_Thickening"] = ["pleural thickening"]
        mapping["Infiltration"] = ["Infiltrate"]
        mapping["Atelectasis"] = ["Atelectases"]

        # Load data
        csv = pd.read_csv(label_file_path)
        csv = csv.replace(np.nan, "-1")

        gt = []
        for pathology in pathologies:
            mask = csv["labels_automatic"].str.contains(pathology.lower())
            if pathology in mapping:
                for syn in mapping[pathology]:
                    mask |= csv["labels_automatic"].str.contains(syn.lower())
            gt.append(mask.values)

        gt = np.asarray(gt).T
        gt = gt.astype(np.float32)

        # Rename pathologies
        pathologies = np.char.replace(pathologies, "Opacity", "Lung Opacity")
        pathologies = np.char.replace(pathologies, "Lesion", "Lung Lesion")

        ## Rename by myself
        pathologies = np.char.replace(pathologies, "Pleural_Thickening", "pleural thickening")
        pathologies = np.char.replace(pathologies, "Infiltration", "Infiltrate")
        pathologies = np.char.replace(pathologies, "Atelectasis", "Atelectases")
        pathologies = pathologies[:-1]
        gt[np.where(np.sum(gt, axis=1) == 0), -1] = 1
        label = gt[:, :-1]

        delete_index = []
        for key, value in delete_dict.items():
            if "openi" or "OpenI" in key:
                delete_index = value
        label = [label[i] for i in range(len(label)) if i not in delete_index]

        label = np.array(label)

    predict = res_process(predict_result)

    if args.evaluation_method == "PNC":
        num_rows, num_cols = predict.shape
        result = np.zeros((num_rows, num_cols // 2))
        for i in range(0, num_cols, 2):
            col1 = predict[:, i]
            col2 = predict[:, i + 1]
            new_col = col1 / (col1 + col2)
            result[:, i // 2] = new_col
        predict = result

    key = [i for i in range(len(pathologies))]
    # AUC, F1, MCC, mAP = calculate_scores(predict, label, key)
    # return AUC, F1, MCC, mAP
    auc_list, f1_list, mcc_list, map_list = calculate_scores_by_class(predict, label, key)
    return auc_list, f1_list, mcc_list, map_list

def triple_Chexpert14_result(predict_result, label_file_path, args):
    csv_head = [
        "path",
        "Atelectasis",
        "Cardiomegaly",
        "Effusion",
        "Infiltration",
        "Lung Mass",
        "Lung Nodule",
        "Pneumonia",
        "Pneumothorax",
        "Consolidation",
        "Edema",
        "Emphysema",
        "Fibrosis",
        "Pleural Thickening",
        "Hernia",
    ]
    df_test = pd.read_csv(label_file_path, sep=" ", names=csv_head)
    key = csv_head[1:]
    label = df_test[key].values

    predict = res_process(predict_result)

    if args.evaluation_method == "PNC":
        num_rows, num_cols = predict.shape
        result = np.zeros((num_rows, num_cols // 2))
        for i in range(0, num_cols, 2):
            col1 = predict[:, i]
            col2 = predict[:, i + 1]
            new_col = col1 / (col1 + col2)
            result[:, i // 2] = new_col
        predict = result

    # auc, f1, mcc, map = calculate_scores(predict, label, key)
    # return auc, f1, mcc, map
    auc_list, f1_list, mcc_list, map_list = calculate_scores_by_class(predict, label, key)
    return auc_list, f1_list, mcc_list, map_list



def triple_ChestXDet10_result(predict_result, label_file_path, args):
    with open(label_file_path, "r") as f:
        data = json.load(f)
    all_path = []
    all_label = []
    for d in data:
        all_path.append(d["file_name"])
        all_label.append(d["syms"])
    if 'ckpt_dir' in args:
        sorted_strings = [
            "Atelectasis",
            "Consolidation",
            "Effusion",
            "Emphysema",
            "Fibrosis",
            "Mass",
            "Nodule",
            "Pneumothorax",
        ]
    else:
        sorted_strings = [
            "Atelectasis",
            "Calcification",
            "Consolidation",
            "Effusion",
            "Emphysema",
            "Fibrosis",
            "Fracture",
            "Mass",
            "Nodule",
            "Pneumothorax",
        ]
    mlb = MultiLabelBinarizer(classes=sorted_strings)
    label = mlb.fit_transform(all_label)
    label = np.asarray(label)

    predict = res_process(predict_result)
    if 'ckpt_dir' in args: # finetune inference
        predict = np.delete(predict, 1, axis=1)
        predict = np.delete(predict, 5, axis=1)

    if args.evaluation_method == "PNC":
        num_rows, num_cols = predict.shape
        result = np.zeros((num_rows, num_cols // 2))
        for i in range(0, num_cols, 2):
            col1 = predict[:, i]
            col2 = predict[:, i + 1]
            new_col = col1 / (col1 + col2)
            result[:, i // 2] = new_col
        predict = result

    # auc, f1, mcc, map = calculate_scores(predict, label, sorted_strings)
    # return auc, f1, mcc, map
    auc_list, f1_list, mcc_list, map_list = calculate_scores_by_class(predict, label, sorted_strings)
    return auc_list, f1_list, mcc_list, map_list


def triple_Chexpert5_result(predict_result, label_file_path, args):
    key = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]

    df_test = pd.read_csv(label_file_path)

    label = df_test[key].values

    predict = res_process(predict_result)

    if args.evaluation_method == "PNC":
        num_rows, num_cols = predict.shape
        result = np.zeros((num_rows, num_cols // 2))
        for i in range(0, num_cols, 2):
            col1 = predict[:, i]
            col2 = predict[:, i + 1]
            new_col = col1 / (col1 + col2)
            result[:, i // 2] = new_col
        predict = result

    pre = np.zeros((predict.shape[0], predict.shape[1]))
    for i in range(predict.shape[0]):
        logit = predict[i]
        ind = np.argmax(logit)
        pre[i, ind] = 1

    # auc, f1, mcc, map = calculate_scores(predict, label, key)
    # return auc, f1, mcc, map
    auc_list, f1_list, mcc_list, map_list = calculate_scores_by_class(predict, label, key)
    return auc_list, f1_list, mcc_list, map_list


def tripple_padchest_rusult_merge(predict_result, label_file_path, delete_dict, args):

    predict = res_process(predict_result)

    if args.evaluation_method == "PNC":
        num_rows, num_cols = predict.shape
        result = np.zeros((num_rows, num_cols // 2))
        for i in range(0, num_cols, 2):
            col1 = predict[:, i]
            col2 = predict[:, i + 1]
            new_col = col1 / (col1 + col2)
            result[:, i // 2] = new_col
        predict = result

    with open(label_file_path, "r") as file:
        data = json.load(file)
    label = []
    key = data.keys()
    for k in key:
        label += data[k]
    unique_label = list(set(label))

    sorted_strings = sorted(unique_label, key=lambda x: (x, label.index(x)))

    index = sorted_strings.index("normal")

    labels = [data[k] for k in key]

    # 创建MultiLabelBinarizer对象
    mlb = MultiLabelBinarizer(classes=sorted_strings)

    # 使用fit_transform()方法进行One-Hot编码
    encoded_labels = mlb.fit_transform(labels)

    encoded_labels = np.delete(encoded_labels, index, axis=1)
    # 删除normal
    sorted_strings.remove("normal")

    delete_index = []
    for _, value in delete_dict.items():
        if 'padchest' in _:
            delete_index = list(set(value))
    encoded_labels = [encoded_labels[i] for i in range(len(encoded_labels)) if i not in delete_index]
    encoded_labels = np.array(encoded_labels)

    # encoded_labels = pd.read_csv(label_file_path)
    # encoded_labels = np.array(encoded_labels.replace(np.nan, "-1"))

    # Notification: 由于标签类别特别大，随机筛选的测试数据中可能不包含全部类被的正例，在计算AUC时需要删除该类别
    invalid_columns = [i for i in range(encoded_labels.shape[1]) if np.unique(encoded_labels[:, i]).size == 1]
    filtered_label = np.delete(encoded_labels, invalid_columns, axis=1)
    filtered_predict = np.delete(predict, invalid_columns, axis=1)

    auc, f1, mcc, map = calculate_scores(filtered_predict, filtered_label, [0]*(192 - len(invalid_columns)))


    # finetune分类时不测试 Padchest20 上的指标
    if hasattr(args, 'ckpt_dir') and 'classification' not in args.ckpt_dir:

        encoded_labels = np.array(encoded_labels)
        predict = np.array(predict)
        n_classes = encoded_labels.shape[1]
        tail_classes = []
        predict_20 = []
        label_20 = []

        for i in range(n_classes):
            # 计算每个类别的正例数目
            positive_count = np.sum(encoded_labels[:, i])

            # 如果正例数目少于10，这是一个tail类别
            if positive_count <= 10 and positive_count > 0:
                tail_classes.append(i)

                # add
                predict_20.append(predict[:, i])
                label_20.append(encoded_labels[:, i])

        print('')
        print("##### Padchest20 #####")

        auc_20, f1_20, mcc_20, map_20 = calculate_scores(np.array(predict_20).T, np.array(label_20).T, [0]*20)

        return auc, f1, mcc, map, auc_20, f1_20, mcc_20, map_20

    else:
        return auc, f1, mcc, map, None, None, None, None