import pickle
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from VECL.paths import dataset_path


def calculate_metrics_with_macro_average(y_true, y_pred):
    """
    计算每列的 precision、recall、f1 分数，并进行宏平均
    :param y_true: numpy.ndarray, 标签结果，形状为 (m, n)
    :param y_pred: numpy.ndarray, 预测结果，形状为 (m, n)
    :return: 字典，包含每列的各个类别指标及宏平均值
    """
    assert y_true.shape == y_pred.shape, "标签结果和预测结果的形状必须相同"

    num_columns = y_true.shape[1]
    classes = [1, -1, 0]  # 分类标签
    results = {}

    for col in range(num_columns):
        column_results = {}
        precisions, recalls, f1s = [], [], []

        for cls in classes:
            # 将当前类别设置为二分类问题 (cls vs rest)
            y_true_binary = (y_true[:, col] == cls).astype(int)
            y_pred_binary = (y_pred[:, col] == cls).astype(int)

            precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
            recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
            f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

            column_results[cls] = {
                "precision": precision,
                "recall": recall,
                "f1": f1
            }

        # 计算宏平均
        column_results["macro_avg"] = {
            "precision": np.mean(precisions),
            "recall": np.mean(recalls),
            "f1": np.mean(f1s)
        }

        results[f"Column {col + 1}"] = column_results

    return results


if __name__ == '__main__':


    filepath = dataset_path('MIMIC', 'captions_drop_0_num_24_.pickle')

    with open(filepath, "rb") as f:
        print(f"Loading captions from {filepath}")
        path2sent, path2label, to_remove = pickle.load(f)

    report_list = []
    llm_label_list = []
    for key, val in tqdm(path2sent.items()):
        i = len(val)
        for I in range(i):
            report_list.append(val[I])
    for key, val in tqdm(path2label.items()):
        i = len(val)
        for I in range(i):
            label = val[I]
            llm_label_list.append(val[I])

    report_list = report_list[:10000]
    llm_label_list = llm_label_list[:10000]

    key = ['25', '17', '4', '19', '5', '8', '15', '6', '1', '3', '2', '18', '16', '20']
    abs_llm_label_list = []
    for item in tqdm(llm_label_list):
        label_item = [0] * len(key)
        item_list = item.split(',')
        for l in item_list:
            l = l.strip()
            if '+' or '-' in l:
                symbol = l[-1]
                num = l[:-1]
                if num not in key:
                    continue
                else:
                    index = key.index(num)
                    if symbol == '+':
                        label_item[index] = 1
                    else:
                        label_item[index] = -1
            else:
                symbol = 0
                num = '25'
                label_item[0] = 1
        abs_llm_label_list.append(label_item)

    CheXlabel_1_path = str(dataset_path('CheXlabeler', 'oral_report_sentences_3000.csv'))
    df_1 = pd.read_csv(CheXlabel_1_path).fillna(0).iloc[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]].values.tolist()
    CheXlabel_2_path = str(dataset_path('CheXlabeler', 'oral_report_sentences_6000.csv'))
    df_2 = pd.read_csv(CheXlabel_2_path).fillna(0).iloc[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]].iloc[1:].values.tolist()
    CheXlabel_3_path = str(dataset_path('CheXlabeler', 'oral_report_sentences_10000.csv'))
    df_3 = pd.read_csv(CheXlabel_3_path).fillna(0).iloc[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]].iloc[1:].values.tolist()

    df_labler = df_1 + df_2 + df_3
    df_labler = df_labler[:10000]


    # 计算每列的指标及宏平均
    metrics_per_column = calculate_metrics_with_macro_average(np.array(df_labler), np.array(abs_llm_label_list))

    # 打印结果
    for column, metrics in metrics_per_column.items():
        print(f"{column}:")
        for cls, scores in metrics.items():
            if cls == "macro_avg":
                print(
                    f"  Macro Average: Precision={scores['precision']:.2f}, Recall={scores['recall']:.2f}, F1={scores['f1']:.2f}")
            else:
                print(
                    f"  Class {cls}: Precision={scores['precision']:.2f}, Recall={scores['recall']:.2f}, F1={scores['f1']:.2f}")