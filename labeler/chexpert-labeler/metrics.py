import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from args import ArgParser
from loader import Loader
from stages import Extractor, Classifier, Aggregator
from constants import CATEGORIES



def process_csv(path):
    df = pd.read_csv(path)
    # 去掉第一行的值
    df = df.drop(0)
    df = df.drop(columns=df.columns[0])
    df = df.reset_index(drop=True)
    return df

def compute_metrics(predicts, labels) -> dict:
    """
    计算chexpert-labeler输出的宏平均 Precision、Recall 和 F1 分数。

    参数:
    - pred_df: pandas.DataFrame，包含预测的标签。
    - label_df: pandas.DataFrame，包含真实的标签。

    返回:
    - metrics: dict，包含 'macro_precision'、'macro_recall' 和 'macro_f1'。
    """

    # 读取CSV文件到DataFrame，并填充缺失值为 -1
    pred_df = predicts.fillna(-1)
    label_df = labels.fillna(-1)
    pred_df.columns = label_df.columns

    # 确保预测和标签的列一致
    if list(pred_df.columns) != list(label_df.columns):
        raise ValueError("预测和标签的列不一致。请确保两者具有相同的标签列。")

    label_columns = label_df.columns.tolist()

    # 初始化列表来存储每个标签的指标
    precision_scores = []
    recall_scores = []
    f1_scores = []

    for label in label_columns:
        y_pred = pred_df[label]
        y_true = label_df[label]

        # 过滤掉真实标签为 -1 的样本（不确定）
        mask = y_true != -1
        y_pred_filtered = y_pred[mask]
        y_true_filtered = y_true[mask]

        # 如果过滤后没有样本，则跳过该标签
        if len(y_true_filtered) == 0:
            print(f"标签 '{label}' 中没有有效样本（所有样本均为不确定）。")
            continue

        # 将预测值和真实值中的 -1 替换为 0（否定）
        y_pred_binary = y_pred_filtered.replace(-1, 0)
        y_true_binary = y_true_filtered.replace(-1, 0)

        # 计算 Precision、Recall 和 F1 分数，忽略零除错误
        try:
            precision = precision_score(y_true_binary, y_pred_binary, average='binary', zero_division=0)
            recall = recall_score(y_true_binary, y_pred_binary, average='binary', zero_division=0)
            f1 = f1_score(y_true_binary, y_pred_binary, average='binary', zero_division=0)

            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
        except ValueError as e:
            print(f"标签 '{label}' 的指标计算失败: {e}")

    # 计算宏平均指标
    if not f1_scores:
        print("没有有效的标签用于计算宏平均指标。")
        return {"macro_precision": 0.0, "macro_recall": 0.0, "macro_f1": 0.0}

    macro_precision = round(sum(precision_scores) / len(precision_scores), 4)
    macro_recall = round(sum(recall_scores) / len(recall_scores), 4)
    macro_f1 = round(sum(f1_scores) / len(f1_scores), 4)

    metrics = {
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1
    }

    return metrics

def label_data(args):
    loader = Loader(args.reports_path,
                    args.sections_to_extract,
                    args.extract_strict)

    extractor = Extractor(args.mention_phrases_dir,
                          args.unmention_phrases_dir,
                          verbose=args.verbose)
    classifier = Classifier(args.pre_negation_uncertainty_path,
                            args.negation_path,
                            args.post_negation_uncertainty_path,
                            verbose=args.verbose)
    aggregator = Aggregator(CATEGORIES,
                            verbose=args.verbose)

    sents = pd.read_csv(args.reports_path)['0'].tolist()

    loader.load(sents)
    # Extract observation mentions in place.
    extractor.extract(loader.collection)
    # Classify mentions in place.
    classifier.classify(loader.collection)
    # Aggregate mentions to obtain one set of labels for each report.
    labels = aggregator.aggregate(loader.collection)

    return pd.DataFrame(labels)

if __name__ == '__main__':

    parser = ArgParser()
    args = parser.parse_args()
    predicts = label_data(args)

    test_report_path = 'labeled_reports.csv'
    labels = process_csv(test_report_path)

    metrics = compute_metrics(predicts, labels)

    print(metrics)