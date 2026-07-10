"""Entry-point script to label radiology reports."""
import pandas as pd
from args import ArgParser
from loader import Loader
from stages import Extractor, Classifier, Aggregator
from constants import *
import pickle
from tqdm import tqdm
import os
import math



def write(reports, labels, output_path, verbose=False):
    """Write labeled reports to specified path."""
    labeled_reports = pd.DataFrame({REPORTS: reports})
    for index, category in enumerate(CATEGORIES):
        labeled_reports[category] = labels[:, index]

    if verbose:
        print(f"Writing reports and labels to {output_path}.")
    labeled_reports[[REPORTS] + CATEGORIES].to_csv(output_path, index=False)

def split_dict_evenly(input_dict, num_subdicts):
    """
    将字典均匀划分成指定数量的子字典

    :param input_dict: 需要划分的字典
    :param num_subdicts: 子字典的数量
    :return: 子字典列表
    """
    # 计算每个子字典应包含的元素数量
    avg_len = len(input_dict) // num_subdicts
    remainder = len(input_dict) % num_subdicts

    # 初始化子字典列表
    subdicts = []

    # 初始化子字典的起始索引
    start_idx = 0

    # 遍历生成子字典
    for i in range(num_subdicts):
        # 计算当前子字典的结束索引
        end_idx = start_idx + avg_len + (1 if i < remainder else 0)

        # 根据索引范围获取子字典中的键值对
        subdict = dict(list(input_dict.items())[start_idx:end_idx])

        # 将子字典添加到列表中
        subdicts.append(subdict)

        # 更新起始索引
        start_idx = end_idx

    return subdicts

class_map = {
    0: '25',
    1: '17',
    2: '4',
    3: '5',
    4: '19',
    5: '8',
    6: '15',
    7: '6',
    8: '1',
    9: '3',
    10: '2',
    11: '18',
    12: '16',
    13: '20'
}

state_map = {
    1.0: '+',
    0.0: '-',
}

def process_label(labels):
    new_labels = []
    for label in labels:
        new_label = []
        for index, label_item in enumerate(label.tolist()):
            if not math.isnan(label_item) and label_item != -1.0 and type(label_item) != str:
                label_class = class_map[index]
                label_state = state_map[label_item]
                if label_class != '25':
                    new_label.append(label_class + label_state)
                else:
                    new_label.append(label_class)
        if not new_label:
            new_labels.append('25')
        else:
            if new_label != ['25'] and '25' in new_label:
                new_label.remove('25')
            new_labels.append(", ".join(new_label))
    return new_labels


def label(args):
    """Label the provided report(s)."""

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

    filename = f'chexpert_labeler_{args.process}.pickle'
    file_path = os.path.join(args.reports_path, filename)
    with open(file_path, 'rb') as file:
        path2sent, path2label = pickle.load(file)

    for path, sents in tqdm(path2sent.items(), desc=f'Labeling Data | process = {args.process}'):

        # Load reports in place.
        loader.load(sents)
        # Extract observation mentions in place.
        extractor.extract(loader.collection)
        # Classify mentions in place.
        classifier.classify(loader.collection)
        # Aggregate mentions to obtain one set of labels for each report.
        labels = aggregator.aggregate(loader.collection)

        # todo label process
        new_labels = process_label(labels)
        path2label[path] = new_labels

    filename = f'chexpert_labeler_output_{args.process}.pickle'
    output_path = os.path.join(args.reports_path, filename)
    with open(output_path, "wb") as f:
        pickle.dump([path2sent, path2label], f, protocol=2)
    print("Save to: ", output_path)


if __name__ == "__main__":
    parser = ArgParser()
    args = parser.parse_args()

    if not os.listdir('/hdd/wuwl/project/VECL/Dataset/MIMIC/chexper_labeler/'):
        with open(args.reports_path, 'rb') as file:
            path2sent, path2label, to_remove = pickle.load(file)
        num_subdicts = 10
        subdicts_sent = split_dict_evenly(path2sent, num_subdicts)
        subdicts_label = split_dict_evenly(path2label, num_subdicts)
        for i, (sent, label) in enumerate(zip(subdicts_sent, subdicts_label)):
            filename = f'chexpert_labeler_{i + 1}.pickle'
            output_path = os.path.join(args.reports_path, filename)
            with open(output_path, "wb") as f:
                pickle.dump([sent, label], f, protocol=2)

    label(args)
