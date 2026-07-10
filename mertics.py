import numpy as np
import pandas as pd
from typing import List
from sklearn.metrics import matthews_corrcoef, roc_curve
from sklearn.utils import resample
from scipy.interpolate import UnivariateSpline
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from tqdm import tqdm


def calculate_best_f1(pred, label):
    best_f1 = 0
    best_threshold = 0
    thresholds = np.arange(0, 1.01, 0.01)
    for threshold in thresholds:
        pred_binary = [1 if p >= threshold else 0 for p in pred]
        f1 = f1_score(label, pred_binary)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return best_f1, best_threshold

def calculate_map(predict, label):
    aps = []
    num_classes = predict.shape[1]
    for i in range(num_classes):
        ap = average_precision_score(label[:, i], predict[:, i])
        aps.append(ap)
    return np.mean(aps)

def calculate_scores(predict, label, key):
    if isinstance(label, list):
        label = np.array(label)
    if isinstance(predict, list):
        predict = np.array(predict)
    # get best thresholds
    best_p_vals = get_best_p_vals(predict, label, key)

    # AUC computation
    macro_auc = roc_auc_score(label, predict, average="macro")
    print(f"AUC: {round(macro_auc, 4)}")

    # F1 computation
    Best_F1_list = []
    Threshold_list = []
    for pred, lab in zip(zip(*predict), zip(*label)):
        best_f1, best_threshold = calculate_best_f1(pred, lab)
        Best_F1_list.append(best_f1)
        Threshold_list.append(best_threshold)
    AVE_F1 = sum(Best_F1_list) / len(Best_F1_list)
    print('F1:', round(AVE_F1, 4))

    # MCC computation
    mcc_cis = compute_mcc(predict, label, key, best_p_vals)
    print(f"MCC: {round(mcc_cis.iloc[0, -1], 4)}")

    # mAP computation
    map_score = calculate_map(predict, label)
    print(f"mAP: {round(map_score, 4)}")

    print('')

    return round(macro_auc, 4), round(AVE_F1, 4), round(mcc_cis.iloc[0, -1], 4), round(map_score, 4)

def calculate_scores_by_class(predict, label, key):
    if isinstance(label, list):
        label = np.array(label)
    if isinstance(predict, list):
        predict = np.array(predict)

        # 初始化结果列表
    auc_list = []
    f1_list = []
    mcc_list = []
    map_list = []

    # 获取最佳阈值
    best_p_vals = get_best_p_vals(predict, label, key)

    # 按类别计算指标
    for i, class_name in enumerate(key):
        # 提取当前类别的预测和标签
        pred_class = predict[:, i]
        label_class = label[:, i]

        # AUC计算
        try:
            auc = roc_auc_score(label_class, pred_class)
        except ValueError:
            auc = np.nan  # 如果类别中没有正样本或负样本，AUC无法计算
        auc_list.append(auc)
        print(f"{class_name} AUC: {round(auc, 4)}")

        # F1计算
        best_f1, best_threshold = calculate_best_f1(pred_class, label_class)
        f1_list.append(best_f1)
        print(f"{class_name} F1: {round(best_f1, 4)}")

        # MCC计算
        mcc = matthews_corrcoef(label_class, (pred_class >= best_p_vals[class_name]).astype(int))
        mcc_list.append(mcc)
        print(f"{class_name} MCC: {round(mcc, 4)}")

        # mAP计算
        try:
            map_score = average_precision_score(label_class, pred_class)
        except ValueError:
            map_score = np.nan  # 如果类别中没有正样本，mAP无法计算
        map_list.append(map_score)
        print(f"{class_name} mAP: {round(map_score, 4)}")

        print('')

    return auc_list, f1_list, mcc_list, map_list

def get_best_p_vals(pred, groundtruth, cxr_labels, metric_func=matthews_corrcoef, spline_k: int = None, verbose: bool = False):
    """
    WARNING: CXR_LABELS must 
    Params: 
    * pred : np arr
        probabilities output by model

    * plot_graphs : bool
        if True, will save plots for metric vs. threshold for 
        each pathology
        
    Note: 
    * `probabilities` value is a linspace of possible probabilities
    """
    probabilities = [val for val in np.arange(0.4, 0.64, 0.0001)]
    best_p_vals = dict()
    for idx, cxr_label in enumerate(cxr_labels):
        y_true = groundtruth[:, idx]
        _, _, probabilities = roc_curve(y_true, pred[:, idx])
        probabilities = probabilities[1:]
        probabilities.sort()
        
        metrics_list = []
        for p in probabilities:
            y_pred = np.where(pred[:, idx] < p, 0, 1)
            metric = metric_func(y_true, y_pred)
            metrics_list.append(metric)
        
        if spline_k is not None: 
            try:
                spl = UnivariateSpline(probabilities, metrics_list, k=spline_k)
                spl_y = spl(probabilities)
                # get optimal thresholds on the spline and on the val_metric_list
                best_index = np.argmax(spl_y)
            except: 
                best_index = np.argmax(metrics_list)
        else:
            best_index = np.argmax(metrics_list)
        
        best_p = probabilities[best_index]
        best_metric = metrics_list[best_index]
        if verbose: 
            print("Best metric for {} is {}. threshold = {}.".format(cxr_label, best_metric, best_p))
        
        best_p_vals[cxr_label] = best_p
    return best_p_vals

def compute_mcc(y_pred: np.array, y_true: np.array, cxr_labels: List, thresholds: dict, label_idx_map: dict = None): 
    def get_mcc_bootstrap(y_pred, y_true, best_p_vals, cxr_labels=cxr_labels, label_idx_map=None):
        stats = {}
        probs = np.copy(y_pred)

        for idx, cxr_label in enumerate(cxr_labels):
            p = best_p_vals[cxr_label]
            probs[:,idx] = np.where(probs[:,idx] < p, 0, 1)

        clip_preds = np.copy(probs)

        for idx, cxr_label in enumerate(cxr_labels):
            if label_idx_map is None: 
                curr_y_true = y_true[:, idx]
            else: 
                curr_y_true = y_true[:, label_idx_map[cxr_label]]

            curr_y_pred = clip_preds[:, idx]
            stats[cxr_label] = [matthews_corrcoef(curr_y_true, curr_y_pred)]
        # compute mean over five major pathologies
        stats["Mean"] = compute_mean(stats, cxr_labels, is_df=False)
        return pd.DataFrame.from_dict(stats)
    
    boot_stats, mcc_cis = f1_mcc_bootstrap(y_pred, y_true, cxr_labels, thresholds, get_mcc_bootstrap, n_samples=1000, label_idx_map=label_idx_map)
    return mcc_cis

def f1_mcc_bootstrap(y_pred, y_true, cxr_labels, best_p_vals, eval_func, n_samples=5000, label_idx_map=None):
    '''
    This function will randomly sample with replacement
    from y_pred and y_true then evaluate `n` times
    and obtain AUROC scores for each.

    You can specify the number of samples that should be
    used with the `n_samples` parameter.

    Confidence intervals will be generated from each
    of the samples.
    '''
    y_pred # (500, 14)
    y_true # (500, 14)

    idx = np.arange(len(y_true))

    boot_stats = []
    for i in tqdm(range(n_samples), desc='Calculating best threshold'):
        sample = resample(idx, replace=True)
        y_pred_sample = y_pred[sample]
        y_true_sample = y_true[sample]

        sample_stats = eval_func(y_pred_sample, y_true_sample, best_p_vals, cxr_labels=cxr_labels, label_idx_map=label_idx_map)
        boot_stats.append(sample_stats)

    boot_stats = pd.concat(boot_stats) # pandas array of evaluations for each sample
    return boot_stats, compute_cis(boot_stats)

def compute_mean(stats, spec_labels, is_df=True):
    # spec_labels = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
    if is_df:
        spec_df = stats[spec_labels]
        res = np.mean(spec_df.iloc[0])
    else:
        # cis is df, within bootstrap
        vals = [stats[spec_label][0] for spec_label in spec_labels]
        res = np.mean(vals)
    return res

def compute_cis(data, confidence_level=0.05):
    """
    FUNCTION: compute_cis
    ------------------------------------------------------
    Given a Pandas dataframe of (n, labels), return another
    Pandas dataframe that is (3, labels).

    Each row is lower bound, mean, upper bound of a confidence
    interval with `confidence`.

    Args:
        * data - Pandas Dataframe, of shape (num_bootstrap_samples, num_labels)
        * confidence_level (optional) - confidence level of interval

    Returns:
        * Pandas Dataframe, of shape (3, labels), representing mean, lower, upper
    """
    data_columns = list(data)
    intervals = []
    for i in data_columns:
        series = data[i]
        sorted_perfs = series.sort_values()
        lower_index = int(confidence_level / 2 * len(sorted_perfs)) - 1
        upper_index = int((1 - confidence_level / 2) * len(sorted_perfs)) - 1
        lower = sorted_perfs.iloc[lower_index].round(4)
        upper = sorted_perfs.iloc[upper_index].round(4)
        mean = round(sorted_perfs.mean(), 4)
        interval = pd.DataFrame({i: [mean, lower, upper]})
        intervals.append(interval)
    intervals_df = pd.concat(intervals, axis=1)
    intervals_df.index = ['mean', 'lower', 'upper']
    return intervals_df