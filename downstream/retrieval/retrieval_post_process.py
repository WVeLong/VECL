import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import pickle
import argparse
import pandas as pd
from tqdm import tqdm
import Levenshtein
import os
from VECL.paths import output_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_path', type=str, default=None)
    parser.add_argument('--MIMIC_data_path', type=str, default=None)
    parser.add_argument('--report_corpus_path', type=str, default=None)
    parser.add_argument('--num_chunks', type=int, default=None)
    args = parser.parse_args()
    return args

def split_list_into_chunks(lst, num_chunks):
    chunk_size = len(lst) // num_chunks
    chunks = [lst[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)]
    remainder = len(lst) % num_chunks
    for i in range(remainder):
        chunks[i].append(lst[num_chunks * chunk_size + i])
    return chunks


def match_report(A, B):
    result = [[] for _ in range(len(A))]
    for i in tqdm(range(len(A)), desc="Retrieving"):
        sublist = A[i]
        min_distance = float('inf')
        best_match = 'error'
        sublist_str = ' '.join(sublist)
        for string in B:
            distance = Levenshtein.distance(sublist_str, string)
            if distance < min_distance:
                min_distance = distance
                best_match = string
        result[i].append(best_match)
    return result


if __name__ == '__main__':

    args = parse_args()

    report_corpus = pd.read_csv(args.MIMIC_data_path)['Report Impression'].tolist()

    report_corpus_chunk = split_list_into_chunks(report_corpus, args.num_chunks)
    chunk_len_list = [len(s) for s in report_corpus_chunk]
    del report_corpus_chunk

    SimR_all = []
    for i in range(args.num_chunks):
        pickle_file = output_path('retrieval_based_report_generation', 'new', f'split_{i+1}.pkl')
        with open(pickle_file, 'rb') as file:
            data = pickle.load(file)
        if i == 0:
            SimR_all = data
        else:
            for idx in range(len(data)):
                best_key = list(SimR_all[idx].keys())[0]
                best_value = SimR_all[idx][best_key]
                new_key = list(data[idx].keys())[0]
                new_value = data[idx][new_key]
                if new_value > best_value:
                    abs_index_key = new_key + sum(chunk_len_list[:i])
                    SimR_all[idx] = {abs_index_key: new_value}

    retrieval_sent_list = []
    for idx in range(len(SimR_all)):
        best_key = list(SimR_all[idx].keys())[0]
        retrieval_sent_list.append(report_corpus[best_key])

    retrieval_report_path = output_path('retrieval_based_report_generation', 'retrieval_report.csv')
    retrieval_report_path.parent.mkdir(parents=True, exist_ok=True)
    retrieval_report = pd.DataFrame(retrieval_sent_list)
    retrieval_report.to_csv(retrieval_report_path, index=False)
    print('检索完成，retrieval report 已保存')