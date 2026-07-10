from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from awq import AutoAWQForCausalLM
import torch
import argparse
from tqdm import tqdm
import pandas as pd
import re
import ast
from LLM_prompt import *
import csv
import os
import warnings
warnings.filterwarnings("ignore")



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_id', type=int, default=1)
    parser.add_argument('--num_chunks', type=int, default=24)
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--use_AWQ_quant', type=bool, default=False)
    parser.add_argument('--report_sentences_path', type=str, default=None)
    parser.add_argument('--report_labels_path', type=str, default=None)
    args = parser.parse_args()
    return args

def load_model(args):
    config = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True).config
    if args.use_AWQ_quant:
        print(f"Loading model in AWQ quantization from {args.model_path} ...")
        model = AutoAWQForCausalLM.from_pretrained(
            args.model_path,
            device_map="auto",  # 自动分配到可用设备
            low_cpu_mem_usage=True,  # 减少 CPU 内存占用
            torch_dtype=torch.float16
        )
    else:
        print(f"Loading model in 4-bit quantization from {args.model_path} ...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,  # 使用 4-bit 量化
            bnb_4bit_quant_type="nf4",  # 量化类型为 nf4
            bnb_4bit_use_double_quant=True,  # 使用双量化
            bnb_4bit_compute_dtype=torch.bfloat16,  # 计算数据类型为 bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            config=config,
            device_map="auto",  # 自动分配到可用设备
            low_cpu_mem_usage=True,  # 减少 CPU 内存占用
            quantization_config=quantization_config,  # 传递量化配置
        )
    return model

def inference(messages, model, tokenizer):
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to('cuda')
    outputs = model.generate(
        input_ids,
        max_new_tokens=9999,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=False,  # 设置为 True 以启用采样模式
    )
    response = outputs[0][input_ids.shape[-1]:]
    response = tokenizer.decode(response, skip_special_tokens=True)
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    response = response.replace('\n', '')
    return response

def parse_string_to_list(string):
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

def split_list_into_chunks(lst, num_chunks):
    chunk_size = len(lst) // num_chunks
    chunks = [lst[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)]
    remainder = len(lst) % num_chunks
    for i in range(remainder):
        chunks[i].append(lst[num_chunks * chunk_size + i])
    return chunks

def append_to_csv(file_path, row):
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)



if __name__ == "__main__":

    args = parse_args()

    # 加载模型
    model = load_model(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # 加载数据并切分工作集
    report = pd.read_csv(args.report_sentences_path, header=None).values.tolist()
    report_corpus = []
    for sent_sample in report:
        report_corpus.append(parse_string_to_list(sent_sample[0]))

    report_corpus_chunk = split_list_into_chunks(report_corpus, args.num_chunks)
    report_corpus_work = report_corpus_chunk[args.split_id - 1]
    del report, report_corpus, report_corpus_chunk, sent_sample

    # 模型推理
    save_path = os.path.join(args.report_labels_path, f'{args.model_name}_{args.split_id}.csv')
    if os.path.exists(save_path):
        history = pd.read_csv(save_path, header=None)
        cache = len(history)
    else:
        cache = 0
    for idx in tqdm(range(len(report_corpus_work)), desc=f'当前子集: {args.split_id}'):
        if idx < cache:
            continue
        label_report = []
        report = report_corpus_work[idx]
        for sent in report:
            input = Category_Classify_Prompt(sent)
            output_category = inference(input, model, tokenizer)
            if '25' not in output_category:
                input = Status_Classify_Prompt(sent)
                output_status = inference(input, model, tokenizer)
                output_category = output_category.split(',')
                label_item = []
                for category in output_category:
                    label_item.append(category + output_status)
                label_item = ', '.join(label_item)
            else:
                label_item = '25'
            print(label_item)
            label_report.append(label_item)
        if len(label_report) < 59:
            for _ in range(59 - len(label_report)):
                label_report.append('0')
        append_to_csv((save_path), label_report)