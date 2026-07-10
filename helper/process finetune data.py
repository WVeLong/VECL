from VECL.paths import dataset_path, remap_image_path
import os
import pandas as pd
from tqdm import tqdm
from VECL.constants import *
from PIL import Image
import cv2

def process_image(path_item, label_item):
    path_item = path_item.replace(
        "/defaultShare/PadChest/manualset/",
        str(dataset_path("PadChest", "images")) + "/",
    )
    if os.path.exists(path_item):
        try:
            img = cv2.imread(path_item)
            if img is not None:  # 确保读取的图像有效
                return path_item, label_item
        except cv2.error:
            pass
    return None, None

if __name__ == '__main__':

    # # test set data
    # data = pd.read_csv(ChestXray14_TRAIN_DATA, sep=' ', names=[ChestXray14_PATH_COL] + ChestXray14_pathologies)
    # label = data[ChestXray14_pathologies].values
    # df = data[ChestXray14_PATH_COL].values
    # label = pd.DataFrame(label, columns=ChestXray14_pathologies)
    # df = pd.DataFrame(df, columns=[ChestXray14_PATH_COL]).values.tolist()
    # new_df = []
    # for item in df:
    #     temp = item[0][11:]
    #     new_df.append(temp)
    #
    # # get train set file
    # train_files = []
    # train_files_abs = []
    # train_labels = []
    # for item in tqdm(os.listdir(directory),desc='process file path'):
    #     if item not in new_df:
    #         train_files.append(item)
    #
    # ChestXray14_pathologies_new = [
    #     'Atelectasis',
    #     'Cardiomegaly',
    #     'Effusion',
    #     'Infiltration',
    #     'Mass',
    #     'Nodule',
    #     'Pneumonia',
    #     'Pneumothorax',
    #     'Consolidation',
    #     'Edema',
    #     'Emphysema',
    #     'Fibrosis',
    #     'Pleural_Thickening',
    #     'Hernia'
    # ]
    #
    # # get train set label
    # index = data['Image Index'].values.tolist()
    # Finding_Labels = data['Finding Labels'].values.tolist()
    # for i in tqdm(range(len(data)), desc='process label'):
    #     index_item = index[i]
    #     if index_item in train_files:
    #         Finding_Labels_item = Finding_Labels[i].split('|')
    #         label_item = [0] * len(ChestXray14_pathologies)
    #         for Finding_Label_item in Finding_Labels_item:
    #             if Finding_Label_item in ChestXray14_pathologies_new:
    #                 label_index = ChestXray14_pathologies_new.index(Finding_Label_item)
    #                 label_item[label_index] = 1
    #         train_labels.append(label_item)
    #
    # df = pd.DataFrame({'file': train_files_abs, 'label': train_labels})
    # df.columns = ['file', 'label']
    # pd.DataFrame(df).to_csv(output_path, index=False, header=False)



    # key = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
    #
    # # read in csv file
    #
    # # prepare data
    # df['Path'] = df['Path'].apply(
    # test_label = df[key]
    # train_label = df_train[key].fillna(0)
    # train_input = df_train['Path'].values.tolist()
    # train_input_abs = []
    #
    # for item in tqdm(train_input, desc='process file path'):
    #     item = item[len('CheXpert-v1.0/train'):]
    #     append = False
    #     for prefix in prefixs:
    #         if os.path.exists(prefix + item):
    #             train_input_abs.append(prefix + item)
    #             append = True
    #     if append == False:
    #         raise ValueError
    #
    # df = pd.DataFrame({'Path': train_input_abs})
    # df.columns = ['Path']
    # pd.DataFrame(df).to_csv(output_path, index=False)


    import json
    from sklearn.preprocessing import MultiLabelBinarizer
    import numpy as np
    import cv2
    import concurrent.futures

    # # read label and data
    # df = pd.read_csv(image_path)
    # df['Path'] = df['Path'].apply(lambda x: x.replace("/defaultShare/PadChest/manualset/",
    #
    # label_image = []
    # with open(label_file, 'r') as f:
    #     data = json.load(f)
    #
    # for k, v in data.items():
    #     label_image.append(v)
    #
    # label = []
    # key = data.keys()
    # for k in key:
    #     label += data[k]
    # unique_label = list(set(label))
    # sorted_strings = sorted(unique_label, key=lambda x: (x, label.index(x)))
    # index = sorted_strings.index("normal")
    #
    # oral_texts = []
    # with open(label_text, 'r') as f:
    #     data_text = json.load(f)
    #
    # for k, v in data_text.items():
    #     oral_texts.append(v[0])
    #
    # mlb = MultiLabelBinarizer(classes=sorted_strings)
    # label = mlb.fit_transform(label_image)
    #
    # # remove normal
    # label = np.delete(label, index, axis=1)
    # sorted_strings.remove("normal")

    label_file_path = dataset_path('PadChest', 'manual_image.json')

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
    labels = mlb.fit_transform(labels)
    labels = np.delete(labels, index, axis=1)
    sorted_strings.remove("normal")


    image_path = "./Dataset/PadChest/padchest_multi_label_image.csv"
    df = pd.read_csv(image_path)


    # remove nan data
    path_list = df['Path'].values.tolist()
    path_new = []
    label_new = []
    cur_error = 0
    # 使用并行处理加速图像读取
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_image, path_list, labels), total=len(path_list), desc='remove nan data'))

        # 将有效图像和标签添加到新列表
        for path_item, label_item in results:
            if path_item is not None:
                path_new.append(path_item)
                label_new.append(label_item)
            else:
                cur_error += 1
        print(cur_error)

    output_path = dataset_path('PadChest', 'train_input.csv')
    df = pd.DataFrame(path_new)
    df.columns = ['Path']
    pd.DataFrame(df).to_csv(output_path, index=False)

    output_path = dataset_path('PadChest', 'train_output.csv')
    df = pd.DataFrame(label_new)
    pd.DataFrame(df).to_csv(output_path, index=False, header=False)
