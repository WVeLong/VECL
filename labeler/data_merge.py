import pickle
import os

pickle_path = '/hdd/wuwl/project/VECL/Dataset/MIMIC/chexper_labeler/'
path2sent_merge, path2label_merge = {}, {}
for item in os.listdir(pickle_path):  # 列出路径下的所有内容
    if 'output' in item:
        item_path = os.path.join(pickle_path, item)  # 拼接完整路径
        with open(item_path, "rb") as file:  # 使用二进制读模式
            path2sent, path2label = pickle.load(file)  # 反序列化数据
            for key, val in path2sent.items():
                path2sent_merge[key] = val
            for key, val in path2label.items():
                path2label_merge[key] = val

for key, val in path2sent_merge.items():
    if len(val) != len(path2label_merge[key]):
        raise ValueError

pickle_path = '/hdd/wuwl/project/VECL/Dataset/MIMIC/captions_drop_0_num_24_.pickle'
with open(pickle_path, "rb") as file:  # 使用二进制读模式
    path2sent, path2label, to_remove = pickle.load(file)

pickle_path = '/hdd/wuwl/project/VECL/Dataset/MIMIC/chexper_labeler/chexpert_labeler_final.pickle'
with open(pickle_path, "wb") as f:
    pickle.dump([path2sent_merge, path2label_merge, to_remove], f, protocol=2)
    print("Save to: ", pickle_path)